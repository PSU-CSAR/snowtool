"""The INSTARR (SPIRES NRT) dataset definition: variables, grid spec, and ingest.

INSTARR is the SPIRES near-real-time snow-property product (Snow-Property
Inversion from Remote Sensing, from MOD09GA surface reflectance) distributed as
one NetCDF per MODIS tile per day, e.g.
``SPIRES_NRT_h09v04_MOD09GA061_<YYYYMMDD>_V1.0.nc``, holding nine snow variables
on the native **MODIS Sinusoidal** 463 m grid (2400x2400 per tile, projected).

The dataset grid is the native sinusoidal lattice, NOT a reprojection: a date's
tiles are mosaicked by simply dropping each into its tile slot (no resampling --
adjacent MODIS tiles abut exactly), so the values stay bit-exact at native
resolution. The grid currently covers the h08-h10 x v04-v05 tile block (7200x4800);
the missing h10v05 corner is just nodata. Because the mosaic is a lossless stitch
of the kept source tiles, widening the grid later is a re-ingest, not a reprocess.

Being projected, INSTARR has a constant cell area (``spec.cell_area``), which is
burned uniformly into every in-basin pixel of its AOI rasters.
"""

from __future__ import annotations

import re

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Self

import numpy
import rasterio
import shapely

from geojson_pydantic.geometries import Geometry
from pydantic import TypeAdapter

from snowtool.exceptions import IngestSourceError
from snowtool.snowdb.ingest import GridAlignedRaster, per_variable_ingest
from snowtool.snowdb.raster.cog import source_tags
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import date
    from pathlib import Path

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.ingest import DateIngest, GridGeometry, WritableRaster

# --- MODIS Sinusoidal grid constants ------------------------------------------

# The MODIS Sinusoidal projection (a sphere of radius 6371007.181 m); no EPSG
# code, so the grid CRS is given as WKT. is_geographic resolves to False and
# cell_area to the constant 463.31^2 m^2 (verified via griffine/pyproj).
MODIS_SINUSOIDAL_WKT = (
    'PROJCS["MODIS Sinusoidal",'
    'GEOGCS["Sphere",DATUM["unnamed",'
    'SPHEROID["unnamed",6371007.181,0]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)

# Canonical MODIS tiling: tile (h, v) upper-left is
# (X_MIN + h*TILE, Y_MAX - v*TILE); each tile is 2400x2400 cells.
_MODIS_X_MIN = -20015109.354
_MODIS_Y_MAX = 10007554.677
_MODIS_TILE_M = 1111950.5196666666
_MODIS_TILE_PX = 2400
_PX_SIZE = 463.3127165693847  # native cell size (m)

# The tile block the grid covers: h08-h10 (3 wide) x v04-v05 (2 tall).
_H_MIN, _H_MAX = 8, 10
_V_MIN, _V_MAX = 4, 5

# The h10v05 (SE) corner of the block is never ingested -- it is permanently
# nodata (see the module docstring). It sits inside the grid's bounding
# rectangle, so without leaving it out a basin over it would be reported as
# fully covered. Left out of the served footprint (a static grid fact, not a
# per-date data gap).
_EMPTY_TILES = frozenset({(10, 5)})


def _modis_tile_polygon(h: int, v: int) -> shapely.Polygon:
    """The MODIS sinusoidal extent of tile ``(h, v)`` as a shapely box."""
    ul_x = _MODIS_X_MIN + h * _MODIS_TILE_M
    ul_y = _MODIS_Y_MAX - v * _MODIS_TILE_M
    return shapely.box(ul_x, ul_y - _MODIS_TILE_M, ul_x + _MODIS_TILE_M, ul_y)


def _instarr_footprint() -> Geometry:
    """The region INSTARR serves: the tile block minus the empty corner.

    A single (multi)polygon in MODIS-sinusoidal coords -- the union of the
    populated tiles (every tile in the h08-h10 x v04-v05 block except the
    permanently-empty ones). This is the coverage footprint; absent it, coverage
    would default to the whole grid rectangle and mis-report basins over the empty
    corner.
    """
    present = [
        _modis_tile_polygon(h, v)
        for h in range(_H_MIN, _H_MAX + 1)
        for v in range(_V_MIN, _V_MAX + 1)
        if (h, v) not in _EMPTY_TILES
    ]
    # Validates the shapely union into the geojson-pydantic Geometry union (the
    # persisted footprint type; DatasetConfig.footprint holds the same).
    return TypeAdapter(Geometry).validate_python(
        shapely.geometry.mapping(shapely.union_all(present)),
    )


# --- INSTARR variables --------------------------------------------------------

# All nine SPIRES variables are intensive per-pixel quantities -> area-weighted
# MEAN (area is the constant sinusoidal cell area). Reads are the native uint8
# (255 nodata) or uint16 (65535 nodata) with no scale/offset. Ingest names each
# mosaic COG via `variable_out_name` (`<distilled-source-stem>__<key>.tif`); the
# glob matches that on the `__<key>` suffix (the doubled delimiter keeps
# `snow_fraction` from also matching `viewable_snow_fraction`).
_PERCENT = Unit(name='percent', scale_factor=1)  # %
_MICRON = Unit(name='um', scale_factor=1)
_PPM = Unit(name='ppm', scale_factor=1)
_DAY = Unit(name='day', scale_factor=1)
_W_PER_M2 = Unit(name='w_per_m2', scale_factor=1)  # W m-2

_U8_NODATA = 255.0
_U16_NODATA = 65535.0


def _variable(key: str, unit: Unit, dtype: str, nodata: float) -> DatasetVariable:
    return DatasetVariable(
        key=key,
        unit=unit,
        reducer=Reducer.MEAN,
        dtype=dtype,
        nodata=nodata,
        glob=f'*__{key}.tif',
    )


INSTARR_VARIABLES = (
    _variable('snow_fraction', _PERCENT, 'uint8', _U8_NODATA),
    _variable('viewable_snow_fraction', _PERCENT, 'uint8', _U8_NODATA),
    _variable('albedo_dirty_flat', _PERCENT, 'uint8', _U8_NODATA),
    _variable('albedo_dirty_terrain_corrected', _PERCENT, 'uint8', _U8_NODATA),
    _variable('deltavis', _PERCENT, 'uint8', _U8_NODATA),
    _variable('grain_size', _MICRON, 'uint16', _U16_NODATA),
    _variable('dust_concentration', _PPM, 'uint16', _U16_NODATA),
    _variable('snow_cover_duration', _DAY, 'uint16', _U16_NODATA),
    _variable('radiative_forcing', _W_PER_M2, 'uint16', _U16_NODATA),
)


# --- INSTARR ingest -----------------------------------------------------------


class InstarrMosaicRaster(GridAlignedRaster):
    """One variable's native-sinusoidal mosaic, ready to write as a COG.

    A :class:`~snowtool.snowdb.ingest.GridAlignedRaster`: it supplies the mosaicked
    array (:meth:`read_array`); the base owns the grid geometry + COG write. At read
    time it allocates the full grid array (filled with the variable's nodata), reads
    each of the date's tile bands for this variable, and drops each into its grid
    slot positioned by the tile's own sinusoidal origin -- a lossless stitch, no
    reprojection. Tiles absent for the date stay nodata. The mosaic is on-grid by
    construction, so the base's shape guard passes trivially (a free check).
    ``source_uris`` are GDAL-readable URIs (the ingester builds
    ``netcdf:<tile>:<variable>``), keeping the NetCDF-format knowledge in the
    ingester.
    """

    def __init__(
        self: Self,
        variable: DatasetVariable,
        source_uris: list[str],
        geometry: GridGeometry,
        *,
        out_name: str,
        tags: dict[str, str] | None = None,
    ) -> None:
        super().__init__(out_name, geometry, nodata=variable.nodata, tags=tags)
        self.variable = variable
        self.source_uris = source_uris

    def read_array(self: Self) -> numpy.ndarray:
        rows, cols = self.geometry.shape
        array = numpy.full(
            (rows, cols),
            self.variable.nodata,
            dtype=self.variable.dtype,
        )

        inverse = ~self.geometry.transform
        for source_uri in self.source_uris:
            with rasterio.open(source_uri) as src:
                # Place by the tile's own sinusoidal origin: the grid transform's
                # inverse maps the tile's upper-left world coord to grid (col, row).
                # round absorbs the sub-mm differences between the stored tile
                # geotransform and the canonical lattice.
                fcol, frow = inverse * (src.bounds.left, src.bounds.top)
                col_off, row_off = round(fcol), round(frow)
                data = src.read(1)
            rows, cols = data.shape
            array[row_off : row_off + rows, col_off : col_off + cols] = data

        return array


class InstarrIngester:
    """Parses a directory of SPIRES NRT tiles into per-date work for the driver.

    The source tiles are tile-major on disk (``h##v##/YYYY/MM/SPIRES_NRT_*.nc``), so
    a date's tiles are scattered; :meth:`plan` scans ``source`` for tiles, groups
    them by date, and yields one :class:`~snowtool.snowdb.ingest.DateIngest` per
    date whose ``build_rasters`` produces one mosaicked COG per variable.
    ``source`` may be a directory (scanned recursively) or a single tile file.
    """

    kind: ClassVar[str] = 'instarr'

    filename_re = re.compile(
        r'SPIRES_NRT_h(?P<h>\d\d)v(?P<v>\d\d)_(?P<collection>MOD09GA\d+)_'
        r'(?P<date>\d{8})_(?P<version>V[\d.]+)\.nc$',
    )

    def plan(
        self: Self,
        source: Path,
        dataset: Dataset,
    ) -> Iterator[DateIngest]:
        candidates = (
            sorted(source.glob('**/SPIRES_NRT_*.nc')) if source.is_dir() else [source]
        )

        # Parse each tile filename once and group the surviving matches by date.
        matches_by_date: dict[date, list[tuple[Path, re.Match[str]]]] = defaultdict(
            list,
        )
        for path in candidates:
            match = self.filename_re.search(path.name)
            if match is None:
                # Every glob-matched SPIRES_NRT_*.nc claims the format, so a file
                # the regex cannot parse is malformed input, not one to silently
                # drop from the mosaic.
                raise IngestSourceError(
                    f'Malformed SPIRES NRT tile filename {path.name!r} (expected '
                    f'{self.filename_re.pattern!r}).',
                )
            h, v = int(match['h']), int(match['v'])
            if not (_H_MIN <= h <= _H_MAX and _V_MIN <= v <= _V_MAX):
                # Placement is by computed offset (see InstarrMosaicRaster), so an
                # out-of-block tile would wrap to a negative slice and silently
                # land in the wrong grid slot instead of failing loudly.
                raise IngestSourceError(
                    f'SPIRES NRT tile h{h:02d}v{v:02d} ({path.name!r}) is outside the '
                    f'configured grid block h{_H_MIN:02d}-h{_H_MAX:02d} x '
                    f'v{_V_MIN:02d}-v{_V_MAX:02d}. Widening the grid is a config '
                    'change plus a re-ingest, not a mosaic-time surprise.',
                )
            tile_date = datetime.strptime(match['date'], '%Y%m%d').date()  # noqa: DTZ007
            matches_by_date[tile_date].append((path, match))

        if not matches_by_date:
            raise IngestSourceError(
                f'No SPIRES NRT tiles found under {source} (expected files like '
                "'SPIRES_NRT_h09v04_MOD09GA061_<YYYYMMDD>_V1.0.nc').",
            )

        geometry = dataset.grid_geometry
        for ingest_date, tile_matches in sorted(matches_by_date.items()):
            tile_paths = [path for path, _ in tile_matches]
            stem, collection, version = self._distilled_stem(tile_matches)
            # Filesystem-visible provenance is the distilled stem; the COG tags
            # carry the full record, including the exact contributing tiles.
            files_tag = ' '.join(sorted(p.name for p in tile_paths))

            def make_raster(
                variable: DatasetVariable,
                out_name: str,
                source_hash: str,
                *,
                tile_paths: list[Path] = tile_paths,
                files_tag: str = files_tag,
                collection: str = collection,
                version: str = version,
                ingest_date: date = ingest_date,
            ) -> WritableRaster:
                return InstarrMosaicRaster(
                    variable,
                    [f'netcdf:{tile}:{variable.key}' for tile in tile_paths],
                    geometry,
                    out_name=out_name,
                    tags=source_tags(
                        dataset=dataset.spec.name,
                        date=ingest_date,
                        variable=variable.key,
                        files=files_tag,
                        source_hash=source_hash,
                        extra={
                            'SOURCE_COLLECTION': collection,
                            'SOURCE_VERSION': version,
                        },
                    ),
                )

            yield per_variable_ingest(
                stem,
                ingest_date,
                tile_paths,
                dataset,
                make_raster,
            )

    def _distilled_stem(
        self: Self,
        tile_matches: list[tuple[Path, re.Match[str]]],
    ) -> tuple[str, str, str]:
        """The mosaic's provenance stem plus its ``(collection, version)``.

        Drops the per-tile ``h##v##`` (the output spans all the date's tiles) and
        keeps the fields shared across them, consuming the matches already parsed
        by :meth:`plan` (no re-parse). Refuses a date whose tiles disagree on
        collection or version -- that would mix products into one mosaic.
        """
        identities = {
            (match['collection'], match['date'], match['version'])
            for _, match in tile_matches
        }

        if len(identities) > 1:
            raise IngestSourceError(
                'INSTARR tiles for one date disagree on collection/version: '
                f'{sorted(identities)}',
            )

        collection, datestr, version = identities.pop()
        return f'SPIRES_NRT_{collection}_{datestr}_{version}', collection, version


# --- INSTARR spec -------------------------------------------------------------

INSTARR_SPEC = DatasetSpec(
    name='instarr',
    grid_params=GridParams(
        origin_x=_MODIS_X_MIN + _H_MIN * _MODIS_TILE_M,
        origin_y=_MODIS_Y_MAX - _V_MIN * _MODIS_TILE_M,
        px_size=_PX_SIZE,
        cols=(_H_MAX - _H_MIN + 1) * _MODIS_TILE_PX,
        rows=(_V_MAX - _V_MIN + 1) * _MODIS_TILE_PX,
        # 512 (not the 256 the geographic datasets use): at the native 463 m cell
        # this is a ~237 km tile edge, matching the ground footprint of a 256-cell
        # tile on the ~925 m SNODAS/SWANN grids, so read windows cover comparable
        # area across datasets.
        tile_size=512,
        crs=MODIS_SINUSOIDAL_WKT,
    ),
    variables=INSTARR_VARIABLES,
    ingester=InstarrIngester(),
    footprint=_instarr_footprint(),
)
