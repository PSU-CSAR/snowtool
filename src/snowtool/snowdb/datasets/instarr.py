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
from typing import TYPE_CHECKING, Self

import numpy
import rasterio
import shapely

from geojson_pydantic.geometries import Geometry
from pydantic import TypeAdapter

from snowtool.exceptions import SnowtoolError
from snowtool.snowdb.dataset import INGEST_FORMAT_VERSION
from snowtool.snowdb.ingest import IngestResult
from snowtool.snowdb.progress import NULL_PROGRESS
from snowtool.snowdb.provenance import hash_files, versioned_hash
from snowtool.snowdb.raster.cog import source_tags, write_cog_guarded
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from affine import Affine

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.progress import ProgressReporter

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

# Validates a GeoJSON geometry mapping into the geojson-pydantic Geometry union
# (the persisted footprint type; DatasetConfig.footprint holds the same).
_GEOMETRY_ADAPTER: TypeAdapter[Geometry] = TypeAdapter(Geometry)


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
    corner. Derived here from the canonical tiling for now; a future cut may
    compute it from the data's real extent. The shapely union is validated into
    the geojson-pydantic Geometry union (the persisted footprint type).
    """
    present = [
        _modis_tile_polygon(h, v)
        for h in range(_H_MIN, _H_MAX + 1)
        for v in range(_V_MIN, _V_MAX + 1)
        if (h, v) not in _EMPTY_TILES
    ]
    return _GEOMETRY_ADAPTER.validate_python(
        shapely.geometry.mapping(shapely.union_all(present)),
    )


# --- INSTARR variables --------------------------------------------------------

# All nine SPIRES variables are intensive per-pixel quantities -> area-weighted
# MEAN (area is the constant sinusoidal cell area). Reads are the native uint8
# (255 nodata) or uint16 (65535 nodata) with no scale/offset. Ingest names each
# mosaic COG `<distilled-source-stem>__<key>.tif`; the glob matches that on the
# `__<key>` suffix (the doubled delimiter keeps `snow_fraction` from also
# matching `viewable_snow_fraction`).
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


class InstarrMosaicRaster:
    """One variable's native-sinusoidal mosaic, ready to write as a COG.

    Implements the :class:`~snowtool.snowdb.ingest.WritableRaster` contract. At
    write time it allocates the full grid array (filled with the variable's
    nodata), reads each of the date's tile bands for this variable, and drops each
    into its grid slot positioned by the tile's own sinusoidal origin -- a lossless
    stitch, no reprojection. Tiles absent for the date stay nodata. ``source_uris``
    are GDAL-readable URIs (the ingester builds ``netcdf:<tile>:<variable>``),
    keeping the NetCDF-format knowledge in the ingester.
    """

    def __init__(
        self: Self,
        variable: DatasetVariable,
        source_uris: list[str],
        grid_params: GridParams,
        *,
        out_name: str,
        transform: Affine,
        crs: rasterio.crs.CRS,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.variable = variable
        self.source_uris = source_uris
        self.grid_params = grid_params
        self.out_name = out_name
        self.transform = transform
        self.crs = crs
        self.tags = tags

    def write_cog(self: Self, output_dir: Path, force: bool = False) -> None:
        gp = self.grid_params
        array = numpy.full(
            (gp.rows, gp.cols),
            self.variable.nodata,
            dtype=self.variable.dtype,
        )

        for source_uri in self.source_uris:
            with rasterio.open(source_uri) as src:
                # Place by the tile's own sinusoidal origin relative to the grid
                # origin; round absorbs the sub-mm differences between the stored
                # tile geotransform and the canonical lattice.
                col_off = round((src.bounds.left - gp.origin_x) / gp.px_size)
                row_off = round((gp.origin_y - src.bounds.top) / gp.px_size)
                data = src.read(1)
            rows, cols = data.shape
            array[row_off : row_off + rows, col_off : col_off + cols] = data

        write_cog_guarded(
            output_dir / self.out_name,
            array,
            force=force,
            transform=self.transform,
            crs=self.crs,
            nodata=self.variable.nodata,
            tile_size=gp.tile_size,
            tags=self.tags,
        )


class InstarrIngester:
    """Ingests a directory of SPIRES NRT tiles into a dataset.

    The INSTARR implementation of :class:`~snowtool.snowdb.ingest.Ingester`. The
    source tiles are tile-major on disk (``h##v##/YYYY/MM/SPIRES_NRT_*.nc``), so a
    date's tiles are scattered; this scans ``source`` for tiles, groups them by
    date, and writes one mosaicked COG per variable per date via the dataset's
    generic :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`. ``source``
    may be a directory (scanned recursively) or a single tile file.
    """

    filename_re = re.compile(
        r'SPIRES_NRT_h(?P<h>\d\d)v(?P<v>\d\d)_(?P<collection>MOD09GA\d+)_'
        r'(?P<date>\d{8})_(?P<version>V[\d.]+)\.nc$',
    )

    def ingest(
        self: Self,
        source: Path,
        dataset: Dataset,
        *,
        force: bool = False,
        progress: ProgressReporter = NULL_PROGRESS,
    ) -> IngestResult:
        candidates = (
            sorted(source.glob('**/SPIRES_NRT_*.nc')) if source.is_dir() else [source]
        )

        tiles_by_date: dict[date, list[Path]] = defaultdict(list)
        for path in candidates:
            match = self.filename_re.search(path.name)
            if match is None:
                continue
            tile_date = datetime.strptime(match['date'], '%Y%m%d').date()  # noqa: DTZ007
            tiles_by_date[tile_date].append(path)

        if not tiles_by_date:
            raise SnowtoolError(
                f'No SPIRES NRT tiles found under {source} (expected files like '
                "'SPIRES_NRT_h09v04_MOD09GA061_<YYYYMMDD>_V1.0.nc').",
            )

        grid_params = dataset.spec.grid_params
        transform = dataset.grid.base_grid.transform
        crs = dataset.grid_crs

        ingested: list[date] = []
        skipped: list[date] = []
        for ingest_date, tile_paths in sorted(tiles_by_date.items()):
            stem, collection, version = self._distilled_stem(tile_paths)
            # Filesystem-visible provenance is the distilled stem; the COG tags
            # carry the full record, including the exact contributing tiles.
            source_files = ' '.join(sorted(p.name for p in tile_paths))
            # One versioned hash per date over the date's sorted contributing tiles,
            # stamped on every COG and compared by the skip check.
            source_hash = versioned_hash(INGEST_FORMAT_VERSION, hash_files(tile_paths))
            rasters = [
                InstarrMosaicRaster(
                    variable,
                    [f'netcdf:{tile}:{variable.key}' for tile in tile_paths],
                    grid_params,
                    out_name=f'{stem}__{variable.key}.tif',
                    transform=transform,
                    crs=crs,
                    tags=source_tags(
                        dataset=dataset.spec.name,
                        date=ingest_date,
                        variable=variable.key,
                        files=source_files,
                        source_hash=source_hash,
                        extra={
                            'SOURCE_COLLECTION': collection,
                            'SOURCE_VERSION': version,
                        },
                    ),
                )
                for variable in dataset.spec.variables.values()
            ]
            wrote = dataset.write_date_cogs(
                ingest_date,
                rasters,
                source_hash=source_hash,
                force=force,
                progress=progress,
            )
            (ingested if wrote else skipped).append(ingest_date)

        return IngestResult(ingested=ingested, skipped=skipped)

    def _distilled_stem(self: Self, tile_paths: list[Path]) -> tuple[str, str, str]:
        """The mosaic's provenance stem plus its ``(collection, version)``.

        Drops the per-tile ``h##v##`` (the output spans all the date's tiles) and
        keeps the fields shared across them. Refuses a date whose tiles disagree
        on collection or version -- that would mix products into one mosaic.
        """
        identities = set()
        for path in tile_paths:
            match = self.filename_re.search(path.name)
            if match is not None:
                identities.add(
                    (match['collection'], match['date'], match['version']),
                )

        if len(identities) > 1:
            raise SnowtoolError(
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
