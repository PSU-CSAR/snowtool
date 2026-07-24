"""The SWANN 800m dataset definition: variables, grid spec, and ingest.

SWANN (University of Arizona "Snow Water Artificial Neural Network") 800m is a
daily CONUS SWE + snow-depth product, distributed as one NetCDF per day
(``UA_SWE_Depth_800m_v1_<YYYYMMDD>_stable.nc``) holding ``SWE`` and ``DEPTH``
int16 variables on a NAD83 (EPSG:4269) geographic grid. The grid literals below
were read straight from a product file's own ``GeoTransform`` (origin/pixel size)
and dimensions, so they are exact, not inferred.

Ingest is rasterio-only: GDAL's NetCDF driver already returns each variable's
array north-up (rows north→south), aligned to the dataset grid, so each band is
written straight out as a grid-aligned COG with no reprojection or flip.
"""

from __future__ import annotations

import re

from datetime import datetime
from typing import TYPE_CHECKING, Self

import numpy
import rasterio

from snowtool.exceptions import IngestSourceError, SnowtoolError
from snowtool.snowdb.ingest import DateIngest, GridAlignedRaster
from snowtool.snowdb.raster.cog import source_tags
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from affine import Affine

    from snowtool.snowdb.dataset import Dataset
    from snowtool.snowdb.ingest import WritableRaster

# --- SWANN 800m variables -----------------------------------------------------

# Both variables are int16 millimetres with the product's -999 fill, reported as
# area-weighted means (intensive depths, like SNODAS swe/depth). Ingest names
# each COG `<source-stem>__<key>.tif` to keep the source provenance in the
# filename; the ``glob`` matches that on the `__<key>` suffix.
_mm = Unit(name='mm', scale_factor=1)
_NODATA = -999.0

SWANN_800M_VARIABLES = (
    DatasetVariable(
        key='swe',
        unit=_mm,
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=_NODATA,
        glob='*__swe.tif',
    ),
    DatasetVariable(
        key='depth',
        unit=_mm,
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=_NODATA,
        glob='*__depth.tif',
    ),
)

# NetCDF subdataset name -> dataset variable key. The ingester reads each
# subdataset and writes it to the matching variable's provenance-named COG.
_SUBDATASET_TO_VARIABLE = {
    'SWE': 'swe',
    'DEPTH': 'depth',
}


# --- SWANN 800m ingest --------------------------------------------------------


class SwannRaster(GridAlignedRaster):
    """One GDAL-readable band, ready to write itself as a grid-aligned COG.

    Written on the dataset grid's own transform/CRS -- the authoritative geometry
    from the spec, not GDAL's lat/lon-derived (float32) geotransform.
    ``expected_shape`` is the dataset grid's ``(rows, cols)``; a source band of any
    other shape (a truncated/regridded UA file) raises rather than write a
    silently mis-aligned COG.
    """

    def __init__(
        self: Self,
        source_uri: str,
        out_name: str,
        *,
        transform: Affine,
        crs: rasterio.crs.CRS,
        tile_size: int,
        nodata: float,
        expected_shape: tuple[int, int],
        tags: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            out_name,
            transform=transform,
            crs=crs,
            tile_size=tile_size,
            nodata=nodata,
            tags=tags,
        )
        self.source_uri = source_uri
        self.expected_shape = expected_shape

    def read_array(self: Self) -> numpy.ndarray:
        with rasterio.open(self.source_uri) as src:
            array = src.read(1)
        if array.shape != self.expected_shape:
            raise IngestSourceError(
                f'SWANN source band {self.source_uri!r} has shape {array.shape}, '
                f'expected the dataset grid shape {self.expected_shape}.',
            )
        return array


class SwannIngester:
    """Parses one SWANN 800m daily NetCDF (one file == one date) for the driver.

    :meth:`plan` parses the date + stage from the filename and yields a single
    :class:`~snowtool.snowdb.ingest.DateIngest` whose ``build_rasters`` produces a
    grid-aligned :class:`SwannRaster` per variable.
    """

    # Temporary policy gate: pin ingest to the `_early` stage so a dataset never
    # mixes revisions.
    PINNED_STAGE = 'early'
    filename_re = re.compile(
        r'UA_SWE_Depth_800m_v1_(?P<date>\d{8})_(?P<stage>early|provisional|stable)\.nc$',
    )

    def plan(
        self: Self,
        source: Path,
        dataset: Dataset,
    ) -> Iterator[DateIngest]:
        if source.is_dir():
            raise SnowtoolError(
                f'Expected a single SWANN 800m NetCDF file (one file == one '
                f'date), got a directory: {source}. Ingest files one per '
                'invocation.',
            )
        match = self.filename_re.search(source.name)
        if match is None:
            raise SnowtoolError(
                f'Not a SWANN 800m file: {source.name!r} does not match '
                f'{self.filename_re.pattern!r}.',
            )
        if match['stage'] != self.PINNED_STAGE:
            raise SnowtoolError(
                f'Refusing to ingest {match["stage"]!r}-stage SWANN file '
                f'{source.name!r}: this dataset pins to the {self.PINNED_STAGE!r} '
                'revision so a date is never a mix of revisions. Remove the '
                'stage pin to ingest finalized data.',
            )
        ingest_date = datetime.strptime(match['date'], '%Y%m%d').date()  # noqa: DTZ007
        stage = match['stage']

        transform = dataset.grid.base_grid.transform
        crs = dataset.grid_crs
        grid_params = dataset.spec.grid_params
        tile_size = grid_params.tile_size
        expected_shape = (grid_params.rows, grid_params.cols)

        # Name each COG after the source file (+ variable) so the provenance is
        # visible in the filesystem; derived from the source path + spec alone, so
        # the skip check has them without opening the NetCDF.
        out_by_key = {
            key: f'{source.stem}__{key}.tif' for key in _SUBDATASET_TO_VARIABLE.values()
        }

        def build_rasters(source_hash: str) -> list[WritableRaster]:
            rasters: list[WritableRaster] = []
            for subdataset, key in _SUBDATASET_TO_VARIABLE.items():
                variable = dataset.spec.variables[key]
                rasters.append(
                    SwannRaster(
                        f'netcdf:{source}:{subdataset}',
                        out_by_key[key],
                        transform=transform,
                        crs=crs,
                        tile_size=tile_size,
                        nodata=variable.nodata,
                        expected_shape=expected_shape,
                        tags=source_tags(
                            dataset=dataset.spec.name,
                            date=ingest_date,
                            variable=variable.key,
                            files=source.name,
                            source_hash=source_hash,
                            extra={'SOURCE_STAGE': stage},
                        ),
                    ),
                )
            return rasters

        yield DateIngest(
            date=ingest_date,
            source_files=[source],
            out_names=frozenset(out_by_key.values()),
            build_rasters=build_rasters,
        )


# --- SWANN 800m spec ----------------------------------------------------------

# Grid literals read directly from a product file's `crs` variable GeoTransform
# (-125.0208 0.008333325394357 ... 49.9375 ...) and lat/lon dimensions
# (3105 x 7025). Geographic NAD83 -> AOI rasters burn per-row geodesic cell area.
SWANN_800M_SPEC = DatasetSpec(
    name='swann-800m',
    grid_params=GridParams(
        origin_x=-125.0208,
        origin_y=49.9375,
        px_size=0.008333325394357,
        cols=7025,
        rows=3105,
        tile_size=256,
        crs=4269,
    ),
    variables=SWANN_800M_VARIABLES,
    ingester=SwannIngester(),
)
