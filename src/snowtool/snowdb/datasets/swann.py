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

import rasterio

from snowtool.exceptions import SNODASError
from snowtool.snowdb.cog import write_cog
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from affine import Affine

    from snowtool.snowdb.dataset import Dataset

# --- SWANN 800m variables -----------------------------------------------------

# Both variables are int16 millimetres with the product's -999 fill, reported as
# area-weighted means (intensive depths, like SNODAS swe/depth). The ``glob`` is
# the literal COG filename ingest writes into each cogs/<date>/ dir.
_mm = Unit(name='mm', scale_factor=1)
_NODATA = -999.0

SWANN_800M_VARIABLES = (
    DatasetVariable(
        key='swe',
        unit=_mm,
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=_NODATA,
        glob='swe.tif',
    ),
    DatasetVariable(
        key='depth',
        unit=_mm,
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=_NODATA,
        glob='depth.tif',
    ),
)

# NetCDF subdataset name -> dataset variable key. The ingester reads each
# subdataset and writes it to the matching variable's COG (variable.glob).
_SUBDATASET_TO_VARIABLE = {
    'SWE': 'swe',
    'DEPTH': 'depth',
}


# --- SWANN 800m ingest --------------------------------------------------------


class SwannRaster:
    """One GDAL-readable band, ready to write itself as a grid-aligned COG.

    Implements the :class:`~snowtool.snowdb.ingest.WritableRaster` contract.
    ``source_uri`` is any URI rasterio can open -- for SWANN the ingester builds a
    ``netcdf:<file>:<SWE|DEPTH>`` string, keeping the NetCDF-format knowledge in
    the (format-aware) ingester rather than here. The array is read at write time
    (GDAL returns the SWANN NetCDF north-up, rows north→south, already aligned to
    the grid) and written on the dataset grid's own transform/CRS -- the
    authoritative geometry from the spec, not GDAL's lat/lon-derived (float32)
    geotransform.
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
    ) -> None:
        self.source_uri = source_uri
        self.out_name = out_name
        self.transform = transform
        self.crs = crs
        self.tile_size = tile_size
        self.nodata = nodata

    def write_cog(self: Self, output_dir: Path, force: bool = False) -> None:
        output_path = output_dir / self.out_name

        if not force and output_path.exists():
            raise FileExistsError(
                f'Unable to write COG: {output_path} already exists. '
                'Remove file and try again or use `force=True`.',
            )

        with rasterio.open(self.source_uri) as src:
            array = src.read(1)

        write_cog(
            output_path,
            array,
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            tile_size=self.tile_size,
            predictor=2,
        )


class SwannIngester:
    """Ingests one SWANN 800m daily NetCDF (one file == one date) into a dataset.

    The SWANN implementation of :class:`~snowtool.snowdb.ingest.Ingester`: it
    parses the date from the filename and hands a :class:`SwannRaster` per
    variable to the dataset's generic
    :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`.
    """

    # The product is published in three processing stages -- `_early` (newest),
    # then `_provisional`, then `_stable` (finalized) -- all the same format.
    # Output COGs are named per variable (swe.tif/depth.tif), so re-ingesting a
    # later stage cleanly overwrites the earlier one for a date (force=True)
    # rather than coexisting as a second file.
    filename_re = re.compile(
        r'UA_SWE_Depth_800m_v1_(?P<date>\d{8})_(?:early|provisional|stable)\.nc$',
    )

    def ingest(
        self: Self,
        source: Path,
        dataset: Dataset,
        *,
        force: bool = False,
    ) -> list[date]:
        match = self.filename_re.search(source.name)
        if match is None:
            raise SNODASError(
                f'Not a SWANN 800m file: {source.name!r} does not match '
                f'{self.filename_re.pattern!r}.',
            )
        ingest_date = datetime.strptime(match['date'], '%Y%m%d').date()  # noqa: DTZ007

        transform = dataset.grid.base_grid.transform
        crs = dataset.grid_crs
        tile_size = dataset.spec.grid_params.tile_size

        rasters = [
            SwannRaster(
                f'netcdf:{source}:{subdataset}',
                variable.glob,
                transform=transform,
                crs=crs,
                tile_size=tile_size,
                nodata=variable.nodata,
            )
            for subdataset, key in _SUBDATASET_TO_VARIABLE.items()
            for variable in (dataset.spec.variables[key],)
        ]

        dataset.write_date_cogs(ingest_date, rasters, force=force)
        return [ingest_date]


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
