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

from snowtool.exceptions import SnowtoolError
from snowtool.snowdb.dataset import INGEST_FORMAT_VERSION
from snowtool.snowdb.ingest import IngestResult
from snowtool.snowdb.provenance import hash_files, versioned_hash
from snowtool.snowdb.raster.cog import source_tags, write_cog_guarded
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from pathlib import Path

    from affine import Affine

    from snowtool.snowdb.dataset import Dataset

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
        tags: dict[str, str] | None = None,
    ) -> None:
        self.source_uri = source_uri
        self.out_name = out_name
        self.transform = transform
        self.crs = crs
        self.tile_size = tile_size
        self.nodata = nodata
        self.tags = tags

    def write_cog(self: Self, output_dir: Path, force: bool = False) -> None:
        with rasterio.open(self.source_uri) as src:
            array = src.read(1)

        write_cog_guarded(
            output_dir / self.out_name,
            array,
            force=force,
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            tile_size=self.tile_size,
            tags=self.tags,
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
    # We pin ingest to `_early`: a dataset must hold a single consistent
    # revision, and `_early` is available first (the latency-over-finality
    # policy SNODAS also follows by pinning to its 05 time-step hour). The regex
    # still matches all three stages so a wrong-stage file earns a precise error
    # (below) rather than a generic "not a SWANN file"; `PINNED_STAGE` does the
    # refusing. Drop the stage check to allow finalized data.
    PINNED_STAGE = 'early'
    filename_re = re.compile(
        r'UA_SWE_Depth_800m_v1_(?P<date>\d{8})_(?P<stage>early|provisional|stable)\.nc$',
    )

    def ingest(
        self: Self,
        source: Path,
        dataset: Dataset,
        *,
        force: bool = False,
    ) -> IngestResult:
        if source.is_dir():
            # Guarded before the filename regex so a directory earns a precise
            # error rather than a misleading "does not match" message (or a raw
            # read failure further in).
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

        # One versioned hash of the source .nc per date (== per file), stamped on
        # every COG and compared by the skip check.
        source_hash = versioned_hash(INGEST_FORMAT_VERSION, hash_files([source]))

        transform = dataset.grid.base_grid.transform
        crs = dataset.grid_crs
        tile_size = dataset.spec.grid_params.tile_size

        # Name each COG after the source file (+ variable) so the provenance is
        # visible in the filesystem; the full record also goes into COG tags.
        rasters = []
        for subdataset, key in _SUBDATASET_TO_VARIABLE.items():
            variable = dataset.spec.variables[key]
            rasters.append(
                SwannRaster(
                    f'netcdf:{source}:{subdataset}',
                    f'{source.stem}__{variable.key}.tif',
                    transform=transform,
                    crs=crs,
                    tile_size=tile_size,
                    nodata=variable.nodata,
                    tags=source_tags(
                        dataset=dataset.spec.name,
                        date=ingest_date,
                        variable=variable.key,
                        files=source.name,
                        source_hash=source_hash,
                        extra={'SOURCE_STAGE': match['stage']},
                    ),
                ),
            )

        wrote = dataset.write_date_cogs(
            ingest_date,
            rasters,
            source_hash=source_hash,
            force=force,
        )
        dates = [ingest_date]
        if wrote:
            return IngestResult(ingested=dates, skipped=[])
        return IngestResult(ingested=[], skipped=dates)


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
