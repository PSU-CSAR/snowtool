"""The SNODAS dataset definition: variables, grid spec, and ingest.

``Product`` is the SNODAS variable enum, including its filename-code mapping
(``to_glob`` / ``from_product_codes``). The SNODAS *filename parser* and the
per-date raster set (``SNODASInputRaster`` / ``SNODASInputRasterSet``) live here
too -- ingest is the only place that parses SNODAS filenames; the read path is
dataset-agnostic (it finds a variable's file by its glob and gets the date from
the ``cogs/<date>/`` directory). ``SNODAS_SPEC`` is the source of truth for the
SNODAS grid/variables; it is collected into the registry in this package's
``__init__``.
"""

from __future__ import annotations

import gzip
import re
import shutil
import tarfile
import tempfile

from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import rasterio

from snowtool.exceptions import SnowtoolError
from snowtool.snowdb.dataset import INGEST_FORMAT_VERSION
from snowtool.snowdb.ingest import IngestResult
from snowtool.snowdb.provenance import hash_files, versioned_hash
from snowtool.snowdb.raster.cog import WGS84, source_tags, write_cog_guarded
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from snowtool.snowdb.dataset import Dataset

HDR_EXTS = ('.Hdr', '.txt')

# --- SNODAS products: the variables + their filename-code mapping --------------

_product_code_to_product_name = {
    1025: 'precip',
    1034: 'swe',
    1036: 'depth',
    1038: 'average_temp',
    1039: 'sublimation_blowing',
    1044: 'runoff',
    1050: 'sublimation',
}

_product_name_to_product_code = {v: k for k, v in _product_code_to_product_name.items()}


_millimeters = Unit(name='mm', scale_factor=1)
_millimeters_100 = Unit(name='mm', scale_factor=100)
_kelvin = Unit(name='k', scale_factor=1)
_kg_per_meter2 = Unit(name='kg_per_m2', scale_factor=10)

_units = {
    'precip_solid': _kg_per_meter2,
    'precip_liquid': _kg_per_meter2,
    'swe': _millimeters,
    'depth': _millimeters,
    'average_temp': _kelvin,
    'sublimation': _millimeters_100,
    'sublimation_blowing': _millimeters_100,
    'runoff': _millimeters_100,
}


class Product(StrEnum):
    PRECIP_SOLID = 'precip_solid'
    PRECIP_LIQUID = 'precip_liquid'
    SNOW_WATER_EQUIVALENT = 'swe'
    SNOW_DEPTH = 'depth'
    AVERAGE_TEMP = 'average_temp'
    SUBLIMATION = 'sublimation'
    SUBLIMATION_BLOWING = 'sublimation_blowing'
    RUNOFF = 'runoff'

    @classmethod
    def from_product_codes(cls: type[Self], product_code: int, vcode: str) -> Self:
        product_name: str = _product_code_to_product_name[product_code]

        if product_name != 'precip':
            return cls(product_name)

        match vcode:
            case 'lL00':
                return cls('precip_liquid')
            case 'lL01':
                return cls('precip_solid')
            case _:
                raise ValueError(
                    f"unknown vcode '{vcode}' for product type 'precip'",
                )

    def to_glob(self: Self) -> str:
        product_name = self.value
        vcode: str = ''

        if product_name.startswith('precip'):
            product_name, precip_type = product_name.split('_')
            match precip_type:
                case 'liquid':
                    vcode = 'lL00'
                case 'solid':
                    vcode = 'lL01'
                case _:
                    raise ValueError(f"unknown precip type '{precip_type}'")

        product_code = _product_name_to_product_code[product_name]
        return ''.join(
            [
                '?' * 6,
                'v[01]',
                str(product_code),
                '?',
                vcode,
                '*TTNATS*.tif',
            ],
        )

    def unit(self: Self) -> Unit:
        return _units[self.value]


# --- SNODAS filename parser ---------------------------------------------------


class Region(StrEnum):
    US = 'us'
    MASKED = 'zz'


class Model(StrEnum):
    SSM = 'ssm'


class Datatype(StrEnum):
    V0 = 'v0'  # driving input
    V1 = 'v1'  # model output


class Timecode(StrEnum):
    T0024 = '0024'  # 24 hr integration
    T0001 = '0001'  # 1 hr snapshot


class Interval(StrEnum):
    HOUR = 'H'
    DAY = 'D'


class Offset(StrEnum):
    P001 = 'P001'  # value is delta over interval or value at interval end
    P000 = 'P000'  # value from interval start


class BaseFileInfo:
    regex: ClassVar[re.Pattern[str]] = re.compile(
        r'^'
        r'(?P<region>[a-z]{2})_'
        r'(?P<model>[a-z]{3})'
        r'(?P<datatype>v\d)'
        r'(?P<product_code>\d{4})'
        r'(?P<scaled>S?)'
        r'(?P<vcode>[a-zA-Z]{2}[\d_]{2})'
        r'[AT](?P<timecode>00(01|24))'
        r'TTNATS'
        r'(?P<year>\d{4})'
        r'(?P<month>\d{2})'
        r'(?P<day>\d{2})'
        r'(?P<hour>\d{2})'
        r'(?P<interval>H|D)'
        r'(?P<offset>P00[01])'
        r'$',
    )

    def __init__(self: Self, path: Path) -> None:
        self.path = path
        self.name = self.path.stem
        info = self._match(self.name).groupdict()

        try:
            self.region = Region(info['region'])
            self.model = Model(info['model'])
            self.datatype = Datatype(info['datatype'])
            self.scaled = bool(info['scaled'])
            self.vcode: str = info['vcode']
            self.timecode = Timecode(info['timecode'])
            self.datetime = datetime(
                year=int(info['year']),
                month=int(info['month']),
                day=int(info['day']),
                hour=int(info['hour']),
                tzinfo=UTC,
            )
            self.interval = Interval(info['interval'])
            self.offset = Offset(info['offset'])
            self.product = Product.from_product_codes(
                int(info['product_code']),
                self.vcode,
            )
        except Exception as e:
            raise ValueError('invalid value in SNODAS file name') from e

    @classmethod
    def _match(cls: type[Self], string: str):
        match = cls.regex.match(string)
        if not match:
            raise ValueError('unable to parse SNODAS file path')
        return match


class SNODASInputRaster(BaseFileInfo):
    def __init__(self: Self, path: Path) -> None:
        super().__init__(path)
        if path.suffix not in HDR_EXTS:
            raise ValueError(
                'SNODAS raster path must be to header file. '
                f"Unknown extension '{path.suffix}'. Valid values: {HDR_EXTS}.",
            )
        # The versioned hash of the source tar, set by the ingester once per date
        # (the same value handed to write_date_cogs) so every COG carries it.
        self.source_hash = ''

    @staticmethod
    def trim_header(hdr: Path) -> None:
        """gdal has a header line length limit of
        256 chars for <2.3.0, or 1024 chars for >=2.3.0,
        but we trim to the smaller size to be safe."""
        line_limit = 255
        lines: list[bytes] = []
        with hdr.open('rb') as f:
            for line in f:
                lines.append(line[: min(len(line), line_limit)] + b'\n')

        with hdr.open('wb') as f:
            f.writelines(lines)

    @property
    def out_name(self: Self) -> str:
        # SNODAS keeps its full parsed source stem as the COG name (its provenance
        # is the filename); satisfies the WritableRaster.out_name contract.
        return f'{self.name}.tif'

    def write_cog(self: Self, output_dir: Path, force: bool = False) -> None:
        # GDAL's SNODAS/raw driver has a header line-length limit; trim first.
        self.trim_header(self.path)

        with rasterio.open(self.path) as src:
            array = src.read(1)
            transform = src.transform
            crs = src.crs or WGS84
            nodata = src.nodata

        write_cog_guarded(
            output_dir / self.out_name,
            array,
            force=force,
            transform=transform,
            crs=crs,
            nodata=nodata,
            tile_size=SNODAS_SPEC.grid_params.tile_size,
            tags=self._provenance_tags(),
        )

    def _provenance_tags(self: Self) -> dict[str, str]:
        # SNODAS already keeps its full source stem as the COG name; these tags
        # add the parsed fields as a structured, queryable record in the file.
        return source_tags(
            dataset=SNODAS_SPEC.name,
            date=self.datetime.date(),
            variable=self.product.value,
            files=self.path.name,
            source_hash=self.source_hash,
            extra={
                'SOURCE_REGION': self.region.value,
                'SOURCE_MODEL': self.model.value,
                'SOURCE_DATATYPE': self.datatype.value,
                'SOURCE_SCALED': str(self.scaled),
                'SOURCE_VCODE': self.vcode,
                'SOURCE_TIMECODE': self.timecode.value,
                'SOURCE_INTERVAL': self.interval.value,
                'SOURCE_OFFSET': self.offset.value,
                'SOURCE_TIMESTEP': self.datetime.isoformat(),
            },
        )


class SNODASInputRasterSet:
    def __init__(
        self: Self,
        rasters: Mapping[Product, SNODASInputRaster],
    ) -> None:
        missing = sorted(product.value for product in Product if product not in rasters)
        if missing:
            raise SnowtoolError(
                f'SNODAS archive missing product(s): {missing}',
            )
        self.rasters = dict(rasters)
        self.date = self.validate_dates(self)
        self.validate_revision(self)

    def __iter__(self: Self) -> Iterator[SNODASInputRaster]:
        return iter(self.rasters.values())

    @staticmethod
    def validate_dates(rasters: Iterable[SNODASInputRaster]) -> date:
        dates = {raster.datetime.date() for raster in rasters}
        if len(dates) > 1:
            raise SnowtoolError(
                'SNODAS rasters not all from same date per filenames',
            )
        return dates.pop()

    # Temporary policy gate: pin ingest to the 05 time-step hour. The hour in a
    # SNODAS filename is the `hh` of the time-step code (TSyyyymmddhh) -- the
    # standard daily product uses 05 for every variable (both the T0001
    # snapshots and the T0024 24-hr integrations). The parser above stays
    # general (any hour), but a dataset must hold a single consistent revision,
    # so we refuse anything but the 05 daily product here (SWANN pins to
    # `_early` for the same latency-over-finality reason). Remove this method
    # (and its call in __init__) to allow other hours.
    PINNED_TIMESTEP_HOUR: ClassVar[int] = 5

    @classmethod
    def validate_revision(
        cls: type[Self],
        rasters: Iterable[SNODASInputRaster],
    ) -> None:
        pinned = cls.PINNED_TIMESTEP_HOUR
        off = sorted({r.datetime.hour for r in rasters if r.datetime.hour != pinned})
        if off:
            raise SnowtoolError(
                f'Refusing SNODAS time-step hour(s) {off}: ingest pins to the '
                f'{cls.PINNED_TIMESTEP_HOUR:02d} time-step (the standard daily '
                'product) so a date never mixes revisions. Remove the revision '
                'pin to allow other hours.',
            )

    @classmethod
    def from_archive(
        cls: type[Self],
        snodas_tar: Path,
        extract_dir: Path,
    ) -> Self:
        rasters: dict[Product, SNODASInputRaster] = {}

        with tempfile.TemporaryDirectory() as _temp:
            temp = Path(_temp)
            with tarfile.open(snodas_tar) as tar:
                tar.extractall(temp, filter='data')

            for f in temp.glob('*.gz'):
                outpath = extract_dir / f.stem
                with gzip.open(f, 'rb') as f_in, outpath.open('wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            for ext in HDR_EXTS:
                for hdr in extract_dir.glob(f'*{ext}'):
                    file_info = SNODASInputRaster(hdr)
                    rasters[file_info.product] = file_info

        return cls(rasters)


# --- SNODAS ingest ------------------------------------------------------------


class SnodasIngester:
    """Ingests a SNODAS tar archive (one archive == one date) into a dataset.

    The SNODAS implementation of :class:`~snowtool.snowdb.ingest.Ingester`: it
    parses the archive into a per-date raster set and hands them to the dataset's
    generic :meth:`~snowtool.snowdb.dataset.Dataset.write_date_cogs`.
    """

    def ingest(
        self,
        source: Path,
        dataset: Dataset,
        *,
        force: bool = False,
    ) -> IngestResult:
        if source.is_dir():
            # Guarded here so a directory earns a precise, typed error instead of
            # tarfile's raw IsADirectoryError.
            raise SnowtoolError(
                f'Expected a single SNODAS tar archive (one archive == one '
                f'date), got a directory: {source}. Ingest archives one per '
                'invocation.',
            )
        # One versioned hash of the source tar per date (== per archive), stamped on
        # every COG and compared by the skip check.
        source_hash = versioned_hash(INGEST_FORMAT_VERSION, hash_files([source]))
        with tempfile.TemporaryDirectory() as extract_dir:
            rasters = SNODASInputRasterSet.from_archive(source, Path(extract_dir))
            for raster in rasters:
                raster.source_hash = source_hash
            wrote = dataset.write_date_cogs(
                rasters.date,
                rasters,
                source_hash=source_hash,
                force=force,
            )
        dates = [rasters.date]
        if wrote:
            return IngestResult(ingested=dates, skipped=[])
        return IngestResult(ingested=[], skipped=dates)


# --- SNODAS variables + spec (the source of truth for SNODAS values) ----------

# SNODAS variables, one per product. All are intensive quantities (depths,
# temperature) reported as area-weighted means; reads are int16 with the SNODAS
# nodata sentinel. (Switching any to a TOTAL basin total is a future, domain-driven
# change -- see the plan's reduced-unit note.)
SNODAS_VARIABLES = tuple(
    DatasetVariable(
        key=product.value,
        unit=product.unit(),
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=-9999.0,
        glob=product.to_glob(),
    )
    for product in Product
)

SNODAS_SPEC = DatasetSpec(
    name='snodas',
    grid_params=GridParams(
        origin_x=-124.733333333333333,
        origin_y=52.875000000000000,
        px_size=0.008333333333333,
        cols=6935,
        rows=3351,
        tile_size=256,
    ),
    variables=SNODAS_VARIABLES,
    ingester=SnodasIngester(),
)
