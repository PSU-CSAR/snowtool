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

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import rasterio

from snowtool.exceptions import IngestSourceError, SnowtoolError
from snowtool.snowdb.ingest import DateIngest
from snowtool.snowdb.raster.cog import WGS84, source_tags, write_cog
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from snowtool.snowdb.dataset import Dataset

HDR_EXTS = ('.Hdr', '.txt')

# --- SNODAS products: the variables + their filename-code mapping --------------

_millimeters = Unit(name='mm', scale_factor=1)
_millimeters_100 = Unit(name='mm', scale_factor=100)
# SNODAS stores snowpack average temperature in tenths of a Kelvin (raw values
# run ~2636-2731, where 2731 = 273.1 K, the 0 degrees C snow melt cap), so the
# reporting scale is 10 despite the NSIDC data-field table listing a scale of 1.
_kelvin = Unit(name='k', scale_factor=10)
_kg_per_meter2 = Unit(name='kg_per_m2', scale_factor=10)


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
        """The product a filename's ``(product_code, vcode)`` identifies.

        Precip shares one code (1025) split by vcode; every other product is
        identified by its code alone (any vcode). The two-tier lookup tries the
        exact ``(code, vcode)`` first (the precip pair) then falls back to the
        vcode-agnostic ``(code, None)`` -- unambiguous, since 1025 registers only
        the precip pair and no ``(code, None)``.
        """
        try:
            return (
                _PRODUCT_BY_CODE_VCODE.get(
                    (product_code, vcode),
                )
                or _PRODUCT_BY_CODE_VCODE[(product_code, None)]
            )
        except KeyError:
            raise ValueError(
                f"no SNODAS product for code {product_code} vcode '{vcode}'",
            ) from None

    def to_glob(self: Self) -> str:
        code, vcode, _ = _PRODUCTS[self]
        return ''.join(
            [
                '?' * 6,
                'v[01]',
                str(code),
                '?',
                vcode,
                '*TTNATS*.tif',
            ],
        )

    def unit(self: Self) -> Unit:
        return _PRODUCTS[self][2]


# The one product identity table: Product -> (filename product code, filename
# vcode, reporting unit). ``vcode`` is the literal glob/parse vcode -- the two
# precip variants share code 1025 and split on it ('lL00' liquid, 'lL01' solid);
# every other product is vcode-agnostic (empty string here; '?'-globbed above).
_PRODUCTS: dict[Product, tuple[int, str, Unit]] = {
    Product.PRECIP_LIQUID: (1025, 'lL00', _kg_per_meter2),
    Product.PRECIP_SOLID: (1025, 'lL01', _kg_per_meter2),
    Product.SNOW_WATER_EQUIVALENT: (1034, '', _millimeters),
    Product.SNOW_DEPTH: (1036, '', _millimeters),
    Product.AVERAGE_TEMP: (1038, '', _kelvin),
    Product.SUBLIMATION_BLOWING: (1039, '', _millimeters_100),
    Product.RUNOFF: (1044, '', _millimeters_100),
    Product.SUBLIMATION: (1050, '', _millimeters_100),
}

# The parse inverse: (code, vcode-or-None) -> Product. Precip registers its two
# real vcodes; every other product registers vcode-agnostically under None (its
# _PRODUCTS vcode is the empty string), which from_product_codes falls back to for
# any parsed vcode (see its two-tier lookup).
_PRODUCT_BY_CODE_VCODE: dict[tuple[int, str | None], Product] = {
    (code, vcode or None): product for product, (code, vcode, _) in _PRODUCTS.items()
}


# --- SNODAS filename parser + per-date raster ---------------------------------


class Region(StrEnum):
    US = 'us'
    MASKED = 'zz'


class Model(StrEnum):
    SSM = 'ssm'


class Datatype(StrEnum):
    V0 = 'v0'  # driving input
    V1 = 'v1'  # model output


class SNODASInputRaster:
    """One parsed SNODAS header file, ready to write itself as a COG.

    Parses the SNODAS filename into its fields (region/model/datatype/product/
    time-step) and implements the :class:`~snowtool.snowdb.ingest.WritableRaster`
    contract. The filename is parsed once, at construction; ``source_hash`` -- the
    driver-computed versioned hash of the source tar -- is stamped later, when
    :meth:`SNODASInputRasterSet.build_rasters` runs, so a COG still can never be
    written without its ``SOURCE_HASH`` provenance tag but the archive is not
    re-parsed to apply it (see :meth:`write_cog`, which requires it be set).
    """

    regex: ClassVar[re.Pattern[str]] = re.compile(
        r'^'
        r'(?P<region>[a-z]{2})_'
        r'(?P<model>[a-z]{3})'
        r'(?P<datatype>v\d)'
        r'(?P<product_code>\d{4})'
        r'(?P<scaled>S?)'
        r'(?P<vcode>[a-zA-Z]{2}[\d_]{2})'
        # timecode: 0024 = 24 hr integration, 0001 = 1 hr snapshot.
        r'[AT](?P<timecode>00(01|24))'
        r'TTNATS'
        r'(?P<year>\d{4})'
        r'(?P<month>\d{2})'
        r'(?P<day>\d{2})'
        r'(?P<hour>\d{2})'
        # interval: H = hour, D = day.
        r'(?P<interval>H|D)'
        # offset: P001 = delta over / value at interval end, P000 = interval start.
        r'(?P<offset>P00[01])'
        r'$',
    )

    def __init__(self: Self, path: Path) -> None:
        self.path = path
        self.name = self.path.stem
        # Stamped at build time by SNODASInputRasterSet.build_rasters; write_cog
        # refuses to run until it is set.
        self.source_hash: str | None = None
        info = self._match(self.name).groupdict()

        try:
            self.region = Region(info['region'])
            self.model = Model(info['model'])
            self.datatype = Datatype(info['datatype'])
            self.scaled = bool(info['scaled'])
            self.vcode: str = info['vcode']
            self.timecode: str = info['timecode']
            self.datetime = datetime(
                year=int(info['year']),
                month=int(info['month']),
                day=int(info['day']),
                hour=int(info['hour']),
                tzinfo=UTC,
            )
            self.interval: str = info['interval']
            self.offset: str = info['offset']
            self.product = Product.from_product_codes(
                int(info['product_code']),
                self.vcode,
            )
        except Exception as e:
            raise IngestSourceError('invalid value in SNODAS file name') from e

        if path.suffix not in HDR_EXTS:
            raise IngestSourceError(
                'SNODAS raster path must be to header file. '
                f"Unknown extension '{path.suffix}'. Valid values: {HDR_EXTS}.",
            )

    @classmethod
    def _match(cls: type[Self], string: str):
        match = cls.regex.match(string)
        if not match:
            raise IngestSourceError('unable to parse SNODAS file path')
        return match

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

    def write_cog(self: Self, output_dir: Path) -> None:
        source_hash = self.source_hash
        if source_hash is None:
            # Upholds the invariant a COG never lands without provenance: the set's
            # build_rasters stamps the hash before handing rasters to the driver.
            raise SnowtoolError(
                'SNODAS raster written without a source hash; '
                'build_rasters must stamp it first',
            )
        # GDAL's SNODAS/raw driver has a header line-length limit; trim first.
        self.trim_header(self.path)

        with rasterio.open(self.path) as src:
            array = src.read(1)
            transform = src.transform
            crs = src.crs or WGS84
            nodata = src.nodata

        write_cog(
            output_dir / self.out_name,
            array,
            transform=transform,
            crs=crs,
            nodata=nodata,
            tile_size=SNODAS_SPEC.grid_params.tile_size,
            tags=self._provenance_tags(source_hash),
        )

    def _provenance_tags(self: Self, source_hash: str) -> dict[str, str]:
        # SNODAS already keeps its full source stem as the COG name; these tags
        # add the parsed fields as a structured, queryable record in the file.
        return source_tags(
            dataset=SNODAS_SPEC.name,
            date=self.datetime.date(),
            variable=self.product.value,
            files=self.path.name,
            source_hash=source_hash,
            extra={
                'SOURCE_REGION': self.region.value,
                'SOURCE_MODEL': self.model.value,
                'SOURCE_DATATYPE': self.datatype.value,
                'SOURCE_SCALED': str(self.scaled),
                'SOURCE_VCODE': self.vcode,
                'SOURCE_TIMECODE': self.timecode,
                'SOURCE_INTERVAL': self.interval,
                'SOURCE_OFFSET': self.offset,
                'SOURCE_TIMESTEP': self.datetime.isoformat(),
            },
        )


class SNODASInputRasterSet:
    """A date's validated set of SNODAS rasters, one per product.

    Built once by :meth:`from_extracted` from an extracted archive's header files:
    it validates that the archive holds every product, is a single date, and sits
    at the pinned time-step. The archive is parsed exactly once here; the driver's
    ``source_hash`` is stamped onto the already-parsed rasters later, by
    :meth:`build_rasters`, which returns them as the date's writable rasters.
    """

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

    @staticmethod
    def extract_archive(snodas_tar: Path, extract_dir: Path) -> None:
        """Extract the tar's gzipped rasters (header + data) into ``extract_dir``.

        The archive nests each raster gzipped inside the tar, so unpack the tar to
        a scratch dir then gunzip each member into ``extract_dir``, leaving the raw
        SNODAS header/data files the raster parser reads.
        """
        with tempfile.TemporaryDirectory() as _temp:
            temp = Path(_temp)
            with tarfile.open(snodas_tar) as tar:
                tar.extractall(temp, filter='data')

            for f in temp.glob('*.gz'):
                outpath = extract_dir / f.stem
                with gzip.open(f, 'rb') as f_in, outpath.open('wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    @classmethod
    def from_extracted(cls: type[Self], extract_dir: Path) -> Self:
        """The validated per-product raster set over already-extracted headers.

        Parses each header file once. The rasters carry no ``source_hash`` yet;
        :meth:`build_rasters` stamps it before they are written.
        """
        rasters: dict[Product, SNODASInputRaster] = {}
        for ext in HDR_EXTS:
            for hdr in extract_dir.glob(f'*{ext}'):
                file_info = SNODASInputRaster(hdr)
                rasters[file_info.product] = file_info
        return cls(rasters)

    def build_rasters(
        self: Self,
        source_hash: str,
    ) -> list[SNODASInputRaster]:
        """Stamp the driver's source hash onto the parsed rasters, then return them.

        The :class:`~snowtool.snowdb.ingest.DateIngest.build_rasters` callback for
        SNODAS: no re-glob, no re-parse -- the archive was parsed once in
        :meth:`from_extracted`; this just applies the hash the driver computed so
        every COG lands with its ``SOURCE_HASH`` provenance tag.
        """
        rasters = list(self)
        for raster in rasters:
            raster.source_hash = source_hash
        return rasters


# --- SNODAS ingest ------------------------------------------------------------


class SnodasIngester:
    """Parses a SNODAS tar archive (one archive == one date) for the ingest driver.

    The SNODAS implementation of :class:`~snowtool.snowdb.ingest.Ingester`: its
    :meth:`plan` extracts and parses the archive exactly once, validates the
    date/product set, and yields a single
    :class:`~snowtool.snowdb.ingest.DateIngest` whose ``build_rasters`` stamps the
    driver-computed source hash onto those already-parsed rasters (each COG stamped
    with that hash). The driver hashes the tar, drives the write, and builds the
    result.
    """

    def plan(
        self,
        source: Path,
        dataset: Dataset,
    ) -> Iterator[DateIngest]:
        if source.is_dir():
            # Guarded here so a directory earns a precise, typed error instead of
            # tarfile's raw IsADirectoryError.
            raise SnowtoolError(
                f'Expected a single SNODAS tar archive (one archive == one '
                f'date), got a directory: {source}. Ingest archives one per '
                'invocation.',
            )

        # Extract the archive once into a tempdir kept alive across the yield by
        # this generator, and parse it once into the validated raster set. The
        # date + product set are validated here; build_rasters stamps the driver's
        # real source hash onto those same parsed rasters -- nothing is extracted
        # or parsed twice.
        with tempfile.TemporaryDirectory() as _extract:
            extract_dir = Path(_extract)
            SNODASInputRasterSet.extract_archive(source, extract_dir)
            raster_set = SNODASInputRasterSet.from_extracted(extract_dir)

            yield DateIngest(
                date=raster_set.date,
                source_files=[source],
                build_rasters=raster_set.build_rasters,
            )


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
