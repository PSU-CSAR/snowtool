"""The SNODAS dataset definition: variables, grid spec, and ingest.

``Product`` is the SNODAS variable enum, including its filename-code mapping
(``to_glob`` / ``from_product_codes``). The SNODAS *filename parser* (``SNODASName``,
a pure parse of a filename stem), its per-date validated set (``SNODASInputRasterSet``,
built from tar member names alone), and the extracted-header writable raster
(``SNODASInputRaster``) all live here -- ingest is the only place that parses SNODAS
filenames; the read path is dataset-agnostic (it finds a variable's file by its glob
and gets the date from the ``cogs/<date>/`` directory). ``SNODAS_SPEC`` is the source
of truth for the SNODAS grid/variables; it is collected into the registry in this
package's ``__init__``.
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

import numpy
import rasterio

from snowtool.exceptions import IngestSourceError, SnowtoolError
from snowtool.snowdb.ingest import DateIngest, GridAlignedRaster
from snowtool.snowdb.raster.cog import source_tags
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.variables import DatasetVariable, Reducer, Unit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from affine import Affine

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

        See :data:`_PRODUCTS` for the code/vcode assignment this inverts.
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
        return f'??????v[01]{code}?{vcode}*TTNATS*.tif'

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

# The parse inverse of _PRODUCTS: (code, vcode-or-None) -> Product.
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


class SNODASName:
    """One parsed SNODAS filename stem -> its typed fields. Pure parse, no I/O.

    The immutable parsed record at the heart of SNODAS ingest: it turns a filename
    stem (region/model/datatype/product/time-step) into typed fields with no
    filesystem access, so a whole date's product set can be identified, validated,
    and named (:attr:`out_name`) straight from the tar's member names -- before a
    single byte is extracted. :class:`SNODASInputRaster` pairs one of these with the
    extracted header path (and the source hash) at write time.
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

    def __init__(self: Self, name: str) -> None:
        self.name = name
        match = self.regex.match(name)
        if not match:
            raise IngestSourceError('unable to parse SNODAS file path')
        info = match.groupdict()

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

    @property
    def out_name(self: Self) -> str:
        # SNODAS keeps its full parsed source stem as the COG name (its provenance
        # is the filename); satisfies the WritableRaster.out_name contract.
        return f'{self.name}.tif'

    def provenance_tags(
        self: Self,
        source_files: str,
        source_hash: str,
    ) -> dict[str, str]:
        # SNODAS already keeps its full source stem as the COG name; these tags
        # add the parsed fields as a structured, queryable record in the file.
        return source_tags(
            dataset=SNODAS_SPEC.name,
            date=self.datetime.date(),
            variable=self.product.value,
            files=source_files,
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


class SNODASInputRaster(GridAlignedRaster):
    """One extracted SNODAS header, ready to write itself as a grid-aligned COG.

    Written on the dataset grid's authoritative transform/CRS from the spec, not
    the header's own geotransform; ``expected_shape`` is the grid's
    ``(rows, cols)``, so a header on any other lattice raises rather than write
    a silently mis-aligned COG.
    """

    def __init__(
        self: Self,
        parsed: SNODASName,
        path: Path,
        source_hash: str,
        *,
        transform: Affine,
        crs: rasterio.crs.CRS,
        tile_size: int,
        expected_shape: tuple[int, int],
    ) -> None:
        if path.suffix not in HDR_EXTS:
            raise IngestSourceError(
                'SNODAS raster path must be to header file. '
                f"Unknown extension '{path.suffix}'. Valid values: {HDR_EXTS}.",
            )
        super().__init__(
            parsed.out_name,
            transform=transform,
            crs=crs,
            tile_size=tile_size,
            nodata=SNODAS_NODATA,
            tags=parsed.provenance_tags(path.name, source_hash),
        )
        self.parsed = parsed
        self.path = path
        self.source_hash = source_hash
        self.expected_shape = expected_shape

    @staticmethod
    def trim_header(hdr: Path) -> None:
        """GDAL raw-driver header lines are length-limited; trim to 255 to be safe."""
        line_limit = 255
        lines: list[bytes] = []
        with hdr.open('rb') as f:
            for line in f:
                # `for line in f` keeps the trailing newline; strip it before
                # truncating so we re-add exactly one (not a doubled newline).
                lines.append(line.rstrip(b'\r\n')[:line_limit] + b'\n')

        with hdr.open('wb') as f:
            f.writelines(lines)

    def read_array(self: Self) -> numpy.ndarray:
        self.trim_header(self.path)
        with rasterio.open(self.path) as src:
            array = src.read(1)
        if array.shape != self.expected_shape:
            raise IngestSourceError(
                f'SNODAS header {self.path.name!r} has shape {array.shape}, '
                f'expected the dataset grid shape {self.expected_shape}.',
            )
        return array


class SNODASInputRasterSet:
    """A date's validated set of parsed SNODAS names, one per product.

    Built from filename stems alone (:meth:`from_names`) -- the tar's member names,
    no extraction -- it validates that the set holds every product, is a single
    date, and sits at the pinned time-step.
    """

    def __init__(
        self: Self,
        names: Mapping[Product, SNODASName],
    ) -> None:
        missing = sorted(product.value for product in Product if product not in names)
        if missing:
            raise SnowtoolError(
                f'SNODAS archive missing product(s): {missing}',
            )
        self.names = dict(names)
        self.date = self._validate_dates()
        self._validate_revision()
        self._validate_region()

    def __iter__(self: Self) -> Iterator[SNODASName]:
        return iter(self.names.values())

    @property
    def out_names(self: Self) -> frozenset[str]:
        """The COG filenames this date will land -- from the parsed names alone."""
        return frozenset(name.out_name for name in self)

    def _validate_dates(self: Self) -> date:
        dates = {name.datetime.date() for name in self}
        if len(dates) > 1:
            raise SnowtoolError(
                'SNODAS rasters not all from same date per filenames',
            )
        return dates.pop()

    # Temporary policy gate: pin ingest to the 05 time-step hour so a dataset never
    # mixes revisions.
    PINNED_TIMESTEP_HOUR: ClassVar[int] = 5

    def _validate_revision(self: Self) -> None:
        pinned = self.PINNED_TIMESTEP_HOUR
        off = sorted({n.datetime.hour for n in self if n.datetime.hour != pinned})
        if off:
            raise SnowtoolError(
                f'Refusing SNODAS time-step hour(s) {off}: ingest pins to the '
                f'{self.PINNED_TIMESTEP_HOUR:02d} time-step (the standard daily '
                'product) so a date never mixes revisions. Remove the revision '
                'pin to allow other hours.',
            )

    def _validate_region(self: Self) -> None:
        off = sorted({n.region.value for n in self if n.region is not Region.MASKED})
        if off:
            raise SnowtoolError(
                f'Refusing SNODAS region(s) {off}: ingest pins to the masked '
                f"('{Region.MASKED.value}') CONUS product -- the unmasked grid is "
                'a different lattice and would not align with the dataset grid.',
            )

    @classmethod
    def from_names(cls: type[Self], names: Iterable[str]) -> Self:
        """The validated per-product set over filename stems (no I/O).

        Each stem is parsed once into a :class:`SNODASName`; the resulting set is
        keyed by product. A repeated product raises rather than last-wins -- the
        archive holds exactly one per product, so tar member ordering must never
        decide which stem a validator sees. Drives the cheap plan-time
        identification.
        """
        parsed: dict[Product, SNODASName] = {}
        for name in names:
            entry = SNODASName(name)
            if entry.product in parsed:
                raise IngestSourceError(
                    f'duplicate SNODAS product {entry.product.value!r} in archive: '
                    f'{parsed[entry.product].name!r} and {entry.name!r}',
                )
            parsed[entry.product] = entry
        return cls(parsed)

    @staticmethod
    def header_stems(member_names: Iterable[str]) -> list[str]:
        """The SNODAS filename stems of an archive's header members.

        Each raster is nested gzipped in the tar as ``<stem>.Hdr.gz`` /
        ``<stem>.txt.gz`` (data files ride alongside as ``.dat.gz``). This strips the
        ``.gz`` and keeps only the header members, returning the bare ``<stem>`` the
        parser reads -- so a date's product set is known from ``tarfile.getnames()``
        without touching the bytes.
        """
        stems: list[str] = []
        for member in member_names:
            name = Path(member).name
            if not name.endswith('.gz'):
                continue
            inner = Path(name[: -len('.gz')])
            if inner.suffix in HDR_EXTS:
                stems.append(inner.stem)
        return stems

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

    def build_rasters(
        self: Self,
        source: Path,
        extract_dir: Path,
        source_hash: str,
        *,
        transform: Affine,
        crs: rasterio.crs.CRS,
        tile_size: int,
        expected_shape: tuple[int, int],
    ) -> list[SNODASInputRaster]:
        """Extract ``source`` into ``extract_dir`` and pair each header with its name.

        The sole place a SNODAS tar is extracted -- one archive, one date. Each
        extracted header is matched by stem to the name already parsed at plan time.
        """
        self.extract_archive(source, extract_dir)

        headers: dict[str, Path] = {}
        for ext in HDR_EXTS:
            for hdr in extract_dir.glob(f'*{ext}'):
                headers[hdr.stem] = hdr

        rasters: list[SNODASInputRaster] = []
        for name in self:
            path = headers.get(name.name)
            if path is None:
                raise IngestSourceError(
                    f'SNODAS archive header missing on extraction for {name.name!r}',
                )
            rasters.append(
                SNODASInputRaster(
                    name,
                    path,
                    source_hash,
                    transform=transform,
                    crs=crs,
                    tile_size=tile_size,
                    expected_shape=expected_shape,
                ),
            )
        return rasters


# --- SNODAS ingest ------------------------------------------------------------


class SnodasIngester:
    """Parses a SNODAS tar archive (one archive == one date) for the ingest driver.

    :meth:`plan` reads only the tar's *member names* (``tarfile.getnames()``) to
    identify and validate the date's product set and derive its ``out_names`` --
    no extraction.
    """

    def plan(
        self,
        source: Path,
        dataset: Dataset,
    ) -> Iterator[DateIngest]:
        if source.is_dir():
            raise SnowtoolError(
                f'Expected a single SNODAS tar archive (one archive == one '
                f'date), got a directory: {source}. Ingest archives one per '
                'invocation.',
            )

        # Read only the member names -- no extraction -- to identify and validate
        # the date's product set. An already-current date is skipped having read
        # only the tar's index.
        with tarfile.open(source) as tar:
            member_names = tar.getnames()
        raster_set = SNODASInputRasterSet.from_names(
            SNODASInputRasterSet.header_stems(member_names),
        )

        transform = dataset.grid.base_grid.transform
        crs = dataset.grid_crs
        grid_params = dataset.spec.grid_params

        # An empty scratch dir for the extraction build_rasters may run, cleaned
        # deterministically when this generator is exhausted/closed. Nothing is
        # extracted here or before the write path decides to build -- so a skipped
        # date leaves the dir empty (zero tar extraction).
        with tempfile.TemporaryDirectory() as _extract:
            extract_dir = Path(_extract)
            yield DateIngest(
                date=raster_set.date,
                source_files=[source],
                out_names=raster_set.out_names,
                build_rasters=lambda source_hash: raster_set.build_rasters(
                    source,
                    extract_dir,
                    source_hash,
                    transform=transform,
                    crs=crs,
                    tile_size=grid_params.tile_size,
                    expected_shape=(grid_params.rows, grid_params.cols),
                ),
            )


# --- SNODAS variables + spec (the source of truth for SNODAS values) ----------

# SNODAS variables, one per product. All are intensive quantities (depths,
# temperature) reported as area-weighted means; reads are int16 with the SNODAS
# nodata sentinel.
SNODAS_NODATA = -9999.0

SNODAS_VARIABLES = tuple(
    DatasetVariable(
        key=product.value,
        unit=product.unit(),
        reducer=Reducer.MEAN,
        dtype='int16',
        nodata=SNODAS_NODATA,
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
