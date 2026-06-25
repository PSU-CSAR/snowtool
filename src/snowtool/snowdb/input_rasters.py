import gzip
import re
import shutil
import tarfile
import tempfile

from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Self

import rasterio

from snowtool.exceptions import SNODASError
from snowtool.snowdb.cog import WGS84, write_cog
from snowtool.snowdb.datasets import SNODAS_SPEC, Product

HDR_EXTS = ('.Hdr', '.txt')


# --- SNODAS filename parser ---------------------------------------------------
# Ingest is the only place that parses SNODAS filenames; the read path finds a
# variable's file by its glob and gets the date from the cogs/<date>/ directory.


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

    def write_cog(self: Self, output_dir: Path, force: bool = False) -> None:
        output_path = output_dir / f'{self.name}.tif'

        if not force and output_path.exists():
            raise FileExistsError(
                f'Unable to write COG: {output_path} already exists. '
                'Remove file and try again or use `force=True`.',
            )

        # GDAL's SNODAS/raw driver has a header line-length limit; trim first.
        self.trim_header(self.path)

        with rasterio.open(self.path) as src:
            array = src.read(1)
            transform = src.transform
            crs = src.crs or WGS84
            nodata = src.nodata

        write_cog(
            output_path,
            array,
            transform=transform,
            crs=crs,
            nodata=nodata,
            tile_size=SNODAS_SPEC.grid_params.tile_size,
            predictor=2,
        )


class SNODASInputRasterSet:
    def __init__(
        self: Self,
        # TODO: make SNODASInputRaster a generic on product and type these strongly
        swe: SNODASInputRaster,
        depth: SNODASInputRaster,
        runoff: SNODASInputRaster,
        sublimation: SNODASInputRaster,
        sublimation_blowing: SNODASInputRaster,
        precip_solid: SNODASInputRaster,
        precip_liquid: SNODASInputRaster,
        average_temp: SNODASInputRaster,
    ) -> None:
        self.swe = swe
        self.snow_depth = depth
        self.runoff = runoff
        self.sublimation = sublimation
        self.sublimation_blowing = sublimation_blowing
        self.precip_solid = precip_solid
        self.precip_liquid = precip_liquid
        self.average_temp = average_temp
        self.date = self.validate_dates(self)
        self.validate_revision(self)

    def __iter__(self: Self) -> Iterator[SNODASInputRaster]:
        yield self.swe
        yield self.snow_depth
        yield self.runoff
        yield self.sublimation
        yield self.sublimation_blowing
        yield self.precip_solid
        yield self.precip_liquid
        yield self.average_temp

    @staticmethod
    def validate_dates(rasters: Iterable[SNODASInputRaster]) -> date:
        dates: set[date] = set()
        for raster in rasters:
            dates.add(raster.datetime.date())

        if len(dates) > 1:
            raise SNODASError(
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
            raise SNODASError(
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
        rasters: dict[str, SNODASInputRaster] = {}

        with tempfile.TemporaryDirectory() as _temp:
            temp = Path(_temp)
            with tarfile.open(snodas_tar) as tar:
                tar.extractall(temp, filter='data')

            for f in temp.glob('*.gz'):
                outpath = extract_dir / f.stem
                with gzip.open(f, 'rb') as f_in, outpath.open('wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            hdrs: list[Path] = []
            for ext in HDR_EXTS:
                hdrs.extend(extract_dir.glob(f'*{ext}'))

            for hdr in hdrs:
                file_info: SNODASInputRaster = SNODASInputRaster(hdr)
                rasters[file_info.product.value] = file_info

        return cls(**rasters)
