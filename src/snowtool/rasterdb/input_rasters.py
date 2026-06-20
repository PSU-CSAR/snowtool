import gzip
import shutil
import tarfile
import tempfile

from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Self

import rasterio

from snowtool.exceptions import SNODASError
from snowtool.rasterdb import constants
from snowtool.rasterdb.cog import WGS84, write_cog
from snowtool.rasterdb.fileinfo import BaseFileInfo

HDR_EXTS = ('.Hdr', '.txt')


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
            tile_size=constants.TILE_SIZE,
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
