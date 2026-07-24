from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from snowtool.snowdb.raster.tiff_cache import DEFAULT_TIFF_CACHE_SIZE
from snowtool.snowdb.zonal_stats import (
    DEFAULT_MAX_CONCURRENT_RASTERS,
    DEFAULT_MAX_ZONE_CELLS,
)


class Settings(BaseSettings):
    def __init__(self, *args, _env_file: None = None, **kwargs) -> None:
        if _env_file is not None:
            raise ValueError('Loading settings from a dotenv file is not supported')
        super().__init__(*args, **kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # We don't support settings from a dotenv file: dotenv files should
        # not be parsed by the application, source with a shell instead. Thus
        # we leave dotenv_settings out of the return.
        return init_settings, env_settings, file_secret_settings

    model_config = SettingsConfigDict(env_prefix='snowtool_')

    # These fields' descriptions are also surfaced as CLI --help text (see
    # snowtool.cli.api).
    snowdb_config: Path = Field(
        description='Snowdb config file or its directory.',
    )

    tiff_cache_size: int = Field(
        DEFAULT_TIFF_CACHE_SIZE,
        description='Max open async-tiff handles kept in the read-path LRU cache.',
    )

    max_zone_cells: int = Field(
        DEFAULT_MAX_ZONE_CELLS,
        description='Cap on a crossed zonal-stats query product size (output rows).',
    )

    max_concurrent_rasters: int = Field(
        DEFAULT_MAX_CONCURRENT_RASTERS,
        description='Cap on concurrent per-raster reductions (bounds peak memory / '
        'fetch fan-out on a wide date range; results are unaffected).',
    )
