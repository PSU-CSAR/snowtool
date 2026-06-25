from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
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
        # We don't support settings from an dotenv file: dotenv files should
        # not be parsed by the application, source with a shell instead. Thus
        # we leave dotenv_settings out of the return.
        return init_settings, env_settings, file_secret_settings

    model_config = SettingsConfigDict(env_prefix='snowtool_')

    # Filesystem location of the snowdb root config (or its directory); the
    # snowdb -- per-dataset COGs, AOI rasters, DEMs, the global AOIs -- is
    # whatever that config defines.
    snowdb_config: Path

    # Max number of open async-tiff handles kept in the read-path LRU cache.
    tiff_cache_size: int = 16384

    # Cap on a crossed zonal-stats query's product size (number of zone cells =
    # output rows). Crossing several fine-grained zone axes multiplies their zone
    # counts; a query over this is rejected before any raster is read. Passed into
    # ZonalStats.calculate by the (future) stats route/command.
    max_zone_cells: int = 10_000
