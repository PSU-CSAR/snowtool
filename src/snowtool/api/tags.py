"""OpenAPI tag names for the routers.

Tag descriptions and external docs (if ever needed) belong at the app-assembly
site, built via ``gazebo.tags.Tag``/``gazebo.tags.tags_metadata`` -- not here.
"""

from enum import StrEnum


class Tags(StrEnum):
    ROOT = 'root'
    DATASETS = 'datasets'
    POURPOINTS = 'pourpoints'
    STATS = 'stats'
