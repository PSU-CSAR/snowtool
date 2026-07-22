"""Shared fixtures, including the synthetic-grid pipeline fixtures.

Everything runs on a tiny 512x512 (2x2 tile) grid so the full pipeline —
resample, area raster, AOI rasterize, zonal stats — exercises real rasterio /
griffine code on hand-computable data, with no system GDAL and no large inputs.
These live at the top level so both the ``snowdb`` and ``cli`` suites reuse them.
"""

import hashlib
import json

from contextlib import contextmanager

import numpy
import pytest
import rasterio

from rasterio.crs import CRS
from rasterio.transform import from_origin

from snowtool.api.settings import Settings
from snowtool.snowdb.constants import DEM_HASH_TAG, NLCD_HASH_TAG
from snowtool.snowdb.dataset import Dataset
from snowtool.snowdb.datasets import SNODAS_VARIABLES
from snowtool.snowdb.db import SnowDb
from snowtool.snowdb.provenance import versioned_hash
from snowtool.snowdb.raster.cog import SOURCE_HASH_TAG, write_cog
from snowtool.snowdb.spec import DatasetSpec, GridParams
from snowtool.snowdb.zones.landcover import FOREST_COVER, LANDCOVER_FORMAT_VERSION
from snowtool.snowdb.zones.terrain import (
    ASPECT_COMPONENT_NODATA,
    ASPECT_ENTROPY,
    ASPECT_FLAT,
    ASPECT_MAJORITY,
    EASTNESS,
    ELEVATION,
    NORTHNESS,
    TERRAIN_FORMAT_VERSION,
)

# gazebo's pytest plugin (assert_problem / assert_has_link / drive_pagination +
# fixtures) is opt-in, not auto-registered.
pytest_plugins = ['gazebo.testing']


class CapturingTask:
    """One task a :class:`CapturingProgress` recorded: its label/total and how far
    it was advanced (so a test can assert a bar was driven to completion)."""

    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.advanced = 0

    def advance(self, n=1):
        self.advanced += n


class CapturingProgress:
    """A ``ProgressReporter`` that records every tracked task instead of drawing.

    Lets a test assert the progress *wiring* -- that an operation opened a task with
    the expected label/total and advanced it -- without a terminal or a real bar.
    """

    def __init__(self):
        self.tasks = []

    @contextmanager
    def track(self, label, *, total=None):
        task = CapturingTask(label, total)
        self.tasks.append(task)
        yield task


# Small synthetic grid parameters.
ORIGIN_X = -120.0
ORIGIN_Y = 45.0
PX = 0.01
SIZE = 512
TILE = 256

DEM_ELEVATION_M = 1000.0  # uniform; 1000 m -> ~3280 ft -> band (3000, 4000) ft
DEM_NODATA = -9999.0
SWE_VALUE = 50  # uniform int16 SWE value
NLCD_FOREST_CLASS = 42  # evergreen forest (in FOREST_CLASSES)
NLCD_NONFOREST_CLASS = 81  # pasture/hay (not forest)
FOREST_PCT_VALUE = 100  # uniform all-forest synthetic land cover


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(snowdb_config=tmp_path)


def make_snowdb(root, specs, **kwargs):
    """Build a SnowDb in code from inline dataset configs (no files written).

    The config-object replacement for the old spec-injection constructor: it
    builds a RootConfig rooted at ``root`` (its ``path`` set but the file not
    written) with each spec embedded as an inline dataset, then constructs the
    SnowDb directly. Use this where a test wants a bound SnowDb without staging
    config files on disk.
    """
    from pathlib import Path

    from snowtool.snowdb.config import CONFIG_FILENAME, InlineDatasetLink, RootConfig
    from snowtool.snowdb.datasets import config_from_spec

    config = RootConfig.create()
    config.path = Path(root) / CONFIG_FILENAME
    for spec in specs:
        config.datasets[spec.name] = InlineDatasetLink(dataset=config_from_spec(spec))
    return SnowDb(config, **kwargs)


def make_manager(root, specs, **kwargs):
    """Build a SnowDbManager over an in-code SnowDb (see :func:`make_snowdb`).

    The write-side counterpart to ``make_snowdb``: where a test exercises a
    management op, wrap the inline-config SnowDb in a manager so writes go through
    the admin surface while reads stay on ``manager.db``.
    """
    from snowtool.snowdb.manager import SnowDbManager

    return SnowDbManager(make_snowdb(root, specs, **kwargs))


def register_dataset_config(manager, name, config, *, active=True):
    """Stage ``config`` at ``data/<name>/dataset.json`` and register its link.

    ``active`` forwards to :meth:`SnowDbManager.register_dataset` so a test can
    register a dataset the reader does not serve (a registered-but-inactive link).
    """
    from snowtool.snowdb.config import DATASET_CONFIG_FILENAME

    ds_dir = manager.db.dataset_dir(name, config)
    ds_dir.mkdir(parents=True, exist_ok=True)
    config_path = ds_dir / DATASET_CONFIG_FILENAME
    config.save(config_path)
    manager.register_dataset(name, config_path, active=active)
    return config_path


def init_with_builtins(root):
    """Initialize ``root`` with every built-in dataset registered (from templates)."""
    from snowtool.snowdb.datasets import DATASET_TEMPLATES
    from snowtool.snowdb.manager import SnowDbManager

    manager = SnowDbManager.initialize(root)
    for name, config in DATASET_TEMPLATES.items():
        register_dataset_config(manager, name, config)
    return manager


def populate_bound_root(
    manager,
    spec,
    pourpoint_geojson,
    *,
    rasterize=True,
    ingest=True,
):
    """Populate an already-initialized, dataset-bound ``manager`` for a query.

    The init-independent population step shared by :func:`populate_synthetic_root`
    and ``test_stats_cli``'s ``populated_root``: imports the AOI, writes uniform
    terrain + land cover, burns the AOI raster, and (optionally) ingests a uniform
    SWE COG for 2018-04-27. Returns the catalog ``SnowDb``.
    """
    from snowtool.snowdb.pourpoint import Pourpoint

    manager.import_pourpoints(pourpoint_geojson)
    dataset = manager.db[spec.name]
    write_terrain(dataset)
    write_landcover(dataset)
    if rasterize:
        dataset.rasterize_aoi(Pourpoint.from_geojson(pourpoint_geojson), rebuild=True)
    if ingest:
        write_swe_cog(dataset)
    return manager.db


def populate_synthetic_root(
    root,
    spec,
    pourpoint_geojson,
    *,
    rasterize=True,
    ingest=True,
):
    """Populate ``root`` end-to-end with the synthetic ``spec`` dataset.

    The shared builder behind the reader and API stats tests: it initializes the
    root, registers + binds the dataset, then runs the shared population step
    (:func:`populate_bound_root`). Returns the catalog ``SnowDb``.
    """
    from snowtool.snowdb.datasets import config_from_spec
    from snowtool.snowdb.manager import SnowDbManager

    manager = SnowDbManager.initialize(root)
    register_dataset_config(manager, spec.name, config_from_spec(spec))
    # Reopen so the freshly-registered dataset is bound.
    manager = SnowDbManager.open(root)
    return populate_bound_root(
        manager,
        spec,
        pourpoint_geojson,
        rasterize=rasterize,
        ingest=ingest,
    )


def make_spec(name, base, **overrides):
    """A second DatasetSpec sharing ``base``'s grid (and variables), renamed.

    The shared "another dataset on the same grid" builder: by default it reuses
    ``base``'s grid params and variables, so a test only names the new dataset and
    overrides what it cares about (e.g. ``variables=()`` for a bare spec).
    """
    fields = {
        'name': name,
        'grid_params': base.grid_params,
        'variables': base.variables.values(),
    }
    fields.update(overrides)
    return DatasetSpec(**fields)


@pytest.fixture
def spec():
    """A tiny synthetic DatasetSpec (2x2 tile geographic grid)."""
    return DatasetSpec(
        name='test',
        grid_params=GridParams(
            origin_x=ORIGIN_X,
            origin_y=ORIGIN_Y,
            px_size=PX,
            cols=SIZE,
            rows=SIZE,
            tile_size=TILE,
        ),
        variables=SNODAS_VARIABLES,
    )


@pytest.fixture
def grid(spec):
    return spec.grid


@pytest.fixture
def source_dem(tmp_path, grid):
    """A uniform-elevation source DEM on the grid extent."""
    path = tmp_path / 'source_dem.tif'
    array = numpy.full((SIZE, SIZE), DEM_ELEVATION_M, dtype=numpy.float32)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=1,
        dtype='float32',
        crs=CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        nodata=DEM_NODATA,
    ) as dst:
        dst.write(array, 1)
    return path


def write_uniform_terrain(
    directory,
    *,
    base_grid,
    crs,
    tile_size: int,
    elevation_value: float = DEM_ELEVATION_M,
    northness_value: float = ASPECT_COMPONENT_NODATA,
    eastness_value: float = ASPECT_COMPONENT_NODATA,
) -> str:
    """Write a uniform terrain set (elevation + flat aspect) into ``directory``.

    The shared primitive behind both the ``dataset`` fixture's :func:`write_terrain`
    and the CLI suite's fake terrain engine: given the raw grid pieces, it writes
    what :func:`generate_terrain` produces -- uniform elevation, all-flat aspect --
    and stamps the DEM provenance hash on every layer. Returns that hash.

    All-flat terrain has no non-flat pixels, so the northness/eastness orientation
    layers default to the finite :data:`ASPECT_COMPONENT_NODATA` sentinel; a test
    that needs a real orientation zone passes uniform ``northness_value`` /
    ``eastness_value`` in ``[-1, 1]``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shape = (base_grid.rows, base_grid.cols)

    elevation = numpy.full(shape, elevation_value, dtype='float32')
    dem_hash = versioned_hash(
        TERRAIN_FORMAT_VERSION,
        hashlib.sha256(elevation.tobytes()).hexdigest(),
    )
    common = {
        'transform': base_grid.transform,
        'crs': crs,
        'tile_size': tile_size,
        'tags': {DEM_HASH_TAG: dem_hash},
    }

    write_cog(
        directory / ELEVATION.filename,
        elevation,
        nodata=ELEVATION.nodata,
        band_descriptions=ELEVATION.band_descriptions,
        **common,
    )
    write_cog(
        directory / ASPECT_MAJORITY.filename,
        numpy.full(shape, ASPECT_FLAT, dtype='uint8'),
        nodata=ASPECT_MAJORITY.nodata,
        band_descriptions=ASPECT_MAJORITY.band_descriptions,
        **common,
    )
    write_cog(
        directory / NORTHNESS.filename,
        numpy.full(shape, northness_value, dtype='float32'),
        nodata=NORTHNESS.nodata,
        band_descriptions=NORTHNESS.band_descriptions,
        **common,
    )
    write_cog(
        directory / EASTNESS.filename,
        numpy.full(shape, eastness_value, dtype='float32'),
        nodata=EASTNESS.nodata,
        band_descriptions=EASTNESS.band_descriptions,
        **common,
    )
    # All-flat terrain -> all aspect mass in the flat class -> zero entropy.
    write_cog(
        directory / ASPECT_ENTROPY.filename,
        numpy.zeros(shape, dtype='float32'),
        nodata=ASPECT_ENTROPY.nodata,
        band_descriptions=ASPECT_ENTROPY.band_descriptions,
        **common,
    )
    return dem_hash


def write_uniform_landcover(
    directory,
    *,
    base_grid,
    crs,
    tile_size: int,
    pct: int = FOREST_PCT_VALUE,
) -> str:
    """Write a uniform percent-forest land-cover layer into ``directory``.

    The shared primitive behind the ``dataset`` fixture's :func:`write_landcover`
    and the CLI suite's fake land-cover engine. Returns the NLCD provenance hash.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shape = (base_grid.rows, base_grid.cols)

    forest = numpy.full(shape, pct, dtype='uint8')
    nlcd_hash = versioned_hash(
        LANDCOVER_FORMAT_VERSION,
        hashlib.sha256(forest.tobytes()).hexdigest(),
    )

    write_cog(
        directory / FOREST_COVER.filename,
        forest,
        transform=base_grid.transform,
        crs=crs,
        tile_size=tile_size,
        nodata=FOREST_COVER.nodata,
        tags={NLCD_HASH_TAG: nlcd_hash},
        band_descriptions=FOREST_COVER.band_descriptions,
    )
    return nlcd_hash


def write_terrain(
    dataset,
    elevation_value: float = DEM_ELEVATION_M,
    northness_value: float = ASPECT_COMPONENT_NODATA,
    eastness_value: float = ASPECT_COMPONENT_NODATA,
) -> str:
    """Write a uniform terrain set onto a dataset's grid (no engine run).

    Convenience wrapper over :func:`write_uniform_terrain` for tests that hold a
    ``dataset`` and just need terrain present (e.g. elevation banding, or a uniform
    northness/eastness orientation zone).
    """
    return write_uniform_terrain(
        dataset.zones['terrain'].directory,
        base_grid=dataset.grid.base_grid,
        crs=dataset.grid_crs,
        tile_size=dataset.spec.grid_params.tile_size,
        elevation_value=elevation_value,
        northness_value=northness_value,
        eastness_value=eastness_value,
    )


def write_landcover(dataset, pct: int = FOREST_PCT_VALUE) -> str:
    """Write a uniform land-cover layer onto a dataset's grid (no engine run)."""
    return write_uniform_landcover(
        dataset.zones['landcover'].directory,
        base_grid=dataset.grid.base_grid,
        crs=dataset.grid_crs,
        tile_size=dataset.spec.grid_params.tile_size,
        pct=pct,
    )


@pytest.fixture
def source_nlcd(tmp_path, grid):
    """A synthetic NLCD land-cover source on the grid extent.

    The left half is forest (class 42), the right half non-forest (class 81), so a
    cell-fraction reduction has a hand-computable, non-uniform result.
    """
    path = tmp_path / 'source_nlcd.tif'
    array = numpy.full((SIZE, SIZE), NLCD_NONFOREST_CLASS, dtype=numpy.uint8)
    array[:, : SIZE // 2] = NLCD_FOREST_CLASS
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=SIZE,
        width=SIZE,
        count=1,
        dtype='uint8',
        crs=CRS.from_epsg(4326),
        transform=grid.base_grid.transform,
        nodata=0,
    ) as dst:
        dst.write(array, 1)
    return path


@pytest.fixture
def dataset(tmp_path, spec):
    """A fully created Dataset: directory, area raster, terrain + land-cover sets."""
    ds, _ = Dataset.create(spec, tmp_path / 'db')
    write_terrain(ds)
    write_landcover(ds)
    return ds


# The synthetic-grid canonical basin: a rectangle well inside the first tile
# (lon -119.9..-119.0, lat 44.9..44.0) with its outflow point at the centre.
_DEFAULT_BOX = (-119.9, 44.9, -119.0, 44.0)
_DEFAULT_POINT = (-119.45, 44.45)


def _rect_ring(box):
    """A closed rectangular ring (lon/lat) from an ``(x0, y0, x1, y1)`` box."""
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def write_pourpoint_record(
    path,
    triplet='10371000:NV:USGS',
    *,
    box=None,
    point=None,
    polygon=None,
    point_only=False,
    properties=None,
):
    """Write a pourpoint-record geojson file and return its path.

    The shared record factory: emits the canonical synthetic-grid shape -- a
    ``GeometryCollection`` of an outflow point plus a basin polygon, ``id`` set to
    ``triplet`` -- or, when ``point_only``, a point-only ``Feature`` (no basin).
    Callers pass their triplet/corners; ``box`` is an ``(x0, y0, x1, y1)`` rectangle
    (default the canonical basin), or pass ``polygon`` for an explicit closed ring.
    """
    point = _DEFAULT_POINT if point is None else point
    point_geom = {'type': 'Point', 'coordinates': list(point)}
    if properties is None:
        properties = {'name': 'Test Basin', 'source': 'test'}
    if point_only:
        feature = {
            'type': 'Feature',
            'id': triplet,
            'geometry': point_geom,
            'properties': properties,
        }
    else:
        ring = polygon if polygon is not None else _rect_ring(box or _DEFAULT_BOX)
        feature = {
            'type': 'GeometryCollection',
            'id': triplet,
            'geometries': [
                point_geom,
                {'type': 'Polygon', 'coordinates': [ring]},
            ],
            'properties': properties,
        }
    path.write_text(json.dumps(feature))
    return path


def write_aoi_record(
    directory,
    triplet,
    *,
    with_polygon=True,
    polygon=None,
    box=None,
    point=None,
    properties=None,
):
    """Write a station AOI record named ``<triplet>.geojson`` into ``directory``.

    The shared AOI-record writer for the suite: a thin ``directory``-oriented
    adapter over :func:`write_pourpoint_record` that names the file from the
    triplet and, by default, stamps the canonical station properties
    (``active``/``basinarea``). ``polygon`` may be a Polygon geometry dict (its
    ring is unwrapped) or a raw ring; ``box``/``point`` follow
    ``write_pourpoint_record``. Pass ``properties`` to override the defaults.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if properties is None:
        properties = {
            'name': triplet,
            'source': 'test',
            'active': True,
            'basinarea': 5.2,
        }
    ring = polygon['coordinates'][0] if isinstance(polygon, dict) else polygon
    return write_pourpoint_record(
        directory / f'{triplet.replace(":", "_")}.geojson',
        triplet,
        box=box,
        point=point,
        polygon=ring,
        point_only=not with_polygon,
        properties=properties,
    )


def write_swe_cog(dataset, date_str: str = '20180427', value: int = SWE_VALUE):
    """Write a uniform int16 SWE COG for ``date_str`` into ``dataset``'s cogs dir."""
    out_dir = dataset._cogs / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{snodas_swe_name(date_str)}.tif'
    write_cog(
        path,
        numpy.full((SIZE, SIZE), value, dtype=numpy.int16),
        transform=dataset.grid.base_grid.transform,
        tile_size=TILE,
        predictor=2,
    )
    return path


@pytest.fixture
def pourpoint_geojson(tmp_path):
    """A pourpoint with a polygon inside tile (0, 0)."""
    return write_pourpoint_record(tmp_path / 'pourpoint.geojson', '12345:MT:USGS')


def snodas_swe_name(date_str: str = '20180427') -> str:
    """A filename matching the SNODAS SWE regex + product glob."""
    # region=us model=ssm datatype=v1 code=1034 scaled=S vcode=lL00
    # T timecode=0001 TTNATS <date> hour=05 interval=H offset=P001
    return f'us_ssmv11034SlL00T0001TTNATS{date_str}05HP001'


def write_marker_cog(path, source_hash: str | None) -> None:
    """Write a tiny real COG at ``path``, tagged with ``source_hash`` if given.

    Real (not name-only) so a header-only ``SOURCE_HASH`` read (the ingest skip
    check) can open it; ``source_hash=None`` simulates a legacy pre-hash COG.
    """
    tags = {SOURCE_HASH_TAG: source_hash} if source_hash is not None else None
    write_cog(
        path,
        numpy.zeros((16, 16), dtype='int16'),
        transform=from_origin(-100.0, 40.0, 0.01, 0.01),
        tile_size=16,
        predictor=2,
        tags=tags,
    )


class FakeRaster:
    """A ``WritableRaster`` that drops a tiny real marker COG into the date dir.

    ``out_name`` is the filename the COG lands under; it carries ``source_hash`` in
    its ``SOURCE_HASH`` tag (what a real ingester stamps, and what the per-date skip
    check reads back), so a fake ingester built on these drives the genuine atomic
    ``Dataset.write_date_cogs`` path end-to-end.
    """

    def __init__(self, out_name: str, source_hash: str) -> None:
        self.out_name = out_name
        self.source_hash = source_hash

    def write_cog(self, output_dir) -> None:
        write_marker_cog(output_dir / self.out_name, self.source_hash)


def _name_for_glob(glob: str) -> str:
    """A concrete filename that matches ``glob`` (``*`` dropped, ``?``/``[..]`` pinned).

    Turns a variable's ``fnmatch`` glob into one deterministic filename: ``*`` -> "",
    ``?`` -> "0", ``[ab..]`` -> its first char. Distinct variable globs (which differ
    by product code) yield distinct names, so a full set resolves one COG per variable.
    """
    out: list[str] = []
    i = 0
    while i < len(glob):
        char = glob[i]
        if char == '*':
            i += 1
        elif char == '?':
            out.append('0')
            i += 1
        elif char == '[':
            close = glob.index(']', i)
            out.append(glob[i + 1])
            i = close + 1
        else:
            out.append(char)
            i += 1
    return ''.join(out)


def full_marker_out_names(dataset) -> frozenset[str]:
    """The COG filenames :func:`full_marker_rasters` will land, one per spec variable.

    A fake ingester's ``DateIngest.out_names``: the write path's per-date skip check
    reads these (with the source hash) *before* ``build_rasters`` runs, so they must
    match what that build produces.
    """
    return frozenset(
        _name_for_glob(variable.glob) for variable in dataset.spec.variables.values()
    )


def full_marker_rasters(dataset, source_hash: str) -> list[FakeRaster]:
    """One :class:`FakeRaster` per spec variable, covering every required variable.

    A fake ingester's ``build_rasters`` returns this so the real ``write_date_cogs``
    completeness check (every variable must resolve to exactly one COG) passes on a
    full SNODAS spec without a real archive.
    """
    return [
        FakeRaster(_name_for_glob(variable.glob), source_hash)
        for variable in dataset.spec.variables.values()
    ]


@pytest.fixture
def swe_cog(dataset):
    """Write a uniform SWE COG for 2018-04-27 into the db's cogs dir."""
    return write_swe_cog(dataset)
