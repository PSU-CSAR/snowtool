# Zone layers and zoning

A **zone layer** is a static raster derived once from an external source and
co-registered onto every dataset grid — elevation and aspect from a DEM, percent
forest cover from NLCD. It is not time-varying data and it is not the thing being
measured; it exists to *stratify* a basin so a query can report a snow statistic
per elevation band, per aspect class, or over any crossing of those axes rather
than as a single basin-wide number. The layers are generated once (the
[generation engines](zone-generation.md) do the expensive source read), stored
under the dataset directory, and read live at query time. Because they sit on the
same grid as the dataset's dated COGs and the [AOI raster](pourpoints.md), the
[query engine](queries.md) can overlay them pixel-for-pixel with no resampling.

Everything about zone layers is pluggable along three axes that interlock, and
reading them together is the point: a *provider* decides what kinds of layers
exist and how they are built, a *source* decides where the input data comes from,
and a *scheme* decides how a layer's pixel values become zones. A new kind of
zone layer touches only the first; a new place to get the same data touches only
the second; a new way to slice the same pixels touches only the third.

## Providers: what layers exist

A `ZoneLayerProvider` (`snowdb/zones/zone_layer.py`) is one kind of zone layer.
It names the layers it writes (each a `ZoneLayer`: filename, dtype, nodata, band
descriptions, a query key, and an optional zoning scheme), the subdirectory its
set lives in, the shared provenance tag its generation stamps, the default source
to read from, and how to run its generation engine. The two built-ins are
`TerrainProvider` (`snowdb/zones/terrain.py`) and `LandCoverProvider`
(`snowdb/zones/landcover.py`), collected into `DEFAULT_ZONE_LAYER_PROVIDERS` in
`snowdb/zones/zone_layer_providers.py` — the registry a `SnowDb` is built with,
passed in rather than reached for as a global so tests and entrypoints can supply
their own. Adding a zone-layer kind is one provider plus one entry in that tuple,
with no edits to `Dataset`, `SnowDb`, the CLI, or diagnostics: the query surface
discovers query-able layers generically through `available_zones`, which keys
every layer that declares a scheme as `'<provider>.<layer.key>'` (e.g.
`'terrain.elevation'`).

## Sources: where the input comes from

A `ZoneLayerSource` is the fine-resolution input a provider reads during
generation. It is a property of the *database*, not of any one dataset — one
source read bins into every grid in a single pass — so it is declared in the root
config and resolved to one source per provider on the `SnowDb`, overridable per
command. Terrain's `DemSource` resolves to either a `LocalFile` DEM or `ThreeDEP`
(USGS 3DEP streamed from S3); land cover's `LandCoverSource` resolves to a
`LocalFile` NLCD raster or `AnnualNLCD` (the MRLC national bundle). Because the
sources are injected through the config and the CLI's `--source` flag rather than
hard-wired into a provider, a test binds a `LocalFile` where production reads the
network. The [generation page](zone-generation.md) covers the sources in depth;
what matters here is that the same provider builds the same layers regardless of
which source fed it.

## Schemes: how pixels become zones

A `ZoneScheme` (`snowdb/zones/zoning.py`) is the rule that turns a layer's native
pixel values into zones. Three kinds are supported:

- `BandedZoning` — contiguous numeric bands aligned to 0 over a fixed domain
  (elevation in feet, the orientation components in percent). Band `i` is
  `[i*step, (i+1)*step)`; aligning to 0 means a given band means the same thing
  regardless of the domain's edges.
- `ThresholdZoning` — a binary split at a threshold: *below* versus *at-or-above*
  it (forest cover forested/unforested, aspect entropy high-/low-signal).
- `CategoricalZoning` — a fixed set of discrete classes keyed by their on-disk
  pixel codes (the aspect classes N/E/S/W/flat).

Every scheme implements one `assign` contract: it maps an array of native pixel
values to per-pixel zone *ordinals*, where `-1` means "out of zone" — a single
sentinel that uniformly covers both layer-nodata pixels and values outside the
scheme's domain, so the [query engine](queries.md) excludes them the same way no
matter which scheme produced them. The zones a scheme enumerates are
self-describing descriptors — a `BandZone` carries its `[min, max)` bounds and
unit, a `ClassZone` its pixel code and label, a `ThresholdZone` its side and the
active threshold — so a crossed query can report each cell of the result as a
typed, labelled zone without the caller switching on the scheme kind.

## Zone layer sets and live reads

A provider's layers live together as a `ZoneLayerSet`: the on-disk directory
under `data/<name>/<subdir>/` plus filesystem operations on it — which layers are
present, a tiled reader for any layer, and the set's shared provenance hash
(read from the first layer's tags, since generation stamps every layer of a set
identically). Terrain's set is `data/<name>/terrain/`, land cover's is
`data/<name>/landcover/`. The layers are read live when a query runs, not folded
into any other artifact: the AOI raster carries only per-pixel cell area and is
deliberately decoupled from the zone layers, so regenerating terrain never forces
an AOI rebuild and vice versa. That decoupling is enforced by the
[provenance](provenance.md) tags — the terrain set carries `SNOWTOOL_DEM_HASH`,
the land-cover set `SNOWTOOL_NLCD_HASH`, each a versioned hash that reads as stale
on either a content change or a format-version bump.

## Elevation bands over one global range

The elevation scheme bands the whole `[MIN_ELEVATION_M, MAX_ELEVATION_M]`
bracket in `snowdb/constants.py` (-100 m to 4500 m), converted to feet, and this
single global range is shared by every dataset rather than being taken from each
dataset's own DEM range. The reason is comparability: bands must line up across
AOIs *and* across datasets, because the AOI is the geographic unit of interest,
not the dataset. The bracket only needs to reach the highest and lowest terrain
any AOI can contain — the values are floored to whole `band_step_ft` bins by the
banding, so a generous bracket costs nothing but a few empty (null) bands at the
extremes. CONUS spans roughly Badwater (-86 m) to Mt. Whitney (4421 m); the
bracket covers it with headroom, and because resampling a source DEM onto the
~1 km dataset grids only ever pulls extremes inward, a bracket valid for the
source can never be exceeded by the resampled layer.

## Query parameters and overrides

A structured scheme (banded, bucketed, threshold) has one tunable parameter: a
band width, a bucket count, or a split point. Each layer's default comes from
the dataset's config `zones` block, where it parses to one of four
single-field member models of the `ZoneLayerParams` union (`snowdb/config.py`)
— `BandStepParams.band_step_ft` for elevation, `BucketParams.buckets` for the
orientation components, `ThresholdParams.threshold_pct` for forest cover,
`EntropyThresholdParams.entropy_threshold` for aspect entropy — folded into
the scheme by `configured`; `None` (an unconfigured layer) falls back to the
scheme's own default, and a param belonging to a different scheme raises
`ZoneParamsError`. A single query can then override that default per axis with
a `LAYER:override` token (the CLI `--zone` flag), parsed in
`parse_zone_selection` (`snowdb/zonal_stats.py`):
the token is delegated to the layer's own scheme, which types it (a banded
axis parses an int band width, a threshold axis a float split point) or
rejects it. A categorical axis takes no override — `parse_override` on the
base scheme raises — so `terrain.aspect` is selected bare.

## The built-in layers

`TerrainProvider` writes five layers, all `float32` except the aspect class,
sharing the `SNOWTOOL_DEM_HASH` tag:

- `elevation.tif` — mean elevation in metres (nodata -9999.0), banded in feet
  (`terrain.elevation`, default step 1000 ft).
- `aspect_majority.tif` — `uint8` majority aspect class per cell (0 N, 1 E, 2 S,
  3 W, 4 flat; nodata 255), categorical (`terrain.aspect`).
- `northness.tif` / `eastness.tif` — mean `cos(aspect)` and mean `sin(aspect)`
  over the cell's non-flat pixels, each a `float32` single-band layer in `[-1, 1]`
  with the finite sentinel -9999.0 where a cell has no non-flat pixels. Each is
  its own banded axis over `[-1, 1]` expressed in percent (`terrain.northness`,
  `terrain.eastness`; default step 50 pct). Two single-band files rather than one
  two-band file because a `ZoneLayer` is one file, one band, one scheme per query
  key.
- `aspect_entropy.tif` — `float32` normalised Shannon entropy of the cell's
  five-class aspect distribution in `[0, 1]` (nodata -1.0), thresholded
  below/at-or-above into a high-signal vs low-signal split
  (`terrain.aspect_entropy`, default threshold 0.5). Meant to be crossed with the
  majority axis to keep only cells whose dominant aspect is well-supported.

`LandCoverProvider` writes one layer under the `SNOWTOOL_NLCD_HASH` tag:

- `forest_cover_pct.tif` — `uint8` percent forest cover, the share of the cell's
  NLCD pixels classed as forest (0..100; nodata 255). It is thresholded, not
  banded — the question is whether a cell is forested — so
  `landcover.forest_cover` splits unforested vs forested at `threshold_pct`
  (default 50%).

How these arrays are actually produced from a DEM and an NLCD raster — the
streaming, the Horn window, the point-in-cell binning, and the sources — is the
subject of [Generating zone layers](zone-generation.md).
