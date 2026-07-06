# Generating zone layers

Zone layers are built by streaming one fine-resolution source once and binning it
into every target grid in the same pass. This is the shape both generation
engines share and the reason `dataset generate-zones` takes several datasets at
once: a provider's source is read a single time over the combined extent of all
the named datasets and dropped into each of their grids, so standing up SNODAS,
SWANN, and INSTARR together pays terrain's DEM reproject and land cover's ~1.5 GB
NLCD download exactly once instead of once per dataset. Passing a single target
degenerates to per-dataset generation; nothing special-cases the count. This page
is the mechanism behind the [zone-layer framework](zones.md); read that first for
what a layer, provider, source, and scheme are.

The machinery that is genuinely common lives in `snowdb/zones/generate_common.py`
so a third provider never copies it. `require_absent_layers` is the pre-flight
existence guard: before the expensive source read, it refuses to generate over
any target that already has the layers (callers only reach it when not forcing).
`finalize_and_stamp` is the digest-then-stamp [provenance](provenance.md) pass —
it finalizes every target's accumulator, folds each target's name and a
caller-chosen digest array into one sha256 (targets sorted by name for a
deterministic hash), turns it into one versioned hash, and stamps that identical
hash on every layer of every target so everything produced together reconciles as
one set. `cells_for_points` and `pixel_centre_coords` are the point-in-cell
binning arithmetic: a pixel centre's coordinates, and the flattened grid cell a
transformed point lands in. Both engines share this exact arithmetic so their
generation hashes stay bit-reproducible.

## Terrain

Terrain generation (`snowdb/zones/terrain_generate.py`) runs in three stages,
because aspect cannot be resampled after the fact — it must be computed from
elevation at the source resolution — and slope and aspect are gradients that come
out distorted on a geographic grid whose pixels are non-square degrees.

1. **Reproject to one fine, projected work grid.** The DEM source is lazily
   reprojected, bilinear, through a `rasterio` `WarpedVRT` into a single common
   lattice: a projected CRS at a fine resolution (`DEFAULT_WORK_CRS = EPSG:5070`,
   CONUS Albers, which makes dx equal dy in real metres; `DEFAULT_WORK_RESOLUTION
   = 10 m`, matching 3DEP). This shared intermediate is the *only* true resample,
   and it is of elevation. A `DemSource` carries the right work CRS and resolution
   for its data, so these constants are only fallbacks.
2. **Derive everything at the work resolution.** The work grid is streamed in
   blocks, each read with a one-pixel halo so the 3×3 Horn window is exact across
   block edges (the Horn pass trims one pixel per edge, leaving the inner block
   aligned with the nominal block, no overlap and no double counting). Each fine
   pixel gets a slope, an aspect, an aspect class (a cardinal quadrant, or flat
   below `FLAT_SLOPE_DEG = 2°`), and `cos`/`sin` of its aspect.
3. **Aggregate fine pixels into target cells.** Each fine pixel's centre is
   transformed with pyproj into each target grid's CRS and assigned to the cell it
   lands in — point-in-cell, not fractional-area. Per cell the engine accumulates
   class counts, an elevation sum, and `cos`/`sin` sums over the non-flat pixels.
   Finalizing reduces those: the class counts give the majority aspect (nodata
   where a cell caught no pixels); the elevation sum divided by the pixel count
   gives mean elevation (an area-mean, since every fine pixel has equal area in
   the projected CRS); the `cos`/`sin` sums divided by the non-flat count give
   northness and eastness (the first circular moment), with a finite sentinel
   where a cell has no non-flat pixels; and the same five class counts give a
   Shannon entropy of the aspect distribution, normalised by `ln(5)` to `[0, 1]`.

Elevation rides this same shared work surface — source, to bilinear work grid, to
per-cell mean — rather than a direct `average` warp. That costs a small,
sub-metre error (bilinear is a mild low-pass and point-in-cell binning is not
area-weighted) accepted in exchange for co-registration: every layer comes from
the one surface and the same binning, so elevation and aspect line up by
construction. The pass is an input-driven scatter — one read, warp, Horn, and
reproject shared across all targets, with each fine pixel dropped into whatever
target cell it lands in — which is exactly what lets the multi-dataset generation
share the expensive target-independent work.

The block reproject and derivation run on a thread pool (the pyproj transform
dominates and releases the GIL), but the binning is serial and in fixed block
order, so the accumulation — and therefore the generation hash — is identical
regardless of worker count. The generation digest is taken over the elevation
array alone: it identifies the generation, not each raster. `generate-zones`
exposes the pool's `--workers` and `--block-size` knobs; `block_size` bounds
per-worker memory (a worker holds a handful of block-sized arrays, so transient
RAM scales with `workers * block_size²`) and neither changes the result.

## Land cover

Land-cover generation (`snowdb/zones/landcover_generate.py`) is simpler because
the layer is categorical: there is no slope or aspect, hence no Horn
neighbourhood and no block halo, and no projected work grid at all. Aspect needs
an undistorted metric grid; a *fraction of pixels* does not, so the engine bins
the source in its native CRS directly. It reads the NLCD raster only over the
combined extent of the target grids, streams it in blocks, and transforms each
valid pixel's centre with pyproj into each target grid's cell. Per cell it
accumulates a count of valid pixels and of forest pixels; the layer value is
`round(100 * forest / valid)` as `uint8`, with 255 where a cell caught no valid
pixels. The forest classes are `FOREST_CLASSES = (41, 42, 43)` — deciduous,
evergreen, mixed — in `snowdb/constants.py`; adding 90 (woody wetlands) there
would count forested wetlands as forest. The generation digest is taken over the
forest-percent array.

## Sources in depth

`ThreeDEP` (`snowdb/zones/terrain_source.py`) streams USGS 3DEP 1/3 arc-second
tiles from the public `prd-tnm` S3 bucket. Discovery is a single concurrent
anonymous pass on the async-tiff store layer the COG read path already uses: each
candidate tile for the extent is opened anonymously, which both proves it exists
and yields its GeoTIFF geo-header, so the existing tiles are stitched into a VRT
mosaic — written by hand as XML, with no `osgeo`/`gdal.BuildVRT` — that rasterio's
bundled GDAL then reads. The actual pixel streaming is GDAL over `/vsis3/`, lazy
and per-block through the terrain engine's `WarpedVRT`, so range reads only ever
touch the intersecting portions of the tiles a target grid actually covers.

`AnnualNLCD` (`snowdb/zones/landcover_source.py`) is download-and-cache rather
than stream, for a concrete reason: unlike 3DEP's cloud-optimized tiles, the
Annual NLCD land cover ships as a single national GeoTIFF inside a `.zip` over an
open MRLC HTTPS download — not a range-readable cloud object — and the `.tif`
inside is deflate-compressed, which `/vsizip/` cannot range-read in place. But it
is one static national file, so it is fetched once (to a `.part` sidecar then
renamed, so an interrupted download never looks complete), extracted to a cache,
and reused. The cache lives under the snowdb root at `.cache/landcover/`, so a
repeated generation reuses the large download.

Both providers also offer a `LocalFile` source for offline or operator-supplied
inputs — a DEM or NLCD raster already on disk. A `LocalFile` is selected either
per command with `--source PROVIDER PATH` (repeatable) or persistently through
the root config's `sources` map, which pins a provider name to a path (see
[configuration](configuration.md)); an unconfigured provider falls back to its
network default. This is the seam that lets a test bind local fixtures where
production reads 3DEP and NLCD.

## Provenance stamping

At the end of a pass, `finalize_and_stamp` digests the generated array (elevation
for terrain, forest-percent for land cover) into one sha256, wraps it in a
versioned hash carrying the provider's format version, and stamps that identical
hash on every layer of every target's set. A later run reads the set as stale if
either the content or the format version has changed, which is what drives a
rebuild without any separate bookkeeping. The [provenance](provenance.md) page
covers the versioned-hash mechanism and the staleness check in full.
