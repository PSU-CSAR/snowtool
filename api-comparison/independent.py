# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "numpy", "rasterio", "pyproj", "shapely"]
# ///
"""Independent zonal-stats recomputation from raw NOHRSC SNODAS grids.

For each arbiter date, downloads the masked SNODAS tar from the NSIDC archive,
reads the raw SWE grid (int16 big-endian, mm, nodata -9999), rasterizes a
basin polygon onto the SNODAS 1/240-degree grid (pixel-center containment),
and computes basin area and area-weighted mean SWE using per-row geodesic
cell areas. Shares no code with either service.

Because the two services host *different* polygons for some pourpoints, stats
are computed for both: the bagis-pourpoints reference polygon (what the old
service holds) and the polygon served by the new API.
"""

from __future__ import annotations

import gzip
import json
import re
import tarfile

from datetime import date
from pathlib import Path

import httpx
import numpy as np

from pyproj import Geod
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape

HERE = Path(__file__).parent
CACHE = HERE / 'cache'
SNODAS_CACHE = CACHE / 'snodas'
REFERENCE = HERE.parent / 'bagis-pourpoints' / 'reference'

TRIPLETS = ['06093200:MT:USGS', '06078600:MT:USGS', '14191000:OR:USGS']
DATES = [date(2023, 12, 1), date(2024, 3, 15), date(2024, 5, 15)]

ARCHIVE = 'https://noaadata.apps.nsidc.org/NOAA/G02158/masked'
GEOD = Geod(ellps='WGS84')


def fetch_swe(d: date) -> tuple[np.ndarray, dict]:
    """(int16 SWE grid, header dict) for one date, tar cached on disk."""
    SNODAS_CACHE.mkdir(parents=True, exist_ok=True)
    tar_path = SNODAS_CACHE / f'SNODAS_{d:%Y%m%d}.tar'
    if not tar_path.exists():
        url = f'{ARCHIVE}/{d.year}/{d:%m}_{d:%b}/SNODAS_{d:%Y%m%d}.tar'
        print(f'downloading {url}')
        with httpx.stream('GET', url, timeout=600.0, follow_redirects=True) as r:
            r.raise_for_status()
            with tar_path.open('wb') as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    stem = f'us_ssmv11034tS__T0001TTNATS{d:%Y%m%d}05HP001'
    with tarfile.open(tar_path) as tar:
        hdr_text = (
            gzip.decompress(tar.extractfile(f'{stem}.txt.gz').read()).decode()
            if f'{stem}.txt.gz' in tar.getnames()
            else tar.extractfile(f'{stem}.txt').read().decode()
        )
        dat_name = (
            f'{stem}.dat.gz' if f'{stem}.dat.gz' in tar.getnames() else f'{stem}.dat'
        )
        raw = tar.extractfile(dat_name).read()
        if dat_name.endswith('.gz'):
            raw = gzip.decompress(raw)

    hdr = {}
    for line in hdr_text.splitlines():
        m = re.match(r'([^:]+):\s*(.*)', line)
        if m:
            hdr[m.group(1).strip()] = m.group(2).strip()

    rows = int(hdr['Number of rows'])
    cols = int(hdr['Number of columns'])
    grid = np.frombuffer(raw, dtype='>i2').reshape(rows, cols)
    return grid, hdr


def cell_area_by_row(lat_max: float, dy: float, dx: float, nrows: int) -> np.ndarray:
    """Geodesic area (m2) of one grid cell in each row, top row first."""
    areas = np.empty(nrows)
    for i in range(nrows):
        top = lat_max - i * dy
        bot = top - dy
        a, _ = GEOD.polygon_area_perimeter([0, dx, dx, 0], [top, top, bot, bot])
        areas[i] = abs(a)
    return areas


def basin_stats(
    grid: np.ndarray,
    hdr: dict,
    geom_json: dict,
) -> dict[str, float]:
    rows, cols = grid.shape
    xmin = float(hdr['Minimum x-axis coordinate'])
    ymax = float(hdr['Maximum y-axis coordinate'])
    dx = float(hdr['X-axis resolution'])
    dy = float(hdr['Y-axis resolution'])
    nodata = int(float(hdr['No data value']))

    transform = from_origin(xmin, ymax, dx, dy)
    mask = rasterize(
        [(shape(geom_json), 1)],
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype='uint8',
    ).astype(bool)

    row_areas = cell_area_by_row(ymax, dy, dx, rows)
    areas = np.broadcast_to(row_areas[:, None], grid.shape)

    in_basin = mask
    valid = in_basin & (grid != nodata)

    basin_area = float(areas[in_basin].sum())
    valid_area = float(areas[valid].sum())
    vals = grid[valid].astype(np.float64)  # SWE mm
    mean_swe = (
        float((vals * areas[valid]).sum() / valid_area) if valid_area else float('nan')
    )

    return {
        'basin_area_m2': basin_area,
        'valid_area_m2': valid_area,
        'pixels': int(in_basin.sum()),
        'valid_pixels': int(valid.sum()),
        'mean_swe_mm': mean_swe,
    }


def api_values(site: str, d: date) -> dict[str, float | None]:
    """Pull the same-date values out of the fetch.py cache for comparison."""
    out: dict[str, float | None] = {
        'old_sql_swe_mm': None,
        'old_zonal_mean_mm': None,
        'old_zonal_area_m2': None,
        'new_mean_mm': None,
        'new_area_m2': None,
    }
    iso = d.isoformat()
    p = CACHE / f'old-basic-{site}.csv'
    if p.exists():
        import csv as _csv

        for row in _csv.DictReader(p.read_text().splitlines()):
            if row['date'] == iso:
                out['old_sql_swe_mm'] = float(row['swe']) * 1000
    p = CACHE / f'old-zonal-{site}-{d:%Y%m}.json'
    if p.exists():
        for res in json.loads(p.read_text())['results']:
            if res['date'] == iso:
                bands = [z for z in res['zones'] if z['area_m2'] > 0]
                warea = sum(z['area_m2'] for z in bands if z['mean_swe_mm'] is not None)
                wsum = sum(
                    z['area_m2'] * z['mean_swe_mm']
                    for z in bands
                    if z['mean_swe_mm'] is not None
                )
                out['old_zonal_area_m2'] = sum(z['area_m2'] for z in bands)
                out['old_zonal_mean_mm'] = wsum / warea if warea else None
    p = CACHE / f'new-basic-{site}-{d:%Y%m}.json'
    if p.exists():
        for res in json.loads(p.read_text())['results']:
            if res['date'] == iso:
                (z,) = res['zones']
                out['new_area_m2'] = z['area_m2']
                out['new_mean_mm'] = z['mean_swe_mm']
    return out


def main() -> None:
    for d in DATES:
        grid, hdr = fetch_swe(d)
        print(f'\n{"=" * 76}\n{d.isoformat()}\n{"=" * 76}')
        for triplet in TRIPLETS:
            site, st, _ = triplet.split(':')
            ref = json.load(open(REFERENCE / f'{site}_{st}_USGS.geojson'))
            ref_poly = next(
                g for g in ref['geometries'] if g['type'] in ('Polygon', 'MultiPolygon')
            )
            new_pp = json.loads((CACHE / f'new-pp-{site}.json').read_text())

            print(f'\n{triplet}')
            api = api_values(site, d)
            for label, poly in [
                ('reference polygon', ref_poly),
                ('new-API polygon', new_pp['geometry']),
            ]:
                s = basin_stats(grid, hdr, poly)
                print(
                    f'  independent, {label:<18} area {s["basin_area_m2"] / 1e6:>10,.2f} km2 '
                    f'(valid {s["valid_area_m2"] / 1e6:>10,.2f})  mean SWE {s["mean_swe_mm"]:>8.2f} mm',
                )
            oza = api['old_zonal_area_m2']
            print(
                f'  old API zonal (ref polygon)      area '
                f'{(oza or float("nan")) / 1e6:>10,.2f} km2                          '
                f'mean SWE {api["old_zonal_mean_mm"] if api["old_zonal_mean_mm"] is not None else float("nan"):>8.2f} mm',
            )
            print(
                f'  old API SQL (ref polygon)                                                     '
                f'swe {api["old_sql_swe_mm"] if api["old_sql_swe_mm"] is not None else float("nan"):>8.2f} mm',
            )
            na = api['new_area_m2']
            print(
                f'  new API stats (new polygon)      area '
                f'{(na or float("nan")) / 1e6:>10,.2f} km2                          '
                f'mean SWE {api["new_mean_mm"] if api["new_mean_mm"] is not None else float("nan"):>8.2f} mm',
            )


if __name__ == '__main__':
    main()
