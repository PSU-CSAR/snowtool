# /// script
# requires-python = ">=3.12"
# dependencies = ["pyproj", "shapely"]
# ///
"""Normalize cached responses from both services and run the check suite.

Reads cache/ written by fetch.py; tolerates missing chunks (reports them as
coverage gaps rather than aborting). Writes per-pourpoint comparison CSVs and
prints the check report to stdout (tee it into a file for the findings).

Old-service SQL stats units: the CSV `swe` column is assumed to be meters
(SNODAS raw SWE integers are mm; the SQL path appears to divide by 1000) and
is reported here as mm. The independent recomputation is the arbiter for
whether those values are correct at all.
"""

from __future__ import annotations

import csv
import json
import math

from datetime import date, timedelta
from pathlib import Path

from pyproj import Geod
from shapely.geometry import shape

HERE = Path(__file__).parent
CACHE = HERE / 'cache'
OUT = HERE / 'out'
REFERENCE = HERE.parent / 'bagis-pourpoints' / 'reference'

TRIPLETS = ['06093200:MT:USGS', '06078600:MT:USGS', '14191000:OR:USGS']
WY_START = date(2023, 10, 1)
WY_END = date(2024, 9, 30)
MI2_M2 = 2_589_988.110336

GEOD = Geod(ellps='WGS84')


def geodesic_area(geom_json: dict) -> float:
    area, _ = GEOD.geometry_area_perimeter(shape(geom_json))
    return abs(area)


def wy_dates() -> list[str]:
    days = (WY_END - WY_START).days + 1
    return [(WY_START + timedelta(days=i)).isoformat() for i in range(days)]


def pct(a: float, b: float) -> str:
    """a as a percentage of b."""
    if not b:
        return 'n/a'
    return f'{100 * a / b:.2f}%'


def load_json(name: str) -> dict | None:
    p = CACHE / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def months() -> list[str]:
    out = []
    cur = WY_START
    while cur <= WY_END:
        out.append(cur.strftime('%Y%m'))
        cur = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
    return out


def load_nwis() -> dict[str, float]:
    """site_no -> drainage area in m2 (from published mi2)."""
    p = CACHE / 'nwis-sites.rdb'
    rows = [
        line.split('\t')
        for line in p.read_text().splitlines()
        if line and not line.startswith('#')
    ]
    header = rows[0]
    site_i = header.index('site_no')
    da_i = header.index('drain_area_va')
    out = {}
    for row in rows[2:]:  # row 1 is the rdb format line
        if row[da_i].strip():
            out[row[site_i]] = float(row[da_i]) * MI2_M2
    return out


# --- normalization ----------------------------------------------------------


def norm_old_basic(site: str) -> dict[str, float]:
    """date -> whole-basin mean SWE in mm (CSV `swe` meters -> mm)."""
    p = CACHE / f'old-basic-{site}.csv'
    if not p.exists():
        return {}
    out = {}
    for row in csv.DictReader(p.read_text().splitlines()):
        out[row['date']] = float(row['swe']) * 1000
    return out


def norm_old_zonal(
    site: str,
) -> dict[str, dict[tuple[float, float], tuple[float, float]]]:
    """date -> {(min_ft, max_ft): (area_m2, mean_swe_mm)}; zero-area bands dropped."""
    out: dict[str, dict] = {}
    for m in months():
        doc = load_json(f'old-zonal-{site}-{m}.json')
        if doc is None:
            continue
        for res in doc['results']:
            bands = {}
            for z in res['zones']:
                if z['area_m2'] > 0:
                    key = (z['min_elevation_ft'], z['max_elevation_ft'])
                    bands[key] = (z['area_m2'], z['mean_swe_mm'])
            out[res['date']] = bands
    return out


def norm_new(site: str, kind: str) -> dict[str, dict]:
    """kind='basic': date -> (area_m2, mean_swe_mm).
    kind='zonal': date -> {(min_ft, max_ft): (area_m2, mean_swe_mm)}."""
    out: dict[str, dict] = {}
    for m in months():
        doc = load_json(f'new-{kind}-{site}-{m}.json')
        if doc is None:
            continue
        for res in doc['results']:
            if kind == 'basic':
                (z,) = res['zones']
                out[res['date']] = (z['area_m2'], z['mean_swe_mm'])
            else:
                bands = {}
                for z in res['zones']:
                    (b,) = z['zone']
                    bands[(float(b['min']), float(b['max']))] = (
                        z['area_m2'],
                        z['mean_swe_mm'],
                    )
                out[res['date']] = bands
    return out


def weighted_mean(bands: dict[tuple, tuple[float, float]]) -> tuple[float, float]:
    """(total area, area-weighted mean) over bands with non-null means."""
    total_area = sum(a for a, _ in bands.values())
    wsum = sum(a * m for a, m in bands.values() if m is not None)
    warea = sum(a for a, m in bands.values() if m is not None)
    return total_area, (wsum / warea if warea else math.nan)


# --- report -----------------------------------------------------------------


def main() -> None:
    OUT.mkdir(exist_ok=True)
    nwis = load_nwis()
    all_dates = wy_dates()

    for triplet in TRIPLETS:
        site, st, _ = triplet.split(':')
        print(f'\n{"=" * 76}\n{triplet}\n{"=" * 76}')

        # -- areas / polygon identity --
        ref = json.load(open(REFERENCE / f'{site}_{st}_USGS.geojson'))
        ref_poly = next(
            g for g in ref['geometries'] if g['type'] in ('Polygon', 'MultiPolygon')
        )
        ref_area = geodesic_area(ref_poly)

        old_pp = load_json(f'old-pp-{site}.json')
        new_pp = load_json(f'new-pp-{site}.json')
        old_area_meta = old_pp['properties']['area_meters'] if old_pp else math.nan
        new_area_meta = new_pp['properties']['area_meters'] if new_pp else math.nan
        new_poly_area = geodesic_area(new_pp['geometry']) if new_pp else math.nan
        nwis_area = nwis.get(site, math.nan)

        old_basic = norm_old_basic(site)
        old_zonal = norm_old_zonal(site)
        new_basic = norm_new(site, 'basic')
        new_zonal = norm_new(site, 'zonal')

        # stats-level areas (should be date-invariant; verify and take first)
        oz_areas = {d: weighted_mean(b)[0] for d, b in old_zonal.items()}
        nb_areas = {d: a for d, (a, _) in new_basic.items()}
        nz_areas = {d: weighted_mean(b)[0] for d, b in new_zonal.items()}
        for label, series in [
            ('old zonal', oz_areas),
            ('new basic', nb_areas),
            ('new zonal sum', nz_areas),
        ]:
            vals = set(round(v, 2) for v in series.values())
            if len(vals) > 1:
                print(f'!! {label} area varies across dates: {sorted(vals)[:5]}...')
        oz_area = next(iter(oz_areas.values()), math.nan)
        nb_area = next(iter(nb_areas.values()), math.nan)

        print('\n[areas]  (reference = geodesic area of shared source polygon)')
        rows = [
            ('source polygon (geodesic)', ref_area, ''),
            ('NWIS published drainage', nwis_area, pct(nwis_area, ref_area)),
            (
                'old API pourpoint area_meters',
                old_area_meta,
                pct(old_area_meta, ref_area),
            ),
            ('old API zonal-stats band sum', oz_area, pct(oz_area, ref_area)),
            (
                'new API served polygon (geodesic)',
                new_poly_area,
                pct(new_poly_area, ref_area),
            ),
            (
                'new API pourpoint area_meters',
                new_area_meta,
                pct(new_area_meta, ref_area),
            ),
            ('new API stats area_m2', nb_area, pct(nb_area, ref_area)),
        ]
        for label, v, p in rows:
            print(f'  {label:<36} {v / 1e6:>12,.2f} km2   {p:>8} of source')

        print('\n[polygon identity]')
        print(f'  new served polygon vs source: {pct(new_poly_area, ref_area)}')
        print(f'  new stats area vs new served polygon: {pct(nb_area, new_poly_area)}')
        print(f'  old area_meters vs source: {pct(old_area_meta, ref_area)}')
        print(f'  old zonal band-sum vs source: {pct(oz_area, ref_area)}')

        # -- internal consistency --
        print('\n[internal consistency]')
        # new: zonal band sum vs basic area; weighted band mean vs basic mean
        diffs_area, diffs_mean = [], []
        for d, (a, m) in new_basic.items():
            if d not in new_zonal:
                continue
            za, zm = weighted_mean(new_zonal[d])
            if a:
                diffs_area.append(abs(za - a) / a)
            if m and not math.isnan(zm):
                diffs_mean.append(abs(zm - m) / m)
        if diffs_area:
            print(
                f'  new: sum(band areas) vs whole-basin area: max rel diff '
                f'{max(diffs_area):.2e} over {len(diffs_area)} dates',
            )
            print(
                f'  new: weighted band mean vs whole-basin mean: max rel diff '
                f'{max(diffs_mean):.2e} over {len(diffs_mean)} dates',
            )
        # old: SQL swe vs zonal weighted mean
        ratios = []
        for d, sql_mm in old_basic.items():
            if d not in old_zonal:
                continue
            _, zm = weighted_mean(old_zonal[d])
            if zm and not math.isnan(zm) and sql_mm:
                ratios.append(sql_mm / zm)
        if ratios:
            ratios.sort()
            mid = ratios[len(ratios) // 2]
            print(
                f'  old: SQL swe (as mm) / zonal weighted mean: median ratio '
                f'{mid:.3f} (n={len(ratios)}, min {ratios[0]:.3f}, max {ratios[-1]:.3f})',
            )

        # -- date coverage --
        print('\n[date coverage over WY2024]')
        for label, series in [
            ('old SQL', old_basic),
            ('old zonal', old_zonal),
            ('new basic', new_basic),
            ('new zonal', new_zonal),
        ]:
            missing = [d for d in all_dates if d not in series]
            print(
                f'  {label:<10} {len(series):>4} dates, {len(missing):>3} missing',
                end='',
            )
            if 0 < len(missing) <= 8:
                print(f'  ({", ".join(missing)})')
            else:
                print()

        # -- band-area comparison (date-invariant; first common date) --
        common = sorted(set(old_zonal) & set(new_zonal))
        if common:
            d0 = common[0]
            print(f'\n[band areas, km2]  ({d0}; old polygon vs new polygon)')
            keys = sorted(set(old_zonal[d0]) | set(new_zonal[d0]))
            print(f'  {"band ft":<14} {"old":>10} {"new":>10} {"new/old":>8}')
            for k in keys:
                oa = old_zonal[d0].get(k, (0, None))[0]
                na = new_zonal[d0].get(k, (0, None))[0]
                r = f'{na / oa:.3f}' if oa else 'n/a'
                print(
                    f'  {f"{k[0]:.0f}-{k[1]:.0f}":<14} {oa / 1e6:>10.2f} {na / 1e6:>10.2f} {r:>8}'
                )

        # -- cross-API SWE series --
        deltas = []
        for d in common:
            _, om = weighted_mean(old_zonal[d])
            nm = new_basic.get(d, (None, None))[1]
            if nm is not None and not math.isnan(om) and om > 0.5 and nm > 0.5:
                deltas.append(nm / om)
        if deltas:
            deltas.sort()
            print(
                f'\n[cross-API whole-basin mean SWE]  new / old-zonal ratio: median '
                f'{deltas[len(deltas) // 2]:.3f} (n={len(deltas)}, '
                f'p10 {deltas[len(deltas) // 10]:.3f}, p90 {deltas[9 * len(deltas) // 10]:.3f})',
            )
            print('  (note: computed over different polygons if identity check failed)')

        # -- per-date comparison CSV --
        out_csv = OUT / f'comparison-{site}.csv'
        with out_csv.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(
                [
                    'date',
                    'old_sql_swe_mm',
                    'old_zonal_area_m2',
                    'old_zonal_mean_swe_mm',
                    'new_area_m2',
                    'new_mean_swe_mm',
                    'new_zonal_weighted_mean_swe_mm',
                ],
            )
            for d in all_dates:
                oz = old_zonal.get(d)
                oza, ozm = weighted_mean(oz) if oz else (None, None)
                nb = new_basic.get(d, (None, None))
                nz = new_zonal.get(d)
                nzm = weighted_mean(nz)[1] if nz else None
                w.writerow(
                    [
                        d,
                        old_basic.get(d),
                        oza,
                        ozm,
                        nb[0],
                        nb[1],
                        nzm,
                    ],
                )
        print(f'\n  wrote {out_csv.relative_to(HERE)}')


if __name__ == '__main__':
    main()
