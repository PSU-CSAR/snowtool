# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Fetch and cache WY2024 SNODAS SWE stats from both services for 3 pourpoints.

Every response is cached under cache/ keyed by request; reruns skip anything
already fetched, so the script is resumable. Failures are recorded in
cache/fetch-failures.json and do not abort the run.
"""

from __future__ import annotations

import json
import sys
import time

from datetime import date, timedelta
from pathlib import Path

import httpx

OLD = 'https://api.snodas.geog.pdx.edu'
NEW = 'https://ebagis.geog.pdx.edu'
NWIS = 'https://waterservices.usgs.gov/nwis/site/'

TRIPLETS = [
    '06093200:MT:USGS',
    '06078600:MT:USGS',
    '14191000:OR:USGS',
]

WY_START = date(2023, 10, 1)
WY_END = date(2024, 9, 30)

CACHE = Path(__file__).parent / 'cache'
FAILURES: dict[str, str] = {}


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cur = start
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunks.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return chunks


def fetch(client: httpx.Client, key: str, url: str, params: dict | None = None) -> None:
    """GET url into cache/<key> unless already present; log failures."""
    out = CACHE / key
    if out.exists():
        print(f'cached  {key}')
        return
    try:
        r = client.get(url, params=params)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f'FAIL    {key}: {e}', file=sys.stderr)
        FAILURES[key] = str(e)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    print(f'fetched {key} ({len(r.content)} bytes)')
    time.sleep(0.5)  # be polite; these are shared university servers


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    chunks = month_chunks(WY_START, WY_END)

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        sites = ','.join(t.split(':')[0] for t in TRIPLETS)
        fetch(
            client,
            'nwis-sites.rdb',
            NWIS,
            {'sites': sites, 'siteOutput': 'expanded', 'format': 'rdb'},
        )

        for triplet in TRIPLETS:
            tkey = triplet.split(':')[0]

            # --- pourpoint metadata (+ basin polygon on the new service) ---
            fetch(
                client, f'old-pp-{tkey}.json', f'{OLD}/pourpoints/by-triplet/{triplet}/'
            )
            fetch(client, f'new-pp-{tkey}.json', f'{NEW}/pourpoints/{triplet}')

            old_pp_file = CACHE / f'old-pp-{tkey}.json'
            if not old_pp_file.exists():
                continue
            old_id = json.loads(old_pp_file.read_text())['id']

            # --- old service: no-zone (SQL) stats; JSON 500s, CSV works ---
            fetch(
                client,
                f'old-basic-{tkey}.csv',
                f'{OLD}/pourpoints/{old_id}/stats/date-range',
                {
                    'start_date': WY_START.isoformat(),
                    'end_date': WY_END.isoformat(),
                    'format': 'csv',
                },
            )
            # keep one JSON attempt on record for the 500 finding
            fetch(
                client,
                f'old-basic-{tkey}.json',
                f'{OLD}/pourpoints/{old_id}/stats/date-range',
                {'start_date': WY_START.isoformat(), 'end_date': WY_END.isoformat()},
            )

            for cstart, cend in chunks:
                mkey = cstart.strftime('%Y%m')

                # --- old service: raster zonal stats, monthly chunks ---
                fetch(
                    client,
                    f'old-zonal-{tkey}-{mkey}.json',
                    f'{OLD}/pourpoints/{old_id}/zonal-stats/date-range',
                    {
                        'products': 'swe',
                        'start_date': cstart.isoformat(),
                        'end_date': cend.isoformat(),
                        'elevation_band_step_ft': 1000,
                    },
                )

                # --- new service: no-zone + elevation-zoned, monthly chunks ---
                dt = f'{cstart.isoformat()}/{cend.isoformat()}'
                fetch(
                    client,
                    f'new-basic-{tkey}-{mkey}.json',
                    f'{NEW}/datasets/snodas/stats/{triplet}/date-range',
                    {'datetime': dt, 'variable': 'swe'},
                )
                fetch(
                    client,
                    f'new-zonal-{tkey}-{mkey}.json',
                    f'{NEW}/datasets/snodas/stats/{triplet}/date-range',
                    {'datetime': dt, 'variable': 'swe', 'zone': 'terrain.elevation'},
                )

    (CACHE / 'fetch-failures.json').write_text(json.dumps(FAILURES, indent=2))
    print(f'\ndone: {len(FAILURES)} failures (cache/fetch-failures.json)')


if __name__ == '__main__':
    main()
