from datetime import date

from snowtool.snowdb.zonal_stat_models import BandZoneRef, CompactStats, CompactZone


def band(lo: int, hi: int) -> BandZoneRef:
    return BandZoneRef(layer='terrain.elevation', min=lo, max=hi, unit='ft')


STATS = CompactStats(
    zone_layers=['terrain.elevation'],
    variables=['mean_snow_fraction_percent'],
    zones=[
        CompactZone(zone=[band(5000, 6000)], area_m2=4_102_331.0),
        CompactZone(zone=[band(6000, 7000)], area_m2=8_874_120.5),
        CompactZone(zone=[band(7000, 8000)], area_m2=6_218_904.2),
        CompactZone(zone=[band(8000, 9000)], area_m2=2_431_887.9),
    ],
    results={
        date(2026, 4, 15): [[71.2], [92.4], [98.1], [99.0]],
        date(2026, 5, 15): [[8.1], [41.6], [88.3], [96.7]],
        date(2026, 6, 15): [[0.0], [2.3], [19.4], [64.8]],
    },
)
