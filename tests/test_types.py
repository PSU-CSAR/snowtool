"""snowtool.types helpers."""

from snowtool import types


def test_triplet_stem_round_trips():
    triplet = '12354500:MT:USGS'
    stem = types.triplet_to_stem(triplet)
    assert stem == '12354500_MT_USGS'  # ':' is not path-safe
    assert types.stem_to_triplet(stem) == triplet


def test_stem_to_triplet_is_lossless_for_hyphenated_ids():
    # A station id may contain a hyphen but never an underscore (STATION_TRIPLET),
    # so the '_' <-> ':' encoding can't collide.
    triplet = 'ABC-1:CO:SNTL'
    assert types.stem_to_triplet(types.triplet_to_stem(triplet)) == triplet
