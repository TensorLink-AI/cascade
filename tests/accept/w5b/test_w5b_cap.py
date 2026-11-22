"""T18e: `cap_source_share` (plan rev6 W5b row rev5, B14, D29).

One source supplying more than `max_source_share` of a frozen set's windows
has the excess dropped deterministically, seeded by the settlement. The cap
drops the fewest windows (README "The cap"); which ones is a seeded choice:
same inputs give the same drop, a different seed a different one.
"""

from __future__ import annotations

import random

from .helpers import capped_counts


def _windows(counts: dict[str, int]) -> list[dict]:
    return [{"window_id": f"{src}-s{k}:{1_790_000_000 + k * 3600}:h64", "source": src}
            for src, n in counts.items() for k in range(n)]


def _ids(kept) -> list[str]:
    return [w["window_id"] for w in kept]


def _counts(kept) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in kept:
        out[w["source"]] = out.get(w["source"], 0) + 1
    return out


DOMINANT = {"big": 40, "a": 10, "b": 10, "c": 10, "d": 10}


def test_t18e_excess_of_a_dominant_source_is_dropped():
    from cascade.validator.lane_d_frozen import cap_source_share

    ws = _windows(DOMINANT)
    kept = cap_source_share(ws, max_share=0.25, seed="lane-d/1:t18e")
    # Q = 13: 13 <= 0.25 * (13 + 40) = 13.25, and 14 > 0.25 * 54 = 13.5.
    assert _counts(kept) == {"big": 13, "a": 10, "b": 10, "c": 10, "d": 10}
    assert set(_ids(kept)) <= set(_ids(ws))
    small = {w["window_id"] for w in ws if w["source"] != "big"}
    assert small <= set(_ids(kept)), "a source within the share lost windows"
    assert max(_counts(kept).values()) <= 0.25 * len(kept)
    assert _ids(kept) == sorted(_ids(kept)), "kept windows are not in window_id order"
    assert len(set(_ids(kept))) == len(kept)


def test_t18e_a_set_within_the_share_is_unchanged():
    from cascade.validator.lane_d_frozen import cap_source_share

    ws = _windows({f"src{j}": 8 for j in range(6)})
    kept = cap_source_share(ws, max_share=0.25, seed="lane-d/1:none")
    assert sorted(_ids(kept)) == sorted(_ids(ws))
    # exactly at the share (4 equal sources, 0.25 each) is not "more than"
    ws4 = _windows({f"src{j}": 5 for j in range(4)})
    assert sorted(_ids(cap_source_share(ws4, max_share=0.25, seed="x"))) == sorted(_ids(ws4))


def test_t18e_same_inputs_give_the_same_drop():
    from cascade.validator.lane_d_frozen import cap_source_share

    ws = _windows(DOMINANT)
    first = _ids(cap_source_share(ws, max_share=0.25, seed="lane-d/1:9000"))
    again = _ids(cap_source_share([dict(w) for w in ws], max_share=0.25, seed="lane-d/1:9000"))
    shuffled = [dict(w) for w in ws]
    random.Random(3).shuffle(shuffled)
    reordered = _ids(cap_source_share(shuffled, max_share=0.25, seed="lane-d/1:9000"))
    assert first == again, "the drop is not a function of its inputs"
    assert first == reordered, "the drop depends on input order"


def test_t18e_a_different_seed_gives_a_different_drop():
    from cascade.validator.lane_d_frozen import cap_source_share

    ws = _windows(DOMINANT)
    a = set(_ids(cap_source_share(ws, max_share=0.25, seed="lane-d/1:9000")))
    b = set(_ids(cap_source_share(ws, max_share=0.25, seed="lane-d/1:9900")))
    assert _counts([w for w in ws if w["window_id"] in a]) == \
        _counts([w for w in ws if w["window_id"] in b])
    assert a != b, "the drop ignores the seed (C(40, 13) choices; a collision is ~0)"


def test_t18e_cap_matches_the_minimum_drop_oracle():
    from cascade.validator.lane_d_frozen import cap_source_share

    rng = random.Random(18)
    for trial in range(60):
        k = rng.randint(1, 8)
        counts = {f"s{j}": rng.randint(1, 30) for j in range(k)}
        if trial % 3 == 0:
            counts["s0"] = rng.randint(40, 90)
        share = rng.choice([0.125, 0.25, 0.375, 0.5])   # dyadic: exact products
        ws = _windows(counts)
        kept = cap_source_share(ws, max_share=share, seed=f"lane-d/1:{trial}")
        want = {s: n for s, n in capped_counts(counts, share).items() if n}
        assert _counts(kept) == want, (trial, counts, share)
        assert set(_ids(kept)) <= set(_ids(ws))
        if kept:
            assert max(_counts(kept).values()) <= share * len(kept)
