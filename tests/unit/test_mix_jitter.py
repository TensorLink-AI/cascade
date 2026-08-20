"""Jittered round mix (the DEC-TB-0003 port; ``cascade.validator.windows``).

The per-round eval draw is unpredictable in realisation but equal-weight per
domain in expectation (capacity-capped — cascade serves one window per series,
so a domain smaller than its uniform share is simply always fully eligible).
Activation is gated on the round's epoch-boundary BLOCK so mixed validator
fleets agree on every round while upgrading, and pre-activation behaviour is
byte-identical to the legacy uniform permutation.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from cascade.validator.windows import (
    MixParams,
    RotatingWindowSource,
    _capped_jittered_split,
    round_composition,
)

# A pool imbalanced the way the production pool is (big transport, tiny econ),
# with dgp_class labels so the class-rotation tier is exercised.
_DOMAINS = {
    "transport": (60, ["flow", "occupancy", "headway"]),
    "nature": (50, ["temp", "wind"]),
    "energy": (44, ["load", "price"]),
    "healthcare": (16, ["admissions"]),
    "econ_fin": (10, ["fx"]),
}

_MIX = MixParams(from_block=100, jitter_alpha=4.0, block_slots=4,
                 class_keep_frac=0.7, series_bag_frac=0.7)


def _window(series_id: str, domain: str, cls: str | None, source: str):
    from cascade.eval.window import EvalWindow

    md = {"domain": domain, "freq": "H", "source": source}
    if cls is not None:
        md["dgp_class"] = cls
    return EvalWindow(
        series_id=series_id,
        history=np.arange(8, dtype=float),
        target=np.ones(4),
        metadata=md,
    )


def _pool(with_classes: bool = True) -> tuple:
    out = []
    for dom, (count, classes) in sorted(_DOMAINS.items()):
        for i in range(count):
            cls = classes[i % len(classes)] if with_classes else None
            out.append(
                _window(f"{dom}_{i:03d}", dom, cls, source=f"{dom}_src{i % 7}")
            )
    return tuple(out)


def _source(mix: MixParams | None = _MIX) -> RotatingWindowSource:
    return RotatingWindowSource(pool=_pool(), mix=mix)


def _ids(windows) -> list[str]:
    return [w.series_id for w in windows]


def _dom_counts(windows) -> Counter:
    return Counter(w.metadata["domain"] for w in windows)


# ─────────────────────────────────────────────── determinism / consensus ────


def test_same_seed_gives_byte_identical_picks():
    a = _source().windows_for_round("round-1", 100, block=500)
    b = _source().windows_for_round("round-1", 100, block=500)
    assert _ids(a) == _ids(b)


def test_different_seeds_give_different_domain_mixes():
    mixes = {
        tuple(sorted(_dom_counts(
            _source().windows_for_round(f"round-{i}", 100, block=500)
        ).items()))
        for i in range(12)
    }
    assert len(mixes) > 6  # realised mix rotates round to round


def test_pre_activation_block_is_byte_identical_to_legacy():
    legacy = _source(mix=None).windows_for_round("round-1", 100, block=99)
    gated = _source().windows_for_round("round-1", 100, block=99)  # < from_block
    assert _ids(gated) == _ids(legacy)


def test_block_none_and_mix_off_stay_legacy():
    legacy = _source(mix=None).windows_for_round("round-1", 100, block=500)
    assert _ids(_source().windows_for_round("round-1", 100, block=None)) == _ids(legacy)
    off = MixParams(from_block=0)
    assert _ids(_source(mix=off).windows_for_round("round-1", 100, block=500)) == _ids(legacy)


# ────────────────────────────────────────────────────── mix properties ────


def test_expected_mix_is_uniform_projected_onto_capacity():
    """Mean per-domain share over many seeds tracks the capacity-capped uniform
    target within ±20% (the TSBench pin, adapted for caps)."""
    n = 100
    totals: Counter = Counter()
    rounds = 60
    for i in range(rounds):
        totals.update(_dom_counts(_source().windows_for_round(f"r{i}", n, block=500)))
    # Capacity-capped uniform target on the BAGGED pool: small domains are
    # capped by eligibility (~bag_frac * size); the surplus flows to the
    # unsaturated (large) domains, which end up above n/k.
    got = {d: totals[d] / rounds for d in _DOMAINS}
    caps = {d: c * _MIX.series_bag_frac for d, (c, _) in _DOMAINS.items()}
    saturated = {d for d in _DOMAINS if caps[d] < n / len(_DOMAINS)}
    spill = n - sum(caps[d] for d in saturated)
    open_share = spill / (len(_DOMAINS) - len(saturated))
    for d in _DOMAINS:
        target = caps[d] if d in saturated else open_share
        assert abs(got[d] - target) / target < 0.20, (d, got[d], target)


def test_every_domain_appears_in_every_round_at_a_block_floor():
    for i in range(50):
        counts = _dom_counts(_source().windows_for_round(f"r{i}", 100, block=500))
        assert set(counts) == set(_DOMAINS)
        for d, c in counts.items():
            cap = _DOMAINS[d][0]
            assert c >= min(_MIX.block_slots, cap)


def test_served_count_is_exactly_the_target_when_capacity_allows():
    for i in range(20):
        w = _source().windows_for_round(f"r{i}", 100, block=500)
        assert len(w) == 100
    # Target above bagged capacity: serves everything eligible, never raises.
    big = _source().windows_for_round("r-big", 5000, block=500)
    assert len(big) <= sum(c for c, _ in _DOMAINS.values())
    assert len(big) >= int(0.5 * sum(c for c, _ in _DOMAINS.values()))


def test_target_windows_overrides_the_callers_n():
    mix = MixParams(from_block=100, target_windows=60,
                    series_bag_frac=1.0, class_keep_frac=1.0)
    w = _source(mix=mix).windows_for_round("round-1", 100, block=500)
    assert len(w) == 60


def test_picks_are_distinct_series_always():
    # cascade's without-replacement invariant: one window per series, never a
    # duplicate — a cell's quota caps at its size instead of duplicate-filling.
    for i in range(30):
        ids = _ids(_source().windows_for_round(f"r{i}", 150, block=500))
        assert len(ids) == len(set(ids))


# ───────────────────────────────────────────────── rotation and the bag ────


def test_class_rotation_misses_classes_per_round_but_covers_all_over_time():
    all_classes = {
        (d, c) for d, (_, classes) in _DOMAINS.items() for c in classes
    }
    seen_missing = False
    covered: set = set()
    for i in range(20):
        served = {
            (w.metadata["domain"], w.metadata["dgp_class"])
            for w in _source().windows_for_round(f"r{i}", 100, block=500)
        }
        covered |= served
        seen_missing = seen_missing or (served != all_classes)
    assert seen_missing            # a single round benches some classes
    assert covered == all_classes  # the week covers everything


def test_series_bag_rotates_which_series_can_appear():
    # Generous target: anything eligible is served, so absence ⇒ bagged out.
    presence = {i: set() for i in range(15)}
    for i in range(15):
        presence[i] = set(_ids(_source().windows_for_round(f"r{i}", 5000, block=500)))
    everything = {w.series_id for w in _pool()}
    union = set().union(*presence.values())
    assert union == everything  # nothing is permanently excluded
    assert any(presence[i] != everything for i in range(15))  # bag bites
    assert len({frozenset(p) for p in presence.values()}) > 10  # and rotates


def test_class_tier_is_inert_without_dgp_class_labels():
    src = RotatingWindowSource(pool=_pool(with_classes=False), mix=_MIX)
    for i in range(10):
        w = src.windows_for_round(f"r{i}", 100, block=500)
        assert len(w) == 100  # no micro-cell starvation (the source-as-class trap)


# ───────────────────────────────────────────────────── allocator pins ────


def test_capped_jittered_split_conserves_floors_and_caps():
    rng = np.random.default_rng(7)
    for _ in range(200):
        caps = [60, 50, 44, 16, 10]
        counts = _capped_jittered_split(100, caps, 4.0, 4, rng)
        assert sum(counts) == 100
        for c, cap in zip(counts, caps):
            assert 0 < c <= cap
            assert c >= min(4, cap)


def test_capped_jittered_split_concentrates_tiny_topups():
    rng = np.random.default_rng(7)
    counts = _capped_jittered_split(6, [50] * 8, 4.0, 4, rng)  # 6 < 8 blocks
    assert sum(counts) == 6
    assert sum(1 for c in counts if c) <= 2  # concentrated, not sprayed


def test_capped_jittered_split_saturates_to_capacity():
    rng = np.random.default_rng(7)
    counts = _capped_jittered_split(500, [60, 50, 10], 4.0, 4, rng)
    assert counts == [60, 50, 10]


# ─────────────────────────────────────────────────────── composition ────


def test_round_composition_reports_the_served_mix():
    w = _source().windows_for_round("round-1", 100, block=500)
    comp = round_composition(w, _MIX)
    assert comp["n_windows"] == 100
    assert sum(comp["domains"].values()) == 100
    assert set(comp["domains"]) == set(_DOMAINS)
    assert 1.0 <= comp["effective_domains"] <= len(_DOMAINS)
    assert comp["n_dgp_classes"] >= len(_DOMAINS)  # ≥1 active class per domain
    assert comp["cadences"] == {"H": 100}
    assert comp["mix"]["from_block"] == 100
    assert comp["mix"]["jitter_alpha"] == 4.0
