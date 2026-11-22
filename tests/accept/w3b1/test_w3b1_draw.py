"""W3b1 (T7b first clause, plan §1.3b "The draw", §16 rule 2 lane D):
``cascade.validator.lane_d_gate.gate_suite_for`` draws one lane D gate suite
per crowning from the settlement boundary's block hash, over ``gate_suites``
minus the era's §16 sentinel, weighted by ``gate_weights`` (equal by
default), deterministic in every process and independent of the sentinel
draw on the same hash."""

from __future__ import annotations

from collections import Counter

import pytest

from .helpers import ALL_SUITES, HASHES, N_HASHES, draw_in_process, reference_sentinels

EQUAL = {s: 1.0 for s in ALL_SUITES}


def _draw(h, suites=ALL_SUITES, weights=None, sentinel=None):
    from cascade.validator.lane_d_gate import gate_suite_for

    return gate_suite_for(h, gate_suites=suites, gate_weights=weights, sentinel=sentinel)


def _counts(suites=ALL_SUITES, weights=None, sentinel=None) -> Counter:
    return Counter(_draw(h, suites, weights, sentinel) for h in HASHES)


def _assert_proportional(counts: Counter, probs: dict[str, float], tol: float = 0.10):
    for s, p in probs.items():
        if p == 0.0:
            assert counts[s] == 0, f"{s} has weight 0 but was drawn {counts[s]} times"
            continue
        want = p * N_HASHES
        assert abs(counts[s] - want) <= tol * want, (
            f"{s} drawn {counts[s]} of {N_HASHES}; expected about {want:.0f}: {dict(counts)}")
    assert set(counts) <= {s for s, p in probs.items() if p > 0}, dict(counts)


# ── membership and the sentinel exclusion ─────────────────────────────────────


@pytest.mark.parametrize("sentinel", [None, *ALL_SUITES])
def test_t7b_draw_is_a_gate_suite_and_never_the_sentinel(sentinel):
    for h in HASHES[:1500]:
        got = _draw(h, sentinel=sentinel)
        assert got in ALL_SUITES, f"draw {got!r} is not one of {ALL_SUITES}"
        assert got != sentinel, f"hash {h}: drew the era's sentinel {sentinel!r}"


def test_t7b_unarmed_sentinel_draws_over_all_gate_suites():
    """``sentinel=None`` (W-S1 not armed): every gate suite is drawable."""
    _assert_proportional(_counts(), {s: 0.25 for s in ALL_SUITES})


def test_t7b_sentinel_outside_gate_suites_removes_nothing():
    """A sentinel the lane D ``gate_suites`` don't list leaves the draw as
    if unarmed, hash by hash."""
    suites = ("gift", "boom", "time")
    for h in HASHES[:1000]:
        assert _draw(h, suites, sentinel="fev") == _draw(h, suites, sentinel=None)


# ── weights ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_t7b_equal_weights_uniform_over_the_rest(sentinel):
    probs = {s: (0.0 if s == sentinel else 1 / 3) for s in ALL_SUITES}
    _assert_proportional(_counts(sentinel=sentinel), probs)


def test_t7b_default_weights_are_equal():
    """``gate_weights=None`` is the equal split: the same draw as explicit
    equal weights, in mapping or ``(suite, weight)`` pair form (the
    ``LaneDConfig.gate_weights`` form), for every hash."""
    pairs = tuple((s, 1.0) for s in ALL_SUITES)
    for sentinel in (None, "boom"):
        for h in HASHES[:1000]:
            a = _draw(h, weights=None, sentinel=sentinel)
            assert a == _draw(h, weights=EQUAL, sentinel=sentinel)
            assert a == _draw(h, weights=pairs, sentinel=sentinel)


def test_t7b_weights_scale_invariant_and_pair_form_matches_mapping():
    w = {"gift": 2.0, "boom": 1.0, "time": 1.0, "fev": 4.0}
    scaled = {s: v * 10 for s, v in w.items()}
    pairs = tuple(w.items())
    for h in HASHES[:1000]:
        a = _draw(h, weights=w, sentinel="time")
        assert a == _draw(h, weights=pairs, sentinel="time")
        assert a == _draw(h, weights=scaled, sentinel="time")


def test_t7b_unequal_weights_are_proportional_unarmed():
    w = {"gift": 2.0, "boom": 1.0, "time": 1.0, "fev": 0.0}
    _assert_proportional(_counts(weights=w), {"gift": 0.5, "boom": 0.25, "time": 0.25,
                                              "fev": 0.0})


def test_t7b_unequal_weights_renormalise_after_removing_the_sentinel():
    """The sentinel's weight is dropped and the rest renormalised:
    gift 3 (sentinel), boom 1, time 2, fev 2 ⇒ 0.2 / 0.4 / 0.4."""
    w = {"gift": 3.0, "boom": 1.0, "time": 2.0, "fev": 2.0}
    _assert_proportional(_counts(weights=w, sentinel="gift"),
                         {"gift": 0.0, "boom": 0.2, "time": 0.4, "fev": 0.4})


def test_t7b_suite_without_a_weight_is_never_drawn():
    """A suite of ``gate_suites`` with no ``gate_weights`` entry has weight 0
    (``_gate_weights`` keeps only the listed suites)."""
    _assert_proportional(_counts(weights=(("gift", 1.0), ("boom", 1.0))),
                         {"gift": 0.5, "boom": 0.5, "time": 0.0, "fev": 0.0})


# ── independence from the sentinel draw (domain separation) ──────────────────


@pytest.mark.parametrize("ref", sorted(reference_sentinels()))
def test_t7b_gate_draw_independent_of_sentinel_draw(ref):
    """Over many boundary hashes, the joint (sentinel drawn from the hash,
    gate drawn from the same hash with that sentinel) covers all 12 valid
    pairs about equally (1/12 each). A gate draw that reuses the sentinel's
    hash-to-index mapping (for example the next suite after the sentinel)
    concentrates on a few pairs."""
    draw_sentinel = reference_sentinels()[ref]
    joint = Counter()
    for h in HASHES:
        s = draw_sentinel(h, ALL_SUITES)
        joint[(s, _draw(h, sentinel=s))] += 1
    want = N_HASHES / 12
    for s in ALL_SUITES:
        for g in ALL_SUITES:
            if g == s:
                assert joint[(s, g)] == 0
                continue
            assert abs(joint[(s, g)] - want) <= 0.20 * want, (
                f"[{ref}] P(sentinel={s}, gate={g}) off 1/12: {joint[(s, g)]} vs "
                f"{want:.0f}; joint {dict(joint)}")


@pytest.mark.parametrize("ref", sorted(reference_sentinels()))
def test_t7b_unarmed_gate_draw_is_not_the_sentinel_mapping(ref):
    """With no sentinel and the same four suites, the gate draw agrees with a
    sentinel-style draw of the same hash only by chance (about 1/4 of the
    hashes, each of the 16 cells near 1/16): the gate hash is domain-separated."""
    draw_sentinel = reference_sentinels()[ref]
    joint = Counter((draw_sentinel(h, ALL_SUITES), _draw(h)) for h in HASHES)
    agree = sum(n for (a, b), n in joint.items() if a == b) / N_HASHES
    assert 0.20 <= agree <= 0.30, f"[{ref}] gate equals the {ref} draw on {agree:.3f}"
    want = N_HASHES / 16
    for cell in ((a, b) for a in ALL_SUITES for b in ALL_SUITES):
        assert abs(joint[cell] - want) <= 0.25 * want, (
            f"[{ref}] cell {cell}: {joint[cell]} vs {want:.0f}")


def test_t7b_draws_independent_between_boundaries():
    """Consecutive crownings (consecutive hashes): a repeat has probability
    1/3 with three eligible suites; no rotation or avoidance."""
    seq = [_draw(h, sentinel="fev") for h in HASHES]
    pairs = list(zip(seq, seq[1:], strict=False))
    repeat = sum(a == b for a, b in pairs) / len(pairs)
    assert 0.28 <= repeat <= 0.39, f"repeat rate {repeat:.3f}, expected about 1/3"


# ── determinism ──────────────────────────────────────────────────────────────


def test_t7b_draw_is_deterministic_and_list_or_tuple():
    first = [_draw(h, sentinel="boom") for h in HASHES[:512]]
    [_draw(h, sentinel="gift") for h in HASHES[512:1024]]
    again = [_draw(h, list(ALL_SUITES), sentinel="boom") for h in HASHES[:512]]
    assert first == again


@pytest.mark.parametrize("weights,sentinel", [(None, None), (None, "time"),
                                              ({"gift": 2.0, "boom": 1.0, "time": 1.0,
                                                "fev": 3.0}, "boom")])
def test_t7b_two_validator_processes_derive_the_same_suite(weights, sentinel):
    """Two interpreters with different PYTHONHASHSEEDs (two validators) and
    this process draw the same suite for every hash: no ``hash()``, no
    process-local state."""
    hashes = list(HASHES[:200])
    here = [_draw(h, weights=weights, sentinel=sentinel) for h in hashes]
    assert draw_in_process(hashes, ALL_SUITES, weights, sentinel, "1") == here
    assert draw_in_process(hashes, ALL_SUITES, weights, sentinel, "2") == here


# ── refusals ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["", "0x", "0x" + "zz" * 32, "0x" + "ab" * 16,
                                 "0x" + "ab" * 33, "ab" * 32])
def test_t7b_malformed_hash_is_refused(bad):
    """A boundary hash that isn't ``0x`` + 64 hex digits never maps to a suite."""
    with pytest.raises(ValueError):
        _draw(bad, sentinel="gift")


@pytest.mark.parametrize("bad", [None, 12345, b"\x00" * 32])
def test_t7b_non_string_hash_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        _draw(bad)


@pytest.mark.parametrize("suites,weights,sentinel", [
    ((), None, None),
    (("gift",), None, "gift"),
    (ALL_SUITES, {"gift": 1.0, "boom": 0.0, "time": 0.0, "fev": 0.0}, "gift"),
    (ALL_SUITES, (("time", 1.0),), "time"),
], ids=["no-suites", "only-the-sentinel", "only-weight-on-sentinel", "only-listed-is-sentinel"])
def test_t7b_empty_eligible_set_is_refused(suites, weights, sentinel):
    with pytest.raises(ValueError):
        _draw(HASHES[0], suites, weights, sentinel)


@pytest.mark.parametrize("weights", [{"gift": -1.0, "boom": 1.0, "time": 1.0, "fev": 1.0},
                                     {"gift": 1.0, "gifteval": 1.0}],
                         ids=["negative", "unknown-suite"])
def test_t7b_bad_weights_are_refused(weights):
    with pytest.raises(ValueError):
        _draw(HASHES[0], weights=weights)
