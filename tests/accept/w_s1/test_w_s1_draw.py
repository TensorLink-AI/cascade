"""W-S1 TS5 and the TS1 draw half: ``cascade.shared.sentinel.sentinel_for_era``
draws one suite of ``sentinel_suites`` from the era-boundary block hash —
deterministic from the hash (the same on every validator and in every
process), uniform over the suites, and independent between eras (a repeat
has probability 1 / len(suites))."""

from __future__ import annotations

from collections import Counter

import pytest

from .helpers import ALL_SUITES, SHIPPED, armed_cfg, draw_in_process, synth_hash, validate_ok

N_HASHES = 4096
SUITE_SETS = (ALL_SUITES, ("boom", "time", "fev"), ("gift", "time"))


def _draws(suites, n=N_HASHES, offset=0) -> list[str]:
    from cascade.shared.sentinel import sentinel_for_era

    return [sentinel_for_era(synth_hash(offset + i), list(suites)) for i in range(n)]


def test_ts5_draw_is_a_member_of_the_suites():
    from cascade.shared.sentinel import sentinel_for_era

    for suites in SUITE_SETS:
        for i in range(256):
            got = sentinel_for_era(synth_hash(i), list(suites))
            assert got in suites, f"draw {got!r} is not one of {suites}"


def test_ts5_draw_is_deterministic_from_the_hash():
    """Same hash, same suites ⇒ same suite, whatever was drawn in between and
    whether the suites come as a list or a tuple."""
    from cascade.shared.sentinel import sentinel_for_era

    first = _draws(ALL_SUITES, n=512)
    _draws(ALL_SUITES, n=512, offset=10_000)
    again = [sentinel_for_era(synth_hash(i), tuple(ALL_SUITES)) for i in range(512)]
    assert first == again


@pytest.mark.parametrize("suites", SUITE_SETS, ids=lambda s: "-".join(s))
def test_ts5_draw_is_uniform_over_the_suites(suites):
    """Over 4096 distinct hashes every suite is drawn within 12% of n / k
    (about 4 standard deviations; the hash set is fixed, so the test is exact)."""
    counts = Counter(_draws(suites))
    want = N_HASHES / len(suites)
    assert set(counts) == set(suites), f"suites never drawn: {set(suites) - set(counts)}"
    for s in suites:
        assert abs(counts[s] - want) <= 0.12 * want, (
            f"{s} drawn {counts[s]} times of {N_HASHES}; uniform is {want:.0f}: {dict(counts)}")


def test_ts5_draws_are_independent_between_eras():
    """Consecutive eras (consecutive hashes): a repeat has probability 1/4, and
    the next era's draw is uniform whatever this era drew (no rotation, no
    avoidance of the previous suite)."""
    seq = _draws(ALL_SUITES)
    pairs = list(zip(seq, seq[1:], strict=False))
    repeat = sum(a == b for a, b in pairs) / len(pairs)
    assert 0.20 <= repeat <= 0.30, f"repeat rate {repeat:.3f}, expected about 0.25"
    cells = Counter(pairs)
    want = len(pairs) / 16
    for a in ALL_SUITES:
        for b in ALL_SUITES:
            assert abs(cells[(a, b)] - want) <= 0.25 * want, (
                f"P(next={b} | this={a}) off uniform: {cells[(a, b)]} vs {want:.0f}")


def test_ts5_single_suite_list_always_draws_it():
    assert set(_draws(("time",), n=64)) == {"time"}


@pytest.mark.parametrize("bad", ["", "0x", "0x" + "zz" * 32, None])
def test_ts1_unreadable_hash_value_is_refused(bad):
    """A malformed hash is never silently mapped to a suite (the satellite then
    replays the recorded value instead, test_w_s1_receipt.py)."""
    from cascade.shared.sentinel import sentinel_for_era

    with pytest.raises((ValueError, TypeError)):
        sentinel_for_era(bad, list(ALL_SUITES))


def test_ts1_empty_suites_are_refused():
    from cascade.shared.sentinel import sentinel_for_era

    with pytest.raises(ValueError):
        sentinel_for_era(synth_hash(0), [])


def test_ts1_two_validator_processes_derive_the_same_sentinel():
    """Two interpreters with different PYTHONHASHSEEDs (two validators) and
    this process draw identically: the draw uses no process-local hashing or
    state."""
    from cascade.shared.sentinel import sentinel_for_era

    hashes = [synth_hash(i) for i in range(128)]
    here = [sentinel_for_era(h, list(ALL_SUITES)) for h in hashes]
    assert draw_in_process(hashes, ALL_SUITES, "1") == here
    assert draw_in_process(hashes, ALL_SUITES, "2") == here


@pytest.mark.parametrize("name", SHIPPED)
def test_ts1_validators_armed_by_overlay_or_file_derive_the_same_record(name):
    """A validator armed by the signed overlay and one whose chain.toml carries
    the same keys stamp the same sentinel record for every era hash; the draw
    depends on the hash and the suites only (mainnet and testnet agree)."""
    from cascade.shared.sentinel import sentinel_record

    from cascade.shared.config import parse_chain_config

    from .helpers import EFFECTIVE, shipped_raw

    by_file = armed_cfg(name)
    by_overlay = validate_ok(parse_chain_config(shipped_raw(name)),
                             {"v2": {"sentinel_from_block": EFFECTIVE}})
    other = armed_cfg("chain.testnet.toml" if name == "chain.toml" else "chain.toml")
    for i in range(32):
        h = synth_hash(i)
        a = sentinel_record(by_file, EFFECTIVE, h)
        b = sentinel_record(by_overlay, EFFECTIVE, h)
        c = sentinel_record(other, EFFECTIVE, h)
        assert a is not None and a == b, f"hash {h}: file {a!r} vs overlay {b!r}"
        assert c is not None and c["suite"] == a["suite"]
