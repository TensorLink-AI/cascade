"""W-S1 TS2: lane A with the sentinel armed. The bench gate
(``evaluate_bench_gate``) and the king check (``king_check_gate``) judge on
the three remaining suites; a candidate regressing only on the sentinel is
blocked iff the ratchet is on (``LCB(sentinel) >= -sentinel_tau``, D34).

The pinned interface: both functions take ``sentinel`` (a suite name, or
``None`` = unarmed), ``sentinel_ratchet`` (bool) and ``sentinel_tau``
(float). The caller passes the era's drawn suite only when
``sentinel_armed(cfg, era_start)``.
"""

from __future__ import annotations

import pytest

from .helpers import (
    ALL_SUITES,
    KC_PARAMS,
    gate,
    geomean,
    king_check,
    scaled_pair,
    verdict_core,
)

BETTER = 0.9          # a clear improvement on a suite
REGRESS = 1.10        # a clear 10% regression (ratio LCB about 1.1 at any alpha)


def _only_sentinel_regresses(sentinel: str) -> tuple[dict, dict]:
    return scaled_pair({s: (REGRESS if s == sentinel else BETTER) for s in ALL_SUITES})


@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_ts2_unarmed_gate_blocks_a_sentinel_only_regression(sentinel):
    """Control: unarmed, all four suites guard, so the fixture fails today."""
    king, chal = _only_sentinel_regresses(sentinel)
    res = gate(king, chal)
    assert res.computed and res.passed is False and sentinel in res.reason, res.reason


@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_ts2_armed_without_ratchet_judges_on_three_suites(sentinel):
    king, chal = _only_sentinel_regresses(sentinel)
    res = gate(king, chal, sentinel=sentinel, sentinel_ratchet=False, sentinel_tau=0.01)
    assert res.computed and res.passed is True, (
        f"sentinel {sentinel}: a regression on the held-out suite blocked with the "
        f"ratchet off: {res.reason}")
    d = res.detail()
    rest = [s for s in ALL_SUITES if s != sentinel]
    assert set(d["suites"]) == set(rest), f"verdict suites {sorted(d['suites'])}, want {rest}"
    assert d["sentinel"]["suite"] == sentinel and d["sentinel"]["ratchet"] is False
    assert d["avg_ratio"] == pytest.approx(geomean(d["suites"][s]["ratio"] for s in rest),
                                           rel=1e-12)
    assert d["win_rate"] == pytest.approx(
        sum(d["suites"][s]["win_rate"] for s in rest) / 3, rel=1e-12)
    assert d["worst_lcb"] == pytest.approx(max(d["suites"][s]["ratio_lcb"] for s in rest),
                                           rel=1e-12)


@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_ts2_armed_with_ratchet_blocks_a_sentinel_only_regression(sentinel):
    king, chal = _only_sentinel_regresses(sentinel)
    res = gate(king, chal, sentinel=sentinel, sentinel_ratchet=True, sentinel_tau=0.01)
    assert res.computed is True and res.passed is False, (
        f"sentinel {sentinel}: ratchet on, a 10% sentinel regression passed")
    assert sentinel in res.reason and "sentinel" in res.reason, res.reason
    assert set(res.detail()["suites"]) == {s for s in ALL_SUITES if s != sentinel}
    assert res.detail()["sentinel"]["ratchet"] is True


def test_ts2_ratchet_tolerance_is_sentinel_tau():
    """The same 10% sentinel regression passes the ratchet at tau 0.5 and is
    blocked at tau 0.05: the bound is sentinel_tau, not [scoring]
    bench_gate_tolerance."""
    king, chal = _only_sentinel_regresses("boom")
    loose = gate(king, chal, sentinel="boom", sentinel_ratchet=True, sentinel_tau=0.5)
    tight = gate(king, chal, sentinel="boom", sentinel_ratchet=True, sentinel_tau=0.05)
    assert loose.computed and loose.passed is True, loose.reason
    assert tight.computed and tight.passed is False, tight.reason


@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_ts2_sentinel_never_selects(sentinel):
    """Ratchet off: the verdict is the same whether the sentinel improves,
    regresses or was never swept. Ratchet on with a non-regressing sentinel:
    the same verdict again (the sentinel blocks, it never helps)."""
    rest = {s: BETTER for s in ALL_SUITES if s != sentinel}
    kw = {"sentinel": sentinel, "sentinel_tau": 0.01}
    cores = []
    for f in (0.5, 1.0, 1.5):
        king, chal = scaled_pair({**rest, sentinel: f})
        cores.append(verdict_core(gate(king, chal, sentinel_ratchet=False, **kw)))
    king, chal = scaled_pair(rest, drop=(sentinel,))
    cores.append(verdict_core(gate(king, chal, sentinel_ratchet=False, **kw)))
    for f in (0.5, 0.97):
        king, chal = scaled_pair({**rest, sentinel: f})
        cores.append(verdict_core(gate(king, chal, sentinel_ratchet=True, **kw)))
    assert cores[0]["passed"] is True
    assert all(c == cores[0] for c in cores), "the sentinel suite moved a verdict"


def test_ts2_ratchet_needs_the_sentinel_rows():
    """Ratchet on and no sentinel sweep: uncomputable (held), never a pass."""
    king, chal = scaled_pair({s: BETTER for s in ALL_SUITES}, drop=("time",))
    res = gate(king, chal, sentinel="time", sentinel_ratchet=True, sentinel_tau=0.01)
    assert res.computed is False and res.passed is None, (res.computed, res.passed, res.reason)


@pytest.mark.parametrize("bad", ["fev", "gift"])
def test_ts2_other_suites_remain_full_guards(bad):
    """§16 rule 3: armed, a regression on a non-sentinel suite still blocks."""
    sentinel = "time"
    king, chal = scaled_pair({s: (REGRESS if s == bad else BETTER) for s in ALL_SUITES})
    for ratchet in (False, True):
        res = gate(king, chal, sentinel=sentinel, sentinel_ratchet=ratchet, sentinel_tau=0.01)
        assert res.computed and res.passed is False and bad in res.reason, res.reason


def test_ts2_unknown_sentinel_suite_is_refused():
    king, chal = scaled_pair({s: BETTER for s in ALL_SUITES})
    with pytest.raises(ValueError):
        gate(king, chal, sentinel="gifteval", sentinel_ratchet=False, sentinel_tau=0.01)


@pytest.mark.parametrize("ratchet", [False, True])
@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_ts2_armed_detail_replays_to_its_verdict(sentinel, ratchet):
    """The audit / satellite re-derive a recorded detail (``detail_verdict``):
    an armed three-suite detail, passing or ratchet-blocked, implies exactly
    the recorded verdict."""
    from cascade.eval.bench_gate import detail_verdict

    king, chal = _only_sentinel_regresses(sentinel)
    res = gate(king, chal, sentinel=sentinel, sentinel_ratchet=ratchet, sentinel_tau=0.01)
    assert detail_verdict(res.detail()) is res.passed, (sentinel, ratchet, res.detail())


# ── king_check_gate (king_check.py:93) ───────────────────────────────────────


@pytest.mark.parametrize("sentinel", ALL_SUITES)
def test_ts2_king_check_drops_the_sentinel(sentinel):
    """The king (challenger) regresses against the init only on the sentinel:
    unarmed the bench half fails, armed without the ratchet it passes on the
    three remaining suites."""
    init, king = _only_sentinel_regresses(sentinel)
    unarmed = king_check(king, init)
    assert unarmed.computed and unarmed.passed is False, unarmed.reason
    armed = king_check(king, init, sentinel=sentinel, sentinel_ratchet=False,
                       sentinel_tau=0.01)
    assert armed.computed and armed.passed is True, armed.reason
    assert set(armed.detail()["suites"]) == {s for s in ALL_SUITES if s != sentinel}


@pytest.mark.parametrize("ratchet", [False, True])
def test_ts2_king_check_is_the_gate_rule_with_the_same_sentinel(ratchet):
    """The king check stays the gate's rule (king as challenger, init as
    reference, seed ``{seed}|king-check``) with the sentinel passed through."""
    from cascade.eval.bench_gate import RULE_V2, evaluate_bench_gate

    init, king = _only_sentinel_regresses("fev")
    kw = {"sentinel": "fev", "sentinel_ratchet": ratchet, "sentinel_tau": 0.01}
    got = king_check(king, init, **kw)
    want = evaluate_bench_gate(
        init, king, None, None, alpha=KC_PARAMS.bootstrap_alpha, B=KC_PARAMS.bootstrap_B,
        seed="ws1|king-check", min_items=KC_PARAMS.bench_gate_min_items, rule=RULE_V2,
        tolerance=KC_PARAMS.bench_gate_tolerance,
        min_win_rate=KC_PARAMS.bench_gate_min_win_rate, **kw)
    assert got.detail() == want.detail()
    assert got.passed is want.passed
