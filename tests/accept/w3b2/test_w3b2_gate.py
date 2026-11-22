"""W3b2: the paired lane D gate on the drawn suite (plan §1.3b "Rule on the
drawn suite", §4 T7b). CRPS and MASE must each satisfy ``LCB_α >= -tau_public``
against KING_D AND against the house, paired per item on rows from ONE host
session, never against stored absolute numbers. The era's sentinel suite is
swept and recorded, never gating. Contract: README.md."""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from .helpers import (
    ALPHA,
    CAND,
    GATE_SUITE,
    HOUSE,
    KING,
    METRICS,
    REFS,
    SENTINEL,
    SESSION,
    TAU,
    gate_kwargs,
    lcb_reference,
    triple,
)

REPO = Path(__file__).resolve().parents[3]


def _gate(cand, king, house, **over):
    from cascade.validator.lane_d_bench import paired_gate

    return paired_gate(cand, king, house, **gate_kwargs(**over))


# ── reuse of the repo's paired LCB ───────────────────────────────────────────


def test_gate_reuses_the_repo_paired_lcb():
    from cascade.eval.expert_lane import paired_geomean_lcb

    mod = importlib.import_module("cascade.validator.lane_d_bench")

    assert getattr(mod, "paired_geomean_lcb", None) is paired_geomean_lcb, (
        "lane_d_bench must import cascade.eval.expert_lane.paired_geomean_lcb (the bench "
        "gate's paired LCB), not re-implement it")


def test_lcbs_match_the_paired_lcb_at_alpha_for_both_refs_and_metrics():
    cand, king, house = triple(king=(1.01, 0.99), house=(0.995, 1.02), noise=0.05)
    res = _gate(cand, king, house, B=4000)
    assert isinstance(res, dict)
    for ref, rec in (("king_d", king), ("house", house)):
        for m in METRICS:
            got = res["lcb"][ref][m]
            want = lcb_reference(rec, cand, GATE_SUITE, m, alpha=ALPHA)
            assert math.isfinite(got)
            assert abs(got - want) < 0.002, (
                f"{ref} {m}: LCB {got} != paired_geomean_lcb(ref, cand, alpha={ALPHA}) "
                f"{want} (uncorrected one-sided LCB_alpha of 1 - geomean(cand)/geomean(ref))")


# ── the four-LCB rule ────────────────────────────────────────────────────────


def test_t7b_clean_improvement_passes():
    cand, king, house = triple(king=(1.03, 1.03), house=(1.03, 1.03))
    res = _gate(cand, king, house)
    assert res["outcome"] == "pass" and res["passed"] is True and res["hold"] is False
    assert res["blocked_by"] == []
    assert all(res["lcb"][r][m] >= -TAU for r in REFS for m in METRICS)
    assert res["suite"] == GATE_SUITE and res["session"] == SESSION
    assert res["digests"] == {"candidate": CAND, "king_d": KING, "house": HOUSE}
    assert res["tau_public"] == TAU and res["alpha"] == ALPHA


def test_t7b_equal_quality_passes_non_inferiority():
    res = _gate(*triple())
    assert res["outcome"] == "pass", res


@pytest.mark.parametrize("ref", REFS)
@pytest.mark.parametrize("metric", METRICS)
def test_t7b_single_metric_regression_against_one_ref_blocks(ref, metric):
    worse = (0.95, 1.03) if metric == "crps" else (1.03, 0.95)
    factors = {"king": (1.03, 1.03), "house": (1.03, 1.03)}
    factors["king" if ref == "king_d" else "house"] = worse
    res = _gate(*triple(**factors))
    assert res["outcome"] == "gate_regression", res
    assert res["passed"] is False and res["hold"] is False
    assert res["blocked_by"] == [ref]
    assert res["lcb"][ref][metric] < -TAU
    other = "mase" if metric == "crps" else "crps"
    assert res["lcb"][ref][other] >= -TAU


def test_t7b_regression_vs_house_blocks_even_when_king_d_passes():
    # A weak (e.g. contaminated-then-dethroned) KING_D must not lower the bar:
    # the candidate beats KING_D by 5% and loses to the house by 5%.
    res = _gate(*triple(king=(1.05, 1.05), house=(0.95, 0.95)))
    assert res["lcb"]["king_d"]["crps"] > 0 and res["lcb"]["king_d"]["mase"] > 0
    assert res["outcome"] == "gate_regression" and res["passed"] is False
    assert res["blocked_by"] == ["house"]


def test_t7b_regression_vs_king_d_blocks_even_when_house_passes():
    res = _gate(*triple(king=(0.95, 0.95), house=(1.05, 1.05)))
    assert res["outcome"] == "gate_regression" and res["blocked_by"] == ["king_d"]


def test_t7b_regression_vs_both_lists_both():
    res = _gate(*triple(king=(0.95, 0.95), house=(0.95, 0.95)))
    assert res["outcome"] == "gate_regression"
    assert res["blocked_by"] == ["house", "king_d"]


def test_threshold_is_inclusive_and_moves_with_tau_public():
    cand, king, house = triple(king=(0.995, 0.995), house=(1.03, 1.03))
    res = _gate(cand, king, house)
    low = min(res["lcb"]["king_d"].values())
    assert -TAU < low < 0, f"fixture should be a mild regression, got LCB {low}"
    at = _gate(cand, king, house, tau_public=-low)
    assert at["outcome"] == "pass", "LCB == -tau_public must pass (>=)"
    assert at["lcb"] == res["lcb"], "the LCBs must not depend on tau_public"
    tight = _gate(cand, king, house, tau_public=-low * 0.999)
    assert tight["outcome"] == "gate_regression" and tight["blocked_by"] == ["king_d"]


# ── a house identical to KING_D ──────────────────────────────────────────────


def test_house_identical_to_king_d_same_object():
    cand, king, _ = triple(king=(1.03, 1.03))
    res = _gate(cand, king, king)
    assert res["outcome"] == "pass"
    assert res["lcb"]["house"] == res["lcb"]["king_d"]
    assert res["digests"]["house"] == res["digests"]["king_d"] == KING


def test_house_identical_to_king_d_copy_blocks_on_both():
    cand, king, _ = triple(king=(0.95, 0.95))
    res = _gate(cand, king, copy.deepcopy(king))
    assert res["outcome"] == "gate_regression"
    assert res["blocked_by"] == ["house", "king_d"]
    assert res["lcb"]["house"] == res["lcb"]["king_d"]


# ── paired, never absolute ───────────────────────────────────────────────────


@pytest.mark.parametrize("scale", [0.7, 1.4])
def test_host_wide_shift_does_not_change_the_verdict(scale):
    # Driver / SKU shift every absolute number on a host; the paired ratio cancels it.
    base = _gate(*triple(king=(0.995, 1.03), house=(1.02, 0.96)))
    shifted = _gate(*triple(king=(0.995, 1.03), house=(1.02, 0.96), scale=scale))
    assert shifted["outcome"] == base["outcome"] == "gate_regression"
    assert shifted["blocked_by"] == base["blocked_by"] == ["house"]
    for r in REFS:
        for m in METRICS:
            assert shifted["lcb"][r][m] == pytest.approx(base["lcb"][r][m], abs=1e-9)


def test_pairs_by_item_not_by_row_order():
    cand, king, house = triple(king=(1.01, 0.99), house=(0.99, 1.01))
    base = _gate(cand, king, house)
    for rec in (cand, king, house):
        rec["suites"][GATE_SUITE]["rows"].reverse()
    king["suites"][GATE_SUITE]["rows"].sort(key=lambda r: r["mase"])
    again = _gate(cand, king, house)
    assert again["outcome"] == base["outcome"]
    for r in REFS:
        for m in METRICS:
            assert again["lcb"][r][m] == pytest.approx(base["lcb"][r][m], abs=1e-12)


# ── the sentinel suite: swept and recorded, never gating ─────────────────────


def test_t7b_sentinel_only_regression_passes_and_is_recorded():
    res = _gate(*triple(king=(1.03, 1.03), house=(1.03, 1.03), sentinel_king=(0.9, 0.9),
                        sentinel_house=(0.9, 0.9)))
    assert res["outcome"] == "pass" and res["passed"] is True
    sen = res["sentinel"]
    assert sen["suite"] == SENTINEL
    assert all(sen["lcb"][r][m] < -TAU for r in REFS for m in METRICS), sen


def test_sentinel_numbers_are_the_paired_lcbs():
    cand, king, house = triple(sentinel_king=(1.02, 0.98), sentinel_house=(0.97, 1.01),
                               noise=0.05)
    res = _gate(cand, king, house, B=4000)
    for ref, rec in (("king_d", king), ("house", house)):
        for m in METRICS:
            want = lcb_reference(rec, cand, SENTINEL, m)
            assert abs(res["sentinel"]["lcb"][ref][m] - want) < 0.002


def test_sentinel_improvement_cannot_rescue_a_gate_regression():
    res = _gate(*triple(house=(0.95, 0.95), sentinel_king=(1.2, 1.2),
                        sentinel_house=(1.2, 1.2)))
    assert res["outcome"] == "gate_regression" and res["blocked_by"] == ["house"]


def test_no_sentinel_records_none():
    res = _gate(*triple(sentinel=None), sentinel=None)
    assert res["outcome"] == "pass" and res["sentinel"] is None


def test_sentinel_equal_to_gate_suite_is_refused():
    with pytest.raises(ValueError):
        _gate(*triple(), sentinel=GATE_SUITE)


# ── argument refusals, JSON and determinism ──────────────────────────────────


@pytest.mark.parametrize("over", [{"suite": "m4"}, {"alpha": 0.0}, {"alpha": 0.5},
                                  {"alpha": -0.1}, {"tau_public": -0.01},
                                  {"tau_public": float("nan")}])
def test_bad_arguments_are_refused(over):
    with pytest.raises((ValueError, TypeError)):
        _gate(*triple(), **over)


@pytest.mark.parametrize("factors", [{}, {"house": (0.95, 0.95)}])
def test_result_is_strict_json(factors):
    res = _gate(*triple(**factors))
    assert json.loads(json.dumps(res, allow_nan=False)) == res


def test_deterministic_in_process():
    a = _gate(*triple(king=(1.0, 0.99), noise=0.05))
    b = _gate(*triple(king=(1.0, 0.99), noise=0.05))
    assert a == b


GATE_SCRIPT = (
    "import json, sys\n"
    "from cascade.validator.lane_d_bench import paired_gate\n"
    "a = json.loads(sys.argv[1])\n"
    "print(json.dumps(paired_gate(*a['records'], **a['kw']), sort_keys=True))\n"
)


def test_two_validators_agree():
    records = triple(king=(1.0, 0.99), house=(0.995, 1.01), noise=0.05)
    kw = gate_kwargs()
    here = _gate(*records)
    payload = json.dumps({"records": list(records), "kw": kw})
    for seed in ("1", "2"):
        env = {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
               "HOME": os.environ.get("HOME", "/tmp")}
        proc = subprocess.run([sys.executable, "-P", "-c", GATE_SCRIPT, payload],
                              capture_output=True, text=True, timeout=120, check=False,
                              env=env, cwd=str(REPO))
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == json.loads(json.dumps(here, sort_keys=True))
