"""W9 (plan rev6.18 A4, F2): the audit re-derives every judged multichannel
window's channel pick from the report, the settlement's block hash and
``[generator] max_channels``, re-stated in `cascade/audit/lane_d.py` (never
imported from the producer, invariant 12), and with the judged snapshots
checks it against the stored series. Contract: README.md "Multichannel
windows (rev6.18 F2)".

On a main without F2 the fixture's `settle_mv` fails on its `max_channels`
assertion, and `check_lane_d` takes no `max_channels` (an `assert` here).
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np

from . import helpers as h
from .f2_fixture import mv

WIDE_EMPTY = (5, 17, 22, 30, 33, 38, 39)


def _run(tmp_path):
    """Built in the test body (never a fixture): on main the build fails on
    settle_mv's assertion, a failure of the test itself."""
    return mv.crowned_mv(tmp_path, channels={"src1": 3, "src3": 40},
                         empty={"src3": WIDE_EMPTY})


def _audit(report: dict, *, max_channels="repo", pool_dirs=None):
    from cascade.audit.lane_d import check_lane_d

    assert "max_channels" in inspect.signature(check_lane_d).parameters, (
        "F2: check_lane_d(..., max_channels=) re-derives the channel picks with "
        "[generator] max_channels")
    kw = {"max_channels": mv.repo_max_channels() if max_channels == "repo" else max_channels}
    if pool_dirs is not None:
        kw["pool_dirs"] = [Path(p) for p in pool_dirs]
    return check_lane_d(copy.deepcopy(report), h.w7a.lane_d_cfg(),
                        signer_hotkey=h.signer().ss58_address, **kw)


def _tampered(report: dict, fn) -> dict:
    out = copy.deepcopy(report)
    w = next(x for x in out["windows"] if x.get("n_channels_raw") == 40)
    fn(w)
    return h.resign(out)


def test_an_honest_multichannel_report_passes(tmp_path):
    run = _run(tmp_path)
    rep = run["s1"].report
    assert any("n_channels" in w for w in rep["windows"])
    res = _audit(rep)
    assert res.status == "PASS", res.detail
    res = _audit(rep, pool_dirs=[run["chain"]["pools"][1]])
    assert res.status == "PASS", res.detail


def test_another_channel_pick_fails(tmp_path):
    run = _run(tmp_path)
    rep = run["s1"].report

    def swap(w):
        left = [c for c in range(40) if c not in WIDE_EMPTY and c not in w["channels"]]
        w["channels"] = sorted([*w["channels"][1:], left[0]])
        w["channels_sha256"] = mv.channels_sha256(w["channels"])

    res = _audit(_tampered(rep, swap))
    assert res.status == "FAIL" and "channels" in res.detail


def test_a_hidden_empty_channel_claim_fails(tmp_path):
    run = _run(tmp_path)
    """Declaring more channels empty changes the candidates, so the pick no
    longer re-derives (and, with the snapshot, the claim is checked)."""
    rep = run["s1"].report

    def hide(w):
        w["empty_channels"] = sorted([*w["empty_channels"], w["channels"][0]])

    res = _audit(_tampered(rep, hide))
    assert res.status == "FAIL"


def test_the_record_must_match_the_stored_series(tmp_path):
    run = _run(tmp_path)
    rep = run["s1"].report

    def fewer(w):
        w["n_channels_raw"] = 39
        w["empty_channels"] = [c for c in w["empty_channels"] if c < 39]
        keep = [c for c in range(39) if c not in w["empty_channels"]]
        w["channels"] = mv.spec_pick(h.w7a.gate_block_hash(h.w7a.BOUNDARY_1), w["series_id"],
                                     keep, mv.repo_max_channels())
        w["n_channels"] = len(w["channels"])
        w["channels_sha256"] = mv.channels_sha256(w["channels"])

    bad = _tampered(rep, fewer)
    assert _audit(bad).status == "PASS", "self-consistent without the snapshot"
    res = _audit(bad, pool_dirs=[run["chain"]["pools"][1]])
    assert res.status == "FAIL" and "snapshot" in res.detail


def test_a_multichannel_series_recorded_univariate_fails_with_the_snapshot(tmp_path):
    run = _run(tmp_path)
    rep = copy.deepcopy(run["s1"].report)
    w = next(x for x in rep["windows"] if x.get("n_channels_raw") == 3)
    for k in mv.CHANNEL_FIELDS:
        w.pop(k)
    bad = h.resign(rep)
    res = _audit(bad, pool_dirs=[run["chain"]["pools"][1]])
    assert res.status == "FAIL" and "univariate" in res.detail


def test_without_max_channels_a_multichannel_report_fails(tmp_path):
    run = _run(tmp_path)
    res = _audit(run["s1"].report, max_channels=None)
    assert res.status == "FAIL" and "max_channels" in res.detail


def test_the_audit_rule_is_the_spec_and_the_producers():
    from cascade.audit.lane_d import channel_pick
    from cascade.eval.lane_d_channels import pick_channels

    rng = np.random.default_rng(3)
    for i in range(40):
        bh = "0x" + rng.bytes(32).hex()
        cand = sorted(rng.choice(64, size=int(rng.integers(1, 64)), replace=False).tolist())
        cap = int(rng.integers(1, 40))
        want = mv.spec_pick(bh, f"s{i}", cand, cap)
        assert channel_pick(bh, f"s{i}", cand, cap) == want
        assert list(pick_channels(bh, f"s{i}", cand, cap)) == want


def test_derive_weights_audits_lane_d_with_generator_max_channels():
    from cascade.audit.checks import _lane_d_max_channels
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(h.w7a.REPO / "chain.toml")
    assert _lane_d_max_channels(cfg) == mv.repo_max_channels() == cfg.generator.max_channels
