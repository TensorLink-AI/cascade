"""F2 (plan rev6.18 A4, DEC-CA-0041): settle_lane_d on multichannel
windows. A ``(C, L)`` window is ONE per-window score per model (channels
averaged), one resampling unit in its source's cluster; the channels are
dropped / capped before any model is scored; univariate results are
unchanged. Contract: README.md "Multichannel windows (rev6.18 F2)".

Fixture: `tests/accept/w7a/mv.py` (W7a's synthetic chain, no torch). Its
`settle_mv` asserts that `settle_lane_d` takes `max_channels`, so on a main
without F2 every test here fails on that assertion.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.accept.w7a import helpers as w7a
from tests.accept.w7a import mv

WIDE = 40
WIDE_EMPTY = (5, 17, 22, 30, 33, 38, 39)     # 33 non-empty channels left: one over the cap


def _scored(report: dict) -> dict:
    return {(r["window_id"], r["digest"]): r for r in report["rows"]}


def test_a_multichannel_window_is_one_row_per_model_scored_as_the_channel_mean(tmp_path):
    r = mv.crowned_mv(tmp_path, channels={"src1": 3, "src3": 4}, zero_target={"src1": (1,)})
    rep, chain = r["s1"].report, r["chain"]
    multi = mv.mv_window_ids(chain) & set(mv.windows_of(rep))
    assert multi, "the multichannel windows are judged"
    rows = _scored(rep)
    pools = {w["window_id"]: w for w in _pool_windows(chain["pools"][1])}
    models = {row["digest"] for row in rep["rows"]}
    for wid in sorted(multi):
        assert sum(1 for k in rows if k[0] == wid) == len(models), "ONE row per model"
        pick = chain["picks"][1][wid]
        w = pools[wid]
        for dg in models:
            want = mv.per_channel_score(chain["forecasts"][1][dg][wid], w["target"][pick],
                                        w["history"][pick], 24)
            assert rows[(wid, dg)]["crps"] == pytest.approx(want[0], rel=1e-12)
            assert rows[(wid, dg)]["mase"] == pytest.approx(want[1], rel=1e-12)


def test_a_multichannel_window_counts_once_in_its_source_cluster(tmp_path):
    r = mv.crowned_mv(tmp_path, channels={"src1": 5})
    rep = r["s1"].report
    cohort = rep["cohorts"][0]
    assert len(cohort["window_ids"]) == len(set(cohort["window_ids"]))
    ws = mv.windows_of(rep)
    src1 = [wid for wid in cohort["window_ids"] if ws[wid]["source"] == "src1"]
    assert src1 and all(ws[wid]["n_channels"] == 5 for wid in src1)
    models = len({row["digest"] for row in rep["rows"]})
    assert len(rep["rows"]) == len(cohort["window_ids"]) * models, (
        "evidence counts windows (and clusters by source), never channels")
    assert rep["crowned"] == r["entry"]["digest"]


def test_a_wider_series_is_cut_to_max_channels_by_the_boundary_block_hash(tmp_path):
    r = mv.crowned_mv(tmp_path, channels={"src3": WIDE}, empty={"src3": WIDE_EMPTY})
    rep = r["s1"].report
    cap = mv.repo_max_channels()
    ws = mv.windows_of(rep)
    wide = [w for w in ws.values() if w["source"] == "src3"]
    assert wide
    for w in wide:
        keep = [c for c in range(WIDE) if c not in WIDE_EMPTY]
        want = mv.spec_pick(w7a.gate_block_hash(w7a.BOUNDARY_1), w["series_id"], keep, cap)
        assert len(want) == cap == w["n_channels"]
        assert w["channels"] == want and w["empty_channels"] == list(WIDE_EMPTY)
        assert w["n_channels_raw"] == WIDE


def test_without_a_block_hash_an_over_cap_series_is_not_scored(tmp_path):
    chain = mv.build_mv_chain(tmp_path / "chain", channels={"src3": WIDE, "src1": 3},
                              empty={"src3": WIDE_EMPTY})
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    state = tmp_path / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True)
    s = mv.settle_mv(chain, state, boundary=w7a.BOUNDARY_1, snap=1, entries=[e],
                     uid_of={e["hotkey"]: w7a.CHALLENGER_UID}, block_hash=None)
    ws = mv.windows_of(s.report)
    assert ws and not any(w["source"] == "src3" for w in ws.values())
    assert any(w["source"] == "src1" and w["n_channels"] == 3 for w in ws.values()), (
        "a series within the cap needs no block hash")


def test_the_legacy_call_scores_no_multichannel_window(tmp_path):
    chain = mv.build_mv_chain(tmp_path / "chain", channels={"src1": 3})
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    state = tmp_path / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True)
    s = mv.settle_mv(chain, state, boundary=w7a.BOUNDARY_1, snap=1, entries=[e],
                     uid_of={e["hotkey"]: w7a.CHALLENGER_UID}, max_channels=None)
    ws = mv.windows_of(s.report)
    assert ws and not any(w["source"] == "src1" for w in ws.values())
    assert not any(k in w for w in ws.values() for k in mv.CHANNEL_FIELDS)


def test_univariate_results_are_unchanged_by_f2(tmp_path):
    """rev6.18 A4: C = 1 scores bit-identically. The same chain stored 1-D
    and as (1, L) / (1, horizon) gives the same rows, point gains and crown
    (the pool bytes differ, so the bootstrap seed and LCBs may)."""
    flat = mv.crowned_mv(tmp_path / "flat", channels={})
    rows2d = mv.crowned_mv(tmp_path / "rows2d", channels={}, one_row=True)
    a, b = flat["s1"].report, rows2d["s1"].report
    assert a["rows"] and a["rows"] == b["rows"] and a["rows_sha256"] == b["rows_sha256"]
    assert [(v["digest"], v["gain"], v["cleared"]) for v in a["verdicts"]] ==         [(v["digest"], v["gain"], v["cleared"]) for v in b["verdicts"]]
    assert a["crowned"] == b["crowned"] is not None
    assert not any(k in w for w in a["windows"] + b["windows"] for k in mv.CHANNEL_FIELDS)
    strip = ("snapshot_sha256",)
    assert ([{k: v for k, v in w.items() if k not in strip} for w in a["windows"]]
            == [{k: v for k, v in w.items() if k not in strip} for w in b["windows"]])


def test_a_univariate_settlement_is_the_same_report_with_or_without_max_channels(tmp_path):
    reports = []
    for name, cap in (("with", "repo"), ("legacy", None)):
        chain = mv.build_mv_chain(tmp_path / name, channels={})
        e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
        state = tmp_path / name / "state" / "lane_d_state.json"
        state.parent.mkdir(parents=True)
        s = mv.settle_mv(chain, state, boundary=w7a.BOUNDARY_1, snap=1, entries=[e],
                         uid_of={e["hotkey"]: w7a.CHALLENGER_UID}, max_channels=cap)
        reports.append({k: v for k, v in s.report.items() if k != "signature"})
    assert reports[0]["rows"] and reports[0] == reports[1]


def test_wiring_passes_generator_max_channels_from_the_config():
    from cascade.validator.lane_d_wiring import _max_channels

    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(w7a.REPO / "chain.toml")
    with open(w7a.REPO / "chain.toml", "rb") as f:
        raw = tomllib.load(f)["generator"]["max_channels"]
    assert _max_channels(cfg) == int(raw) == cfg.generator.max_channels == mv.repo_max_channels()


def test_inputs_send_the_eval_host_only_the_picked_channels():
    from cascade.validator.lane_d_inputs import LaneDInputsError, LaneDInputsSource

    hist = np.vstack([np.ones((3, 50)), np.full((1, 50), np.nan), np.ones((36, 50))])
    pools = [(Path("p"), {"windows": [
        {"window_id": "u", "history": np.ones(50), "target": np.ones(8)},
        {"window_id": "m", "series_id": "m-sid", "history": hist[:4],
         "target": np.ones((4, 8))},
        {"window_id": "w", "series_id": "w-sid", "history": hist,
         "target": np.ones((40, 8))}]})]
    bh = "0x" + "12" * 32
    reads = []
    reader = SimpleNamespace(block_hash=lambda b: reads.append(b) or bh)
    src = SimpleNamespace(cfg=SimpleNamespace(generator=SimpleNamespace(max_channels=32)),
                          _reader=lambda: reader)
    try:
        got = {w["window_id"]: w["history"]
               for w in LaneDInputsSource._windows(src, pools, w7a.BOUNDARY_1)}
    except (TypeError, LaneDInputsError) as e:
        raise AssertionError(f"F2: the inputs source does not pick channels: {e}") from e
    assert got["u"].shape == (50,)
    assert got["m"].shape == (3, 50), "the empty channel drops"
    want = mv.spec_pick(bh, "w-sid", [c for c in range(40) if c != 3], 32)
    assert got["w"].shape == (32, 50) and np.array_equal(got["w"], hist[want])
    assert reads == [w7a.BOUNDARY_1] and src.channel_block_hash == bh


def _pool_windows(pool_dir):
    from cascade.validator.lane_d import load_pool

    return load_pool(pool_dir, w7a.HORIZON)["windows"]
