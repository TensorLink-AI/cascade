"""W3b5 through the real `settle_lane_d`: the always-run A10 low-frequency
non-inferiority check on the private pool (plan rev6 §1.3b, §1.5, §1.7, T4b).
README.md has the contract.

Every state comes from real settlements. A candidate X first clears against
the genesis house H0 on snapshot 1 at boundary(1) and confirms on snapshot 2
at boundary(2): there it is a crowning candidate and the A10 check runs on
snapshot 2's low-frequency series (years old, so never forward-eligible).
The miner-king tests crown Y first (boundaries 1-2), then judge X against Y
and the house (boundaries 3-4).

On main the slice is counted inside the frozen set (always 0 here) and
never decides anything, so every test that expects `pass`/`fail` or a
`lowfreq_regression` fails on an AssertionError.
"""

from __future__ import annotations

import shutil

import pytest

from .helpers import (
    METRICS,
    REFS,
    T_FIRST,
    T_LATER,
    boundary,
    check_lowfreq_block,
    entry,
    entry_of,
    expected_lcbs,
    flat,
    hourly_forecasts,
    house,
    lane_d_cfg,
    lowfreq_forecasts,
    lowfreq_records,
    merge,
    read_state,
    settle,
    snapshot,
    uids,
    verdict_of,
    wids,
)

TOL = 1e-6
FLAT_1 = flat(1.0)


def _house_king_run(tmp_path, *, x_lf, house_lf=FLAT_1, n_lf=8, freq="D", cfg=None,
                    jitter=0.0, x_quality="sharp", lf_in_pool1=False):
    """X first-clears at boundary(1) and confirms at boundary(2) against the
    genesis house; snapshot 2 holds `n_lf` low-frequency series of `freq`."""
    cfg = cfg or lane_d_cfg()
    h0, x = house(), entry("x", reveal_ts=T_FIRST)
    hd, xd = h0["digest"], x["digest"]
    path = tmp_path / "state" / "lane_d_state.json"
    lf1 = lowfreq_records("lf1", n_lf, freq) if lf_in_pool1 else []
    pool1, recs1 = snapshot(tmp_path, 1, T_FIRST, lf1)
    fc1 = merge({hd: hourly_forecasts(recs1, "bad", "house-0"),
                 xd: hourly_forecasts(recs1, "sharp", "x")},
                {hd: lowfreq_forecasts(lf1, house_lf), xd: lowfreq_forecasts(lf1, x_lf)})
    s1 = settle(cfg, boundary(1), pool1, [x], h0, fc1, uids(x), path)
    after_s1 = tmp_path / "after_s1.json"
    shutil.copy(path, after_s1)
    lf2 = lowfreq_records("lf2", n_lf, freq)
    pool2, recs2 = snapshot(tmp_path, 2, T_FIRST, lf2)
    fc2 = merge({hd: hourly_forecasts(recs2, "bad", "house-0"),
                 xd: hourly_forecasts(recs2, x_quality, "x")},
                {hd: lowfreq_forecasts(lf2, house_lf),
                 xd: lowfreq_forecasts(lf2, x_lf, jitter=jitter, name="x")})
    s2 = settle(cfg, boundary(2), pool2, [x], h0, fc2, uids(x), path)
    return {"cfg": cfg, "h0": h0, "x": x, "path": path, "s1": s1, "s2": s2,
            "lf2": lf2, "recs2": recs2, "pool2": pool2, "fc2": fc2, "after_s1": after_s1}


def _confirm_verdict(run):
    v = verdict_of(run["s2"].report, run["x"]["digest"])
    assert v is not None, "X was not judged on its confirmation set"
    assert v["stage"] == "confirm" and v["cleared"] is True, \
        f"fixture: X must clear its confirmation duel on the hourly set: {v!r}"
    return v


def _assert_blocked(run, s, xd, king_digest):
    v = verdict_of(s.report, xd)
    assert v["outcome"] == "lost", f"a lowfreq_regression candidate is lost, not {v['outcome']!r}"
    assert s.report["crowned"] is None, "a low-frequency regression must not crown"
    assert s.report["king_after"]["digest"] == king_digest
    assert s.report["king_after"] == s.report["king_before"]
    e = entry_of(s.report, xd)
    assert (e["status"], e["reason"]) == ("lost", "lowfreq_regression"), e
    rec = read_state(run["path"])["entries"][xd]
    assert (rec["status"], rec["reason"]) == ("lost", "lowfreq_regression"), rec
    assert read_state(run["path"])["king"]["digest"] == king_digest
    assert xd in s.report["judged"], "a blocked candidate was judged; it is not a skip"
    assert xd not in {k["digest"] for k in s.report["skipped"]}


def _assert_crowned(s, xd, x_uid):
    v = verdict_of(s.report, xd)
    assert v["outcome"] == "crowned", v
    assert s.report["crowned"] == xd
    assert s.pay["court_uids"][0] == x_uid


def _validate(report):
    from cascade.validator.lane_d_report import validate_report

    validate_report(report)


# ── T4b: regression only on low-frequency series ────────────────────────────


def test_t4b_a_model_that_regresses_only_on_low_frequency_series_is_blocked(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.3))
    xd, hd = run["x"]["digest"], run["h0"]["digest"]
    v = _confirm_verdict(run)
    lf = v["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["result"] == "fail", lf
    assert lf["n_windows"] == len(run["lf2"]), \
        "the slice is the private pool's low-frequency windows, not the frozen set's"
    want = expected_lcbs(flat(1.3), flat(1.0))
    for ref in REFS:   # the house holds KING_D: both references are H0
        for m in METRICS:
            assert lf["lcb"][ref][m] == pytest.approx(want[m], abs=TOL), (ref, m, lf["lcb"])
    _assert_blocked(run, run["s2"], xd, hd)
    assert run["s2"].pay["court_uids"][0] is None, "the house keeps KING_D (its slot burns)"
    _validate(run["s2"].report)


def test_a_candidate_not_worse_on_low_frequency_series_is_crowned_with_pass(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(0.8))
    x = run["x"]
    v = _confirm_verdict(run)
    lf = v["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["result"] == "pass" and lf["n_windows"] == len(run["lf2"]), lf
    want = expected_lcbs(flat(0.8), flat(1.0))
    for ref in REFS:
        for m in METRICS:
            assert lf["lcb"][ref][m] == pytest.approx(want[m], abs=TOL)
    _assert_crowned(run["s2"], x["digest"], uids(x)[x["hotkey"]])
    assert read_state(run["path"])["entries"][x["digest"]]["reason"] == "crowned"
    _validate(run["s2"].report)


def test_a_regression_within_tau_public_passes_non_inferiority(tmp_path):
    # LCB = 1 - 1.005 = -0.005 >= -0.01: non-inferior, not a regression.
    run = _house_king_run(tmp_path, x_lf=flat(1.005))
    x = run["x"]
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["lcb"]["house"]["crps"] == pytest.approx(-0.005, abs=TOL)
    assert lf["result"] == "pass", lf
    _assert_crowned(run["s2"], x["digest"], uids(x)[x["hotkey"]])


def test_a_regression_just_beyond_tau_public_fails(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.02))
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["lcb"]["house"]["crps"] == pytest.approx(-0.02, abs=TOL)
    assert lf["result"] == "fail", lf
    _assert_blocked(run, run["s2"], run["x"]["digest"], run["h0"]["digest"])


def test_tau_public_is_read_from_config(tmp_path):
    # The same 3% regression: blocked at the default 0.01, non-inferior at 0.05.
    cfg = lane_d_cfg(tau_public=0.05)
    run = _house_king_run(tmp_path, x_lf=flat(1.03), cfg=cfg)
    x = run["x"]
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, cfg)
    assert lf["tau_public"] == 0.05
    assert lf["result"] == "pass", lf
    _assert_crowned(run["s2"], x["digest"], uids(x)[x["hotkey"]])


def test_a_mase_only_regression_is_blocked(tmp_path):
    # Better CRPS (weight 2.55 vs 4.5), but the median is 1.5x further off.
    spec = (0.1, 1.5, 1.5)
    run = _house_king_run(tmp_path, x_lf=spec)
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    want = expected_lcbs(spec, flat(1.0))
    assert want["crps"] > 0 > want["mase"]
    assert lf["lcb"]["house"]["crps"] == pytest.approx(want["crps"], abs=TOL)
    assert lf["lcb"]["house"]["mase"] == pytest.approx(want["mase"], abs=TOL)
    assert lf["result"] == "fail", "CRPS and MASE must EACH be non-inferior"
    _assert_blocked(run, run["s2"], run["x"]["digest"], run["h0"]["digest"])


def test_a_crps_only_regression_is_blocked(tmp_path):
    # A sharper median (MASE x0.5) but wide upper quantiles (CRPS x1.5).
    spec = (0.5, 0.5, 5.0)
    run = _house_king_run(tmp_path, x_lf=spec)
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    want = expected_lcbs(spec, flat(1.0))
    assert want["mase"] > 0 > want["crps"]
    assert lf["lcb"]["house"]["crps"] == pytest.approx(want["crps"], abs=TOL)
    assert lf["lcb"]["house"]["mase"] == pytest.approx(want["mase"], abs=TOL)
    assert lf["result"] == "fail", lf
    _assert_blocked(run, run["s2"], run["x"]["digest"], run["h0"]["digest"])


# ── the slice: the configured frequencies, the private pool, no floor ───────


def test_the_slice_follows_lowfreq_freqs(tmp_path):
    # Weekly series with a 30% regression: in the slice under ["W"] ...
    cfg_w = lane_d_cfg(lowfreq_freqs=["W"])
    run = _house_king_run(tmp_path / "w", x_lf=flat(1.3), freq="W", cfg=cfg_w)
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, cfg_w)
    assert lf["freqs"] == ["W"] and lf["n_windows"] == len(run["lf2"])
    assert lf["result"] == "fail"
    _assert_blocked(run, run["s2"], run["x"]["digest"], run["h0"]["digest"])


def test_series_outside_lowfreq_freqs_are_not_in_the_slice(tmp_path):
    # ... and outside it under ["D"]: no slice window, so nothing blocks.
    cfg_d = lane_d_cfg(lowfreq_freqs=["D"])
    run = _house_king_run(tmp_path, x_lf=flat(1.3), freq="W", cfg=cfg_d)
    x = run["x"]
    lf = _confirm_verdict(run)["lowfreq"]
    assert lf == {"result": "no_windows", "freqs": ["D"], "n_windows": 0, "tau_public": 0.01}
    _assert_crowned(run["s2"], x["digest"], uids(x)[x["hotkey"]])


def test_the_slice_is_the_private_pool_not_the_forward_only_set(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.3))
    s2 = run["s2"].report
    lf_ids = set(wids(run["lf2"]))
    (cohort,) = s2["cohorts"]
    assert not lf_ids & set(cohort["window_ids"]), \
        "fixture: the low-frequency series must not be forward-eligible"
    assert cohort["refused"].get("before_margin", 0) >= len(lf_ids)
    assert not lf_ids & {r["window_id"] for r in s2["rows"]}, \
        "the A10 slice does not enter the duel's rows"
    lf = verdict_of(s2, run["x"]["digest"])["lowfreq"]
    assert lf["n_windows"] == len(lf_ids), \
        "§1.7: A10 uses the private pool because daily-or-slower series rarely qualify"


def test_a_single_low_frequency_window_is_still_checked(tmp_path):
    # The plan states no floor for the slice: one window is a slice.
    run = _house_king_run(tmp_path, x_lf=flat(1.3), n_lf=1)
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["n_windows"] == 1 and lf["result"] == "fail", lf
    _assert_blocked(run, run["s2"], run["x"]["digest"], run["h0"]["digest"])


def test_an_empty_slice_is_no_windows_and_never_blocks(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.3), n_lf=0)
    x = run["x"]
    cfg = run["cfg"]
    lf = _confirm_verdict(run)["lowfreq"]
    assert lf == {"result": "no_windows", "freqs": [str(f) for f in cfg.lowfreq_freqs],
                  "n_windows": 0, "tau_public": float(cfg.tau_public)}
    _assert_crowned(run["s2"], x["digest"], uids(x)[x["hotkey"]])


# ── who is checked ──────────────────────────────────────────────────────────


def test_first_clears_are_not_checked_and_go_confirm_pending(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.3), lf_in_pool1=True)
    cfg, xd = run["cfg"], run["x"]["digest"]
    v1 = verdict_of(run["s1"].report, xd)
    assert v1["stage"] == "first" and v1["outcome"] == "confirm_pending", v1
    assert v1["lowfreq"] == {"result": "not_candidate",
                             "freqs": [str(f) for f in cfg.lowfreq_freqs], "n_windows": 0,
                             "tau_public": float(cfg.tau_public)}
    e = entry_of(run["s1"].report, xd)
    assert (e["status"], e["reason"]) == ("confirm_pending", "confirm_pending")


def test_a_confirmation_loser_is_lost_confirmation_not_lowfreq_regression(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.3), x_quality="bad")
    cfg, xd = run["cfg"], run["x"]["digest"]
    v = verdict_of(run["s2"].report, xd)
    assert v["stage"] == "confirm" and v["cleared"] is False, v
    assert v["lowfreq"]["result"] == "not_candidate" and v["lowfreq"]["n_windows"] == 0
    assert v["lowfreq"]["tau_public"] == float(cfg.tau_public)
    e = entry_of(run["s2"].report, xd)
    assert (e["status"], e["reason"]) == ("lost", "lost_confirmation")


def test_a_blocked_entry_is_not_judged_again(tmp_path):
    run = _house_king_run(tmp_path, x_lf=flat(1.3))
    x, h0, cfg = run["x"], run["h0"], run["cfg"]
    _assert_blocked(run, run["s2"], x["digest"], h0["digest"])
    pool3, recs3 = snapshot(tmp_path, 3, T_FIRST)
    fc3 = {h0["digest"]: hourly_forecasts(recs3, "bad", "house-0"),
           x["digest"]: hourly_forecasts(recs3, "sharp", "x")}
    s3 = settle(cfg, boundary(3), pool3, [x], h0, fc3, uids(x), run["path"])
    assert x["digest"] not in s3.report["judged"]
    assert s3.report["crowned"] is None
    rec = read_state(run["path"])["entries"][x["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "lowfreq_regression")


def _runner_up_run(tmp_path, order=("x", "z")):
    """X (sharp) and Z (good) both first-clear the house at boundary(1) and
    confirm at boundary(2); on snapshot 2's slice X regresses (x1.3, jittered)
    and Z is better (x0.7). The 5% jitter keeps every window's ratio on its
    side of `-tau_public`, so the result holds for any paired bootstrap.
    `order` is the entries' input order."""
    cfg = lane_d_cfg()
    h0 = house()
    named = {"x": entry("x", reveal_ts=T_FIRST), "z": entry("z", reveal_ts=T_FIRST + 60)}
    x, z = named["x"], named["z"]
    ents = [named[n] for n in order]
    hd, xd, zd = h0["digest"], x["digest"], z["digest"]
    uid_of = uids(x, z)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1, T_FIRST + 60)
    fc1 = {hd: hourly_forecasts(recs1, "bad", "house-0"),
           xd: hourly_forecasts(recs1, "sharp", "x"), zd: hourly_forecasts(recs1, "good", "z")}
    s1 = settle(cfg, boundary(1), pool1, ents, h0, fc1, uid_of, path)
    assert {verdict_of(s1.report, d)["outcome"] for d in (xd, zd)} == {"confirm_pending"}
    lf2 = lowfreq_records("lf2", 8)
    pool2, recs2 = snapshot(tmp_path, 2, T_FIRST + 60, lf2)
    fc2 = merge({hd: hourly_forecasts(recs2, "bad", "house-0"),
                 xd: hourly_forecasts(recs2, "sharp", "x"),
                 zd: hourly_forecasts(recs2, "good", "z")},
                {hd: lowfreq_forecasts(lf2, flat(1.0)),
                 xd: lowfreq_forecasts(lf2, flat(1.3), jitter=0.05, name="x"),
                 zd: lowfreq_forecasts(lf2, flat(0.7), jitter=0.05, name="z")})
    s2 = settle(cfg, boundary(2), pool2, ents, h0, fc2, uid_of, path)
    return {"cfg": cfg, "x": x, "z": z, "path": path, "s2": s2, "uid_of": uid_of}


def test_every_crowning_candidate_is_checked_and_a_passing_runner_up_crowns(tmp_path):
    """X, the max-T best on the hourly duel, regresses on the low-frequency
    slice; Z clears too and is non-inferior there. X is blocked, Z crowns."""
    run = _runner_up_run(tmp_path)
    cfg, x, z, path, s2 = run["cfg"], run["x"], run["z"], run["path"], run["s2"]
    xd, zd = x["digest"], z["digest"]
    vx, vz = verdict_of(s2.report, xd), verdict_of(s2.report, zd)
    assert vx["cleared"] and vz["cleared"] and vx["stage"] == vz["stage"] == "confirm"
    assert vx["lcb"] > vz["lcb"], "fixture: X must be the max-T best"
    check_lowfreq_block(vx["lowfreq"], cfg)
    check_lowfreq_block(vz["lowfreq"], cfg)
    assert vx["lowfreq"]["result"] == "fail" and vz["lowfreq"]["result"] == "pass"
    assert vx["outcome"] == "lost" and vz["outcome"] == "crowned"
    assert s2.report["crowned"] == zd
    assert entry_of(s2.report, xd)["reason"] == "lowfreq_regression"
    assert read_state(path)["king"]["digest"] == zd
    assert s2.pay["court_uids"][0] == run["uid_of"][z["hotkey"]]


# ── a miner KING_D: the house is still a reference ──────────────────────────


def _miner_king_run(tmp_path, *, x_lf, y_lf4=FLAT_1, house_lf4=FLAT_1):
    """Y is crowned at boundary(2) (non-inferior on snapshot 2's slice). X,
    revealed later, first-clears against Y and the house at boundary(3) and
    confirms at boundary(4); snapshot 4 holds the slice X is checked on."""
    cfg = lane_d_cfg()
    h0 = house()
    y = entry("y", reveal_ts=T_FIRST)
    x = entry("x", reveal_ts=T_LATER, after=2)
    hd, yd, xd = h0["digest"], y["digest"], x["digest"]
    uid_of = uids(y, x)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1, T_FIRST)
    settle(cfg, boundary(1), pool1, [y], h0,
           {hd: hourly_forecasts(recs1, "bad", "house-0"),
            yd: hourly_forecasts(recs1, "good", "y")}, uid_of, path)
    lf2 = lowfreq_records("lf2", 8)
    pool2, recs2 = snapshot(tmp_path, 2, T_FIRST, lf2)
    s2 = settle(cfg, boundary(2), pool2, [y], h0,
                merge({hd: hourly_forecasts(recs2, "bad", "house-0"),
                       yd: hourly_forecasts(recs2, "good", "y")},
                      {hd: lowfreq_forecasts(lf2, flat(1.0)),
                       yd: lowfreq_forecasts(lf2, flat(1.0))}), uid_of, path)
    assert s2.report["crowned"] == yd, "fixture: Y must be crowned at boundary(2)"
    pool3, recs3 = snapshot(tmp_path, 3, T_LATER)
    s3 = settle(cfg, boundary(3), pool3, [y, x], h0,
                {hd: hourly_forecasts(recs3, "bad", "house-0"),
                 yd: hourly_forecasts(recs3, "good", "y"),
                 xd: hourly_forecasts(recs3, "sharp", "x")}, uid_of, path)
    assert verdict_of(s3.report, xd)["outcome"] == "confirm_pending", \
        "fixture: X must first-clear against Y and the house"
    lf4 = lowfreq_records("lf4", 8)
    pool4, recs4 = snapshot(tmp_path, 4, T_LATER, lf4)
    s4 = settle(cfg, boundary(4), pool4, [y, x], h0,
                merge({hd: hourly_forecasts(recs4, "bad", "house-0"),
                       yd: hourly_forecasts(recs4, "good", "y"),
                       xd: hourly_forecasts(recs4, "sharp", "x")},
                      {hd: lowfreq_forecasts(lf4, house_lf4), yd: lowfreq_forecasts(lf4, y_lf4),
                       xd: lowfreq_forecasts(lf4, x_lf)}), uid_of, path)
    return {"cfg": cfg, "h0": h0, "x": x, "y": y, "path": path, "s2": s2, "s4": s4,
            "uid_of": uid_of}


def test_with_a_miner_king_a_regression_against_the_house_alone_blocks(tmp_path):
    # X beats KING_D Y on the slice (+5%) but is 36% worse than the house.
    run = _miner_king_run(tmp_path, x_lf=flat(0.95), house_lf4=flat(0.7))
    xd, yd = run["x"]["digest"], run["y"]["digest"]
    v = verdict_of(run["s4"].report, xd)
    assert v["stage"] == "confirm" and v["cleared"] and v["king_digest"] == yd, v
    lf = v["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    vs_y, vs_h = expected_lcbs(flat(0.95), flat(1.0)), expected_lcbs(flat(0.95), flat(0.7))
    for m in METRICS:
        assert lf["lcb"]["king_d"][m] == pytest.approx(vs_y[m], abs=TOL)
        assert lf["lcb"]["house"][m] == pytest.approx(vs_h[m], abs=TOL)
    assert lf["result"] == "fail"
    _assert_blocked(run, run["s4"], xd, yd)
    assert run["s4"].pay["court_uids"][0] == run["uid_of"][run["y"]["hotkey"]], \
        "KING_D Y keeps its slot"


def test_with_a_miner_king_a_regression_against_king_d_alone_blocks(tmp_path):
    run = _miner_king_run(tmp_path, x_lf=flat(1.0), y_lf4=flat(0.8), house_lf4=flat(1.2))
    xd, yd = run["x"]["digest"], run["y"]["digest"]
    lf = verdict_of(run["s4"].report, xd)["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["lcb"]["king_d"]["crps"] == pytest.approx(1.0 - 1.0 / 0.8, abs=TOL)
    assert lf["lcb"]["house"]["crps"] == pytest.approx(1.0 - 1.0 / 1.2, abs=TOL)
    assert lf["result"] == "fail"
    _assert_blocked(run, run["s4"], xd, yd)


def test_with_a_miner_king_non_inferior_on_both_crowns(tmp_path):
    run = _miner_king_run(tmp_path, x_lf=flat(0.9))
    x, yd = run["x"], run["y"]["digest"]
    # Y's own crowning ran the check against the house on snapshot 2.
    vy = verdict_of(run["s2"].report, yd)
    check_lowfreq_block(vy["lowfreq"], run["cfg"])
    assert vy["lowfreq"]["result"] == "pass"
    lf = verdict_of(run["s4"].report, x["digest"])["lowfreq"]
    check_lowfreq_block(lf, run["cfg"])
    assert lf["result"] == "pass"
    _assert_crowned(run["s4"], x["digest"], run["uid_of"][x["hotkey"]])
    state = read_state(run["path"])
    assert [p["digest"] for p in state["prior_kings"]] == [yd]
    _validate(run["s4"].report)


# ── determinism ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("x_lf", [flat(0.7), flat(1.3)], ids=["pass", "fail"])
def test_the_check_is_deterministic_across_a_restart(tmp_path, x_lf):
    """A jittered slice (per-window ratios vary, so the bootstrap is not
    degenerate): boundary(2) settled twice from copies of the same
    boundary(1) state gives the same A10 record, outcome, entries and
    report digests."""
    run = _house_king_run(tmp_path, x_lf=x_lf, jitter=0.05)
    x, h0, cfg = run["x"], run["h0"], run["cfg"]
    lf = _confirm_verdict(run)["lowfreq"]
    check_lowfreq_block(lf, cfg)
    replay = tmp_path / "replay" / "lane_d_state.json"
    replay.parent.mkdir(parents=True)
    shutil.copy(run["after_s1"], replay)
    again = settle(cfg, boundary(2), run["pool2"], [x], h0, run["fc2"], uids(x), replay)
    r1, r2 = run["s2"].report, again.report
    assert verdict_of(r2, x["digest"]) == verdict_of(r1, x["digest"])
    for key in ("rows_sha256", "crowned", "entries", "king_after", "judged", "skipped"):
        assert r2[key] == r1[key], key
    assert read_state(replay)["entries"][x["digest"]] == \
        read_state(run["path"])["entries"][x["digest"]]


def test_the_check_is_independent_of_entry_input_order(tmp_path):
    a = _runner_up_run(tmp_path / "a", order=("x", "z"))
    b = _runner_up_run(tmp_path / "b", order=("z", "x"))
    for key in ("verdicts", "rows_sha256", "crowned", "entries", "judged"):
        assert a["s2"].report[key] == b["s2"].report[key], key
