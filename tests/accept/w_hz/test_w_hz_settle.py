"""W-HZ: a band settlement end to end, its report, our audit, and the OLD
checker other validators run (owner 2026-11-27 consensus constraint: only
our main validator scores lane D; every other validator follows the signed
report through the deployed release's ``lane_d_findings``). Contract:
README.md."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from .helpers import (
    BOUNDARY_1,
    CHALLENGER_UID,
    HASH_A,
    HOUR,
    OLD_RELEASES,
    REPO,
    T0,
    band_cfg,
    entry,
    feeds_for,
    fixture_key,
    forecasts,
    house,
    minute_fleet,
    need,
    old_raw,
    old_tree,
    run_old_checker,
    settle,
    write_band_pool,
)

BOUND = T0 + 7 * HOUR


def _settled(tmp_path, cfg, **kw):
    return _settle_out(tmp_path, cfg, **kw).report


def _settle_out(tmp_path, cfg, **kw):
    lane_d = need("cascade.validator.lane_d", "load_pool_for")
    fleet = minute_fleet(bound=BOUND, sources=4, per_source=6)
    d = write_band_pool(tmp_path / "pool", fleet, feeds=feeds_for(fleet))
    e, h = entry("alice"), house()
    fc = forecasts([lane_d.load_pool_for(d, cfg)], [e["digest"]], h["digest"])
    out = settle(cfg, BOUNDARY_1, d, [e], h, fc, {e["hotkey"]: CHALLENGER_UID},
                 tmp_path / "state.json", HASH_A, **kw)
    assert out.report is not None
    return out


def test_a_band_report_judges_windows_at_their_band_horizons(tmp_path):
    cfg = band_cfg()
    rep = _settled(tmp_path, cfg)
    assert rep["windows"], "fixture: the cohort judged windows"
    table = rep["horizon_bands"]["table"]
    assert table["T"] == {"short": 64, "medium": 480, "long": 736}
    for w in rep["windows"]:
        assert w["h"] == table["T"][w["band"]] and w["window_id"].endswith(f":h{w['h']}")
    (c,) = rep["cohorts"]
    assert set(c["cells"]) == {"subdaily|short", "subdaily|medium", "subdaily|long"}
    assert len(set(c["cells"].values())) == 1 and sum(c["cells"].values()) == len(
        c["window_ids"])
    (v,) = rep["verdicts"]
    assert set(v["cells"]) == {f"live|{k}" for k in c["cells"]}
    assert all(x["n"] == next(iter(c["cells"].values())) for x in v["cells"].values())
    # the config echo is the pre-band one: no band key in params
    assert rep["params"]["horizon"] == 64 and rep["params"]["horizons"] == [64]
    assert not {"horizon_bands", "max_horizon", "band_slots", "min_cell_windows"} & set(
        rep["params"])


def test_equal_cells_make_the_unweighted_statistic_the_equal_weight_one(tmp_path):
    from cascade.eval.bootstrap import clamped_log_ratios, log_gain

    cfg = band_cfg()
    rep = _settled(tmp_path, cfg)
    (c,), (v,) = rep["cohorts"], rep["verdicts"]
    rows = {(r["window_id"], r["digest"]): r for r in rep["rows"]}
    kd, m = c["king_digest"], v["digest"]
    cell = {w["window_id"]: w["band"] for w in rep["windows"]}
    x = clamped_log_ratios(*[np.asarray([rows[(w, d)][k] for w in c["window_ids"]])
                             for d in (kd, m) for k in ("crps", "mase")], floor=1e-9)
    per = {b: float(np.mean([xi for xi, w in zip(x, c["window_ids"], strict=True)
                             if cell[w] == b])) for b in set(cell.values())}
    assert np.isclose(v["halves"]["live"]["gain"], log_gain(float(np.mean(list(per.values())))))


def test_our_audit_passes_a_band_report_clean(tmp_path):
    cfg = band_cfg()
    rep = _settled(tmp_path, cfg)
    from cascade.audit.lane_d import lane_d_findings

    f = lane_d_findings(rep, cfg, signer_hotkey=fixture_key("signer").ss58_address)
    assert f.findings == [] and not f.transient


def test_our_chain_replay_matches_a_band_settlement(tmp_path):
    from cascade.audit.lane_d_chain import MATCH, replay_chain

    cfg = band_cfg()
    out = _settle_out(tmp_path, cfg)
    assert out.receipt is not None
    (res,) = replay_chain({BOUNDARY_1: out.report}, [out.receipt], cfg,
                          signer_ss58=fixture_key("signer").ss58_address)
    assert res.mismatches == () and res.status == MATCH, res.mismatches


def _deployed_checker_blobs() -> set[str]:
    """The three deployed releases' ``cascade/audit/lane_d.py`` blobs (one
    blob: one of them stands for all; re-checked, not assumed)."""
    blobs = set()
    for rel in OLD_RELEASES:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                              f"{rel}:cascade/audit/lane_d.py"],
                             capture_output=True, text=True)
        assert out.returncode == 0, f"release {rel} is not in this checkout: {out.stderr}"
        blobs.add(out.stdout.strip())
    return blobs


@pytest.mark.parametrize("public", [False, True])
def test_the_old_checker_pays_a_band_report(tmp_path, monkeypatch, public):
    """Backward compatibility: the pre-band checker (deployed release
    09f19baa) verifies a NEW band report under its own config (no band keys):
    no burn-class finding, so lane D's signed court is followed. Its only
    finding is the alert-class ``ld_embargo`` (it expects h = 64 on every
    window), which never stops pay."""
    over = {}
    extra = {}
    if public:
        over = {"public_weight": 0.5, "window_ratio_floor": 0.5}
        extra = _public_inputs(monkeypatch)
    cfg = band_cfg(**over)
    rep = _settled(tmp_path, cfg, **extra)
    assert any(w["h"] != 64 for w in rep["windows"]), "fixture: band horizons in the report"
    if public:
        assert any(w.get("band") for s in rep["public"]["sources"] for w in s["windows"])
    signer = fixture_key("signer").ss58_address
    assert len(_deployed_checker_blobs()) == 1, "the deployed lane D checkers differ"
    tree = old_tree(tmp_path)
    got = run_old_checker(tree, rep, old_raw(**over), signer, tmp_path)
    assert got["transient"] == "", got
    assert got["burn"] is None, got
    assert {code for code, _ in got["findings"]} <= {"ld_embargo"}, got
    assert got["n_rows"] == len(rep["rows"]) and got["n_cohorts"] == len(rep["cohorts"])


def _public_inputs(monkeypatch) -> dict:
    """Two public sources served from in-memory index rows (the draw reads
    the index alone) and every digest's per-window scores."""
    import cascade.validator.lane_d_public as lp
    from cascade.eval.lane_d_public_index import PUBLIC_SOURCES

    rows = {src: [{"ordinal": i, "series_id": f"{src}-s{i}", "dataset": f"ds{i % 3}",
                   "family": f"fam{i % 3}", "freq": "H", "length": 3000 + 37 * i,
                   "n_channels": 1} for i in range(40)] for src in PUBLIC_SOURCES}
    monkeypatch.setattr(lp, "iter_index", lambda d: iter(rows[str(d)]))
    monkeypatch.setattr(lp, "load_manifest", lambda d: {"source": str(d),
                                                        "index_sha256": "a" * 64,
                                                        "files_sha256": "b" * 64})
    from cascade.validator.lane_d_public import draw_public

    cfg = band_cfg(public_weight=0.5, window_ratio_floor=0.5)
    draws = [draw_public(src, source=src, boundary_block_hash=HASH_A, horizon=64,
                         bands=cfg.horizon_bands, max_horizon=cfg.max_horizon)
             for src in sorted(PUBLIC_SOURCES)]
    scores = {}
    for dg, crps in ((entry("alice")["digest"], 1.0), (house()["digest"], 1.3)):
        scores[dg] = {d["source"]: [{"window_id": w["window_id"], "family": w["family"],
                                     "crps": crps, "mase": crps} for w in d["windows"]]
                      for d in draws}
    return {"public_indexes": {src: src for src in PUBLIC_SOURCES},
            "public_scores": {"scores": scores, "dropped": {d["source"]: [] for d in draws}}}
