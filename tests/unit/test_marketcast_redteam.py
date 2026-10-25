"""MarketCast red-team fixes (2026-10-20, DEC-CA-0064 §Red-team fixes).

The red-team simulations, ported: against an ORACLE base (the true
predictive distribution) every attack must FAIL to win —

* F1   sharpening / narrowing (all knots, inner knots only with the 10/90 %
  knots kept exact so the central gate passes, collapsed tails) and the
  log-RV vol shrink: the per-window log CRPS rewarded them; the per-cell
  MEAN-CRPS ratio does not;
* F1b  collapsed tails fail the tail gate; vol is calibration-gated;
* F2   miscalibrating only the thin classes (fx / metals / commodities) fails
  the class-weighted gate;
* F3   a lattice forecaster moving mass off zero gained from the
  realised-target filter; no filter reads the target now, zero-scale cells
  are masked and recorded;
* F4   a 1-ULP-nudged copy of a senior entry is a near duplicate;

while an HONEST better forecaster (vs a mis-scaled base) still wins. Plus
F5 (leg contract), F6 (owner-weighted family, coldkey leg cap) and F7
(deterministic forward faults), and the audit rejecting tampered new fields.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from cascade.eval import marketcast as M
from cascade.shared.config.marketcast import DEFAULT_ASSETS, TIMESTRAL_KNOTS, parse_marketcast
from cascade.shared.marketcast_report import (
    MktEntryRows,
    MktWindowRow,
    behaviour_distance,
    load_mkt_report,
    mkt_text_digest,
    score_windows,
    sign_mkt_report,
    verdict_from_rows,
)

K = np.asarray(TIMESTRAL_KNOTS)
HZ = {"1m": (1, 1), "5m": (5, 1), "1h": (60, 1), "4h": (60, 4), "1d": (1440, 1)}
CLASSES = ("crypto", "fx", "metals", "commodities", "equity_etf")
THIN = {"fx", "metals", "commodities"}
SIM_MC = parse_marketcast({"bootstrap_B": 400})


def _sim(transform, *, seed=0, df=3.0, base_tf=None, suite="returns", z=None, target_fn=None,
         per_cell=24, mc=SIM_MC):
    """The red-team market: per asset a lognormal scale, per (class, 6 h
    block) a lognormal regime, Student-t returns; the base is the ORACLE
    (``base_tf`` mis-scales it for the honest case). Returns ``(verdict,
    pooled arithmetic CRPS gain)``."""
    zq = (stats.t.ppf(K, df) / np.sqrt(df / (df - 2))) if z is None else z
    rng = np.random.default_rng(seed)
    t0 = 1_800_000_000 - (1_800_000_000 % 86400)
    span = 6 * 86400
    regime: dict = {}
    ys, qb, qf, wids = [], [], [], []
    for cls in CLASSES:
        for a in DEFAULT_ASSETS[cls]:
            sig_a = rng.lognormal(0, 0.5)
            for h, (fm, steps) in HZ.items():
                for i in range(per_cell):
                    start = t0 + int(i * (span - fm * 60 * steps) / (per_cell - 1)) // (
                        fm * 60) * (fm * 60)
                    bv = regime.setdefault((cls, start // 21600), rng.lognormal(0, 0.4))
                    sig = sig_a * bv * np.sqrt(fm)
                    y = sig * rng.standard_t(df, steps) / np.sqrt(df / (df - 2))
                    if target_fn is not None:
                        y = target_fn(y, rng)
                    q = np.tile(sig * zq, (steps, 1))
                    ys.append(y)
                    qb.append(q if base_tf is None else base_tf(q))
                    qf.append(transform(q, cls))
                    wids.append(f"{cls}:{a}|{suite}|{h}|{start}")
    sc = score_windows(ys, qb, qf, knots=K, coverage_levels=mc.coverage_levels, gated=True,
                       tail_levels=mc.tail_levels)
    rows = MktEntryRows("hk", "ptr", f"mkt:{suite}", "judge", 0.0, windows=tuple(
        MktWindowRow(w, b, f, ci, cn, float("nan"), tl)
        for w, (b, f, ci, cn, tl) in zip(wids, sc, strict=True)))
    v = verdict_from_rows(rows, mc, suite=suite, k=1, seed=f"s{seed}")
    lb, lf = np.array([r[0] for r in sc]), np.array([r[1] for r in sc])
    return v, float(1.0 - lf.mean() / lb.mean())


def _shrink(s, tails=None):
    def f(q, _cls):
        med = q[:, 10:11]
        out = med + s * (q - med)
        if tails is not None:
            for j in (0, 1, 19, 20):
                out[:, j] = med[:, 0] + tails * s * (q[:, j] - med[:, 0])
            out[:, 1] = np.minimum(out[:, 1], out[:, 2])
            out[:, 0] = np.minimum(out[:, 0], out[:, 1])
            out[:, 19] = np.maximum(out[:, 19], out[:, 18])
            out[:, 20] = np.maximum(out[:, 20], out[:, 19])
        return out
    return f


def _gated_preserving(tail, inner):
    """sharp2.py: the 0.10 / 0.25 / 0.75 / 0.90 knots kept EXACT (so the
    central 50 / 80 % gate passes) while the tails and the inner knots are
    pulled in."""
    def f(q, _cls):
        out, med = q.copy(), q[:, 10]
        for j in (0, 1):
            out[:, j] = q[:, 2] + tail * (q[:, j] - q[:, 2])
        for j in (19, 20):
            out[:, j] = q[:, 18] + tail * (q[:, j] - q[:, 18])
        for j in range(6, 15):
            if j != 10:
                out[:, j] = med + inner * (q[:, j] - med)
        for j in (3, 4):
            out[:, j] = q[:, 5] + inner * (q[:, j] - q[:, 5])
        for j in (16, 17):
            out[:, j] = q[:, 15] + inner * (q[:, j] - q[:, 15])
        return out
    return f


def _ident(q, _cls):
    return q


def _wide(s):
    return lambda q: q[:, 10:11] + s * (q - q[:, 10:11])


# ── F1 / F1b: sharpening never beats an oracle; honest still wins ────────────


@pytest.mark.parametrize("name,tf", [
    ("shrink 0.85", _shrink(0.85)),
    ("shrink 0.9 + tails 0", _shrink(0.9, 0.0)),
    ("inner x0.5 (won before: LCB 0.0195)", _gated_preserving(1.0, 0.5)),
    ("tails 0 + inner 0.5 (won before: LCB 0.031)", _gated_preserving(0.0, 0.5)),
    ("tails to the 10/90 knots", _gated_preserving(0.0, 1.0)),
])
def test_sharpening_never_beats_an_oracle_base(name, tf):
    v, arith = _sim(tf)
    assert not v.passed, (name, v.reason)
    assert v.gain <= 0.002 and v.gain_lcb < SIM_MC.gain_margin, (name, v.gain, v.gain_lcb)
    assert arith < 0                                      # it IS worse in mean CRPS


def test_collapsed_tails_fail_the_tail_gate_even_with_central_coverage_exact():
    v, _ = _sim(_gated_preserving(0.0, 1.0))
    bad = {r.key for r in v.coverage.rows if r.gated and (r.under or r.over)}
    assert any("tail" in k for k in bad) and not any(
        r.kind == "central" and (r.under or r.over) for r in v.coverage.rows if r.gated)
    assert any(r.horizon == "" and r.kind == "tail" for r in v.coverage.rows)   # pooled row


def test_an_oracle_passes_every_calibration_row_and_cannot_win():
    for seed in (0, 1, 2):
        v, _ = _sim(_ident, seed=seed)
        assert v.coverage.passed, [r.key for r in v.coverage.rows if r.under or r.over]
        assert not v.passed and v.gain == pytest.approx(0.0, abs=1e-12)


def test_an_honest_better_forecaster_still_wins():
    v, arith = _sim(_ident, base_tf=_wide(1.3))           # the base's quantiles 30 % too wide
    assert v.passed and v.gain_lcb >= SIM_MC.gain_margin and arith > 0, v.reason


def test_vol_shrink_never_wins_and_vol_is_calibration_gated():
    zn = stats.norm.ppf(K)
    for s in (0.8, 0.6, 0.5):                             # vol.py: +4.5 % / +7.3 % / +7.9 % before
        v, arith = _sim(_shrink(s), df=1e6, suite="vol", z=zn)
        assert not v.passed and v.gain < 0 and arith < 0, (s, v.gain)
        assert v.coverage is not None and not v.coverage.passed
    v, _ = _sim(_ident, df=1e6, suite="vol", z=zn, base_tf=_wide(1.4))
    assert v.passed, v.reason


# ── F2: thin classes are weighted like the gain ──────────────────────────────


def test_miscalibrating_only_the_thin_classes_fails_the_gate():
    def thin(q, cls):
        return _shrink(0.7)(q, cls) if cls in THIN else q
    v, _ = _sim(thin)
    # raw-pooled coverage stayed above 0.45 at 50 % (thin.py passed); the
    # class-weighted rate does not
    assert not v.passed and not v.coverage.passed
    assert any(r.key.endswith("@0.5") and r.under for r in v.coverage.rows)


# ── F3: no target filter; zero-scale cells masked and recorded ───────────────


def test_lattice_forecaster_moving_mass_off_zero_loses():
    vals = np.array([-2, -1, 0, 1, 2.0])
    p = np.array([0.05, 0.15, 0.6, 0.15, 0.05])

    def lattice(pp):
        c = np.cumsum(pp)
        return np.array([vals[np.searchsorted(c, k - 1e-12)] for k in K])

    rng = np.random.default_rng(0)
    smooth = np.quantile(rng.choice(vals, 10 ** 6, p=p) + rng.uniform(-0.5, 0.5, 10 ** 6), K)
    off = p.copy()
    off[2] = 0.0
    attack = lattice(off / off.sum())
    v, arith = _sim(lambda q, _c: np.tile(attack, (q.shape[0], 1)),
                    target_fn=lambda y, r: r.choice(vals, y.size, p=p),
                    base_tf=lambda q: np.tile(smooth, (q.shape[0], 1)))
    assert not v.passed and v.gain < -0.2 and arith < 0       # lattice.py: +18.7 % before


def test_cut_windows_never_reads_the_target():
    s = np.concatenate([np.random.default_rng(1).normal(0, 1, 100), np.zeros(100)])
    s[150:] = 0.0
    kw = dict(cls="c", asset="a", suite="returns", horizon="1m", freq_minutes=1, steps=1,
              t0=0, context_length=50, after=-1, now=1e9, min_context=10)
    ws = M.cut_windows(s, **kw)
    assert any(float(np.abs(w.target).sum()) == 0.0 for w in ws)   # zero targets stay
    # the stale-CONTEXT rule (known at the start) drops windows only once the
    # last min_context context values are flat
    stale = M.cut_windows(s, stale_context=True, **kw)
    assert {w.index for w in stale} == set(range(10, 110))
    with pytest.raises(ValueError, match="min_target_abs_bp was removed"):
        parse_marketcast({"min_target_abs_bp": 0.1})


def test_zero_scale_cells_are_masked_and_ratios_capped():
    cl = [f"fx|{i}" for i in range(10) for _ in range(5)]
    hz = ["1m"] * 50
    b = np.ones(50)
    b[:5] = 1e-9                                            # one cell's base loss ~0
    f = b.copy()
    f[:5] = 0.0                                             # a point mass on a flat cell
    Sb, Sf, _c, _h, masked = M.cell_sums(b, f, clusters=cl, horizons=hz)
    assert masked == ["fx|0|1m"] and Sb[0, 0] == 0 and Sf[0, 0] == 0
    g, _lcb = M.clustered_gain(b, f, clusters=cl, horizons=hz, alpha=0.05, B=200, seed="s")
    assert g == pytest.approx(0.0)                          # never a gain from that cell
    # one CLASS whose ratio is 0 (a point mass that happened to hit): capped
    cls = [f"c{i}|0" for i in range(10) for _ in range(5)]
    f2 = np.ones(50)
    f2[5:10] = 0.0
    g2, _ = M.clustered_gain(np.ones(50), f2, clusters=cls, horizons=hz, alpha=0.05, B=200,
                             seed="s")
    assert g2 == pytest.approx(1.0 - np.exp(-np.log(4.0) / 10))
    rows = MktEntryRows("h", "p", "mkt:range", "judge", 0.0, windows=tuple(
        MktWindowRow(f"fx:EUR|range|1m|{21600 * int(c.split('|')[1]) + j * 60}", float(x),
                     float(y)) for j, (c, x, y) in enumerate(zip(cl, b, f, strict=True))))
    v = verdict_from_rows(rows, replace(SIM_MC, min_windows=10, min_clusters=2,
                                        min_time_blocks=2), suite="range", k=1, seed="s")
    assert v.diagnostics["masked_cells"] == 1 and v.diagnostics["masked"] == ["fx|0|1m"]


# ── F4: behavioural near duplicates ──────────────────────────────────────────


def test_behaviour_distance_and_a_one_ulp_copy():
    rng = np.random.default_rng(0)
    base = [np.sort(rng.normal(0, 1, (2, 21)), axis=-1) for _ in range(30)]
    ft = [q * 1.1 for q in base]
    copy = [np.nextafter(q.astype(np.float32), np.float32(np.inf)).astype(np.float64)
            for q in ft]
    d, mx = behaviour_distance(base, ft, copy)
    assert 0 < d < 1e-5 and mx < 1e-4
    d2, _ = behaviour_distance(base, ft, [q * 1.2 for q in base])
    assert d2 > 0.02


# ── the validator + audit end to end ─────────────────────────────────────────


def _flow(cfg, tmp_path, *, factors, blocks, extra=None):
    from .test_marketcast_validator import (
        BLOCK,
        EB,
        _entry,
        _history,
        _ptr,
        _recorded,
        _rows,
        _runner,
        _settle,
        _snapshot,
        _Store,
    )

    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry(hk) for hk in factors]
    if extra:
        es = [replace(e, marketcast={**e.marketcast, **extra.get(e.miner_hotkey, {})})
              for e in es]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es, blocks))
    for hk, fc in factors.items():
        r._calls[_ptr(hk)] = fc
    rows1 = _rows(r, _settle(store, "1001", es))
    rows2 = _rows(r, _settle(store, "1002", [], prev="1001", block=BLOCK + EB,
                             reports={"1001": _recorded(store, "1001")}))
    return SimpleNamespace(r=r, store=store, s1=s1, rows=(rows1, rows2), BLOCK=BLOCK, EB=EB)


def _audit_inputs(f):
    reps = {k: v for k, v in f.store.objs.items() if k.endswith(".json")
            and k.startswith("marketcast/")}
    inputs = {"lane_c_manifests": {k.split("round-")[1][:-5]: v for k, v in f.store.objs.items()
                                   if k.startswith("lane_c/")},
              "quantiles": {k: v for k, v in f.store.objs.items() if k.endswith(".npy")},
              "snapshots": {f.s1.digest: f.s1}}
    return reps, inputs


def _with(rc, **kw):
    return SimpleNamespace(**{**rc.__dict__, **kw})


class _Scaled:
    """``_Fc``'s Gaussian quantiles × ``factor``, optionally nudged by a
    relative ``eps`` (a 1-ULP-scale weight edit)."""
    quantile_levels = TIMESTRAL_KNOTS

    def __init__(self, factor, eps=0.0, raise_with=None):
        from .test_marketcast_validator import Z

        self.factor, self.eps, self.raise_with, self.Z, self.n = factor, eps, raise_with, Z, 0

    def forecast_quantiles(self, context, horizon, *, past_covariates=None,
                           future_covariates=None):
        self.n += 1
        if self.raise_with is not None:
            raise self.raise_with
        c = np.asarray(context)
        sd = float(np.nanstd(c[np.isfinite(c)])) or 1.0
        return np.tile(self.Z * sd * self.factor * (1.0 + self.eps), (int(horizon), 1))


def test_a_near_copy_loses_to_the_earlier_commit_and_the_audit_replays_it(cfg, tmp_path):
    from cascade.audit.checks import check_marketcast

    from .test_marketcast_validator import BLOCK, _rc

    f = _flow(cfg, tmp_path, factors={"alice": _Scaled(1.0), "dave": _Scaled(1.0, eps=1e-7),
                                      "carol": _Scaled(1.6)},
              blocks={"alice": BLOCK - 900, "dave": BLOCK - 400, "carol": BLOCK - 300})
    rows2 = f.rows[1]
    by = {x["hotkey"]: x for x in rows2 if str(x.get("target", "")).startswith("mkt:")}
    assert by["dave"]["stage"] == "duplicate" and "near duplicate" in by["dave"]["reason"]
    nd = by["dave"]["mkt"]["near_dup"]
    assert nd["hotkey"] == "alice" and nd["mean_rel"] < 1e-5
    assert by["alice"]["stage"] == "verdict" and by["alice"]["passed"]
    assert by["carol"]["stage"] == "verdict" and "near_dup" in by["carol"]["mkt"]
    assert by["carol"]["mkt"]["near_dup"]["mean_rel"] > SIM_MC.near_dup_tol
    # the owner-weighted family: three owners, one entry each
    assert {x["k"] for x in by.values()} == {3}
    reps, inputs = _audit_inputs(f)
    rc = _rc("1002", BLOCK + f.EB, rows2)
    res = check_marketcast(rc, f.r.cfg, reps, **inputs)
    assert res.status == "PASS", res.detail
    # tamper in the receipt: the dup judged as a verdict, a moved near_dup, k, masked_cells
    rows = [{**x, "stage": "verdict"} if x.get("hotkey") == "dave" else x for x in rows2]
    assert check_marketcast(_with(rc, lane_c=rows), f.r.cfg, reps, **inputs).status == "FAIL"
    rows = [{**x, "mkt": {**x["mkt"], "near_dup": {**nd, "mean_rel": 0.5}}}
            if x.get("hotkey") == "dave" else x for x in rows2]
    assert check_marketcast(_with(rc, lane_c=rows), f.r.cfg, reps, **inputs).status == "FAIL"
    rows = [{**x, "k": 1} if x.get("hotkey") == "alice" else x for x in rows2]
    res = check_marketcast(_with(rc, lane_c=rows), f.r.cfg, reps, **inputs)
    assert res.status == "FAIL" and "F6" in res.detail
    rows = [{**x, "mkt": {**x["mkt"], "masked_cells": 3}} if x.get("hotkey") == "alice" else x
            for x in rows2]
    res = check_marketcast(_with(rc, lane_c=rows), f.r.cfg, reps, **inputs)
    assert res.status == "FAIL" and "F3" in res.detail
    # tamper by the main validator (re-signed, re-bound): the distance and the
    # tail tallies are recomputed from the sidecars
    rep = load_mkt_report(f.store.objs["marketcast/round-1001.json"])

    def _resigned(mutate, who):
        ents = [mutate(e) if e.hotkey == who else e for e in rep.entries]
        text = sign_mkt_report(replace(rep, entries=tuple(ents), signature=""),
                               _main_kp()).to_json()
        mut = next(e for e in ents if e.hotkey == who)
        rows = [{**x, "mkt": {**x["mkt"], "report_sha256": mkt_text_digest(text),
                              **({"near_dup": dict(mut.near_dup)} if mut.near_dup
                                 and x.get("hotkey") == who else {})}}
                if str(x.get("target", "")).startswith("mkt:") else x for x in rows2]
        return check_marketcast(_with(rc, lane_c=rows), f.r.cfg,
                                {**reps, "marketcast/round-1001.json": text}, **inputs)

    res = _resigned(lambda e: replace(e, near_dup={**e.near_dup, "mean_rel": 1e-9}), "carol")
    assert res.status == "FAIL"
    res = _resigned(lambda e: replace(e, near_dup={**e.near_dup, "max_rel": 1e-9}), "dave")
    assert res.status == "FAIL" and "near_dup recomputes" in res.detail
    w0 = lambda e: replace(e, windows=(replace(e.windows[0], tail=(9, 9)),) + e.windows[1:])  # noqa: E731
    res = _resigned(w0, "alice")
    assert res.status == "FAIL" and "tail" in res.detail
    # without the reference sidecar the distance is not verified: WARN
    q = {k: v for k, v in inputs["quantiles"].items() if "/r-" not in k}
    assert check_marketcast(rc, f.r.cfg, reps, **{**inputs, "quantiles": q}).status == "WARN"


def test_sybils_of_one_coldkey_share_one_owner_budget_end_to_end(cfg, tmp_path):
    from cascade.audit.checks import check_lane_c, check_marketcast

    from .test_marketcast_validator import BLOCK, _rc

    f = _flow(cfg, tmp_path, factors={"alice": _Scaled(1.0), "erin": _Scaled(2.4),
                                      "carol": _Scaled(1.6)},
              blocks=None, extra={"alice": {"coldkey": "COLD"}, "erin": {"coldkey": "COLD"}})
    rows2 = f.rows[1]
    by = {x["hotkey"]: x for x in rows2 if str(x.get("target", "")).startswith("mkt:")}
    # 2 owners: COLD's two hotkeys at alpha / (2 x 2), carol at alpha / 2
    assert {h: x["k"] for h, x in by.items()} == {"alice": 4, "erin": 4, "carol": 2}
    assert by["alice"]["mkt"]["owner"] == "ck:COLD" and by["carol"]["mkt"]["owner"] == "hk:carol"
    assert by["alice"]["alpha"] == pytest.approx(SIM_MC.alpha / 4)
    reps, inputs = _audit_inputs(f)
    rc = _rc("1002", BLOCK + f.EB, rows2)
    assert check_lane_c(rc, f.r.cfg).status == "PASS", check_lane_c(rc, f.r.cfg).detail
    assert check_marketcast(rc, f.r.cfg, reps, **inputs).status == "PASS"
    # a forged owner (splitting the sybils) moves k: both checks catch it
    rows = [{**x, "mkt": {**x["mkt"], "owner": "ck:OTHER"}} if x.get("hotkey") == "erin" else x
            for x in rows2]
    assert check_marketcast(_with(rc, lane_c=rows), f.r.cfg, reps, **inputs).status == "FAIL"
    assert check_lane_c(_with(rc, lane_c=rows), f.r.cfg).status == "FAIL"


def _main_kp():
    from .test_marketcast_validator import MAIN_KP

    return MAIN_KP


def test_the_earliest_commit_holds_whatever_the_uid():
    from cascade.validator.marketcast import _near_dup_refs

    r = MktEntryRows("dave", "pd", "mkt:returns", "judge", 0.0)
    datas = {("alice", "pa", "judge"): (b"", (100, 5, "alice")),
             ("zed", "pz", "judge"): (b"", (300, 5, "zed")),
             ("dave", "pd", "judge"): (b"", (200, 5, "dave"))}
    assert _near_dup_refs(r, (200, 5, "dave"), datas, {}) == [("alice", "pa", 100)]
    holders = {"mkt:returns": ("hold", "ph", -1)}
    assert ("hold", "ph", -1) in _near_dup_refs(r, (200, 5, "dave"), datas, holders)
    assert _near_dup_refs(r, (50, 5, "dave"), datas, {}) == []


# ── F6: owners, not hotkeys ──────────────────────────────────────────────────


def test_family_counts_owners_and_sybils_share_one_owner_budget():
    from cascade.shared.lane_c_manifest import LaneCEntry
    from cascade.shared.manifest import format_trained_pointer
    from cascade.validator.marketcast import mkt_family_sizes, mkt_owner

    def e(hk, ck=""):
        return LaneCEntry(hk, 1, "mkt:returns", f"{hk}/r@sha256:" + "1" * 64,
                          format_trained_pointer(f"c/{hk}@sha256:" + hashlib.sha256(
                              hk.encode()).hexdigest()), 1,
                          marketcast={"coldkey": ck} if ck else {})

    ents = [e("s1", "COLD"), e("s2", "COLD"), e("s3", "COLD"), e("honest", "OTHER"), e("solo")]
    ks = mkt_family_sizes(ents)
    by = {k[1]: v for k, v in ks.items()}
    # 3 owners: the sybil coldkey's 3 hotkeys each at alpha / (3 x 3), the others at alpha / 3
    assert by == {"s1": 9, "s2": 9, "s3": 9, "honest": 3, "solo": 3}
    assert sum(1.0 / v for v in by.values()) == pytest.approx(1.0)   # Σ levels = alpha
    assert mkt_owner(ents[-1]) == "hk:solo" and mkt_owner(ents[0]) == "ck:COLD"


def test_trainer_caps_open_mkt_legs_per_coldkey_and_stamps_it(tmp_path):
    from cascade.funding.queue import LaneCQueue
    from cascade.shared.chain import Commitment
    from cascade.shared.config.lane_c import LaneCConfig
    from cascade.shared.lane_c_manifest import format_finetune_commit, load_lane_c_manifest
    from cascade.shared.manifest import format_trained_pointer
    from cascade.trainer.lane_c_driver import LaneCDriver
    from cascade.trainer.marketcast_runner import MktLegRecipe

    FROM = 9000
    king = format_trained_pointer("cascade/ckpt@sha256:" + "a" * 64)
    mc = replace(SIM_MC, from_block=FROM)
    cfg = SimpleNamespace(lane_c=LaneCConfig(from_block=FROM, targets=("mkt:returns",),
                                             legs_per_target=5, max_base_age_eras=1),
                          marketcast=mc)
    refs = {hk: f"{hk.lower()}/recipe@sha256:" + hashlib.sha256(hk.encode()).hexdigest()
            for hk in ("ALFA", "BRAV")}
    pending: list = []
    published: list = []

    class Ops:
        q = LaneCQueue(tmp_path / "q.json", clock=lambda: 1_000_000.0)
        commits = [Commitment(uid=i, hotkey=hk, coldkey="COLD", commit_block=100,
                              payload=format_finetune_commit("mkt:returns", refs[hk],
                                                             mkt_suites=("returns", "vol")))
                   for i, hk in enumerate(refs)]

        def commitments(self):
            return self.commits

        def queue(self):
            return self.q

        def lane_a_busy(self, hk):
            return False

        def fetch_recipe(self, r, target, **kw):
            return MktLegRecipe(target=target, payload="{}", has_generator=False,
                                train_time=kw["train_time"])

        def train_leg(self, leg, recipe):
            return {"trained_pointer": format_trained_pointer(
                        "c/" + leg["hotkey"] + "@sha256:" + "c" * 64),
                    "corpus_digest": "c" * 64, "train_block": leg["started_block"],
                    "marketcast": {"base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
                                   "checkpoint_sha256": "d" * 64}}

        def sign(self, m):
            return replace(m, signature="ab" * 32)

        def publish(self, key, text):
            published.append(text)

        def spawn(self, fn, name):
            pending.append(fn)

        def now(self):
            return 1_234_567.0

    ops = Ops()
    for hk in refs:
        ops.q.add_lane_c(hk, "mkt:returns", refs[hk], 100)
    drv = LaneCDriver(cfg, ops, tmp_path / "drv.json")
    started = drv.tick(FROM + 10, 1, king, FROM)
    assert [leg["hotkey"] for leg in started] == ["ALFA"] and started[0]["coldkey"] == "COLD"
    assert drv.tick(FROM + 11, 1, king, FROM) == []        # BRAV waits: same coldkey
    pending.pop()()                                        # ALFA's leg finishes
    assert [leg["hotkey"] for leg in drv.tick(FROM + 12, 1, king, FROM)] == ["BRAV"]
    assert drv.settle(round_id="r1", block=FROM + 7200, era_index=2, base_pointer=king)
    m = load_lane_c_manifest(published[-1])
    assert [e.marketcast.get("coldkey") for e in m.entries] == ["COLD"]


# ── F5 / F7 ──────────────────────────────────────────────────────────────────


def test_mkt_leg_contract_is_the_lane_c_leg_not_the_throne():
    from cascade.shared.config import load_chain_config
    from cascade.trainer.marketcast_runner import mkt_leg_contract

    c = load_chain_config("chain.toml")
    base = c.throne_contracts()[0]
    leg = mkt_leg_contract(c, base)
    assert leg.corpus_mode == "stream_cpu"
    assert leg.max_train_seconds < base.max_train_seconds
    assert leg.max_train_seconds == pytest.approx(min(
        base.max_train_seconds, max(c.round.heat_guard_factor * c.lane_c.leg_hours * 3600,
                                    c.round.heat_guard_floor_seconds)), abs=1)


def test_a_deterministic_forward_exception_is_a_fault_infra_stays_infra(cfg, tmp_path):
    from cascade.validator.marketcast import MktForwardError, infra_like

    assert infra_like(RuntimeError("CUDA error: an illegal memory access"))
    assert infra_like(OSError("disk")) and not infra_like(IndexError("index 3"))
    assert issubclass(MktForwardError, Exception)
    f = _flow(cfg, tmp_path, factors={
        "alice": _Scaled(1.0),
        "carol": _Scaled(1.0, raise_with=RuntimeError("mat1 and mat2 shapes cannot be "
                                                      "multiplied")),
        "erin": _Scaled(1.0, raise_with=RuntimeError("CUDA out of memory"))},
        blocks=None)
    rep = load_mkt_report(f.store.objs["marketcast/round-1001.json"])
    by = {e.hotkey: e for e in rep.entries}
    assert "forward failed identically" in by["carol"].fault
    assert by["erin"].deferred and not by["erin"].fault
    assert not by["alice"].fault and not by["alice"].deferred
