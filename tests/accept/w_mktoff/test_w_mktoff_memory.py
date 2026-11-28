"""W-MKTOFF box guards: the pod judge keeps the window arrays off the box,
every round logs its RSS (ERROR over budget), an in-process judge over the
box cap is refused, and IR010 flags the MarketCast materialisers."""

from __future__ import annotations

import importlib.util
import logging
import sys
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from tests.accept.w_mktoff import helpers as H

ROOT = Path(__file__).resolve().parents[3]


def _peak(tmp_path, snap, *, on_pod, mc=None):
    v, store, sess = H.rig(tmp_path, snap, on_pod=on_pod)
    if mc is not None:
        v.cfg = SimpleNamespace(marketcast=mc)
    tracemalloc.start()
    try:
        assert v._mkt_publish_round(H.item(), holders=H.holders()) is True
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        sess.close()


def test_box_allocation_does_not_grow_with_the_context(tmp_path):
    """The in-process judge's box peak grows with the context length (the
    covariate rows); the pod judge's stays flat — the windows live on the pod."""
    snap = H.snapshot(tmp_path / "snap", minutes=6000)
    wide = replace(H.mc(), eval_context_length=1024)
    old = _peak(tmp_path / "o", snap, on_pod=False, mc=wide)
    new = _peak(tmp_path / "n", snap, on_pod=True, mc=wide)
    assert new * 4 < old, (new, old)


def test_each_round_logs_its_rss_and_errors_over_budget(tmp_path, caplog):
    snap = H.snapshot(tmp_path / "snap")
    v, _store, sess = H.rig(tmp_path / "r", snap)
    seq = iter([100 << 20] + [900 << 20] * 10_000)
    v.mkt_rss_probe = lambda: next(seq)
    v.mkt_rss_interval = 0.01
    caplog.set_level(logging.INFO, logger="cascade.validator")
    try:
        assert v._mkt_publish_round(H.item(), holders=H.holders()) is True
    finally:
        sess.close()
    msgs = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any("rss start 100 MB, peak 900 MB (+800 MB)" in m for _l, m in msgs), msgs
    assert any(lv == logging.ERROR and "over the 512 MB box budget" in m for lv, m in msgs)


def test_in_process_judge_over_the_box_cap_is_refused(tmp_path, caplog):
    snap = H.snapshot(tmp_path / "snap")
    v, store, sess = H.rig(tmp_path / "r", snap, on_pod=False)
    v.mkt_box_judge_cap_bytes = 1 << 20
    caplog.set_level(logging.ERROR, logger="cascade.validator")
    try:
        assert v._mkt_publish_round(H.item(), holders=H.holders()) is False
    finally:
        sess.close()
    assert store.puts == []
    assert any("over the 1 MB box cap: refused" in r.getMessage() for r in caplog.records)


def test_the_10_54_geometry_is_refused_in_process(tmp_path, caplog):
    """The deployed geometry (58 perp assets, returns + vol, 4096 steps,
    mkt:market): the in-process estimate is ~16 GB, refused before a window
    is cut — a pod whose worker predates the judge lands here."""
    from cascade.shared.config import load_chain_config

    mc = load_chain_config(ROOT / "chain.toml").marketcast
    snap = H.snapshot(tmp_path / "snap")
    big = SimpleNamespace(root=snap.root, meta=snap.meta, digest=snap.digest,
                          assets=lambda cls: list(dict(mc.assets).get(cls, ())))
    v, store, sess = H.rig(tmp_path / "r", snap, on_pod=False)
    v.cfg = SimpleNamespace(marketcast=mc)
    v.mkt_snapshot_fn = lambda block: (big, "k", "e" * 64)
    it = H.item()
    for c in it["candidates"]:
        c["entry"]["target"] = "mkt:market"
    caplog.set_level(logging.ERROR, logger="cascade.validator")
    try:
        assert v._mkt_publish_round(it, holders={}) is False
    finally:
        sess.close()
    msg = next(r.getMessage() for r in caplog.records if "box cap: refused" in r.getMessage())
    mb = int(msg.split("needs ~", 1)[1].split(" MB", 1)[0])
    assert mb > 10_000, msg


def _cir():
    spec = importlib.util.spec_from_file_location("check_incident_rules_mktoff",
                                                  ROOT / "scripts" / "check_incident_rules.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ir010_flags_the_marketcast_judge_materialisers():
    cir = _cir()
    for call in ("target_windows(snap, mc, s, after=0)", "window_covariates(snap, w, n)",
                 "_covariate_rows(ws, snap, mc, covs)", "cut_windows(s)",
                 "build_covariates(n, block_starts=t, n_context=4)"):
        text = f"def f(snap, mc, s, w, n, ws, covs, t):\n    return {call}\n"
        for rel in ("cascade/validator/marketcast.py", "cascade/validator/mkt_x.py"):
            assert [h[0] for h in cir.check_ir010(rel, text)] == [2], (rel, call)
        # elsewhere (the pod-side pool module, other validator files) not a hit
        assert cir.check_ir010("cascade/validator/lane_b.py", text) == []


def test_ir010_repo_is_clean_with_the_reviewed_judge_sites():
    cir = _cir()
    rels = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "cascade" / "validator").glob("*.py")
                  if p.name.startswith(("marketcast", "mkt_")))
    raw = [h for rel in rels for h in cir.check_ir010(rel, (ROOT / rel).read_text())]
    judge = [h for h in raw if "materialises MarketCast windows" in h[1]]
    assert len(judge) >= 6, "the judge's call sites are hits (allowlisted)"
    assert [h for h in cir.check(ROOT, paths=rels) if h.rule == "IR010"] == []
