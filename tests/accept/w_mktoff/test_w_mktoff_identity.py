"""W-MKTOFF: the pod judge publishes exactly the in-process path's bytes,
and the box never cuts a window or builds a covariate row."""

from __future__ import annotations

import threading

import pytest

from tests.accept.w_mktoff import helpers as H


def _run(tmp_path, snap, *, on_pod, monkeypatch):
    from cascade.pool import marketcast as P
    from cascade.validator import marketcast as M

    calls = []
    for mod, name in ((P, "target_windows"), (P, "window_covariates"),
                      (M, "_covariate_rows"), (M, "compute_entry_rows")):
        orig = getattr(mod, name)

        def spy(*a, _o=orig, _n=name, **k):
            calls.append((_n, threading.current_thread().name))
            return _o(*a, **k)

        monkeypatch.setattr(mod, name, spy)
    v, store, sess = H.rig(tmp_path, snap, on_pod=on_pod)
    try:
        assert v._mkt_publish_round(H.item(), holders=H.holders()) is True
    finally:
        sess.close()
        monkeypatch.undo()
    return store, calls


def test_pod_judge_report_and_sidecars_are_byte_identical(tmp_path, monkeypatch):
    from cascade.shared.marketcast_report import INFRA_DEFERRED, load_mkt_report

    snap = H.snapshot(tmp_path / "snap")
    old, old_calls = _run(tmp_path / "old", snap, on_pod=False, monkeypatch=monkeypatch)
    new, new_calls = _run(tmp_path / "new", snap, on_pod=True, monkeypatch=monkeypatch)
    key = "marketcast/round-7001.json"
    rep = load_mkt_report(old.objs[key])
    by = {e.hotkey: e for e in rep.entries}
    # the fixture exercises every branch the judge has
    assert by["alice"].windows and by["carol"].windows
    assert by["carol"].near_dup and by["carol"].holder and by["alice"].holder
    assert "not the base architecture" in by["dave"].fault
    assert "fine-tune forward failed identically" in by["frank"].fault
    assert by["gina"].status == INFRA_DEFERRED and "RuntimeError: registry 503" in \
        by["gina"].infra_reason
    # byte identity: the report and every sidecar, in publish order
    assert new.puts == old.puts
    for k in old.puts:
        assert new.objs[k] == old.objs[k], k
    # the reference path cut windows in the validator process; the pod judge never
    assert any(n == "target_windows" for n, _t in old_calls)
    assert new_calls == [], new_calls


def test_pod_snapshot_view_must_match_the_box(tmp_path, monkeypatch):
    """A pod whose snapshot differs from the box's is refused: the round waits."""
    import shutil

    snap = H.snapshot(tmp_path / "snap")
    v, store, sess = H.rig(tmp_path / "r", snap)
    orig = H.local_run

    def tamper(argv, timeout):
        got = orig(argv, timeout)
        if argv[0] == "scp":
            dest = argv[-1].split(":", 1)[1]
            f = next(p for p in sorted(__import__("pathlib").Path(dest).rglob("close.npy")))
            shutil.copy(f.parent / "high.npy", f)          # a changed bar file
        return got

    sess.run = tamper
    try:
        assert v._mkt_publish_round(H.item(), holders=H.holders()) is False
    finally:
        sess.close()
    assert store.puts == []


@pytest.mark.parametrize("op", ["judge_rows", "judge_near_dup", "judge_holder"])
def test_judge_ops_need_a_round(tmp_path, op):
    from cascade.validator.mkt_worker import Worker

    with pytest.raises(ValueError, match="before judge_round"):
        Worker("cpu", tmp_path).handle({"op": op}, {})
