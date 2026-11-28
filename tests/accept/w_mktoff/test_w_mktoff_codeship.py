"""W-MKTOFF follow-up (2026-11-28 14:01): the MarketCast pod serves the
VALIDATOR's release worker (judge ops included), whatever its image ships.

14:01:35 the rc3 image's own worker answered ``hello`` without the judge, the
validator refused the 1 TB in-process judge and round 7488676168881524362
was never judged. The validator now ships its own ``cascade`` package to the
pod by content hash and starts the worker from it (as lane B's pods)."""

from __future__ import annotations

import threading

import pytest

from tests.accept.w_mktoff import helpers as H


def _session(tmp_path, *, started=None, make_worker=None, workdir=None):
    from cascade.trainer.remote import RemoteHost
    from cascade.validator.mkt_offload import MktSession, cache_root_of
    from cascade.validator.mkt_worker import Worker

    host = RemoteHost(name="pod", host="localhost", workdir=str(workdir or tmp_path / "pod"))
    root = cache_root_of(host)
    __import__("pathlib").Path(root).mkdir(parents=True, exist_ok=True)

    def mk():
        w = make_worker(root) if make_worker else Worker("cpu", root)
        w.models.update(H.models())
        return w

    runs: list = []

    def run(argv, timeout):
        runs.append(list(argv))
        return H.local_run(argv, timeout)

    sess = MktSession(host, run=run, popen=H.pod_popen(mk, started=started))
    return sess, host, runs


def test_the_pod_worker_runs_the_validators_shipped_release_code(tmp_path):
    from cascade.trainer.lane_b_offload import code_bundle

    started: list[str] = []
    sess, host, _runs = _session(tmp_path, started=started)
    sess.start()
    try:
        sha, _blob = code_bundle()
        code = f"{host.workdir}/_mkt_code/{sha}"
        assert len(started) == 1
        assert f"PYTHONPATH={code} " in started[0] and " -P -m cascade.validator.mkt_worker " \
            in started[0], f"the worker is not started from the shipped package: {started[0]}"
        assert sess.hello.get("code") == sha
        assert sess.has_judge, "the pod's worker serves no judge: the image's code answered"
    finally:
        sess.close()


def test_the_judge_runs_on_the_pod_with_byte_identical_reports(tmp_path, monkeypatch):
    """The 14:01 shape: the pod's IMAGE worker has no judge ops. The shipped
    release worker judges there; the box cuts no window and the report and
    sidecars are the in-process path's bytes."""
    from cascade.pool import marketcast as P
    from cascade.validator import marketcast as M

    snap = H.snapshot(tmp_path / "snap")

    def publish(sub, on_pod):
        calls = []
        if on_pod:
            for mod, name in ((P, "target_windows"), (P, "window_covariates"),
                              (M, "_covariate_rows"), (M, "compute_entry_rows")):
                orig = getattr(mod, name)

                def spy(*a, _o=orig, _n=name, **k):
                    calls.append((_n, threading.current_thread().name))
                    return _o(*a, **k)

                monkeypatch.setattr(mod, name, spy)
        v, store, sess = H.rig(tmp_path / sub, snap, on_pod=on_pod)
        try:
            judge = sess.has_judge
            assert v._mkt_publish_round(H.item(), holders=H.holders()) is True
        finally:
            sess.close()
            monkeypatch.undo()
        return store, calls, judge

    old, _c, _j = publish("ref", False)
    new, calls, judge = publish("pod", True)
    assert judge, "the pod served the image's worker (no judge ops)"
    assert calls == [], f"the box judged in-process: {calls[:4]}"
    assert new.puts == old.puts
    for k in old.puts:
        assert new.objs[k] == old.objs[k], k


def test_the_package_is_shipped_once_per_content_hash(tmp_path):
    workdir = tmp_path / "pod"
    s1, _h, runs1 = _session(tmp_path, workdir=workdir)
    s1.start()
    s1.close()
    assert any(a[0] == "scp" for a in runs1), "the release package was never shipped"
    s2, _h, runs2 = _session(tmp_path, workdir=workdir)
    s2.start()
    try:
        assert not any(a[0] == "scp" for a in runs2), "a cached package was shipped again"
        assert s2.has_judge
    finally:
        s2.close()


def test_a_worker_on_other_code_is_refused(tmp_path):
    """A worker that did not import the shipped package (another checkout
    shadowing it) never serves the round, even one answering the judge op."""
    from cascade.validator.mkt_offload import MktOffloadError
    from cascade.validator.mkt_worker import Worker

    class Shadowed(Worker):
        def __setattr__(self, k, v):
            if k == "code_id" and getattr(self, "code_id", None) is not None:
                return                       # stays the checkout's own name
            super().__setattr__(k, v)

    sess, _h, _r = _session(tmp_path, make_worker=lambda root: Shadowed("cpu", root))
    try:
        with pytest.raises(MktOffloadError, match="not this release's shipped package"):
            sess.start()
    finally:
        sess.close()
