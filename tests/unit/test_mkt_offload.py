"""The MarketCast GPU offload (cascade.validator.mkt_offload / mkt_worker):
the framed worker protocol, error classification (a pod failure is infra,
never the miner's), the request-file lease, the validator's background
publisher (a pod failure postpones the round, it never defers or charges an
entry), and — with the real Timestral base — remote forecasts identical to
local ones through a real worker subprocess."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.trainer.remote import RemoteHost
from cascade.validator.marketcast import MktForwardError, infra_like
from cascade.validator.mkt_offload import (
    MktLease,
    MktOffloadError,
    MktSession,
    RemoteForecaster,
    request_path_for,
)
from cascade.validator.mkt_worker import SCHEMA, Worker, decode_frame, encode_frame

ROOT = Path(__file__).resolve().parents[2]


class _Fake:
    quantile_levels = (0.1, 0.5, 0.9)

    def __init__(self, fail=None):
        self.fail = fail

    def forecast_batch(self, ctx, horizon, *, past_covariates=None, future_covariates=None):
        if self.fail is not None:
            raise self.fail
        last = np.asarray(ctx)[:, -1:, None]
        return np.broadcast_to(last * np.array([0.9, 1.0, 1.1]),
                               (ctx.shape[0], horizon, 3)).copy()


class _PipeProc:
    """A Popen stand-in wired to an in-thread :class:`Worker`."""

    def __init__(self, worker):
        a_r, a_w = os.pipe()          # session → worker
        b_r, b_w = os.pipe()          # worker → session
        self.stdin = os.fdopen(a_w, "wb")
        self.stdout = os.fdopen(b_r, "rb")
        self._win, self._wout = os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb")
        self.t = threading.Thread(target=self._serve, args=(worker,), daemon=True)
        self.t.start()

    def _serve(self, worker):
        try:
            worker.serve(self._win, self._wout)
        except (OSError, ValueError):
            pass                           # the session side hung up mid-frame
        finally:
            self._wout.close()

    def wait(self, timeout=None):
        self.t.join(timeout)
        return 0

    def kill(self):
        self._wout.close()


def _session(models, tmp_path=None):
    w = Worker("cpu", tmp_path or Path("."))
    w.models.update(models)
    s = MktSession(RemoteHost(name="pod", host="h"), popen=lambda *a, **k: _PipeProc(w))
    s.start()
    return s, w


def test_frames_carry_arrays_without_pickle():
    h, arr = decode_frame(encode_frame({"op": "x", "n": 3}, {"a": np.arange(4.0), "b": None})[8:])
    assert h == {"op": "x", "n": 3} and list(arr) == ["a"] and arr["a"].tolist() == [0, 1, 2, 3]


def test_remote_forecaster_matches_the_local_model():
    s, _w = _session({"m": _Fake()})
    try:
        fc = RemoteForecaster(s, "m", _Fake.quantile_levels)
        ctx = np.random.default_rng(0).normal(100, 1, size=(5, 32))
        got = fc.forecast_batch(ctx, 7)
        assert np.array_equal(got, _Fake().forecast_batch(ctx, 7))
        assert fc.forecast_quantiles(ctx[0], 7).shape == (7, 3)
    finally:
        s.close()


def test_worker_errors_come_back_classified():
    s, _w = _session({"bad": _Fake(ValueError("covariates shape")),
                      "oom": _Fake(RuntimeError("CUDA out of memory")),
                      "bug": _Fake(KeyError("x"))})
    try:
        with pytest.raises(ValueError, match="covariates"):
            RemoteForecaster(s, "bad", _Fake.quantile_levels).forecast_batch(np.ones((1, 4)), 2)
        with pytest.raises(MktOffloadError) as ei:          # infra: the pod's, never the miner's
            RemoteForecaster(s, "oom", _Fake.quantile_levels).forecast_batch(np.ones((1, 4)), 2)
        assert infra_like(ei.value)
        with pytest.raises(RuntimeError) as ei:             # a model bug: the rows code's F7 path
            RemoteForecaster(s, "bug", _Fake.quantile_levels).forecast_batch(np.ones((1, 4)), 2)
        assert not infra_like(ei.value) and not isinstance(ei.value, MktOffloadError)
        with pytest.raises(ValueError, match="unknown op"):
            s.call({"op": "nope"})
    finally:
        s.close()


def test_a_dead_session_is_infra():
    s, _w = _session({"m": _Fake()})
    s.proc.stdout.close()
    s.proc.kill()
    s2, _ = _session({"m": _Fake()})
    s2.proc._wout.close()                                   # worker side gone
    with pytest.raises(MktOffloadError):
        RemoteForecaster(s2, "m", _Fake.quantile_levels).forecast_batch(np.ones((1, 4)), 2)
    assert issubclass(MktOffloadError, ConnectionError) and not issubclass(
        MktOffloadError, MktForwardError)


def test_a_wrong_shape_reply_is_infra():
    s, _w = _session({"m": _Fake()})
    try:
        fc = RemoteForecaster(s, "m", (0.1, 0.5))            # the pod's model has 3 levels
        with pytest.raises(MktOffloadError, match="shape"):
            fc.forecast_batch(np.ones((2, 4)), 3)
    finally:
        s.close()


def test_the_worker_refuses_a_checkpoint_outside_its_root_or_with_another_sha(tmp_path):
    w = Worker("cpu", tmp_path)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="escapes"):
        w.handle({"op": "load", "model": "m", "dir": "../", "sha256": "0"}, {})
    with pytest.raises(ValueError, match="sha256"):
        w.handle({"op": "load", "model": "m", "dir": "c", "sha256": "0" * 64}, {})
    assert w.handle({"op": "hello"}, {})[0]["schema"] == SCHEMA


def test_the_lease_requests_waits_and_releases(tmp_path):
    t = [0.0]
    seen = iter([None, None, "HOST"])
    lease = MktLease(tmp_path / "mkt_hosts.toml", host_fn=lambda: next(seen),
                     clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    assert lease.acquire("r1", wait_s=60) == "HOST"
    assert request_path_for(tmp_path / "mkt_hosts.toml").is_file()
    lease.release()
    assert not lease.request.exists()
    lease2 = MktLease(tmp_path / "x.toml", host_fn=lambda: None,
                      clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    assert lease2.acquire("r2", wait_s=30, poll_s=10) is None


# ── the validator's background publisher ─────────────────────────────────────


class _Store:
    def __init__(self, have=()):
        self.have = set(have)

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.have:
            raise ObjectNotFound(key)
        return "{}"


def _publisher(tmp_path, *, host="HOST", have=(), publish=None):
    from cascade.validator.marketcast import MarketCastMixin

    events = []

    class _Lease:
        def __init__(self, path):
            events.append(("lease", str(path)))

        def acquire(self, rid, wait_s):
            events.append(("acquire", rid))
            return host

        def touch(self, rid):
            events.append(("touch", rid))

        def release(self):
            events.append(("release",))

    class _Sess:
        def __init__(self, h):
            events.append(("session", h))

        def start(self):
            events.append(("start",))

        def close(self):
            events.append(("close",))

    class V(MarketCastMixin):
        pass

    v = V()
    v.mkt_eval_hosts = tmp_path / "mkt_hosts.toml"
    v.mkt_lease_fn, v.mkt_session_fn = _Lease, _Sess
    v._mkt_store_obj = _Store(have)
    v._mkt_round_ready = lambda item: True          # the pod-free prechecks pass

    def _pub(item, *, holders=None):
        events.append(("publish", item["round_id"], "_mkt_sess" in v.__dict__))
        return True if publish is None else publish(item)

    v._mkt_publish_round = _pub
    return v, events


def test_publisher_leases_one_pod_for_every_unreported_round(tmp_path):
    from cascade.shared.marketcast_report import mkt_report_key

    v, ev = _publisher(tmp_path, have={mkt_report_key("r1")})
    assert v._mkt_publish_remote([{"round_id": "r1"}, {"round_id": "r2"}, {"round_id": "r3"}],
                                 {}) == 2
    pubs = [e for e in ev if e[0] == "publish"]
    assert pubs == [("publish", "r2", True), ("publish", "r3", True)]   # inside the session
    assert ev[-2:] == [("close",), ("release",)] and "_mkt_sess" not in v.__dict__


def test_publisher_without_a_pod_publishes_nothing_and_releases(tmp_path):
    v, ev = _publisher(tmp_path, host=None)
    assert v._mkt_publish_remote([{"round_id": "r1"}], {}) == 0
    assert not [e for e in ev if e[0] in ("publish", "session")] and ev[-1] == ("release",)


def test_nothing_due_never_requests_a_pod(tmp_path):
    from cascade.shared.marketcast_report import mkt_report_key

    v, ev = _publisher(tmp_path, have={mkt_report_key("r1")})
    assert v._mkt_publish_remote([{"round_id": "r1"}], {}) == 0 and ev == []


def test_one_background_run_at_a_time(tmp_path):
    gate = threading.Event()
    v, ev = _publisher(tmp_path, publish=lambda item: gate.wait(10))
    assert v._mkt_publish_async([{"round_id": "r1"}], {}) is True
    assert v._mkt_publish_async([{"round_id": "r2"}], {}) is False      # still running
    gate.set()
    v.__dict__["_mkt_pub_thread"].join(10)
    assert v._mkt_publish_async([], {}) is False


def test_a_pod_failure_postpones_the_round_instead_of_deferring_entries(monkeypatch):
    """_mkt_publish_round with a session whose forward raises MktOffloadError
    returns False and publishes nothing (no INFRA_DEFERRED rows)."""
    from cascade.validator import marketcast as M

    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise MktOffloadError("pod gone")

    monkeypatch.setattr(M, "compute_entry_rows", boom)

    class V(M.MarketCastMixin):
        pass

    v = V()
    put = []
    v._mkt_store_obj = SimpleNamespace(
        get_text=lambda k: (_ for _ in ()).throw(__import__(
            "cascade.shared.hippius", fromlist=["ObjectNotFound"]).ObjectNotFound(k)),
        put_text=lambda *a, **k: put.append(a), put_bytes=lambda *a, **k: put.append(a))
    v.mkt_eval_hosts = Path("x")
    v.__dict__["_mkt_sess"] = object()
    v.cfg = SimpleNamespace(marketcast=SimpleNamespace(max_infra_deferrals=8))
    v.mkt_snapshot_fn = lambda block: (SimpleNamespace(), "k", "s")
    v.mkt_forecaster_fn = lambda p: (_Fake(), ())
    entry = SimpleNamespace(miner_hotkey="hk", trained_pointer="p", target="mkt:market")
    monkeypatch.setattr(M, "_unpack", lambda c: (entry, "judge", None))
    monkeypatch.setattr(M, "entry_after", lambda *a, **k: 0.0)
    monkeypatch.setattr(M, "entry_draw", lambda *a, **k: {})
    v.mkt_block_hash_fn = lambda b: "0x" + "00" * 32
    assert v._mkt_publish_round({"round_id": "r9", "block": 1, "candidates": [{}]}) is False
    assert calls == [1] and put == []
    # without a session the offloaded path never computes locally
    del v.__dict__["_mkt_sess"]
    assert v._mkt_publish_round({"round_id": "r9", "block": 1, "candidates": [{}]}) is False
    assert calls == [1]


# ── the real thing: Timestral base through a real worker subprocess ──────────


def _local_ssh(tmp_path):
    """run/popen stand-ins that execute the "remote" command locally."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    def run(argv, timeout):
        if argv[0] == "scp":
            dest = Path(argv[-1].split(":", 1)[1])
            for a in argv[1:-1]:
                if os.path.isfile(a):
                    shutil.copy(a, dest / Path(a).name)
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.run(["bash", "-c", argv[-1]], capture_output=True, text=True,
                              timeout=timeout, env=env)

    def popen(argv, **kw):
        return subprocess.Popen(["bash", "-c", argv[-1]], env=env, **kw)

    return run, popen


def test_real_base_forecasts_identically_through_the_worker(tmp_path):
    pytest.importorskip("timestral")
    from cascade.shared.config import load_chain_config

    mc = load_chain_config(ROOT / "chain.toml").marketcast
    try:
        from cascade.eval.timestral_adapter import resolve_pinned_base

        base = resolve_pinned_base(mc.base_ref, mc.base_sha256)
    except Exception as e:  # noqa: BLE001 — no network / no cache
        pytest.skip(f"pinned base unavailable: {e}")
    from cascade.validator.mkt_offload import qualify

    run, popen = _local_ssh(tmp_path)
    host = RemoteHost(name="local", host="localhost", workdir=str(tmp_path),
                      remote_python=sys.executable)
    s = MktSession(host, run=run, popen=popen, device="cpu")
    s.start()
    try:
        rec = qualify(host, base, n=8, context=256, horizon=12, session=s)
        # thread counts differ between the processes: equal to ~1e-5 of the
        # audit's re-forecast tolerance, not bit-for-bit
        assert rec["ok"] and rec["levels_match"] and rec["worst_fraction_of_tolerance"] < 0.01
        scp = []
        s.run = lambda argv, t: (scp.append(argv[0]), run(argv, t))[1]
        s.forecaster(base, key="base")                        # cached on the pod: no upload
        assert "scp" not in scp
    finally:
        s.close()
