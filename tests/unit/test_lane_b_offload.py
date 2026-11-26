"""Lane B model work runs on the EVAL POD, never on the trainer box (owner
rule 2026-11-26, ABSOLUTE: no checkpoint download, no torch model load, no CPU
forward on the 7 GB box).

Torch-free on the box: the pod is a fake (shell steps run locally against a
temp dir standing in for the pod workdir; the worker step runs
``lane_b_offload.pod_main`` in-process with the Hub pull and the forward
stubbed). The real-torch encode/decode equivalence runs in CI only
(``importorskip``)."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.trainer import expert_lane as tel
from cascade.trainer import lane_b_offload as off
from cascade.trainer.expert_lane import RoutingProbe


def _probe(seed=0, n=37, layers=3, E=16, k=4) -> RoutingProbe:
    rng = np.random.default_rng(seed)
    topk = [rng.integers(0, E, size=(n, k)).astype(np.uint8) for _ in range(layers)]
    aff = np.stack([np.bincount(t.reshape(-1), minlength=E) for t in topk]).astype(np.int64)
    return RoutingProbe(topk=topk, affinity=aff)


def _same(a: RoutingProbe, b: RoutingProbe) -> bool:
    return (len(a.topk) == len(b.topk)
            and all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
                    for x, y in zip(a.topk, b.topk, strict=True))
            and a.affinity.dtype == b.affinity.dtype
            and a.affinity.tobytes() == b.affinity.tobytes())


def _fake_forward(king_dir, contexts, cfg, target):
    """Deterministic stand-in for probe_target_routing (a function of the
    checkpoint bytes, the slice and the target)."""
    seed = int.from_bytes((Path(king_dir) / "weights.safetensors").read_bytes()[:4], "little")
    seed += sum(int(np.asarray(c, dtype=np.float64).sum() * 1000) for c in
                contexts[: cfg.affinity_windows]) + len(target)
    return _probe(seed=seed % (2 ** 32))


class FakePod:
    """ssh/scp doubles: ``host.workdir`` is a local temp dir; every shell step
    really runs (mkdir, tar, test, rm), the worker step runs pod_main here."""

    def __init__(self, tmp: Path, monkeypatch, *, fail_worker=False):
        self.workdir = tmp / "pod"
        self.workdir.mkdir()
        self.commands: list[str] = []
        self.fetched: list[str] = []
        self.fail_worker = fail_worker
        from cascade.trainer.remote import RemoteHost

        self.host = RemoteHost(name="valeval-fake", host="203.0.113.9",
                               workdir=str(self.workdir), remote_python="python")
        ckpts = tmp / "registry"

        def pod_fetch(pointer, root):
            self.fetched.append(pointer)
            d = ckpts / pointer.split("@")[-1].replace(":", "-")
            d.mkdir(parents=True, exist_ok=True)
            (d / "weights.safetensors").write_bytes(pointer.encode()[-8:])
            return d

        monkeypatch.setattr(off, "_pod_fetch", pod_fetch)
        monkeypatch.setattr(tel, "probe_target_routing", _fake_forward)

    def run(self, argv, timeout, stdin_text=""):
        assert argv[0] == "ssh"
        cmd = argv[-1]
        self.commands.append(cmd)
        if "-m cascade.trainer.lane_b_offload" in cmd:
            assert "flock -w" in cmd and "env PYTHONPATH=" in cmd
            if self.fail_worker:
                return subprocess.CompletedProcess(argv, 1, "", "CUDA error")
            words = shlex.split(cmd.split("-m cascade.trainer.lane_b_offload", 1)[1]
                                .split("&&")[0])
            op, run_dir, root = words[:3]
            off.pod_main([op, run_dir, root])
            out = (Path(run_dir) / "result.json").read_text()
            return subprocess.CompletedProcess(argv, 0, out, "")
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                              timeout=60)
        return proc

    def scp(self, host, local, remote, timeout):
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, remote)
        return subprocess.CompletedProcess([], 0, "", "")

    def runner(self, **kw):
        return off.LaneBPodRunner(lambda: self.host, run=self.run, scp=self.scp, **kw)


CTX = [np.cumsum(np.random.default_rng(i).normal(size=32)) for i in range(6)]
KING = "cascade/king@sha256:" + "b" * 64


# ── codec ────────────────────────────────────────────────────────────────────


def test_probe_codec_is_byte_exact():
    p = _probe(3)
    back = off.decode_probe(__import__("json").loads(__import__("json").dumps(
        off.encode_probe(p))))
    assert _same(p, back)
    with pytest.raises(ValueError):
        off.decode_probe({"schema": "other"})
    bad = off.encode_probe(p)
    bad["affinity"]["shape"] = [99, 99]
    with pytest.raises(ValueError):
        off.decode_probe(bad)


def test_contexts_roundtrip_dtype_exact(tmp_path):
    ctx = [np.arange(5, dtype=np.float32), np.cumsum(np.ones(7)), np.array([1e-300, 3.3])]
    off.write_contexts(tmp_path / "c.npz", ctx)
    back = off.read_contexts(tmp_path / "c.npz")
    assert [a.dtype for a in back] == [a.dtype for a in ctx]
    assert all(a.tobytes() == b.tobytes() for a, b in zip(ctx, back, strict=True))


# ── remote runner ────────────────────────────────────────────────────────────


def test_probe_runs_on_the_pod_and_decodes_identically(tmp_path, monkeypatch):
    pod = FakePod(tmp_path, monkeypatch)
    r = pod.runner()
    got = r.probe(KING, CTX, 4, "energy")
    # what the local path would compute for the same checkpoint bytes + slice
    local = _fake_forward(tmp_path / "registry" / ("sha256-" + "b" * 64),
                          CTX, SimpleNamespace(affinity_windows=4), "energy")
    assert _same(got, local)
    assert pod.fetched == [KING]                        # checkpoint fetched ON the pod
    # uuid run dir removed; code bundle cached by hash; the second call skips upload
    runs = pod.workdir / off.OFFLOAD_DIR / "runs"
    assert not any(runs.iterdir())
    sha, _ = off.code_bundle()
    assert (pod.workdir / off.OFFLOAD_DIR / "code" / sha / ".ok").is_file()
    assert (pod.workdir / off.OFFLOAD_DIR / "code" / sha / "cascade" / "trainer"
            / "lane_b_offload.py").is_file()
    n_unpack = sum("tar -xzf" in c for c in pod.commands)
    assert _same(r.probe(KING, CTX, 4, "energy"), got)
    assert sum("tar -xzf" in c for c in pod.commands) == n_unpack == 1
    # never a glob rm / foreign cache eviction on the shared pod
    for c in pod.commands:
        assert ".part-*" not in c and "_private_eval_offload" not in c


def test_lineage_read_on_the_pod(tmp_path, monkeypatch):
    pod = FakePod(tmp_path, monkeypatch)
    seen = {}

    def read_measurement(d, binding):
        seen["dir"] = d
        return 1234

    import cascade.trainer.lineage as lin

    monkeypatch.setattr(lin, "read_measurement", read_measurement)
    assert pod.runner().lineage(KING, {"pointer": KING}) == 1234
    assert pod.fetched == [KING] and "registry" in str(seen["dir"])


def test_pod_failure_and_missing_host_refuse(tmp_path, monkeypatch):
    pod = FakePod(tmp_path, monkeypatch, fail_worker=True)
    with pytest.raises(off.LaneBOffloadUnavailable):
        pod.runner().probe(KING, CTX, 4, "energy")
    for host_fn in (None, lambda: None):
        with pytest.raises(off.LaneBOffloadUnavailable):
            off.LaneBPodRunner(host_fn).probe(KING, CTX, 4, "energy")
    assert issubclass(off.LaneBOffloadUnavailable, OSError)     # transient I/O class


# ── the guard on RunnerLaneBOps ──────────────────────────────────────────────


def _ops(monkeypatch, offload=None):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    calls = []
    runner = SimpleNamespace(
        cfg=SimpleNamespace(lane_b=SimpleNamespace(affinity_windows=4)),
        _fetch_checkpoint_dir=lambda p: calls.append(p) or Path("/nonexistent"))
    return RunnerLaneBOps(runner, None, None, offload=offload), calls


def test_guard_refuses_local_work_by_default(monkeypatch):
    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)

    def boom(*a, **k):
        raise AssertionError("local torch path taken")

    monkeypatch.setattr(tel, "probe_target_routing", boom)
    ops, calls = _ops(monkeypatch)
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.fetch(KING)
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.probe_routing(KING, CTX, "energy")
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.lineage_measurement(KING, {})
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.judge(SimpleNamespace(), "1", 1)
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.merge_scores(KING, 1)
    assert calls == []                                  # nothing downloaded locally


def test_probe_goes_remote_never_local(tmp_path, monkeypatch):
    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    pod = FakePod(tmp_path, monkeypatch)
    ops, calls = _ops(monkeypatch, offload=pod.runner())
    got = ops.probe_routing(KING, CTX, "energy")
    assert isinstance(got, RoutingProbe) and pod.fetched == [KING] and calls == []


def test_local_opt_in_only_with_env(monkeypatch):
    monkeypatch.setenv(off.LOCAL_EVAL_ENV, "1")
    seen = []
    monkeypatch.setattr(tel, "probe_target_routing",
                        lambda d, c, cfg, t: seen.append(d) or _probe(1))
    ops, calls = _ops(monkeypatch)
    assert _same(ops.probe_routing(KING, CTX, "energy"), _probe(1))
    assert calls == [KING] and seen == [Path("/nonexistent")]
    for v in ("", "0", "true", "yes"):
        monkeypatch.setenv(off.LOCAL_EVAL_ENV, v)
        assert not off.local_eval_allowed()


# ── the driver: a refusing probe leaves the entry queued, uncharged ──────────


def test_driver_probe_prefers_ops_probe_routing(tmp_path, monkeypatch):
    from cascade.trainer.lane_b_driver import LaneBDriver

    monkeypatch.setattr(tel, "probe_target_routing",
                        lambda *a: (_ for _ in ()).throw(AssertionError("local")))
    cfg = SimpleNamespace(lane_b=SimpleNamespace(affinity_windows=4, domains=("energy",),
                                                 targets=("energy",)))
    seen = []

    class Ops:
        def contexts(self, d, b):
            return CTX

        def fetch(self, p):
            raise AssertionError("fetched locally")

        def probe_routing(self, p, ctx, d):
            seen.append((p, d, len(ctx)))
            return _probe(2)

    drv = LaneBDriver(cfg, Ops(), tmp_path / "s.json")
    assert _same(drv._probe(KING, KING, "energy", 1), _probe(2))
    assert _same(drv._probe(KING, KING, "energy", 1), _probe(2))    # cached
    assert seen == [(KING, "energy", 4)]


GEN = "alice/metro-gen@sha256:" + "a" * 64


class RefusingOps:
    """Torch-free driver ops whose probes refuse (no eval pod)."""

    def __init__(self, fq):
        from cascade.shared.config import parse_lane_b

        self.cfg = SimpleNamespace(lane_b=parse_lane_b(
            {"from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
             "routing_jaccard": 0.7, "min_experts": 2, "max_experts": 4}))
        self.fq, self.probes, self.dispatched = fq, 0, []
        self.miner_funded = True

    def funded_queue(self):
        return self.fq

    def hotkey_busy(self, hk):
        return False

    def commitments(self):
        from cascade.shared.expert_manifest import format_expert_commit

        return [SimpleNamespace(hotkey="alice", uid=7, commit_block=5,
                                payload=format_expert_commit("energy", GEN))]

    def contexts(self, domain, block):
        return CTX

    def fetch(self, p):
        raise AssertionError("checkpoint fetched on the box")

    def probe_routing(self, p, ctx, d):
        self.probes += 1
        raise off.LaneBOffloadUnavailable("no eval pod")

    def dispatch_leg(self, *a):
        self.dispatched.append(a)


@pytest.mark.parametrize("latched", [False, True])
def test_tick_with_refusing_probe_keeps_entry_queued_uncharged(tmp_path, monkeypatch, latched):
    from cascade.funding.queue import LaneBQueue
    from cascade.trainer.lane_b_driver import LaneBDriver

    monkeypatch.setattr(tel, "probe_target_routing",
                        lambda *a: (_ for _ in ()).throw(AssertionError("local torch")))
    fq = LaneBQueue(tmp_path / "lane_b_queue.json")
    fq.add_lane_b("alice", ("energy",), GEN, 5)
    ops = RefusingOps(fq)
    drv = LaneBDriver(ops.cfg, ops, tmp_path / "lane_b.json")
    drv.add_lineage_tokens(1000)
    prev, king = "cascade/prev@sha256:" + "c" * 64, KING
    assert drv.tick(1, 0, prev) == []
    if latched:
        drv.st.gate_opened_block = 1
        assert drv.tick(2, 1, king) == []          # probes refused: nothing dispatched
    else:
        with pytest.raises(off.LaneBOffloadUnavailable):
            drv.tick(2, 1, king)                   # closed co-gate: the tick retries
        assert drv.st.gate_opened_block == 0
    assert ops.probes >= 1 and ops.dispatched == [] and not drv.queue.open
    assert [s.hotkey for s in drv.queue.pending] == ["alice"]
    e = fq.get("alice")
    assert e.status == "queued" and not getattr(e, "attempts", 0)


# ── CI only (real torch): the pod worker equals the local probe ──────────────


def test_real_probe_local_equals_pod_worker_roundtrip(tmp_path, monkeypatch):
    """CI: a tiny real MoE king, probed by the local path and by the pod
    worker (``pod_main``, Hub pull stubbed to the same dir) → JSON → decode:
    byte-identical arrays. Never run on the trainer box (loads torch)."""
    pytest.importorskip("torch")
    import dataclasses
    import json

    from .test_moe_trainer import _C, _train

    _train(tmp_path / "king", dataclasses.replace(_C(), copula_loss_cap=0.0), budget=600)
    cfg = SimpleNamespace(affinity_windows=16)
    rng = np.random.default_rng(0)
    ctx = [np.cumsum(rng.normal(size=32)) for _ in range(20)]
    local = tel.probe_target_routing(tmp_path / "king", ctx, cfg, "energy")
    run = tmp_path / "run"
    run.mkdir()
    (run / "request.json").write_text(json.dumps(
        {"op": "probe", "pointer": KING, "target": "energy", "affinity_windows": 16}))
    off.write_contexts(run / "contexts.npz", ctx)
    (tmp_path / "root" / "code").mkdir(parents=True)
    monkeypatch.setattr(off, "_pod_fetch", lambda p, root: tmp_path / "king")
    assert off.pod_main(["probe", str(run), str(tmp_path / "root")]) == 0
    remote = off.decode_probe(json.loads((run / "result.json").read_text()))
    assert _same(local, remote)
    assert remote.select(SimpleNamespace(esft_p=0.5, min_experts=2, max_experts=6)) == \
        local.select(SimpleNamespace(esft_p=0.5, min_experts=2, max_experts=6))
