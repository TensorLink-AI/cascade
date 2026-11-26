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

        self.profiles: list[str] = []

        def pod_fetch(pointer, root, keep=None, **kw):
            self.fetched.append(pointer)
            self.profiles.append(kw.get("profile"))
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
    monkeypatch.setattr(off, "_pod_fetch", lambda p, root, keep=None, **kw: tmp_path / "king")
    assert off.pod_main(["probe", str(run), str(tmp_path / "root")]) == 0
    remote = off.decode_probe(json.loads((run / "result.json").read_text()))
    assert _same(local, remote)
    assert remote.select(SimpleNamespace(esft_p=0.5, min_experts=2, max_experts=6)) == \
        local.select(SimpleNamespace(esft_p=0.5, min_experts=2, max_experts=6))


# ── judge / merge_scores on the pod ──────────────────────────────────────────


def _scores(seed=0, n=5):
    from cascade.eval.scoring import WindowScore

    rng = np.random.default_rng(seed)
    out = [WindowScore(series_id=f"h64-s{i}", mase=float(rng.random()),
                       qloss_per_q=rng.random(9), abs_target=float(rng.random() * 10),
                       channel=i % 2, domain="energy", source=None if i % 2 else f"src{i}")
           for i in range(n)]
    out[0] = __import__("dataclasses").replace(out[0], mase=float("nan"))
    return out


def _same_scores(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if (x.series_id, x.channel, x.domain, x.source) != (y.series_id, y.channel, y.domain,
                                                            y.source):
            return False
        if float(x.mase).hex() != float(y.mase).hex() \
                or float(x.abs_target).hex() != float(y.abs_target).hex():
            return False
        qa, qb = np.asarray(x.qloss_per_q), np.asarray(y.qloss_per_q)
        if qa.dtype != qb.dtype or qa.tobytes() != qb.tobytes():
            return False
        if [float(q).hex() for q in x.quantile_levels] != \
                [float(q).hex() for q in y.quantile_levels]:
            return False
    return True


def test_scores_codec_is_exact_including_nan():
    s = _scores(1)
    back = off.decode_scores(__import__("json").loads(__import__("json").dumps(
        off.encode_scores(s))))
    assert _same_scores(s, back)
    with pytest.raises(ValueError):
        off.decode_scores({"schema": "x"})


def _windows():
    from cascade.eval.window import EvalWindow

    rng = np.random.default_rng(4)
    return [EvalWindow(series_id=f"w{i}", history=rng.normal(size=32), target=rng.normal(size=8),
                       metadata={"freq": "H"}) for i in range(3)]


def _fake_eval(ckpt, windows, *, num_samples, device, contract, quantile_head, quantile_batch):
    seed = int.from_bytes((Path(ckpt) / "weights.safetensors").read_bytes()[:4], "little")
    seed += int(sum(float(w.history.sum()) for w in windows) * 1e6) % 1000
    seed += num_samples + int(quantile_head) * 7 + quantile_batch + len(str(contract)) + len(device)
    return _scores(seed % (2 ** 32), n=len(windows))


def test_eval_runs_on_the_pod_and_decodes_identically(tmp_path, monkeypatch):
    import cascade.validator.evaluator as ev

    pod = FakePod(tmp_path, monkeypatch)
    seen = {}

    def fake(ckpt, windows, **kw):
        seen.update(kw, n=len(windows), ids=[w.series_id for w in windows])
        return _fake_eval(ckpt, windows, **kw)

    monkeypatch.setattr(ev, "evaluate_checkpoint", fake)
    monkeypatch.delenv(off.POD_DEVICE_ENV, raising=False)
    w = _windows()
    got = pod.runner().evaluate(KING, w, num_samples=100, contract={"c": 1},
                                quantile_head=True, quantile_batch=4)
    local = _fake_eval(tmp_path / "registry" / ("sha256-" + "b" * 64), w, num_samples=100,
                       device="cpu", contract={"c": 1}, quantile_head=True, quantile_batch=4)
    assert _same_scores(got, local)
    assert seen["device"] == "cpu" and seen["ids"] == ["w0", "w1", "w2"]
    assert seen["contract"] == {"c": 1} and seen["quantile_batch"] == 4
    assert pod.fetched == [KING]


def test_merge_check_runs_on_the_pod(tmp_path, monkeypatch):
    import cascade.validator.expert_lane as vel

    pod = FakePod(tmp_path, monkeypatch)
    calls = []

    def fake(kd, cd, sel, *, strict):
        calls.append((Path(kd).name, Path(cd).name, sel, strict))
        return None if sel == {0: [1]} else "MergeError: rows differ"

    monkeypatch.setattr(vel, "merge_check_dirs", fake)
    cand = "cascade/cand@sha256:" + "d" * 64
    r = pod.runner()
    assert r.merge_check(KING, cand, {0: [1]}, True) is None
    assert r.merge_check(KING, cand, {0: [2]}, False) == "MergeError: rows differ"
    assert calls[0] == ("sha256-" + "b" * 64, "sha256-" + "d" * 64, {0: [1]}, True)
    assert pod.fetched == [KING, cand, KING, cand]


def _lb_entry():
    from cascade.shared.config import parse_lane_b
    from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest
    from cascade.validator.expert_lane import _scored_weights_rule

    lb = parse_lane_b({"from_block": 1, "domains": ["energy"], "min_experts": 1,
                       "max_experts": 4})
    ptr = "metro-v1:trained:hippius:org/ckpt@sha256:" + "b" * 64
    e = ExpertEntry(miner_hotkey="hk", miner_uid=1, gen_ref=GEN, domain="energy",
                    selection={0: [1, 2]}, exclude={}, base_king_pointer=ptr,
                    rebased_pointer=ptr, corpus_digest="cd", train_block=5, commit_block=4,
                    gpu_name="NVIDIA H100 80GB HBM3", scored_weights=_scored_weights_rule(lb))
    m = ExpertManifest(round_id="r1", created_block=10, king_pointer=ptr, lineage_tokens=1,
                       entries=(e,))
    return lb, e, m, ptr


def test_judge_entry_merge_check_hook_replaces_the_local_fetch():
    from cascade.validator.expert_lane import judge_entry

    lb, e, m, ptr = _lb_entry()

    def no_fetch(p):
        raise AssertionError("fetched locally")

    seen = []
    j = judge_entry(e, manifest=m, king_pointer=ptr, king_scores=[],
                    evaluate=lambda p: pytest.fail("evaluated"), fetch_dir=no_fetch, cfg=lb,
                    seed="1", strict=True,
                    merge_check=lambda k, c, sel, strict: seen.append((k, c, sel, strict))
                    or "MergeError: bad rows")
    assert (j.passed, j.stage, j.reason) == (False, "merge", "MergeError: bad rows")
    assert seen == [(ptr, ptr, {0: [1, 2]}, True)]

    class Fetched(Exception):
        pass

    def marks(p):
        raise Fetched(p)

    with pytest.raises(Fetched):                 # validators (no hook): unchanged path
        judge_entry(e, manifest=m, king_pointer=ptr, king_scores=[],
                    evaluate=lambda p: pytest.fail("evaluated"), fetch_dir=marks, cfg=lb,
                    seed="1")


def _runner_ops(monkeypatch, offload):
    """RunnerLaneBOps with the config reads its judge/merge_scores make stubbed."""
    import cascade.shared.config as sc
    import cascade.shared.era as era
    import cascade.validator.expert_lane as vel
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    lb, _, _, _ = _lb_entry()
    contract = SimpleNamespace(primary_size="CONTRACT")
    cfg = SimpleNamespace(
        lane_b=lb, eval=SimpleNamespace(num_samples=50),
        training=SimpleNamespace(at_block=lambda b: contract),
        round=None, quantile_head_at=lambda b: True,
        merge_guard_at_settlement=None)
    lb_proxy = SimpleNamespace(**{**lb.__dict__, "merge_guard_at_settlement": lambda b: True})
    cfg.lane_b = lb_proxy
    monkeypatch.setattr(era, "settlement_contract_block", lambda r, b: b)
    monkeypatch.setattr(sc, "shared_ladder_quantile_batch", lambda e, b: 3)
    monkeypatch.setattr(vel, "verdict_windows", lambda cfg, src, seed, block: _windows())
    runner = SimpleNamespace(cfg=cfg, _fetch_checkpoint_dir=lambda p: pytest.fail("box fetch"))
    return RunnerLaneBOps(runner, None, None, offload=offload)


class RecordingPod:
    def __init__(self):
        self.evals, self.merges = [], []

    def evaluate(self, pointer, windows, **kw):
        self.evals.append((pointer, [w.series_id for w in windows], kw))
        return _scores(len(self.evals))

    def merge_check(self, k, c, sel, strict):
        self.merges.append((k, c, sel, strict))
        return None


def test_runner_judge_and_merge_scores_go_to_the_pod(monkeypatch):
    import cascade.validator.evaluator as ev
    import cascade.validator.expert_lane as vel

    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    monkeypatch.setattr(ev, "evaluate_checkpoint", lambda *a, **k: pytest.fail("local eval"))
    pod = RecordingPod()
    ops = _runner_ops(monkeypatch, pod)
    captured = {}

    def fake_judge_manifest(manifest, **kw):
        captured.update(kw)
        kw["evaluate"]("cand")
        kw["merge_check"]("k", "c", {0: [1]}, True)
        return ["J"]

    monkeypatch.setattr(vel, "judge_manifest", fake_judge_manifest)
    _, _, m, ptr = _lb_entry()
    assert ops.judge(m, "7", 100) == ["J"]
    assert captured["merge_check"] == pod.merge_check and captured["strict"] is True
    assert _same_scores(captured["king_scores"], _scores(1))
    assert [e[0] for e in pod.evals] == [ptr, "cand"]
    assert pod.evals[0][2] == {"num_samples": 50, "contract": "CONTRACT", "quantile_head": True,
                               "quantile_batch": 3}
    assert pod.merges == [("k", "c", {0: [1]}, True)]
    got = ops.merge_scores(ptr, 100)
    assert _same_scores(got, _scores(3)) and pod.evals[-1][1] == ["w0", "w1", "w2"]


def test_runner_judge_refuses_without_pod_before_drawing_windows(monkeypatch):
    import cascade.validator.expert_lane as vel

    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    ops = _runner_ops(monkeypatch, None)
    monkeypatch.setattr(vel, "verdict_windows", lambda *a, **k: pytest.fail("windows drawn"))
    _, _, m, ptr = _lb_entry()
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.judge(m, "7", 100)
    with pytest.raises(off.LaneBOffloadUnavailable):
        ops.merge_scores(ptr, 100)


# ── settlement rebase / era-init merges / bench staging on the pod ───────────
#
# Owner rule (ABSOLUTE): the trainer box never downloads a checkpoint nor loads
# torch. The rebase, the era init's merge checks / acceptance stages / merged
# build and the bench's pointer staging are pod steps; their pushes go pod →
# Hub. Torch-free here: the torch functions are stubbed with byte-deterministic
# stand-ins run by the SAME LaneBModel code on both sides (local and pod), so
# the bytes compare; the real-torch equivalence is CI-only (importorskip).

import hashlib  # noqa: E402
import json  # noqa: E402

BASE = "cascade/base@sha256:" + "c" * 64
LEG = "cascade/leg@sha256:" + "e" * 64
SETTINGS = {"suites": ["gift-eval", "boom", "time", "fev"], "max_series": 0,
            "data_dir": "/data/bench", "num_samples": 100, "batch_size": 512}


def _mk_ckpt(d: Path, payload: bytes, leg_rec: dict | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"arch": "moe"}
    if leg_rec is not None:
        cfg["lane_b_leg"] = leg_rec
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "model.py").write_text("# model\n", encoding="utf-8")
    (d / "forecast_wrapper.py").write_text("# fw\n", encoding="utf-8")
    (d / "weights.safetensors").write_bytes(payload)
    return d


class Registry:
    """The Hub stand-in: digest-named dirs; ``push`` records what was pushed
    (bytes + target + which side pushed) and returns a content pointer."""

    def __init__(self, tmp: Path, leg_rec: dict | None = None):
        self.root = tmp / "hub"
        self.pushes: list[dict] = []
        _mk_ckpt(self.root / ("sha256-" + "b" * 64), b"KING" * 4)
        _mk_ckpt(self.root / ("sha256-" + "c" * 64), b"BASE" * 4)
        _mk_ckpt(self.root / ("sha256-" + "e" * 64), b"LEGW" * 4, leg_rec)

    def dir_of(self, pointer: str) -> Path:
        return self.root / pointer.rsplit("@", 1)[-1].replace(":", "-")

    def push(self, local, full, hub=None, *, hf_repo=None, hf_token=None):
        files = {p.name: p.read_bytes() for p in sorted(Path(local).iterdir()) if p.is_file()}
        h = hashlib.sha256(b"".join(k.encode() + v for k, v in files.items())).hexdigest()
        dest = self.root / f"sha256-{h}"
        if not dest.exists():
            shutil.copytree(local, dest)
        self.pushes.append({"full": full, "hf_repo": hf_repo, "files": files,
                            "namespace": getattr(hub, "namespace", None)})
        return SimpleNamespace(ref=SimpleNamespace(immutable_ref=f"{full}@sha256:{h}"),
                               size_bytes=sum(map(len, files.values())))


def _fake_settle_leg(king, base, leg, sel, out, *, refuse_nonfinite=False):
    """Byte-deterministic stand-in for expert_lane.settle_leg."""
    if refuse_nonfinite and sel == {0: [9]}:
        from cascade.trainer.moe_model import MergeError

        raise MergeError("non-finite delta on MoE layer 0 expert row 9")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "model.py", "forecast_wrapper.py"):
        shutil.copyfile(Path(king) / f, out / f)
    w = hashlib.sha256(b"|".join((Path(d) / "weights.safetensors").read_bytes()
                                 for d in (king, base, leg)) + repr(sorted(
                                     (int(k), list(v)) for k, v in sel.items())).encode())
    (out / "weights.safetensors").write_bytes(w.digest() * 8)
    return out


@pytest.fixture
def pod_env(tmp_path, monkeypatch):
    """A FakePod whose Hub pull/push is the Registry; settle_leg stubbed; a
    torch-free ``moe_model`` stand-in (its MergeError) for the refusal path."""
    import sys
    import types

    import cascade.shared.hippius as hip
    from cascade.trainer import lane_b_driver

    reg = Registry(tmp_path, leg_rec={"steps": 10, "tokens": 640, "stop": "tokens"})
    pod = FakePod(tmp_path, monkeypatch)
    pod.stdins = []
    pod.scps = []
    orig_run, orig_scp = pod.run, pod.scp

    def run(argv, timeout, stdin_text=""):
        if "-m cascade.trainer.lane_b_offload" in argv[-1]:
            pod.stdins.append((argv[-1].split("-m cascade.trainer.lane_b_offload", 1)[1]
                               .split()[0], stdin_text))
        return orig_run(argv, timeout, stdin_text)

    def scp(host, local, remote, timeout):
        pod.scps.append(Path(local).name)
        return orig_scp(host, local, remote, timeout)

    pod.run, pod.scp = run, scp
    fetched_on_pod = []
    profiles_on_pod = []

    def pod_fetch(pointer, root, keep=None, **kw):
        fetched_on_pod.append(pointer)
        profiles_on_pod.append((pointer, kw.get("profile")))
        d = reg.dir_of(pointer)
        if keep is not None:
            keep.add(d)
        return d

    monkeypatch.setattr(off, "_pod_fetch", pod_fetch)
    monkeypatch.setattr(hip, "upload_dir_to_hub_or_hf", reg.push)
    monkeypatch.setattr(lane_b_driver, "settle_leg", _fake_settle_leg)
    moe = types.ModuleType("cascade.trainer.moe_model")
    moe.MergeError = type("MergeError", (ValueError,), {})
    monkeypatch.setitem(sys.modules, "cascade.trainer.moe_model", moe)
    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    return SimpleNamespace(pod=pod, reg=reg, fetched=fetched_on_pod, tmp=tmp_path,
                           profiles=profiles_on_pod)


def _push(repo):
    full = f"cascade/{repo}"
    return {"full": full, "hf_repo": f"hfns/{repo}", "registry_url": "https://r",
            "namespace": "cascade"}


def _local_model(reg, tmp):
    from cascade.trainer.lane_b_model import LaneBModel

    def upload(d, repo):
        full = _push(repo)["full"]
        up = reg.push(d, full, SimpleNamespace(namespace="cascade"), hf_repo=f"hfns/{repo}")
        from cascade.shared.manifest import format_trained_pointer

        return format_trained_pointer(up.ref.immutable_ref)

    return LaneBModel(reg.dir_of, upload)


def test_rebase_runs_on_the_pod_and_pushes_pod_to_hub(pod_env, monkeypatch):
    from cascade.trainer.lane_b_bench import content_key

    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "hub_secret_value")
    env = pod_env
    r = env.pod.runner()
    model = off.PodLaneBModel(r, SimpleNamespace(settings=SETTINGS), _push)
    res = model.rebase(king_pointer=KING, base_pointer=BASE, leg_pointer=LEG,
                       selection={0: [1, 2], 3: [4]}, out_dir=env.tmp / "never-used",
                       repo="ckpt-r1-expert-energy-u7", refuse_nonfinite=True, limits=True)
    assert env.fetched == [KING, LEG, BASE]                       # pulled ON the pod
    assert res.actuals == {"steps": 10, "tokens": 640, "stop": "tokens"}
    pod_push = env.reg.pushes[-1]
    assert pod_push["full"] == "cascade/ckpt-r1-expert-energy-u7"
    assert pod_push["hf_repo"] == "hfns/ckpt-r1-expert-energy-u7"
    # the local path (same LaneBModel code, same inputs) pushes the same bytes
    local = _local_model(env.reg, env.tmp).rebase(
        king_pointer=KING, base_pointer=BASE, leg_pointer=LEG, selection={0: [1, 2], 3: [4]},
        out_dir=env.tmp / "local-rebased", repo="ckpt-r1-expert-energy-u7",
        refuse_nonfinite=True, limits=True)
    assert env.reg.pushes[-1]["files"] == pod_push["files"]
    assert res.pointer == local.pointer and res.actuals == local.actuals
    # the bench key the pod computed == the box's old content key of those bytes
    assert isinstance(res.bench, off.PodCkpt) and res.bench.pointer == res.pointer
    assert res.bench.content_key == content_key(env.tmp / "local-rebased", SETTINGS)
    # nothing but code / request JSON ever left the box; no checkpoint left on the pod
    assert set(env.pod.scps) <= {"code.tgz", "request.json"}
    assert not any((env.pod.workdir / off.OFFLOAD_DIR / "runs").iterdir())
    assert not (env.tmp / "never-used").exists()
    # credentials over stdin only (the HF fallback token only for pushing steps)
    op, stdin = env.pod.stdins[-1]
    assert op == "rebase" and "HF_TOKEN=" in stdin and "HIPPIUS_HUB_TOKEN=" in stdin
    assert all("secret_value" not in c for c in env.pod.commands)
    env.pod.runner().probe(KING, CTX, 4, "energy")
    assert env.pod.stdins[-1][0] == "probe" and "HF_TOKEN" not in env.pod.stdins[-1][1]


def test_rebase_limits_leg_without_record_is_not_rebased(tmp_path, pod_env):
    env = pod_env
    _mk_ckpt(env.reg.dir_of(LEG), b"LEGW" * 4)                  # no lane_b_leg record
    model = off.PodLaneBModel(env.pod.runner(), None, _push)
    res = model.rebase(king_pointer=KING, base_pointer=BASE, leg_pointer=LEG,
                       selection={0: [1]}, out_dir=tmp_path / "x", repo="r",
                       refuse_nonfinite=False, limits=True)
    assert (res.actuals, res.pointer, res.bench) == ({}, None, None)
    assert env.reg.pushes == [] and BASE not in env.fetched


def test_rebase_refusal_comes_back_as_merge_refused(pod_env):
    from cascade.trainer.lane_b_model import MergeRefused

    model = off.PodLaneBModel(pod_env.pod.runner(), None, _push)
    with pytest.raises(MergeRefused, match="non-finite"):
        model.rebase(king_pointer=KING, base_pointer=BASE, leg_pointer=LEG, selection={0: [9]},
                     out_dir=pod_env.tmp / "x", repo="r", refuse_nonfinite=True, limits=False)
    assert pod_env.reg.pushes == []


def test_model_steps_refuse_without_a_pod(tmp_path):
    from cascade.trainer.lane_b_model import RebaseResult  # noqa: F401 — import is torch-free

    for host_fn in (None, lambda: None):
        model = off.PodLaneBModel(off.LaneBPodRunner(host_fn), None, _push)
        with pytest.raises(off.LaneBOffloadUnavailable):
            model.rebase(king_pointer=KING, base_pointer=BASE, leg_pointer=LEG,
                         selection={0: [1]}, out_dir=tmp_path, repo="r",
                         refuse_nonfinite=False, limits=False)
        with pytest.raises(off.LaneBOffloadUnavailable):
            model.finite(KING, [{}])
        with pytest.raises(off.LaneBOffloadUnavailable):
            model.stage(KING, [], out_dir=tmp_path, repo="r", strict=True)
        with pytest.raises(off.LaneBOffloadUnavailable):
            model.build(KING, [], out_dir=tmp_path, repo="r", strict=True, era_index=1)


def _stub_merge_math(monkeypatch):
    """Byte-deterministic stand-ins for LaneBModel's torch methods (they still
    fetch and push through the model's own fetch / upload)."""
    from cascade.trainer import lane_b_model as lbm

    def finite(self, init_pointer, merges):
        self._fetch(init_pointer)
        return ["legnan" not in str(m["rebased_pointer"]) for m in merges]

    def _write(self, init_pointer, merges, out_dir, tag):
        init = self._fetch(init_pointer)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in lbm.CODE_FILES:
            shutil.copyfile(init / f, out / f)
        parts = [(init / "weights.safetensors").read_bytes()]
        parts += [(self._fetch(m["rebased_pointer"]) / "weights.safetensors").read_bytes()
                  + json.dumps(m["selection"], sort_keys=True).encode() for m in merges]
        (out / "weights.safetensors").write_bytes(
            hashlib.sha256(tag + b"|".join(parts)).digest() * 4)
        return out

    def stage(self, init_pointer, merges, *, out_dir, repo, strict):
        if strict and len(merges) > 1 and merges[0]["selection"] == merges[-1]["selection"]:
            raise lbm.MergeRefused("lane-B merges overlap on MoE layer 0 expert row 1")
        out = _write(self, init_pointer, merges, out_dir, b"stage")
        return self._upload(out, repo), out

    def build(self, init_pointer, chosen, *, out_dir, repo, strict, era_index):
        return self._upload(_write(self, init_pointer, chosen, out_dir, b"build"), repo)

    monkeypatch.setattr(lbm.LaneBModel, "finite", finite)
    monkeypatch.setattr(lbm.LaneBModel, "stage", stage)
    monkeypatch.setattr(lbm.LaneBModel, "build", build)


def _merge(ptr, rows, rid="r1"):
    return {"rebased_pointer": ptr, "king_pointer": KING,
            "selection": {"0": list(rows)}, "round_id": rid}


def test_merge_steps_run_on_the_pod_and_match_the_local_model(pod_env, monkeypatch):
    _stub_merge_math(monkeypatch)
    env = pod_env
    r = env.pod.runner()
    m1, m2 = _merge(LEG, [1]), _merge(BASE, [2])
    assert r.merge_finite(KING, [m1, m2]) == [True, True]
    ptr, key = r.merge_stage(KING, [m1], strict=True, push=_push("ckpt-acceptance-20-0"),
                             bench_settings=SETTINGS)
    staged = env.reg.pushes[-1]
    local_ptr, local_dir = _local_model(env.reg, env.tmp).stage(
        KING, [m1], out_dir=env.tmp / "local-stage", repo="ckpt-acceptance-20-0", strict=True)
    assert ptr == local_ptr and env.reg.pushes[-1]["files"] == staged["files"]
    from cascade.trainer.lane_b_bench import content_key

    assert key == content_key(local_dir, SETTINGS)
    from cascade.trainer.lane_b_model import MergeRefused

    with pytest.raises(MergeRefused, match="overlap"):
        r.merge_stage(KING, [m1, dict(m1)], strict=True, push=_push("x"), bench_settings=None)
    built = r.merge_build(KING, [m1, m2], strict=True, push=_push("ckpt-era2-merged-init"),
                          era_index=2)
    assert built == _local_model(env.reg, env.tmp).build(
        KING, [m1, m2], out_dir=env.tmp / "local-build", repo="ckpt-era2-merged-init",
        strict=True, era_index=2)
    assert set(env.pod.scps) <= {"code.tgz", "request.json"}
    assert not any((env.pod.workdir / off.OFFLOAD_DIR / "runs").iterdir())


def test_pod_fetch_never_evicts_the_steps_own_checkpoints(tmp_path, monkeypatch):
    import cascade.shared.hippius as hip

    def fetch_from_hub(ref, dest, hub=None, **kw):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / hip.FETCH_COMPLETE_MARKER).touch()
        return Path(dest)

    monkeypatch.setattr(hip, "fetch_from_hub", fetch_from_hub)
    monkeypatch.setattr(hip, "hub_snapshot_bytes", lambda ref, **kw: 0)
    monkeypatch.setattr(off, "DISK_HEADROOM_BYTES", 0)
    root = tmp_path / "root"
    keep: set = set()
    ptrs = [f"cascade/m{i}@sha256:{i:064x}" for i in range(off.CKPT_KEEP + 3)]
    dirs = [off._pod_fetch(p, root, keep=keep) for p in ptrs]
    assert all(d.is_dir() for d in dirs) and keep == set(dirs)
    off._pod_fetch("cascade/z@sha256:" + "f" * 64, root)        # another step evicts again
    assert len([p for p in (root / "ckpt").iterdir()]) == off.CKPT_KEEP


# ── the driver end to end on the pod model (no box fetch, no box torch) ──────


class PodOps:
    """Driver ops whose model work is the pod (fetch / upload / bench_rows
    on the box are test failures)."""

    def __init__(self, model):
        self.model, self.published = model, []

    def lane_b_model(self):
        return self.model

    def fetch(self, p):
        raise AssertionError(f"checkpoint fetched on the box: {p}")

    def upload(self, d, repo):
        raise AssertionError("checkpoint uploaded from the box")

    def bench_rows(self, d):
        raise AssertionError("bench rows read from a box dir")

    def prebench(self, pointer):
        pass

    def publish_expert_manifest(self, m):
        self.published.append(m)

    def judge(self, m, round_id, block, **kw):
        return []

    def commitments(self):
        return []

    def merge_scores(self, pointer, block):
        return ["scores-of", pointer]


class RowsBench:
    """LaneBBench stand-in: rows by pointer / content key."""

    settings = SETTINGS

    def __init__(self, ready=True):
        self.asked, self.ready = [], ready

    def rows_remote(self, pointer, key=None):
        from cascade.eval.benchmarks import item_rows_from_report

        self.asked.append((pointer, key))
        if not self.ready:
            return None
        return item_rows_from_report({
            "suites": [{"suite": s, "status": "ok",
                        "rows": [{"full": f"{s}/i{i}", "CRPS": 0.4, "MASE": 1.0}
                                 for i in range(12)]} for s in SETTINGS["suites"]],
            "data_revisions": {s: "r1" for s in SETTINGS["suites"]}})


def _settle_driver(tmp_path, ops, *, require_public=True, king=KING):
    from cascade.shared.config import parse_lane_b
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.lane_b_driver import FinishedExpertLeg, LaneBDriver

    spec = {"from_block": 1, "domains": ["energy", "web"], "open_tokens": 0,
            "merge_guard_from_block": 1}
    if require_public:
        spec.update(require_public=["gift", "boom", "time", "fev"], min_public_items=10)
    cfg = SimpleNamespace(lane_b=parse_lane_b(spec))
    drv = LaneBDriver(cfg, ops, tmp_path / "lane_b.json")
    fins = []
    for hk, dom, sel in (("alice", "energy", {0: [1, 2]}), ("bob", "web", {0: [9]})):
        leg = OpenLeg(sub=LaneBSubmission(hk, 7, GEN, dom, 5), selection=sel,
                      base_king_ref=king, base_era=0)
        fins.append(asdict_(FinishedExpertLeg(leg=asdict_(leg), leg_pointer=LEG,
                                              corpus_digest="c", train_block=3)))
    drv.st.finished = fins
    drv.save()
    return drv


def asdict_(x):
    from dataclasses import asdict

    return asdict(x)


def test_settlement_rebases_on_the_pod_and_never_touches_the_box(pod_env, monkeypatch):
    from cascade.shared.manifest import format_trained_pointer

    env = pod_env
    king = format_trained_pointer(KING)
    bench = RowsBench()
    model = off.PodLaneBModel(env.pod.runner(), bench, _push)
    ops = PodOps(model)
    drv = _settle_driver(env.tmp, ops, king=king)
    drv.settle(round_id="r1", block=20, era_index=0, king_pointer=king,
               work_dir=env.tmp / "w")
    assert len(ops.published) == 1
    entries = ops.published[0].entries
    # alice rebased + pushed from the pod; bob's refused rebase dropped
    assert [e.miner_hotkey for e in entries] == ["alice"]
    assert entries[0].rebased_pointer == "metro-v1:trained:hippius:" + \
        env.reg.pushes[-1]["full"] + "@" + entries[0].rebased_pointer.rsplit("@", 1)[-1]
    assert [o["outcome"] for o in drv.st.outcomes if o.get("hotkey") == "bob"] == \
        ["merge_refused"]
    # king rows by pointer; the candidate's rows by the pod-computed content key
    assert bench.asked[0] == (king, None)
    assert bench.asked[1][0] == entries[0].rebased_pointer and bench.asked[1][1]
    assert set(env.pod.scps) <= {"code.tgz", "request.json"}


def test_settlement_without_a_pod_fails_closed_and_keeps_the_legs(tmp_path, monkeypatch):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    runner = SimpleNamespace(cfg=SimpleNamespace(), _fetch_checkpoint_dir=lambda p: pytest.fail(
        "box fetch"))
    ops = RunnerLaneBOps(runner, None, None, bench=RowsBench(), offload=None)
    drv = _settle_driver(tmp_path, ops)
    before = list(drv.st.finished)
    with pytest.raises(off.LaneBOffloadUnavailable):
        drv.settle(round_id="r1", block=20, era_index=0, king_pointer=KING,
                   work_dir=tmp_path / "w")
    assert drv.st.finished == before and "r1" not in drv.st.settled


def test_runner_ops_model_is_the_pod_by_default_local_only_with_env(monkeypatch):
    from cascade.trainer.lane_b_driver import RunnerLaneBOps
    from cascade.trainer.lane_b_model import LaneBModel

    hub = SimpleNamespace(namespace="cascade", registry_url="https://r")
    runner = SimpleNamespace(cfg=SimpleNamespace(), hub=lambda: hub,
                             _hf_ckpt_repo=lambda full: "hfns/" + full.rsplit("/", 1)[-1],
                             _fetch_checkpoint_dir=lambda p: pytest.fail("box fetch"))
    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    pod = object()
    ops = RunnerLaneBOps(runner, None, None, offload=pod)
    m = ops.lane_b_model()
    assert isinstance(m, off.PodLaneBModel) and m.runner is pod
    assert ops.push_target("ckpt-x") == {"full": "cascade/ckpt-x", "hf_repo": "hfns/ckpt-x",
                                         "registry_url": "https://r", "namespace": "cascade"}
    with pytest.raises(off.LaneBOffloadUnavailable):
        RunnerLaneBOps(runner, None, None, offload=None).lane_b_model()
    monkeypatch.setenv(off.LOCAL_EVAL_ENV, "1")
    assert isinstance(RunnerLaneBOps(runner, None, None).lane_b_model(), LaneBModel)


def _pending_driver(tmp_path, ops, *, acceptance):
    from cascade.shared.config import parse_lane_b
    from cascade.trainer.lane_b_driver import LaneBDriver

    spec = {"from_block": 1, "domains": ["energy", "web"]}
    if acceptance:
        spec["acceptance_from_block"] = 1
    return LaneBDriver(SimpleNamespace(lane_b=parse_lane_b(spec)), ops, tmp_path / "d.json")


def test_era_init_merges_on_the_pod_legacy_path(pod_env, monkeypatch):
    _stub_merge_math(monkeypatch)
    env = pod_env
    model = off.PodLaneBModel(env.pod.runner(), None, _push)
    ops = PodOps(model)
    drv = _pending_driver(env.tmp, ops, acceptance=False)
    bad = _merge("cascade/legnan@sha256:" + "d" * 64, [5])
    _mk_ckpt(env.reg.dir_of(bad["rebased_pointer"]), b"NANS")
    good, overlap = _merge(LEG, [1]), _merge(BASE, [1])
    drv.st.pending_merges = [good, overlap, bad]
    ptr, merges = drv.init_for(2, KING, env.tmp / "w", era_start=20)
    assert merges == [good]
    assert env.reg.pushes[-1]["full"] == "cascade/ckpt-era2-merged-init"
    assert ptr.endswith(ptr.rsplit("@", 1)[-1]) and "ckpt-era2-merged-init" in ptr
    assert drv.st.pending_merges == [overlap]          # the overlap waits; NaN dropped
    assert drv.init_for(2, KING, env.tmp / "w", era_start=20) == (ptr, merges)


def test_era_init_acceptance_stages_on_the_pod(pod_env, monkeypatch):
    import cascade.eval.public_guards as pg
    import cascade.trainer.merge_acceptance as tma

    _stub_merge_math(monkeypatch)
    monkeypatch.setattr(pg, "pair_public_rows", lambda *a, **k: None)
    judged = []

    def judge_acceptance(qualified, **kw):
        judged.append(kw)
        return list(qualified), {"ok": True}

    monkeypatch.setattr(tma, "judge_acceptance", judge_acceptance)
    env = pod_env
    bench = RowsBench()
    model = off.PodLaneBModel(env.pod.runner(), bench, _push)
    ops = PodOps(model)
    drv = _pending_driver(env.tmp, ops, acceptance=True)
    win = {"hotkey": "alice", "uid": 7, "domain": "energy", "gen_ref": GEN, "commit_block": 5,
           "round_id": "r1", "merge_block": 10, "rebased_pointer": LEG,
           "king_pointer": KING, "selection": {"0": [1]}}
    from cascade.shared.lane_b import DomainWin

    fields = set(DomainWin.__dataclass_fields__)
    win = {k: v for k, v in win.items() if k in fields}
    m = {**_merge(LEG, [1]), "qualified_wins": [win]}
    drv.st.pending_merges = [m]
    ptr, merges = drv.init_for(2, KING, env.tmp / "w", era_start=20)
    assert [x["rebased_pointer"] for x in merges] == [LEG]
    stage_ptr = merges[0]["acceptance"]["candidate"]
    assert "ckpt-acceptance-20-0" in stage_ptr and "ckpt-era2-merged-init" in ptr
    assert judged[0]["candidate_scores"] == ["scores-of", stage_ptr]
    assert bench.asked[0] == (KING, None)                  # anchor rows by pointer
    assert bench.asked[1][0] == stage_ptr and bench.asked[1][1]     # stage by content key
    assert set(env.pod.scps) <= {"code.tgz", "request.json"}


# ── bench: pointer benches stage on the pod, under the guard ─────────────────


def _lb_bench(tmp_path, **kw):
    from cascade.trainer.lane_b_bench import LaneBBench

    ecfg = SimpleNamespace(cascade_bench_max_series=0, gift_gate_data_dir="/data/bench",
                           bench_workers="auto")
    return LaneBBench(hosts_path=tmp_path / "targon-eval.toml", cache_dir=tmp_path / "rows",
                      eval_cfg=ecfg, start_worker=False, **kw)


def _bench_report(status="ok"):
    return {"suites": [{"suite": s, "status": status,
                        "rows": [{"full": f"{s}/i{i}", "CRPS": 0.4, "MASE": 1.0}
                                 for i in range(12)]} for s in SETTINGS["suites"]],
            "data_revisions": {s: "r1" for s in SETTINGS["suites"]}}


def _drain(lb):
    while not lb._q.empty():
        tag, item = lb._q.get()
        lb.run_one(item)
        lb._pending.discard(tag)


def test_prebench_without_pod_is_refused_never_fetched(tmp_path, monkeypatch):
    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    fetched = []
    lb = _lb_bench(tmp_path, fetch=lambda p: fetched.append(p) or tmp_path,
                   host_fn=lambda: "pod", bench_fn=lambda *a, **k: pytest.fail("benched"))
    lb.prebench(KING)
    tag, item = lb._q.get()
    with pytest.raises(off.LaneBOffloadUnavailable):
        lb.run_one(item)
    assert fetched == []


def test_pointer_bench_stages_on_the_pod_and_sweeps_in_place(tmp_path, pod_env):
    env = pod_env
    r = env.pod.runner()
    swept = []

    def pinned(host, pin, **kw):
        assert sorted(p.name for p in Path(pin).iterdir()) == [
            "config.json", "forecast_wrapper.py", "model.py", "weights.safetensors"]
        swept.append((host.name, pin, kw))
        return _bench_report()

    lb = _lb_bench(tmp_path, fetch=lambda p: pytest.fail("box fetch"), pod=r,
                   pinned_bench_fn=pinned)
    assert lb.rows_remote(KING) is None                   # queued, never inline
    _drain(lb)
    assert len(swept) == 1 and swept[0][2]["settings"] == lb.settings
    assert not Path(swept[0][1]).exists()                 # pin removed after the sweep
    from cascade.trainer.lane_b_bench import content_key

    key = content_key(env.reg.dir_of(KING), lb.settings)
    assert lb.cached(key) is not None and lb.rows_remote(KING) is not None
    # a restarted trainer: the persisted pointer → key map, no re-stage
    lb2 = _lb_bench(tmp_path, pod=SimpleNamespace(
        bench_stage=lambda *a: pytest.fail("re-staged")))
    assert lb2.rows_remote(KING) is not None
    lb2.prebench(KING)
    assert lb2._q.empty()
    # a pod-computed key (a rebased candidate) hits the cache without a stage
    assert lb2.rows_remote("cascade/other@sha256:" + "9" * 64, key=key) is not None
    assert set(env.pod.scps) <= {"code.tgz", "request.json"}


def test_bench_pinned_runs_the_upload_paths_command_on_the_pin(tmp_path):
    from cascade.trainer.lane_b_bench import bench_pinned
    from cascade.trainer.remote import RemoteHost
    from cascade.validator.eval_offload import build_bench_remote_command

    host = RemoteHost(name="valeval", host="203.0.113.9", workdir="/w", remote_python="python")
    cmds = []

    def run(argv, timeout, stdin=""):
        cmd = argv[-1]
        cmds.append(cmd)
        if cmd.startswith("cat "):
            return subprocess.CompletedProcess(argv, 0, json.dumps(_bench_report()), "")
        return subprocess.CompletedProcess(argv, 0, "CASCADE_FREE=999999999999\n", "")

    pin = "/w/_lane_b_offload/pinned/abc"
    rep = bench_pinned(host, pin, settings=SETTINGS, workers="auto", timeout_s=9000,
                       label="laneb-x", run=run)
    assert rep == _bench_report()
    sweep = [c for c in cmds if "cascade-benchmark" in c]
    assert len(sweep) == 1
    report = shlex.split(sweep[0].split("cascade-benchmark", 1)[1])[1]
    assert sweep[0] == build_bench_remote_command(
        host, pin, report, suites="gift-eval,boom,time,fev", num_samples=100, batch_size=512,
        max_series=0, data_dir="/data/bench", device="cuda", workers="auto",
        timeout_s=9000 - 15)
    assert cmds[-1].startswith("rm -rf ") and "pinned" not in cmds[-1]


def test_bench_stage_pins_hardlinks_and_matches_the_box_key(pod_env):
    from cascade.trainer.lane_b_bench import content_key

    env = pod_env
    r = env.pod.runner()
    host, key, pin = r.bench_stage(KING, SETTINGS)
    src = env.reg.dir_of(KING)
    assert key == content_key(src, SETTINGS)
    assert (Path(pin) / "weights.safetensors").read_bytes() == \
        (src / "weights.safetensors").read_bytes()
    assert (Path(pin) / "weights.safetensors").stat().st_ino == \
        (src / "weights.safetensors").stat().st_ino            # a link, not a copy
    r.unpin(host, pin)
    assert not Path(pin).exists()


# ── CI only (real torch): the pod worker's bytes equal the local model's ─────


def test_real_rebase_and_build_pod_worker_equal_local(tmp_path, monkeypatch):
    """CI: a tiny real MoE king; the rebase and the merged-init build run by
    the pod worker (``pod_main``; Hub pull/push stubbed to dirs) push the
    same bytes as the local LaneBModel (JSON round trip of the selection and
    merge records included). Never run on the trainer box (loads torch)."""
    pytest.importorskip("torch")
    import dataclasses

    import cascade.shared.hippius as hip

    from .test_moe_trainer import _C, _train

    _train(tmp_path / "king", dataclasses.replace(_C(), copula_loss_cap=0.0), budget=600)
    reg = Registry(tmp_path)
    for name in ("b", "c", "e"):
        d = reg.root / ("sha256-" + name * 64)
        shutil.rmtree(d)
        shutil.copytree(tmp_path / "king", d)
    pod = FakePod(tmp_path, monkeypatch)
    monkeypatch.setattr(off, "_pod_fetch", lambda p, root, keep=None, **kw: reg.dir_of(p))
    monkeypatch.setattr(hip, "upload_dir_to_hub_or_hf", reg.push)
    model = off.PodLaneBModel(pod.runner(), None, _push)
    sel = {0: [1, 2]}
    res = model.rebase(king_pointer=KING, base_pointer=BASE, leg_pointer=LEG, selection=sel,
                       out_dir=tmp_path / "x", repo="rb", refuse_nonfinite=True, limits=False)
    pushed = reg.pushes[-1]["files"]
    local = _local_model(reg, tmp_path).rebase(
        king_pointer=KING, base_pointer=BASE, leg_pointer=LEG, selection=sel,
        out_dir=tmp_path / "lr", repo="rb", refuse_nonfinite=True, limits=False)
    assert reg.pushes[-1]["files"] == pushed and res.pointer == local.pointer
    merge = {"rebased_pointer": LEG,
             "king_pointer": KING, "selection": {"0": [1, 2]}, "round_id": "r1"}
    assert model.finite(KING, [merge]) == _local_model(reg, tmp_path).finite(KING, [merge])
    built = model.build(KING, [merge], out_dir=tmp_path / "y", repo="mi", strict=True,
                        era_index=2)
    pod_files = reg.pushes[-1]["files"]
    assert built == _local_model(reg, tmp_path).build(
        KING, [merge], out_dir=tmp_path / "ly", repo="mi", strict=True, era_index=2)
    assert reg.pushes[-1]["files"] == pod_files


_TORCH_FREE_SCRIPT = r'''
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "safetensors") or name == "cascade.trainer.moe_model":
            raise ModuleNotFoundError(f"blocked: {name}", name=name)
sys.meta_path.insert(0, _Block())
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from cascade.shared.config import parse_lane_b
from cascade.shared.manifest import format_trained_pointer
from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
from cascade.trainer.lane_b_driver import FinishedExpertLeg, LaneBDriver
from cascade.trainer.lane_b_model import MergeRefused, RebaseResult
from cascade.trainer.lane_b_offload import PodCkpt

tmp = Path(sys.argv[1])
KING = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
GEN = "alice/metro-gen@sha256:" + "a" * 64

class Model:
    def rows(self, p): return None
    def bench(self, h): return None
    def rebase(self, *, selection, **kw):
        if selection == {0: [9]}:
            raise MergeRefused("refused")
        p = format_trained_pointer("cascade/rb@sha256:" + "d" * 64)
        return RebaseResult({}, p, PodCkpt(p, "k" * 64))
    def finite(self, init, merges): return [True] * len(merges)
    def build(self, init, chosen, **kw):
        return format_trained_pointer("cascade/mi@sha256:" + "f" * 64)

class Ops:
    published = []
    def lane_b_model(self): return Model()
    def fetch(self, p): raise AssertionError("box fetch")
    def publish_expert_manifest(self, m): self.published.append(m)
    def judge(self, m, rid, block, **kw): return []
    def commitments(self): return []

cfg = SimpleNamespace(lane_b=parse_lane_b({"from_block": 1, "domains": ["energy", "web"],
                                           "open_tokens": 0}))
ops = Ops()
drv = LaneBDriver(cfg, ops, tmp / "d.json")
for hk, dom, sel in (("alice", "energy", {0: [1]}), ("bob", "web", {0: [9]})):
    leg = OpenLeg(sub=LaneBSubmission(hk, 7, GEN, dom, 5), selection=sel,
                  base_king_ref=KING, base_era=0)
    drv.st.finished.append(asdict(FinishedExpertLeg(leg=asdict(leg), leg_pointer=KING,
                                                     corpus_digest="c", train_block=3)))
drv.settle(round_id="r1", block=20, era_index=0, king_pointer=KING, work_dir=tmp / "w")
assert len(ops.published) == 1 and len(ops.published[0].entries) == 1
drv.st.pending_merges = [{"rebased_pointer": KING, "king_pointer": KING,
                          "selection": {"0": [1]}, "round_id": "r1"}]
ptr, merges = drv.init_for(1, KING, tmp / "w", era_start=20)
assert "mi@" in ptr and len(merges) == 1
assert not any(m.split(".")[0] in ("torch", "safetensors") for m in sys.modules)
print("TORCH_FREE_OK")
'''


def test_settlement_and_era_init_are_torch_free_on_the_box(tmp_path):
    """In a fresh interpreter with torch / safetensors / moe_model import-
    blocked, a settlement (one rebase, one refused) and a legacy era init run
    through a pod model — the box path imports none of them."""
    import sys

    proc = subprocess.run([sys.executable, "-P", "-c", _TORCH_FREE_SCRIPT, str(tmp_path)],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(Path(__file__).resolve().parents[2]),
                          env={**__import__("os").environ,
                               "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
    assert "TORCH_FREE_OK" in proc.stdout, proc.stderr[-3000:]


# ── deployment: a trainer restart between a leg's finish and its settlement ──


def test_restart_between_finish_and_settle_settles_once_without_redispatch(tmp_path):
    """The fix ships by a trainer restart AFTER the first lane B leg returned
    and BEFORE it settled. The finished leg (its trained pointer) is on disk
    in ``lane_b_driver.json``; the restarted process (new driver AND a new
    funding-queue object over the same files) neither requeues, re-dispatches
    nor re-charges it, and settles it exactly once."""
    from tests.accept.w_s2 import helpers as ws2
    from tests.accept.wb18 import helpers as wb18

    sc = ws2.trainer_scenario(tmp_path, sentinel_from_block=0, require_public=())
    hk = wb18.HK_E
    finished = json.loads(json.dumps(sc.driver.st.finished))     # its on-disk form
    assert [f["leg"]["sub"]["hotkey"] for f in finished] == [hk]
    entry = sc.fq.get(hk)
    assert entry.status == "in_flight"
    attempt, n_dispatched = entry.attempt_id, len(sc.ops.dispatched)
    on_disk = json.loads((tmp_path / "lane_b_driver.json").read_text())
    assert on_disk["finished"] == finished and finished[0]["leg_pointer"]
    assert json.loads(on_disk["queue"])["open"] == []          # nothing left "open"

    sc.restart_all()                                           # nothing in memory survives
    assert json.loads(json.dumps(sc.driver.st.finished)) == finished
    for _ in range(3):
        assert sc.tick(1, ws2.TRAINER_KING) == []
    assert len(sc.ops.dispatched) == n_dispatched              # never re-dispatched
    e = sc.fq.get(hk)
    assert (e.status, e.attempt_id) == ("in_flight", attempt)  # never requeued / re-charged
    assert json.loads(json.dumps(sc.driver.st.finished)) == finished

    ws2.settle(sc, tmp_path)
    assert len(sc.ops.published) == 1
    assert [x.miner_hotkey for x in sc.ops.published[0].entries] == [hk]
    assert sc.driver.st.finished == [] and str(ws2.SETTLE_BLOCK) in sc.driver.st.settled
    assert sc.fq.get(hk).status != "in_flight"                 # spent by the settlement
    ws2.settle(sc, tmp_path)                                   # idempotent
    sc.restart_all()
    ws2.settle(sc, tmp_path)
    assert len(sc.ops.published) == 1
    assert len(sc.ops.dispatched) == n_dispatched


# ── own model hosts, cache root, disk guard, selective fetch (2026-11-26) ────
# The 14 GB king (optimizer.safetensors 4.97 GB) filled the shared 43 GB eval
# pod to 98%; lane B settlement pulls king + base + leg there.


def _hosts_toml(path: Path, name: str, workdir: Path, cache_root: str | None = None) -> Path:
    extra = f'cache_root = "{cache_root}"\n' if cache_root is not None else ""
    path.write_text(f'[[host]]\nname = "{name}"\nhost = "203.0.113.7"\nstage = "final"\n'
                    f'workdir = "{workdir}"\n{extra}', encoding="utf-8")
    return path


def test_model_hosts_flag_routes_model_work_bench_stays_on_bench_hosts(tmp_path):
    from cascade.trainer import main as tm

    bench = _hosts_toml(tmp_path / "bench.toml", "valeval-bench", tmp_path / "b")
    model = _hosts_toml(tmp_path / "model.toml", "lium-volume", tmp_path / "m", "/workspace")
    args = tm._build_parser().parse_args(["--lane-b-bench-hosts", str(bench),
                                          "--lane-b-model-hosts", str(model),
                                          "--lane-b-pod-headroom-gb", "12.5"])
    r = tm._lane_b_offload(args)
    assert r.host().name == "lium-volume" and r.source == "--lane-b-model-hosts"
    assert r.headroom_bytes == 12_500_000_000
    assert off.LaneBPodRunner.root_of(r.host()) == "/workspace/" + off.OFFLOAD_DIR
    # bench staging + its sidecar stay on the bench hosts (the pin is pod-local)
    cfg = SimpleNamespace(lane_b=SimpleNamespace(require_public=True),
                          eval=SimpleNamespace(cascade_bench_max_series=0,
                                               gift_gate_data_dir="", bench_workers="1"))
    args.work_root = tmp_path / "work"
    lb = tm._lane_b_bench(args, cfg, SimpleNamespace(_fetch_checkpoint_dir=None))
    assert lb._pod.host().name == "valeval-bench" and lb._pod.headroom_bytes == 12_500_000_000
    # unset ⇒ the bench hosts, as before; default headroom 10 GB
    args = tm._build_parser().parse_args(["--lane-b-bench-hosts", str(bench)])
    r = tm._lane_b_offload(args)
    assert r.host().name == "valeval-bench" and r.source == "--lane-b-bench-hosts"
    assert r.headroom_bytes == 10 * 10**9
    assert off.LaneBPodRunner.root_of(r.host()) == f"{tmp_path / 'b'}/{off.OFFLOAD_DIR}"
    assert tm._lane_b_offload(tm._build_parser().parse_args([])) is None


def test_cache_root_holds_the_offload_dir(tmp_path, monkeypatch):
    import dataclasses

    from cascade.trainer.remote import load_hosts

    vol = tmp_path / "volume"
    h = load_hosts(_hosts_toml(tmp_path / "h.toml", "x", tmp_path, str(vol)))[0]
    assert h.cache_root == str(vol)
    assert load_hosts(_hosts_toml(tmp_path / "g.toml", "y", tmp_path))[0].cache_root == ""
    pod = FakePod(tmp_path, monkeypatch)
    pod.host = dataclasses.replace(pod.host, cache_root=str(vol))
    pod.runner().probe(KING, CTX, 4, "energy")
    assert (vol / off.OFFLOAD_DIR / "code").is_dir()
    assert not (pod.workdir / off.OFFLOAD_DIR).exists()
    worker = [c for c in pod.commands if "-m cascade.trainer.lane_b_offload" in c][-1]
    assert f"{vol}/{off.OFFLOAD_DIR}" in worker and f"cd {pod.workdir}" in worker


# a checkpoint with its lineage files (the king's 14 GB shape, scaled down)
_CKPT_FILES = {"config.json": 10, "model.py": 10, "forecast_wrapper.py": 10,
               "training_measurement.json": 5, "weights.safetensors": 100,
               "weights_stable.safetensors": 100, "optimizer.safetensors": 300,
               "power_ema.safetensors": 50}


class FilteringHub:
    """``fetch_from_hub`` / ``hub_snapshot_bytes`` over digest-named dirs,
    honouring allow/ignore patterns exactly like the registry pull."""

    def __init__(self, tmp: Path):
        self.root = tmp / "filtering-hub"
        self.pulls: list[tuple[str, list[str]]] = []

    def add(self, pointer: str, files: dict | None = None, seed: bytes = b"") -> str:
        d = self.root / pointer.rsplit("@", 1)[-1].replace(":", "-")
        d.mkdir(parents=True, exist_ok=True)
        for name, n in (files or _CKPT_FILES).items():
            if name == "config.json":
                (d / name).write_text(json.dumps({"arch": "moe", "pad": "x" * n}))
            else:
                (d / name).write_bytes((seed + name.encode()) * n)
        return pointer

    def _names(self, ref, allow, ignore):
        import fnmatch

        import cascade.shared.hippius as hip

        allow, ignore = hip._fetch_patterns(allow, ignore)
        d = self.root / str(ref).rsplit("@", 1)[-1].replace(":", "-")
        return d, [p.name for p in sorted(d.iterdir())
                   if any(fnmatch.fnmatchcase(p.name, a) for a in allow)
                   and not any(fnmatch.fnmatchcase(p.name, i) for i in ignore)]

    def fetch(self, ref, dest, hub=None, *, allow_patterns=None, ignore_patterns=None):
        import cascade.shared.hippius as hip

        src, names = self._names(ref, allow_patterns, ignore_patterns)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copyfile(src / n, dest / n)
        (dest / hip.FETCH_COMPLETE_MARKER).touch()
        self.pulls.append((str(ref), names))
        return dest

    def size(self, ref, *, allow_patterns=None, ignore_patterns=None):
        src, names = self._names(ref, allow_patterns, ignore_patterns)
        return sum((src / n).stat().st_size for n in names)


def _du(disk: Path, cap: int):
    def used() -> int:
        return sum(p.stat().st_size for p in disk.rglob("*") if p.is_file())

    return lambda path: SimpleNamespace(free=cap - used())


def _entry(ckpt: Path, name: str, nbytes: int, age: float) -> Path:
    import os
    import time

    d = ckpt / name
    d.mkdir(parents=True)
    (d / "weights.safetensors").write_bytes(b"x" * nbytes)
    t = time.time() - age
    os.utime(d, (t, t))
    return d


def test_disk_guard_evicts_lru_unneeded_entries_only(tmp_path, monkeypatch):
    import cascade.shared.hippius as hip

    hub = FilteringHub(tmp_path)
    monkeypatch.setattr(hip, "fetch_from_hub", hub.fetch)
    disk = tmp_path / "disk"
    root = disk / off.OFFLOAD_DIR
    ckpt = root / "ckpt"
    oldest = _entry(ckpt, "sha256-" + "1" * 64, 400, age=300)
    older = _entry(ckpt, "sha256-" + "2" * 64, 400, age=200)
    mine = _entry(ckpt, "sha256-" + "3" * 64, 400, age=900)     # this step holds it
    newest = _entry(ckpt, "sha256-" + "4" * 64, 400, age=10)
    stale_tmp = _entry(ckpt, ".sha256-x.fetch-1-ab", 50, age=3 * 3600)
    validator = disk / "_eval_offload" / "cache" / "k"           # the validator's cache
    validator.mkdir(parents=True)
    (validator / "w").write_bytes(b"v" * 400)
    pin = root / "pinned" / "p1"
    pin.mkdir(parents=True)
    (pin / "w").write_bytes(b"p" * 100)
    ptr = hub.add("cascade/leg@sha256:" + "e" * 64)
    need = hub.size(ptr, ignore_patterns=list(off.LINEAGE_FILES))     # weights profile
    # room for the pull + 50 B headroom only after evicting the stale temp + 2 LRU
    cap = sum(p.stat().st_size for p in disk.rglob("*") if p.is_file()) - 850 + need + 50
    keep = {mine}
    d = off._pod_fetch(ptr, root, keep=keep, profile=off.FETCH_WEIGHTS,
                       disk={"headroom_bytes": 50}, size_fn=hub.size, disk_usage=_du(disk, cap))
    assert d.name.endswith("~weights") and mine in keep and d in keep
    assert not stale_tmp.exists() and not oldest.exists() and not older.exists()
    assert newest.exists() and mine.exists()                   # LRU stopped once it fit
    assert (validator / "w").exists() and (pin / "w").exists()  # never outside ckpt/
    assert not (d / "optimizer.safetensors").exists()


def test_disk_guard_refuses_when_still_short_and_never_touches_outside(tmp_path, monkeypatch):
    import cascade.shared.hippius as hip

    hub = FilteringHub(tmp_path)
    monkeypatch.setattr(hip, "fetch_from_hub", hub.fetch)
    disk = tmp_path / "disk"
    root = disk / off.OFFLOAD_DIR
    mine = _entry(root / "ckpt", "sha256-" + "3" * 64, 400, age=900)
    other = _entry(root / "ckpt", "sha256-" + "5" * 64, 100, age=900)
    outside = disk / "uv-cache" / "big"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"u" * 5000)
    ptr = hub.add("cascade/king@sha256:" + "b" * 64)
    with pytest.raises(off.PodDiskShort, match="pod disk short"):
        off._pod_fetch(ptr, root, keep={mine}, profile=off.FETCH_FULL,
                       disk={"headroom_bytes": 10**6}, size_fn=hub.size,
                       disk_usage=_du(disk, 6000))
    assert mine.exists() and outside.exists() and not other.exists()
    assert hub.pulls == []                                     # nothing pulled
    # unknown size ⇒ the configured fallback guards the pull
    with pytest.raises(off.PodDiskShort):
        off._pod_fetch(ptr, root, profile=off.FETCH_META,
                       disk={"headroom_bytes": 0, "size_fallback_bytes": 10**9},
                       size_fn=lambda ref, **kw: None, disk_usage=_du(disk, 6000))
    assert hub.pulls == [] and outside.exists()


def test_disk_short_on_the_pod_is_a_transient_refusal(tmp_path, monkeypatch, caplog):
    import cascade.shared.hippius as hip

    pod = FakePod(tmp_path, monkeypatch)
    monkeypatch.undo()                     # the real _pod_fetch, a real pull path
    monkeypatch.setattr(tel, "probe_target_routing", _fake_forward)
    hub = FilteringHub(tmp_path)
    hub.add(KING)
    monkeypatch.setattr(hip, "fetch_from_hub", hub.fetch)
    monkeypatch.setattr(hip, "hub_snapshot_bytes", lambda ref, **kw: 10**15)   # 1 PB
    with caplog.at_level("ERROR"), pytest.raises(off.LaneBOffloadUnavailable,
                                                 match="pod disk short"):
        pod.runner(headroom_bytes=0).probe(KING, CTX, 4, "energy")
    assert hub.pulls == [] and "pod disk short" in caplog.text
    assert not any((pod.workdir / off.OFFLOAD_DIR / "runs").iterdir())     # cleaned up


def _real_fetch_pod(tmp_path, monkeypatch):
    """A FakePod over the REAL _pod_fetch and a FilteringHub (the selective
    fetch end to end), Hub push = the Registry, settle_leg stubbed."""
    import sys
    import types

    import cascade.shared.hippius as hip
    from cascade.trainer import lane_b_driver

    pod = FakePod(tmp_path, monkeypatch)
    monkeypatch.undo()
    monkeypatch.setattr(tel, "probe_target_routing", _fake_forward)
    hub = FilteringHub(tmp_path)
    for ptr, seed in ((KING, b"K"), (BASE, b"B"), (LEG, b"L")):
        hub.add(ptr, seed=seed)
    reg = Registry(tmp_path)
    for ptr in (KING, BASE, LEG):          # the Registry's copies = the full checkpoints
        shutil.rmtree(reg.dir_of(ptr))
        shutil.copytree(hub.root / ptr.rsplit("@", 1)[-1].replace(":", "-"), reg.dir_of(ptr))
    monkeypatch.setattr(hip, "fetch_from_hub", hub.fetch)
    monkeypatch.setattr(hip, "hub_snapshot_bytes", hub.size)
    monkeypatch.setattr(hip, "upload_dir_to_hub_or_hf", reg.push)
    monkeypatch.setattr(lane_b_driver, "settle_leg", _fake_settle_leg)
    moe = types.ModuleType("cascade.trainer.moe_model")
    moe.MergeError = type("MergeError", (ValueError,), {})
    monkeypatch.setitem(sys.modules, "cascade.trainer.moe_model", moe)
    monkeypatch.delenv(off.LOCAL_EVAL_ENV, raising=False)
    return pod, hub, reg


def _pulled(hub, ptr):
    return [names for r, names in hub.pulls if r.endswith(ptr.rsplit("@", 1)[-1])]


def test_selective_fetch_rebase_skips_lineage_files_same_bytes_and_key(tmp_path, monkeypatch):
    from cascade.trainer.lane_b_bench import content_key

    pod, hub, reg = _real_fetch_pod(tmp_path, monkeypatch)
    r = pod.runner(headroom_bytes=0)
    model = off.PodLaneBModel(r, SimpleNamespace(settings=SETTINGS), _push)
    res = model.rebase(king_pointer=KING, base_pointer=BASE, leg_pointer=LEG,
                       selection={0: [1]}, out_dir=tmp_path / "x", repo="rb",
                       refuse_nonfinite=True, limits=False)
    for ptr in (KING, BASE, LEG):
        (names,) = _pulled(hub, ptr)
        assert not set(names) & set(off.LINEAGE_FILES) and "weights.safetensors" in names
    pushed = reg.pushes[-1]["files"]
    local = _local_model(reg, tmp_path).rebase(       # the full checkpoints, box path
        king_pointer=KING, base_pointer=BASE, leg_pointer=LEG, selection={0: [1]},
        out_dir=tmp_path / "lr", repo="rb", refuse_nonfinite=True, limits=False)
    assert reg.pushes[-1]["files"] == pushed and res.pointer == local.pointer
    assert res.bench.content_key == content_key(tmp_path / "lr", SETTINGS)
    # the bench stage of a full checkpoint: the same key from the weights-only copy
    host, key, pin = r.bench_stage(KING, SETTINGS)
    assert key == content_key(reg.dir_of(KING), SETTINGS)
    r.unpin(host, pin)
    assert len(_pulled(hub, KING)) == 1                       # cached, not pulled again


def test_selective_fetch_profiles_per_op(tmp_path, monkeypatch):
    import cascade.trainer.lineage as lin
    import cascade.validator.expert_lane as vel

    pod, hub, reg = _real_fetch_pod(tmp_path, monkeypatch)
    r = pod.runner(headroom_bytes=0)
    monkeypatch.setattr(lin, "read_measurement", lambda d, b: sorted(
        p.name for p in Path(d).iterdir() if not p.name.startswith(".")).__len__())
    # lineage reads training_measurement.json only ⇒ JSON files only
    assert r.lineage(KING, {}) == 2
    assert _pulled(hub, KING) == [["config.json", "training_measurement.json"]]
    seen = {}

    def merge_check_dirs(kd, cd, sel, *, strict):
        seen["dirs"] = (kd.name, cd.name)
        return None

    monkeypatch.setattr(vel, "merge_check_dirs", merge_check_dirs)
    assert r.merge_check(KING, LEG, {0: [1]}, True) is None
    assert all(n.endswith("~weights") for n in seen["dirs"])
    # a weights fetch after the meta one pulls the weights; the probe reuses it
    r.probe(KING, CTX, 4, "energy")
    assert len(_pulled(hub, KING)) == 2
    ckpt = pod.workdir / off.OFFLOAD_DIR / "ckpt"
    assert not any((d / f).exists() for d in ckpt.iterdir() for f in off.LINEAGE_FILES)


def test_selective_fetch_merge_build_init_full_merges_weights(tmp_path, monkeypatch):
    import inspect

    from cascade.trainer import lane_b_model as lbm

    # the real build reads the init's lineage files through fetch_full
    src = inspect.getsource(lbm.LaneBModel.build)
    assert "init_dir = self._fetch_full(init_pointer)" in src
    assert "self._fetch(init_pointer)" not in src.split("init_dir = self._fetch_full", 1)[0]
    pod, hub, reg = _real_fetch_pod(tmp_path, monkeypatch)
    got = {}

    def build(self, init_pointer, chosen, *, out_dir, repo, strict, era_index):
        got["init"] = sorted(p.name for p in self._fetch_full(init_pointer).iterdir())
        got["merge"] = sorted(p.name for p in self._fetch(chosen[0]["rebased_pointer"])
                              .iterdir())
        got["init_again"] = self._fetch(init_pointer).name
        return "cascade/mi@sha256:" + "a" * 64

    monkeypatch.setattr(lbm.LaneBModel, "build", build)
    r = pod.runner(headroom_bytes=0)
    assert r.merge_build(KING, [{"rebased_pointer": LEG, "king_pointer": KING,
                                 "selection": {"0": [1]}, "round_id": "r1"}],
                         strict=True, push=_push("mi"), era_index=2).startswith("cascade/mi@")
    assert set(off.LINEAGE_FILES) <= set(got["init"])          # the init carries them
    assert not set(off.LINEAGE_FILES) & set(got["merge"])
    assert "~" not in got["init_again"]                      # the full copy serves weights


def test_full_fetch_drops_the_redundant_light_copy(tmp_path, monkeypatch):
    import cascade.shared.hippius as hip

    hub = FilteringHub(tmp_path)
    monkeypatch.setattr(hip, "fetch_from_hub", hub.fetch)
    hub.add(KING)
    root = tmp_path / "pod" / off.OFFLOAD_DIR
    kw = {"disk": {"headroom_bytes": 0}, "size_fn": hub.size}
    light = off._pod_fetch(KING, root, profile=off.FETCH_WEIGHTS, **kw)
    full = off._pod_fetch(KING, root, profile=off.FETCH_FULL, **kw)
    assert not light.exists() and (full / "optimizer.safetensors").is_file()
    assert off._pod_fetch(KING, root, profile=off.FETCH_META, **kw) == full
    assert len(hub.pulls) == 2


def test_hub_snapshot_bytes_sums_the_matching_layers(monkeypatch):
    pytest.importorskip("hippius_hub")
    import hippius_hub._oci as oci
    import hippius_hub.auth as auth

    import cascade.shared.hippius as hip

    layers = [{"size": n, "annotations": {"org.opencontainers.image.title": name}}
              for name, n in _CKPT_FILES.items()]
    layers.append({"size": 7, "annotations": {"org.opencontainers.image.title":
                                              "weights_ema_0.99.safetensors"}})
    seen = {}

    def fetch_manifest(registry, repo, rev, token, **kw):
        seen["rev"] = rev
        return SimpleNamespace(manifest={"layers": layers}, digest=rev)

    monkeypatch.setattr(oci, "fetch_manifest", fetch_manifest)
    monkeypatch.setattr(oci, "iter_titled_layers", lambda m: (
        (lay["annotations"]["org.opencontainers.image.title"], lay) for lay in m["layers"]))
    monkeypatch.setattr(auth, "get_oci_bearer_token", lambda repo, token, **kw: "t")
    monkeypatch.setattr(hip, "_resolve_hub_token_for_pull", lambda action: False)
    total = sum(_CKPT_FILES.values())
    assert hip.hub_snapshot_bytes(KING) == total              # UNPUBLISHED never counted
    assert seen["rev"] == "sha256:" + "b" * 64
    assert hip.hub_snapshot_bytes(KING, ignore_patterns=list(off.LINEAGE_FILES)) == \
        total - 300 - 100 - 50
    assert hip.hub_snapshot_bytes(KING, allow_patterns=["*.json"]) == 15
    assert hip.hub_snapshot_bytes("cascade/x@hf:" + "a" * 40) is None
    monkeypatch.setattr(oci, "fetch_manifest", lambda *a, **k: 1 / 0)
    assert hip.hub_snapshot_bytes(KING) is None               # advisory, never raises
