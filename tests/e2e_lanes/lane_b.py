"""Lane B full cycle on fake infrastructure (W-E2E, retrospective P1a).

One cycle, every stage the REAL cascade code:

    submit     the miner's commitment (``format_expert_commit``) on the fake
               chain + the CLI's signed fund headers (``build_lane_b_headers``)
    intake     ``FundingIntake.lane_b_fund`` over HTTP (``make_server``) with
               the production reveal oracle (``ChainRevealResolver.lane_b``)
    fund/queue the shared ``lane_b_queue.json`` (``LaneBQueue``), read by the
               trainer through ``lane_b_queue_of`` (``[round] funded_queue_path``)
    dispatch   ``LaneBDriver.tick`` (gate, routing probe ON THE EVAL POD,
               selection, ``_mark_funded``) → ``RunnerLaneBOps.dispatch_leg``
               → ``_dispatch_funded`` → ``TrainerRunner._run_funded_leg`` (the
               real admission, money-gate, image and re-attach checks)
    generation ``runner_funded_legs.leg_corpus`` → a fake Shadeform CPU VM
    train      ``TrainerRunner._rent_funded_host`` → a fake Lium pod (the
               worker is a torch-free stand-in writing a real-layout
               checkpoint: the release's pyramid code, a valid safetensors)
    harvest    the box pushes the leg checkpoint with the real
               ``upload_dir_to_hub_or_hf`` into an in-memory Hippius registry
               (a fake ``hippius_hub`` module) and returns the leg pointer
    settlement ``LaneBDriver.settle``: king + candidate bench rows
               (``LaneBBench`` on the pod), the prepare-ahead worker
               (``LaneBPrepare``), rebase / merge check / eval on the eval pod
               (``LaneBPodRunner`` → ``pod_main`` in-process), publish
               (``RunnerLaneBOps.publish_expert_manifest`` → S3 stand-in), the
               self-judge (``RunnerLaneBOps.judge`` → ``judge_manifest``)
    validator  ``ValidatorRunner._lane_b_prepare`` / ``_lane_b_commit``: reads
               the expert manifest, fetches every checkpoint with the real
               ``fetch_from_hub`` (chunked-v2 reassembly included), scores with
               the real scorer, folds the win into ``LaneBState``
    miner      ``cascade.miner.submission_status.lane_queue_status`` (what
               ``cascade lane-b status`` prints) over the intake's HTTP queue

The fakes sit only at the infrastructure boundary: chain, S3, the Hippius
registry, the Lium pod / Shadeform VM, ssh (the eval pod runs ``pod_main`` in
this process), the model math (``settle_leg`` / ``probe_target_routing`` /
the forecaster are byte-deterministic stand-ins that run ON THE POD only — a
box-side call reaches the real torch code and :func:`harness.box_guard`
refuses it).

Box-side stages run inside :func:`harness.box_guard`. Nothing here needs a
key, a pod, a chain or the network."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .harness import BlockClock, Cycle, box_budget, box_guard

REPO = Path(__file__).resolve().parents[2]
HK = "5E2eLaneBMinerAliceAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
GEN = "e2e/laneb-gen@sha256:" + hashlib.sha256(b"e2e lane b generator").hexdigest()
DOMAINS = ("energy", "web")
TARGET = "energy"
GPU = "L40S"
NETUID = 91
KING_ERR = 1.0          # the king's forecast error scale
LEG_ERR = 0.6           # what the miner's data teaches the expert rows
POINTER_MT = "application/vnd.hippius.pointer.v2"
PACK_MT = "application/vnd.hippius.pack.v1"
CHATTER = "Downloading weights.safetensors (parallel)...\nFetching 3 files: 100%|##| 3/3\n"

STAGES = ("submit", "intake", "fund_queue", "dispatch", "generation", "train",
          "harvest_upload", "settlement", "published_verdict", "validator_verify",
          "miner_status")
# per box stage: what an in-process load of pod data would blow (bytes)
BOX_BUDGET = 64 * 1024 * 1024


def _sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


# ── checkpoints (the release's code files + a real safetensors) ─────────────


def weights_bytes(err: float, tag: str, n: int = 64) -> bytes:
    """A valid safetensors file: one F32 tensor + ``__metadata__`` carrying
    the forecast error scale the fake forecaster reads."""
    data = struct.pack(f"<{n}f", *[float(i % 7) for i in range(n)])
    header = json.dumps({"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, len(data)]},
                         "__metadata__": {"e2e_err": repr(float(err)), "e2e_tag": tag}},
                        sort_keys=True).encode()
    return struct.pack("<Q", len(header)) + header + data


def read_err(ckpt: Path) -> tuple[float, str]:
    raw = (Path(ckpt) / "weights.safetensors").read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    meta = json.loads(raw[8:8 + n])["__metadata__"]
    return float(meta["e2e_err"]), str(meta["e2e_tag"])


def write_ckpt(d: Path, err: float, tag: str, *, leg_rec: dict | None = None) -> Path:
    from cascade.eval.checkpoint_guard import expected_checkpoint_code

    d.mkdir(parents=True, exist_ok=True)
    for name, code in expected_checkpoint_code("cascade_pyramid").items():
        (d / name).write_bytes(code)
    cfg = {"arch": "moe", "e2e": tag}
    if leg_rec is not None:
        cfg["lane_b_leg"] = leg_rec
    (d / "config.json").write_text(json.dumps(cfg, sort_keys=True), encoding="utf-8")
    (d / "weights.safetensors").write_bytes(weights_bytes(err, tag))
    return d


# ── the forecaster (the "model math", pod side only) ────────────────────────


def _truth(history: np.ndarray, horizon: int) -> np.ndarray:
    h = np.asarray(history, dtype=np.float64)
    slope = float(np.mean(np.diff(h[-8:]))) if h.size > 8 else 0.0
    return h[-1] + slope * np.arange(1, horizon + 1)


def forecast(ckpt: Path, history, horizon: int, num_samples: int) -> np.ndarray:
    """``(C, num_samples, horizon)`` samples: the window's truth plus noise
    scaled by the checkpoint's error, seeded by (weights, history)."""
    err, tag = read_err(ckpt)
    hist = np.asarray(history, dtype=np.float64)
    hist = hist if hist.ndim == 2 else hist[None, :]
    out = np.empty((hist.shape[0], num_samples, horizon))
    for c in range(hist.shape[0]):
        seed = int.from_bytes(hashlib.sha256(tag.encode() + hist[c].tobytes()).digest()[:8],
                              "little")
        rng = np.random.default_rng(seed)
        out[c] = _truth(hist[c], horizon) + err * rng.normal(size=(num_samples, horizon))
    return out


def score_ckpt(ckpt: Path, windows, num_samples: int):
    """What the eval pod's ``evaluate_checkpoint`` returns: the REAL scorer
    over the fake forecaster (after the real torch-free ingest check)."""
    from cascade.eval.checkpoint_guard import CheckpointTampered, safetensors_header
    from cascade.eval.scoring import score_joint_forecaster_on_windows
    from cascade.validator.evaluator import EvaluatorError

    # the ingest check's torch-free half (verify_checkpoint's code-identity and
    # state-spec halves import the model module: torch, which this process
    # never loads) — the half a chunked-v2 pointer saved as the weights fails
    try:
        safetensors_header(Path(ckpt) / "weights.safetensors")
    except CheckpointTampered as e:
        raise EvaluatorError(f"checkpoint_tampered: {e}") from e
    return score_joint_forecaster_on_windows(
        lambda h, H, ns: forecast(ckpt, h, H, ns), windows, num_samples)


class WindowSource:
    """The private pool stand-in: ``windows_for_round`` / ``windows_for_verdict``
    over a fixed, seeded set (each domain over several sources)."""

    def __init__(self, n_per_domain: int = 48, sources: int = 6):
        from cascade.eval.window import EvalWindow

        rng = np.random.default_rng(91)
        self.windows = []
        for d in DOMAINS:
            for i in range(n_per_domain):
                h = np.cumsum(rng.normal(size=48)) + (10.0 if d == "energy" else -5.0)
                self.windows.append(EvalWindow(
                    series_id=f"{d}-s{i % sources}-w{i}", history=h, target=_truth(h, 8),
                    metadata={"freq": "H", "domain": d, "source": f"{d}-src{i % sources}"}))

    def windows_for_round(self, seed, n, block=None):
        return list(self.windows)

    windows_for_verdict = windows_for_round


# ── the in-memory Hippius registry (a fake hippius_hub module) ──────────────


class MemHub:
    """OCI-shaped objects by manifest digest: single-blob (what the release
    writes) or chunked-v2 (what hippius_hub >= 0.6 writes for big files —
    2026-11-28: an eval pod whose venv drifted off the lock pushed one).
    ``snapshot_download`` behaves like the PINNED hippius_hub 0.5.1: titled
    layers only, a chunked file's POINTER saved under the file's name."""

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.pushes: list[dict] = []
        self.chunked_repos: tuple[str, ...] = ()    # repo substrings pushed chunked-v2
        self.downloads: list[str] = []

    # push side (``hippius_hub.upload_folder``)
    def upload_folder(self, *, repo_id, folder_path, allow_patterns=None, ignore_patterns=None,
                      token=None, **_):
        files = {p.name: p.read_bytes() for p in sorted(Path(folder_path).iterdir())
                 if p.is_file() and not p.name.startswith(".")}
        # the release pushes single-blob (HIPPIUS_CHUNKED_WRITE=0); a pod venv
        # that drifted off the lock (hippius_hub >= 0.6) writes chunked-v2
        chunked = any(s in str(repo_id) for s in self.chunked_repos)
        manifest, titled, packs = (_chunked_layout(files) if chunked
                                   else _single_layout(files))
        digest = _sha(manifest)
        self.objects[digest] = {"repo": str(repo_id), "files": files, "titled": titled,
                                "packs": packs, "manifest": manifest, "chunked": chunked}
        self.pushes.append({"repo": str(repo_id), "digest": digest, "chunked": chunked})
        return SimpleNamespace(oid=digest)

    def seed(self, repo: str, d: Path) -> str:
        """A pre-existing object (the king), pushed single-blob."""
        return f"{repo}@{self.upload_folder(repo_id=repo, folder_path=d).oid}"

    # fetch side (``hippius_hub.snapshot_download`` + the raw blob reader)
    def snapshot_download(self, *, repo_id, revision, local_dir, allow_patterns=None,
                          ignore_patterns=None, **_):
        import fnmatch

        obj = self.objects.get(str(revision))
        if obj is None:
            raise RuntimeError(f"404 manifest unknown: {repo_id}@{revision}")
        self.downloads.append(str(revision))
        d = Path(local_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, data in obj["titled"].items():
            if allow_patterns and not any(fnmatch.fnmatch(name, p) for p in allow_patterns):
                continue
            if ignore_patterns and any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
                continue
            (d / name).write_bytes(data)

    def reader(self, ref):
        obj = self.objects[str(ref.digest)]

        class _Reader:
            def manifest_bytes(self, digest):
                assert digest == _sha(obj["manifest"])
                return obj["manifest"]

            def blob_to(self, digest, fh):
                fh.write(obj["packs"][digest])

        return _Reader()

    def snapshot_bytes(self, ref, **_):
        obj = self.objects.get(str(getattr(ref, "digest", "") or str(ref).rsplit("@", 1)[-1]))
        return None if obj is None else sum(len(v) for v in obj["files"].values())

    def install(self, mp) -> None:
        import cascade.shared.hippius as hip

        mod = types.ModuleType("hippius_hub")
        mod.upload_folder = self.upload_folder
        mod.snapshot_download = self.snapshot_download
        mp.setitem(sys.modules, "hippius_hub", mod)
        mp.setattr(hip, "_chunked_registry_reader", self.reader, raising=False)
        mp.setattr(hip, "hub_snapshot_bytes", self.snapshot_bytes)
        mp.setenv("HIPPIUS_HUB_TOKEN", "e2e-fake-token")
        mp.setattr(hip, "UPLOAD_BACKOFF_BASE_S", 0.0, raising=False)


def _single_layout(files: dict) -> tuple[bytes, dict, dict]:
    layers = [{"mediaType": "application/octet-stream", "size": len(v), "digest": _sha(v),
               "annotations": {"org.opencontainers.image.title": k}}
              for k, v in sorted(files.items())]
    manifest = json.dumps({"schemaVersion": 2, "layers": layers}, sort_keys=True).encode()
    return manifest, dict(files), {}


def _chunked_layout(files: dict) -> tuple[bytes, dict, dict]:
    """hippius_hub >= 0.6's chunked-v2 layout for weights.safetensors."""
    file_bytes = files["weights.safetensors"]
    step = max(1, len(file_bytes) // 5)
    cuts = sorted({0, *range(step, len(file_bytes), step), len(file_bytes)})
    pieces = [file_bytes[a:b] for a, b in itertools.pairwise(cuts)]
    half = max(1, len(pieces) // 2)
    groups = [list(range(half)), list(range(half, len(pieces)))]
    packs, rows = {}, [None] * len(pieces)
    for g in (x for x in groups if x):
        order = g if g[0] == 0 else list(reversed(g))
        blob, offs = b"", {}
        for i in order:
            offs[i] = len(blob)
            blob += pieces[i]
        pd = _sha(blob)
        packs[pd] = blob
        for i in g:
            rows[i] = {"digest": _sha(pieces[i]), "offset": offs[i], "pack": pd,
                       "size": len(pieces[i])}
    pointer = json.dumps({"chunks": rows, "file": {"digest": _sha(file_bytes),
                                                   "size": len(file_bytes)},
                          "version": "chunked-v2"}, separators=(",", ":"),
                         sort_keys=True).encode()
    small = {k: v for k, v in files.items() if k != "weights.safetensors"}
    layers = [{"mediaType": "application/octet-stream", "size": len(v), "digest": _sha(v),
               "annotations": {"org.opencontainers.image.title": k}}
              for k, v in sorted(small.items())]
    layers.append({"mediaType": POINTER_MT, "size": len(pointer), "digest": _sha(pointer),
                   "annotations": {"org.opencontainers.image.title": "weights.safetensors",
                                   "com.hippius.file.size": str(len(file_bytes)),
                                   "com.hippius.file.digest": _sha(file_bytes),
                                   "com.hippius.chunk.count": str(len(pieces))}})
    layers += [{"mediaType": PACK_MT, "size": len(b), "digest": d} for d, b in packs.items()]
    manifest = json.dumps({"schemaVersion": 2,
                           "mediaType": "application/vnd.oci.image.manifest.v1+json",
                           "annotations": {"com.hippius.layout": "chunked-v2"},
                           "layers": layers}).encode()
    return manifest, dict(small, **{"weights.safetensors": pointer}), packs


class MemStore:
    """The S3 manifest store stand-in (``put_text`` / ``get_text``)."""

    def __init__(self):
        self.objs: dict[str, str] = {}
        self.acl: dict[str, str] = {}

    def put_text(self, key, text, content_type=None, acl=None, **_):
        self.objs[str(key)] = str(text)
        self.acl[str(key)] = str(acl or "")

    def get_text(self, key):
        from cascade.shared.hippius import StorageError

        if key not in self.objs:
            raise StorageError(f"404 {key}")
        return self.objs[key]


class FakeChain:
    """Commitments + the block clock (read-only: no extrinsic exists here)."""

    def __init__(self, clock: BlockClock):
        self.clock = clock
        self.rows: list[SimpleNamespace] = []

    def commit(self, hotkey: str, uid: int, payload: str) -> None:
        self.rows.append(SimpleNamespace(hotkey=hotkey, uid=uid, payload=payload,
                                         commit_block=self.clock.block))

    def poll_commitments(self, include_history=False):
        return list(self.rows)

    def current_block(self):
        return self.clock.block

    def block_hash(self, block):
        return self.clock.block_hash(block)


# ── the eval pod (ssh/scp doubles; the worker step runs pod_main here) ───────


class EvalPod:
    """A ``LaneBPodRunner`` target. Shell steps really run in ``workdir``; the
    worker step runs ``lane_b_offload.pod_main`` in-process with the pod-side
    model stand-ins applied ONLY for that call (the box never sees them)."""

    def __init__(self, world, name: str):
        from cascade.trainer.remote import RemoteHost

        self.world = world
        self.workdir = world.tmp / name
        self.workdir.mkdir()
        self.host = RemoteHost(name=f"valeval-{name}", host="203.0.113.9",
                               workdir=str(self.workdir), remote_python="python")
        self.ops: list[str] = []
        self.chatter = False                       # #983: download chatter on stdout

    @contextlib.contextmanager
    def pod_side(self):
        """The pod's environment: its model functions are the stand-ins."""
        import cascade.validator.evaluator as ev
        import cascade.validator.expert_lane as vel
        from cascade.trainer import expert_lane as tel
        from cascade.trainer import lane_b_driver
        from cascade.trainer import lane_b_offload as off

        w = self.world
        moe = types.ModuleType("cascade.trainer.moe_model")
        moe.MergeError = type("MergeError", (ValueError,), {})
        with w.mp.context() as m:
            m.setitem(sys.modules, "cascade.trainer.moe_model", moe)
            m.setattr(off, "_pod_fetch", self._pod_fetch)
            m.setattr(tel, "probe_target_routing", _fake_probe)
            m.setattr(lane_b_driver, "settle_leg", _fake_settle_leg)
            m.setattr(vel, "merge_check_dirs", lambda kd, cd, sel, strict: None)
            m.setattr(ev, "evaluate_checkpoint",
                      lambda ckpt, windows, *, num_samples, **kw: score_ckpt(
                          Path(ckpt), windows, num_samples))
            yield

    def _pod_fetch(self, pointer, root, keep=None, **kw):
        from cascade.shared import hippius
        from cascade.shared.manifest import parse_trained_pointer

        ref = hippius.HubRef.parse(parse_trained_pointer(pointer) or pointer)
        dest = Path(root) / "ckpt" / ref.digest.replace(":", "-")
        hippius.fetch_from_hub(ref, dest)
        if keep is not None:
            keep.add(dest)
        return dest

    def run(self, argv, timeout, stdin_text=""):
        from cascade.trainer import lane_b_offload as off

        assert argv[0] == "ssh"
        cmd = argv[-1]
        if "-m cascade.trainer.lane_b_offload" in cmd:
            assert "flock -w" in cmd and "env PYTHONPATH=" in cmd
            words = shlex.split(cmd.split("-m cascade.trainer.lane_b_offload", 1)[1]
                                .split("&&")[0])
            op, run_dir, root = words[:3]
            self.ops.append(op)
            with self.pod_side():
                off.pod_main([op, run_dir, root])
            out = (Path(run_dir) / "result.json").read_text()
            if self.chatter or self.world.pod_chatter:
                out = CHATTER + out
            return subprocess.CompletedProcess(argv, 0, out, "")
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60)

    def scp(self, host, local, remote, timeout):
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, remote)
        return subprocess.CompletedProcess([], 0, "", "")

    def runner(self, **kw):
        from cascade.trainer import lane_b_offload as off

        return off.LaneBPodRunner(lambda: self.host, run=self.run, scp=self.scp, **kw)


def _fake_probe(king_dir, contexts, cfg, target):
    """The routing probe stand-in: a function of the king bytes + the slice."""
    from cascade.trainer.expert_lane import RoutingProbe

    seed = int.from_bytes(hashlib.sha256(
        (Path(king_dir) / "weights.safetensors").read_bytes()
        + b"".join(np.asarray(c, dtype=np.float64).tobytes() for c in contexts)
        + str(target).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    n, layers, experts, k = 64, 2, 16, 4
    topk = [rng.integers(0, experts, size=(n, k)).astype(np.uint8) for _ in range(layers)]
    aff = np.stack([np.bincount(t.reshape(-1), minlength=experts) for t in topk]).astype(np.int64)
    return RoutingProbe(topk=topk, affinity=aff)


def _fake_settle_leg(king, base, leg, sel, out, *, refuse_nonfinite=False):
    """The rebase stand-in: the king's code + config, the LEG's error scale
    (the expert rows carry what the miner's data taught them)."""
    err, tag = read_err(Path(leg))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "model.py", "forecast_wrapper.py"):
        shutil.copyfile(Path(king) / f, out / f)
    stamp = hashlib.sha256(b"|".join((Path(d) / "weights.safetensors").read_bytes()
                                     for d in (king, base, leg))
                           + repr(sorted((int(k), list(v)) for k, v in sel.items())).encode())
    (out / "weights.safetensors").write_bytes(weights_bytes(err, "rebased-" + stamp.hexdigest()[:16]))
    return out


# ── the bench sweep (rows a function of the pinned bytes' error) ────────────

SUITES = ("gift-eval", "boom", "time", "fev")


class Sweeps:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, host, pin, *, settings, workers, timeout_s, label):
        self.calls.append(label)
        err, _ = read_err(Path(pin))
        return {"suites": [{"suite": s, "status": "ok",
                            "rows": [{"full": f"{s}/i{i}", "CRPS": 0.3 + 0.05 * err + i / 1000,
                                      "MASE": 1.0 + 0.05 * err + i / 1000}
                                     for i in range(12)]} for s in SUITES],
                "data_revisions": {s: "r1" for s in SUITES}}


# ── the GPU pod (fake Lium) + the CPU VM (fake Shadeform) ───────────────────


class _PodTrained(Exception):   # noqa: N818 — a control-flow carrier, not an error
    """Carries the harvested leg out of the rent seam (see :class:`Lium`)."""

    def __init__(self, entry):
        super().__init__("trained")
        self.entry = entry


class WorkerFailed(RuntimeError):
    """A worker exit as the dispatcher reports it (rc, final traceback line)."""

    def __init__(self, msg, *, returncode, final_exception, role="expert"):
        super().__init__(msg)
        self.returncode, self.final_exception, self.role = returncode, final_exception, role


OOM_LINE = ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has "
            "a total capacity of 44.40 GiB of which 1.10 GiB is free. Of the allocated memory "
            "41.20 GiB is allocated by PyTorch, and 1.00 GiB is reserved by PyTorch but "
            "unallocated.")
NAN_LINE = "FloatingPointError: non-finite pyramid gradient"


class Lium:
    """``TrainerRunner._rent_funded_host`` stand-in: rents, runs the worker,
    harvests and pushes the leg checkpoint from the BOX with the real
    ``upload_dir_to_hub_or_hf``; then leaves the real leg body through
    :class:`_PodTrained` (the rest of ``_run_funded_leg_scoped`` is ssh /
    harvest plumbing that has its own unit tests)."""

    def __init__(self, world):
        self.world = world
        self.rents: list[str] = []
        self.ledger: list[SimpleNamespace] = []      # the funded pod ledger
        self.fail: str | None = None                 # "oom" | "nan"
        self.trained = 0

    def rent(self, runner, round_id, gen, gpu_choice=None, **kw):
        from cascade.provision.funded import funded_pod_name
        from cascade.trainer.leg_reattach import REATTACH_REFUSED, attach_only

        name = funded_pod_name(str(round_id), gen.hotkey, NETUID) + "-0"
        if attach_only() and name not in {p.instance_id for p in self.ledger}:
            # the funded rent path under attach-only: never a fresh rent
            msg = f"{REATTACH_REFUSED}: the leg's pod is not on the funded ledger"
            runner._record_funded_failure(gen.hotkey, msg, miner_fault=False,
                                          error_class="no_capacity", burn=False)
            from cascade.trainer.faults import _FundedLegSkip

            raise _FundedLegSkip(gen.hotkey)
        self.rents.append(name)
        self.ledger.append(SimpleNamespace(instance_id=name))
        self.world.cycle.ran("train", f"{gpu_choice.gpu if gpu_choice else '?'} pod {name}")
        if self.fail:
            self._fail(runner, gen, round_id)
        loc = runner.__dict__["_leg_local_obj"]
        loc.funded_pod_sku = f"1x{GPU}"
        self.trained += 1
        d = write_ckpt(self.world.tmp / f"harvest-{self.trained}", LEG_ERR, f"leg-{gen.hotkey[:8]}",
                       leg_rec={"steps": 10, "tokens": 640, "stop": "tokens"})
        from cascade.shared.hippius import upload_dir_to_hub_or_hf
        from cascade.shared.manifest import format_trained_pointer

        full = f"{runner.hub().namespace}/ckpt-{round_id}-expert-{gen.hotkey[:12]}"
        up = upload_dir_to_hub_or_hf(d, full, runner.hub())
        ptr = format_trained_pointer(up.ref.immutable_ref)
        self.world.cycle.ran("harvest_upload", ptr)
        raise _PodTrained(SimpleNamespace(trained_pointer=ptr, corpus_digest="corpus-" + gen.ref[-8:],
                                          train_block=self.world.clock.block, gpu_name=GPU))

    def _fail(self, runner, gen, round_id):
        """The worker died: the real classifier decides, as at
        ``runner_funded_legs`` (the dispatch-failure branch of the leg body)."""
        from cascade.trainer.faults import GPU_OOM_CLASS, classify_leg_exception, gpu_oom_message

        line = OOM_LINE if self.fail == "oom" else NAN_LINE
        e = WorkerFailed(f"worker exited 1: {line}", returncode=1, final_exception=line)
        seeds = SimpleNamespace(base_seed=str(round_id))
        # retro 2026-11-28 (P7): the fund-time fit evidence of the queue entry,
        # as runner_funded_legs reads it (M-OOM-FIT needs it to be the miner's)
        from cascade.trainer import fault_class

        q = runner._funded_queue()
        prior = q.get(gen.hotkey) if q is not None else None
        miner_fault, cls, burn = classify_leg_exception(
            runner, gen, seeds, e, rc=1, text=str(e), stalled_before=False, shards_leg=False,
            role="expert", fit_evidence=fault_class.fit_evidence_of(prior, "expert"))
        text = str(e)
        if cls == GPU_OOM_CLASS:
            text = f"{gpu_oom_message(f'1x{GPU}')}: {text}"
        runner._record_funded_failure(gen.hotkey, text, miner_fault=miner_fault,
                                      error_class=cls, burn=burn)
        raise e


# ── the world ────────────────────────────────────────────────────────────────


def lane_b_section(**over) -> dict:
    """chain.toml's [lane_b], compressed for the cycle (small window counts,
    the ``tokens`` opening rule at 0 tokens, two domain slots)."""
    raw = tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))["lane_b"]
    raw.update({"from_block": 1, "domains": list(DOMAINS), "open_rule": "tokens",
                "open_tokens": 0, "bootstrap_B": 500, "min_target_windows": 24,
                "min_target_clusters": 3, "min_guard_windows": 12, "min_experts": 2,
                "max_experts": 6, "affinity_windows": 8, "merge_guard_from_block": 1})
    raw.update(over)
    return raw


class LaneBWorld:
    def __init__(self, cfg, tmp: Path, mp):
        self.tmp, self.mp = Path(tmp), mp
        self.cycle = Cycle("b")
        self.clock = BlockClock(block=0, epoch_blocks=900)
        self.chain = FakeChain(self.clock)
        self.hub = MemHub()
        self.store = MemStore()
        self.source = WindowSource()
        self.generations = 0
        self.pod_chatter = False            # #983: every pod reply carries download chatter
        self._threads: list[threading.Thread] = []
        self._setup_cfg(cfg)

    # ── configuration (the repo template, compressed) ────────────────────
    def _setup_cfg(self, cfg):
        from cascade.shared.config import parse_lane_b
        from cascade.shared.era import era_length_blocks

        v2 = int(cfg.training.v2_from_block)
        era = era_length_blocks(cfg.round, v2)
        self.era_blocks = era
        self.base_block = (v2 // era + 4) * era          # an era boundary under v2
        self.clock.block = self.base_block
        # the production layout: the queue files and the driver state in the
        # trainer's work root ([round] funded_queue_path is work_root-relative)
        self.work_root = self.tmp / "train"
        self.work_root.mkdir(parents=True)
        self.queue_path = self.work_root / "funded_queue.json"
        self.cfg = dataclasses.replace(
            cfg, lane_b=parse_lane_b(lane_b_section()),
            round=dataclasses.replace(cfg.round, funded_mode="required",
                                      funded_queue_path="funded_queue.json"),
            eval=dataclasses.replace(cfg.eval, scored_from_block=0, num_samples=16),
            validator=dataclasses.replace(cfg.validator,
                                          state_db_path=str(self.tmp / "val" / "state.json")))

    # ── stage: submit (the miner) ────────────────────────────────────────
    def submit(self, uid: int = 7):
        from cascade.miner.lane_b import build_lane_b_headers
        from cascade.shared.expert_manifest import format_expert_commit

        self.chain.commit(HK, uid, format_expert_commit(TARGET, GEN))
        self.fund_headers = build_lane_b_headers(
            "lane-b-fund", HK, (TARGET,), GEN, "lium-e2e-key", lambda m: b"\x01",
            now=time.time, gpu=GPU)
        self.cycle.ran("submit", f"commit {TARGET}:{GEN[-12:]}")

    # ── stage: intake (HTTP, the real routes) ────────────────────────────
    def start_intake(self):
        from cascade.funding.intake import FundingIntake
        from cascade.funding.main import ChainRevealResolver
        from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
        from cascade.funding.vault import PayerKeyVault

        resolver = ChainRevealResolver(self.chain, clock=lambda: 0.0)
        lb = self.cfg.lane_b
        self.intake = FundingIntake(
            FundedQueue(self.queue_path), PayerKeyVault(dir=None),
            resolve_reveal=resolver, require_signature=False,
            lane_b_queue=LaneBQueue(lane_b_queue_path(self.queue_path)),
            resolve_lane_b_reveal=lambda hk, t, r: resolver.lane_b(
                hk, t, r, slots=tuple(lb.domains), max_targets=int(lb.max_domains_per_leg or 1)),
            lane_b_targets=tuple(lb.domains), lane_b_max_targets=int(lb.max_domains_per_leg or 1))
        self.srv = self.intake.make_server("127.0.0.1", 0, per_ip_rpm=0)
        t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        t.start()
        self.intake_url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def fund(self):
        import urllib.request

        req = urllib.request.Request(f"{self.intake_url}/v1/lane-b/fund", method="POST",
                                     headers=self.fund_headers)
        with box_guard(), urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            assert resp.status == 202, body
        self.cycle.ran("intake", f"HTTP {resp.status} {body.get('status')}")

    def close(self):
        srv = getattr(self, "srv", None)
        if srv is not None:
            srv.shutdown()
            srv.server_close()

    # ── the trainer (real TrainerRunner, RunnerLaneBOps, LaneBDriver) ────
    def start_trainer(self, *, restart: bool = False):
        import cascade.funding.lane_b_gpus as lbg
        import cascade.trainer.leg_pins as leg_pins
        import cascade.trainer.runner_funded_legs as rfl
        from cascade.trainer.lane_b_bench import LaneBBench
        from cascade.trainer.lane_b_driver import LaneBDriver, RunnerLaneBOps
        from cascade.trainer.lane_b_prepare import LaneBPrepare
        from cascade.trainer.loop import TrainerRunner

        mp = self.mp
        if not restart:
            self.lium = Lium(self)
            self.eval_pod = EvalPod(self, "trainer-eval-pod")
            self.val_pod = EvalPod(self, "validator-eval-host")
            self.hub.install(mp)
            mp.setattr(lbg, "lane_b_gpus_from_env", lambda cfg, **kw: lbg.lane_b_gpus(cfg, None))
            mp.setattr(leg_pins, "contract_runs_after", lambda cfg, c: False)
            mp.setattr(rfl, "_reupload_pending", lambda *a, **k: None)
            mp.setattr(rfl, "_pre_corpus_money_gates", lambda *a, **k: None)
            mp.setattr(rfl, "leg_corpus", self._shadeform_generate)
            mp.setattr(TrainerRunner, "_dispatcher", lambda self: object())
            real = TrainerRunner._run_funded_leg

            def run_funded_leg(runner, *a, **kw):
                try:
                    return real(runner, *a, **kw)
                except _PodTrained as done:
                    return done.entry
            mp.setattr(TrainerRunner, "_run_funded_leg", run_funded_leg)
            mp.setattr(TrainerRunner, "_rent_funded_host",
                       lambda runner, rid, gen, **kw: self.lium.rent(runner, rid, gen, **kw))
            self._king_setup()
        r = TrainerRunner(cfg=self.cfg, base_trainer=object(), work_root=self.work_root,
                          trainer_spec="m:C")
        era = SimpleNamespace(base_seed=777, start_block=self.base_block, index=self.era_index)
        r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(current=era))
        r._rolling_seeds = lambda e: SimpleNamespace(base_seed=777)
        r._load_funded_ledger = lambda: list(self.lium.ledger)
        r._operator_leg_pod_prefix = lambda rid, hk: f"cascade-n{NETUID}-{rid}-op-{hk[:12]}"
        store = self.store
        r.manifest_store = lambda: store
        runner = self.eval_pod.runner()
        self.sweeps = Sweeps()
        bench = LaneBBench(hosts_path=self.tmp / "targon-eval.toml", cache_dir=self.tmp / "rows",
                           eval_cfg=SimpleNamespace(cascade_bench_max_series=0,
                                                    gift_gate_data_dir="/data/bench",
                                                    bench_workers="1"),
                           start_worker=False, pod=runner, pinned_bench_fn=self.sweeps)
        ops = RunnerLaneBOps(r, self.chain, self.source, bench=bench, offload=runner)
        tick = itertools.count(0, 30)
        ops._preparer = LaneBPrepare(runner, bench, start_worker=False,
                                     clock=lambda: float(next(tick)))
        self.runner, self.ops, self.bench = r, ops, bench
        self.drv = LaneBDriver(self.cfg, ops, self.work_root / "lane_b_driver.json")
        # the real dispatch spawns a leg thread: keep a handle to join it
        real_thread = threading.Thread

        def spawn(*a, **kw):
            t = real_thread(*a, **kw)
            self._threads.append(t)
            return t
        mp.setattr("cascade.trainer.lane_b_driver.threading.Thread", spawn)

    era_index = 1

    def _king_setup(self):
        from cascade.shared.manifest import format_trained_pointer

        kd = write_ckpt(self.tmp / "king", KING_ERR, "king")
        self.king = format_trained_pointer(self.hub.seed("cascade/king-e2e", kd))

    def _shadeform_generate(self, runner, gen, seeds, contract, **kw):
        """The CPU-tier corpus on a (fake) Shadeform VM."""
        self.generations += 1
        self.cycle.ran("generation", f"shadeform cpu vm #{self.generations}")
        return {"mode": "shards", "digest": "corpus-" + gen.ref[-8:]}

    def join_legs(self, timeout=30.0):
        for t in list(self._threads):
            t.join(timeout)
            assert not t.is_alive(), "a lane B leg thread hung"
        self._threads.clear()

    # ── stage: dispatch (tick on the funded queue) ───────────────────────
    def dispatch(self, *, hold=False):
        """One driver tick at the era start: the gate, the probe on the eval
        pod, the funded entry marked in flight, the leg thread. ``hold`` = the
        trainer dies right after the dispatch (the leg never reached a rent)."""
        self._watch_dispatch(hold=hold)
        with box_guard(), box_budget("dispatch", BOX_BUDGET):
            started = self.drv.tick(self.clock.block, self.era_index, self.king)
            self.join_legs()
        return started

    def _watch_dispatch(self, *, hold=False):
        """Record fund/queue + dispatch at the moment the driver hands the leg
        over (the funded entry is in flight then); ``hold`` drops the leg."""
        ops = self.ops
        real = type(ops).dispatch_leg

        def dispatch_leg(leg, era, block, done):
            e = ops.funded_queue().get(leg.sub.hotkey)
            self.cycle.ran("fund_queue", f"entry {e.status if e else None}")
            self.cycle.ran("dispatch", f"{leg.sub.hotkey[:8]} rows "
                           f"{ {k: len(v) for k, v in leg.selection.items()} }")
            if not hold:
                real(ops, leg, era, block, done)
        real_re = type(ops).reattach_leg

        def reattach_leg(leg, era, block, done, rec):
            e = ops.funded_queue().get(leg.sub.hotkey)
            self.cycle.ran("fund_queue", f"entry {e.status if e else None} (re-attach)")
            self.cycle.ran("dispatch", f"{leg.sub.hotkey[:8]} re-attach after restart")
            real_re(ops, leg, era, block, done, rec)
        ops.dispatch_leg = dispatch_leg
        ops.reattach_leg = reattach_leg

    def restart_trainer(self):
        """A trainer restart: a new process over the same state files."""
        self.start_trainer(restart=True)
        self._watch_dispatch()
        with box_guard():
            started = self.drv.tick(self.clock.block, self.era_index, self.king)
            self.join_legs()
        return started

    # ── stage: settlement (prepare-ahead + in-window retries) ────────────
    def settle(self, *, attempts: int = 6):
        """The settlement at the next boundary, with prepare-ahead steps and
        the bench queue drained between in-window retries."""
        self.clock.to_next_boundary()
        self.round_id = str(self.clock.block * 7 + 3)
        drv, ops = self.drv, self.ops
        prep = ops._preparer
        errors = []
        with box_guard(), box_budget("settlement", BOX_BUDGET):
            ops.prepare_ahead(drv.prepare_specs(self.clock.block, self.era_index, self.king),
                              paused=drv._settling.is_set)
            for _ in range(12):
                prep.run_due()
                _drain(self.bench)
            judgements = None
            for _ in range(attempts):
                try:
                    judgements = drv.settle(round_id=self.round_id, block=self.clock.block,
                                            era_index=self.era_index, king_pointer=self.king,
                                            work_dir=self.tmp / "settle")
                    break
                except Exception as e:  # noqa: BLE001 — the in-window retry path
                    errors.append(e)
                    _drain(self.bench)
        self.settle_errors = errors
        if judgements is None:
            return None
        self.judgements = judgements
        self.cycle.ran("settlement", f"round {self.round_id}: {len(judgements)} judged")
        from cascade.shared.expert_manifest import expert_manifest_key

        key = expert_manifest_key(self.round_id)
        if key in self.store.objs:
            self.cycle.ran("published_verdict", key)
        return judgements

    # ── stage: validator verification ────────────────────────────────────
    def validate(self):
        """A validator judges the published expert manifest from the bytes in
        storage: its own fetches (real fetch_from_hub), its own scoring."""
        import cascade.validator.eval_offload as eo
        from cascade.shared.manifest import TrainedEntry, TrainingManifest
        from cascade.validator.loop import ValidatorRunner
        from cascade.validator.state import genesis

        (self.tmp / "val").mkdir(exist_ok=True)
        r = ValidatorRunner(cfg=self.cfg, state=genesis("K", 0), verify_signatures=False,
                            cache_dir=self.tmp / "val" / "ckpts")
        r._lane_b_store = self.store
        r._lane_b_window_source = self.source
        r.private_eval_host_fn = lambda: self.val_pod.host
        r.lane_b_merge_ops = self.val_pod.runner(source="--eval-hosts")

        def forecasts_via_host(host, checkpoint, windows, *, num_samples, **kw):
            score_ckpt(Path(checkpoint), [], num_samples)   # the ingest check only
            return [forecast(Path(checkpoint), w.history, int(np.asarray(w.target).shape[-1]),
                             num_samples) for w in windows]
        self.mp.setattr(eo, "forecasts_via_host", forecasts_via_host)
        m = TrainingManifest(
            round_id=self.round_id, created_block=self.clock.block, contract_digest="1" * 64,
            base_arch_digest="2" * 64, eval_dataset="d",
            entries=[TrainedEntry("K", 0, "king", GEN, self.king, "3" * 64, self.clock.block - 1)],
            prev_round_id="", expert_manifest_sha256=self.drv.expert_manifest_digest(self.round_id))
        with box_guard(), box_budget("validator", BOX_BUDGET):
            r._lane_b_block = r._epoch_start_block(m)
            rounds = r._lane_b_prepare(m)
            r._lane_b_commit(rounds)
        self.validator = r
        self.val_rounds = rounds
        self.cycle.ran("validator_verify", f"{len(rounds or [])} round(s) folded")
        return rounds

    # ── stage: the miner's view ──────────────────────────────────────────
    def miner_status(self) -> str:
        from cascade.miner.submission_status import lane_queue_status

        args = SimpleNamespace(intake_url=self.intake_url, hotkey=HK, cmd="lane-b status")
        buf = io.StringIO()
        with box_guard(), contextlib.redirect_stdout(buf):
            lane_queue_status(args, "b")
        out = args._reporter
        text = buf.getvalue() + "\n".join(str(x) for x in getattr(out, "lines", []) or [])
        self.cycle.ran("miner_status", text.splitlines()[0] if text else "")
        return text

    def queue_row(self) -> dict:
        _, doc = self.intake.lane_b_queue_view()
        return next(r for r in doc["entries"] if r["hotkey"] == HK)

    def stored_entry(self) -> dict:
        """The raw queue file entry (what ``mark_done`` stored)."""
        raw = json.loads((self.work_root / "lane_b_queue.json").read_text(encoding="utf-8"))
        rows = raw.get("entries", raw) if isinstance(raw, dict) else raw
        rows = rows.values() if isinstance(rows, dict) else rows
        return next(r for r in rows if isinstance(r, dict) and r.get("hotkey") == HK)


def _drain(bench) -> int:
    n = 0
    while not bench._q.empty():
        tag, item = bench._q.get()
        bench.run_tagged(tag, item)
        n += 1
    return n


def trainer_rows(world) -> str:
    from cascade.validator.expert_lane import judgements_record

    return json.dumps(judgements_record(world.judgements), sort_keys=True)


def validator_rows(world) -> str:
    (rnd,) = world.val_rounds
    return json.dumps(rnd.rows, sort_keys=True)


def full_cycle(world, *, dispatch_hold: bool = False) -> LaneBWorld:
    """submit → … → miner status. Stops (without raising) at the first stage
    that does not complete; the caller asserts on ``world.cycle``."""
    world.submit()
    world.start_intake()
    world.fund()
    world.start_trainer()
    if dispatch_hold:
        world.dispatch(hold=True)
        world.restart_trainer()
    else:
        world.dispatch()
    if not world.drv.st.finished:
        return world
    if world.settle() is None:
        return world
    world.validate()
    world.miner_status()
    return world
