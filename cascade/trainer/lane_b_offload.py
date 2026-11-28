"""Lane B model work on the EVAL POD — never on the trainer box.

The trainer box (7 GB RAM, no GPU) co-hosts the validator and OOM'd before;
owner rule (ABSOLUTE): no checkpoint downloads to its disk, no torch model
loads, no CPU forwards there. Lane B's model work therefore runs on the eval
pod(s) named by the same eval-hosts TOML as the lane-B bench
(``--lane-b-bench-hosts``, e.g. ``targon-eval.toml``, re-read per call):

* **Routing probe** (:meth:`LaneBPodRunner.probe`): the king checkpoint is
  fetched ON THE POD (digest-pinned Hub pull into a pod-side cache this module
  owns), :func:`cascade.trainer.expert_lane.probe_target_routing` runs there
  on the pod's CPU — the same function, the same code bytes (this release's
  ``cascade`` package is shipped per content hash, so a stale pod checkout
  never answers), the same device as the old local path — and the
  :class:`~cascade.trainer.expert_lane.RoutingProbe` comes back as JSON
  carrying the raw array bytes (dtype + shape + base64), so the decoded
  object is byte-identical to what the pod computed.
* **Lineage measurement** (:meth:`LaneBPodRunner.lineage`): the checkpoint's
  measurement JSON is read on the pod; only the integer returns.
* **Settlement / era-init model work** (:class:`PodLaneBModel`, WB27
  follow-up): the lane B settlement rebase (king / base / leg fetched on the
  pod, ``settle_leg``, the limits record read), the era init's merge checks,
  acceptance stages and the merged init build all run as pod steps whose
  worker drives :class:`cascade.trainer.lane_b_model.LaneBModel` over the
  pod's own Hub pull and Hub PUSH — a rebased / staged / merged checkpoint
  goes from the pod to the Hub directly (credentials over ssh stdin), and
  only its pointer and bench content key come back. Never checkpoint bytes
  through the box.
* **Bench staging** (:meth:`LaneBPodRunner.bench_stage`): the lane-B bench's
  pointer path (#909 prebench) fetches on the pod, computes the same bench
  content key the box computed, and hard-links the bench files into a pinned
  dir the sidecar sweeps in place (no upload from the box).
* **Hard guard.** No hosts file, no host, an unreachable pod or a failed pod
  step raises :class:`LaneBOffloadUnavailable` (an ``OSError``: the driver
  treats it like any transient I/O failure — the entry stays queued, nothing
  is charged, the next tick retries). Nothing ever falls back to local torch
  or a local download. The local path exists only behind the explicit opt-in
  ``CASCADE_LANE_B_LOCAL_EVAL=1`` (tests / dev boxes with a GPU).
* **Shared pod.** ``valeval-blackwell4`` also serves the validator's evals
  with no lock. Every call uses a uuid-unique run dir (removed afterwards);
  the only cache evicted is this module's own ``_lane_b_offload/ckpt``; the
  heavy step runs under a pod-side ``flock`` (lane B serialises with itself).
* **Own hosts, own disk.** ``--lane-b-model-hosts`` names the pod(s) for the
  model work (default: the bench hosts); bench staging stays with the bench
  sidecar. A host's ``cache_root`` roots ``_lane_b_offload`` (e.g. a volume).
  Each step fetches only the files it reads (:data:`LINEAGE_FILES` are
  skipped except for the merged init's init), and every pull is disk-guarded
  (:func:`_ensure_room`: LRU-evict unneeded cache entries, else refuse —
  transient).

Torch-free on the box: the pod-side half (:func:`pod_main`) imports torch only
on the pod.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import shlex
import tarfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger("cascade.trainer.lane_b_offload")

LOCAL_EVAL_ENV = "CASCADE_LANE_B_LOCAL_EVAL"
PROBE_SCHEMA = "lane-b-routing-probe/1"
OFFLOAD_DIR = "_lane_b_offload"
# pod-side checkpoint cache entries kept (this module's own dir only)
CKPT_KEEP = 4
CODE_KEEP = 3
PROBE_TIMEOUT_S = 1800
LOCK_WAIT_S = 1200
STDERR_TAIL = 800
# Hub pull credentials forwarded over ssh STDIN (never argv / files) when set
# here; unset ⇒ the pod pulls anonymously, as validators do.
_FORWARD_ENV = ("HIPPIUS_HUB_TOKEN", "HIPPIUS_TOKEN", "CASCADE_HIPPIUS_TOKEN",
                "HIPPIUS_HUB_USERNAME", "HIPPIUS_REGISTRY_USERNAME",
                "HIPPIUS_HUB_PASSWORD", "HIPPIUS_REGISTRY_PASSWORD")
# ops that PUSH a checkpoint from the pod also get the HF fallback token
# (``upload_dir_to_hub_or_hf``), again over stdin only
_UPLOAD_FORWARD_ENV = ("HF_TOKEN", "HUGGINGFACE_API_KEY")
UPLOAD_OPS = frozenset({"rebase", "merge_stage", "merge_build"})
# rebase / merge steps: three checkpoint pulls + one push, CPU arithmetic
MODEL_TIMEOUT_S = 3600
# pinned bench dirs older than this are abandoned (a box that died mid-sweep)
PIN_STALE_S = 24 * 3600

# ── selective fetch + disk guard (2026-11-26: the 14 GB king filled the shared
# 43 GB eval pod to 98% — optimizer.safetensors alone is ~5 GB) ─────────────
# What each pod step reads of a checkpoint (traced in code):
#   META    lineage: lineage.read_measurement reads training_measurement.json.
#   WEIGHTS probe (expert_lane.load_king: config.json + model code + weights),
#           eval (evaluator.load_*forecaster), merge_check (merge_check_dirs), merge_verify
#           (merged_rebuild_check: weights + config.json of every pointer),
#           rebase (settle_leg + leg_actuals: config.json, model.py,
#           forecast_wrapper.py, weights.safetensors — and it WRITES only
#           those), merge_finite / merge_stage (weights + CODE_FILES), every
#           merge record's pointers in merge_build (LaneBModel._load: weights),
#           bench_stage (content_key / pin: BENCH_UPLOAD_FILES).
#   FULL    merge_build's INIT only: it carries optimizer.safetensors (moments
#           reset), weights_stable.safetensors and power_ema.safetensors
#           (deltas applied) into the pushed init.
# The lineage files skipped below are never read by a WEIGHTS step, so the
# bytes a step pushes and the bench content keys are unchanged.
FETCH_FULL, FETCH_WEIGHTS, FETCH_META = "full", "weights", "meta"
LINEAGE_FILES = ("optimizer.safetensors", "weights_stable.safetensors",
                 "power_ema.safetensors")
# strongest first: an entry of a stronger profile serves every weaker one
_PROFILES = (FETCH_FULL, FETCH_WEIGHTS, FETCH_META)
_PROFILE_PATTERNS = {
    FETCH_FULL: (None, None),
    FETCH_WEIGHTS: (None, list(LINEAGE_FILES)),
    FETCH_META: (["*.json"], None),
}
# free space kept on the cache filesystem beyond the pull (configurable:
# ``--lane-b-pod-headroom-gb``); the validator's evals share this disk
DISK_HEADROOM_BYTES = 10 * 10**9
# expected pull size when the registry cannot say (hf: / vault refs, a
# registry error): one full Cascade MoE checkpoint
SIZE_FALLBACK_BYTES = 16 * 10**9
# a hidden fetch temp dir this old is a killed step's leftover (steps hold the
# heavy flock, so a live one is never this old)
FETCH_TMP_STALE_S = 2 * 3600


class LaneBOffloadUnavailable(OSError):
    """Lane B model work could not run on an eval pod (none configured, none
    reachable, or the pod step failed). Transient: never a miner fault, never
    a reason to run the work on the trainer box."""


def local_eval_allowed() -> bool:
    """The explicit opt-in for lane B model work on THIS machine (tests/dev)."""
    return os.environ.get(LOCAL_EVAL_ENV, "").strip() == "1"


def refuse_local(what: str) -> None:
    """Raise unless the local opt-in is set."""
    if not local_eval_allowed():
        raise LaneBOffloadUnavailable(
            f"lane B {what} refused on the trainer box: model work runs on the eval pod "
            f"only ({LOCAL_EVAL_ENV}=1 opts a dev/test box in)")


# ── RoutingProbe codec (exact: raw bytes, dtype, shape) ──────────────────────


def _enc_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def _dec_array(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["b64"].encode("ascii"), validate=True)
    dt = np.dtype(d["dtype"])
    shape = tuple(int(x) for x in d["shape"])
    if len(raw) != int(np.prod(shape, dtype=np.int64)) * dt.itemsize:
        raise ValueError("lane B probe reply: array byte count does not match its shape")
    return np.frombuffer(raw, dtype=dt).reshape(shape).copy()


def encode_probe(probe) -> dict:
    return {"schema": PROBE_SCHEMA, "topk": [_enc_array(t) for t in probe.topk],
            "affinity": _enc_array(probe.affinity)}


def decode_probe(doc: dict):
    from .expert_lane import RoutingProbe

    if not isinstance(doc, dict) or doc.get("schema") != PROBE_SCHEMA:
        raise ValueError("lane B probe reply: unknown schema")
    topk = [_dec_array(t) for t in doc["topk"]]
    aff = _dec_array(doc["affinity"])
    if aff.dtype != np.int64 or aff.ndim != 2 or len(topk) != aff.shape[0]:
        raise ValueError("lane B probe reply: malformed affinity")
    return RoutingProbe(topk=topk, affinity=aff)


def write_contexts(path: Path, contexts: list) -> None:
    """The slice, dtype-exact (npz, one array per context, in order)."""
    arrs = {f"c{i:06d}": np.asarray(c) for i, c in enumerate(contexts)}
    np.savez(path, **arrs)


def read_contexts(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return [z[k] for k in sorted(z.files)]


# ── code bundle (this release's package, content-addressed) ──────────────────

_BUNDLE: tuple[str, bytes] | None = None


def code_bundle(root: Path | None = None) -> tuple[str, bytes]:
    """``(sha256, tar.gz bytes)`` of this release's ``cascade`` package (no
    ``__pycache__``/website), deterministic so the pod caches it by hash."""
    global _BUNDLE
    if root is None and _BUNDLE is not None:
        return _BUNDLE
    pkg = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    buf = io.BytesIO()
    import gzip

    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for p in sorted(pkg.rglob("*")):
            rel = p.relative_to(pkg.parent)
            if not p.is_file() or "__pycache__" in rel.parts or "website" in rel.parts \
                    or p.suffix in (".pyc", ".pyo"):
                continue
            data = p.read_bytes()
            ti = tarfile.TarInfo(str(rel))
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
            tar.addfile(ti, io.BytesIO(data))
    blob = buf.getvalue()
    out = (hashlib.sha256(blob).hexdigest()[:32], blob)
    if root is None:
        _BUNDLE = out
    return out


# ── box side ─────────────────────────────────────────────────────────────────


class LaneBPodRunner:
    """Runs lane B model work on the eval pod named by ``host_fn``."""

    def __init__(self, host_fn: Callable[[], object] | None, *, run=None, scp=None,
                 timeout_s: int = PROBE_TIMEOUT_S, bundle: Callable[[], tuple] | None = None,
                 headroom_bytes: int = DISK_HEADROOM_BYTES,
                 size_fallback_bytes: int = SIZE_FALLBACK_BYTES,
                 source: str = "--lane-b-bench-hosts", where: str = "the trainer box"):
        self._host_fn = host_fn
        self._run = run
        self._scp = scp
        self.timeout_s = int(timeout_s)
        self._bundle = bundle or code_bundle
        self.headroom_bytes = max(0, int(headroom_bytes))
        self.size_fallback_bytes = max(0, int(size_fallback_bytes))
        self.source = str(source)
        self.where = str(where)

    @classmethod
    def from_hosts_path(cls, path: Path | str | None, **kw) -> LaneBPodRunner:
        host_fn = None
        if path is not None:
            from ..validator.eval_offload import make_eval_host_fn

            host_fn = make_eval_host_fn(Path(path), local_fallback=False)
        return cls(host_fn, **kw)

    def host(self):
        host = self._host_fn() if self._host_fn is not None else None
        if host is None:
            raise LaneBOffloadUnavailable(
                f"lane B: no eval pod configured/listed ({self.source}) — model work "
                f"never runs on {self.where}; retried next tick")
        return host

    @staticmethod
    def root_of(host) -> str:
        """``<cache_root or workdir>/_lane_b_offload`` on ``host``."""
        base = (getattr(host, "cache_root", "") or "").strip() or host.workdir
        return f"{base.rstrip('/') or '/'}/{OFFLOAD_DIR}"

    # transport (injectable)
    def _ssh(self, host, command: str, timeout: float, stdin_text: str = ""):
        from .remote import build_ssh_argv, run_ssh

        run = self._run or run_ssh
        return run(build_ssh_argv(host, command), timeout, stdin_text)

    def _put(self, host, local: Path, remote: str, timeout: float):
        if self._scp is not None:
            return self._scp(host, local, remote, timeout)
        from ..validator.eval_offload import build_scp_argv
        from .remote import run_ssh

        return (self._run or run_ssh)(build_scp_argv(host, str(local), remote), timeout, "")

    def _call(self, op: str, pointer: str, request: dict, contexts: list | None,
              payload: object = None, timeout_s: int | None = None, host=None) -> dict:
        import tempfile

        host = host if host is not None else self.host()
        deadline = time.monotonic() + int(timeout_s or self.timeout_s)
        q = shlex.quote
        root = self.root_of(host)
        run_dir = f"{root}/runs/{uuid.uuid4().hex}"
        sha, blob = self._bundle()
        code = f"{root}/code/{sha}"

        def left() -> float:
            r = deadline - time.monotonic()
            if r <= 0:
                raise LaneBOffloadUnavailable(f"lane B {op}: pod deadline expired")
            return r

        def ok(proc, what):
            if proc.returncode:
                tail = (getattr(proc, "stderr", "") or "")[-STDERR_TAIL:]
                log.error("lane B %s: %s failed on %s (exit %s): %s", op, what,
                          getattr(host, "name", host), proc.returncode, tail)
                raise LaneBOffloadUnavailable(f"lane B {op}: pod {what} failed")
            return proc

        try:
            with tempfile.TemporaryDirectory(prefix="cascade-laneb-offload-") as td:
                local = Path(td)
                (local / "request.json").write_text(json.dumps(
                    {**request, "op": op, "pointer": pointer,
                     "disk": {"headroom_bytes": self.headroom_bytes,
                              "size_fallback_bytes": self.size_fallback_bytes}}),
                    encoding="utf-8")
                if contexts is not None:
                    write_contexts(local / "contexts.npz", contexts)
                if payload is not None:
                    # box -> OUR pod only (the reply is JSON, never a pickle)
                    import pickle

                    (local / "payload.pkl").write_bytes(pickle.dumps(payload, protocol=4))
                have = ok(self._ssh(host, f"umask 077 && mkdir -p {q(run_dir)} && "
                                    f"{{ test -f {q(code + '/.ok')} && echo HAVE_CODE || true; }}",
                                    left()), "run dir setup")
                if "HAVE_CODE" not in (have.stdout or ""):
                    (local / "code.tgz").write_bytes(blob)
                    ok(self._put(host, local / "code.tgz", f"{run_dir}/code.tgz", left()),
                       "code upload")
                    part = f"{code}.part-{uuid.uuid4().hex[:8]}"
                    ok(self._ssh(host, f"mkdir -p {q(part)} && tar -xzf {q(run_dir + '/code.tgz')} "
                                 f"-C {q(part)} && touch {q(part + '/.ok')} && "
                                 f"{{ mv -T {q(part)} {q(code)} 2>/dev/null || rm -rf {q(part)}; }} "
                                 f"&& test -f {q(code + '/.ok')}", left()), "code unpack")
                ok(self._put(host, local / "request.json", f"{run_dir}/request.json", left()),
                   "request upload")
                if contexts is not None:
                    ok(self._put(host, local / "contexts.npz", f"{run_dir}/contexts.npz",
                                 left()), "contexts upload")
                if payload is not None:
                    ok(self._put(host, local / "payload.pkl", f"{run_dir}/payload.pkl",
                                 left()), "payload upload")
                names = _FORWARD_ENV + (_UPLOAD_FORWARD_ENV if op in UPLOAD_OPS else ())
                fwd = {k: os.environ[k] for k in names if os.environ.get(k)}
                src = "set -a && . /dev/stdin && set +a && " if fwd else ""
                stdin = "".join(f"{k}={q(v)}\n" for k, v in fwd.items())
                py = (f"env PYTHONPATH={q(code)} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
                      f"timeout --signal=TERM --kill-after=15s "
                      f"{max(1, int(left()) - 20)}s {host.remote_python} -P -m "
                      f"cascade.trainer.lane_b_offload {q(op)} {q(run_dir)} {q(root)}")
                cmd = (f"{src}cd {q(host.workdir)} && mkdir -p {q(root)} && "
                       f"flock -w {LOCK_WAIT_S} {q(root + '/.heavy.lock')} {py} && "
                       f"cat {q(run_dir + '/result.json')}")
                proc = ok(self._ssh(host, cmd, left(), stdin), "worker")
                try:
                    doc = pod_reply_json(proc.stdout)
                except ValueError as e:
                    raise LaneBOffloadUnavailable(f"lane B {op}: pod reply is not JSON") from e
                if isinstance(doc, dict) and "offload_unavailable" in doc:
                    # the pod refused the step itself (disk guard): transient
                    msg = str(doc["offload_unavailable"])
                    log.error("lane B %s: refused on %s: %s — nothing charged, retried next "
                              "tick", op, getattr(host, "name", host), msg)
                    raise LaneBOffloadUnavailable(f"lane B {op}: {msg}")
                return doc
        except LaneBOffloadUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 — any transport failure is transient
            raise LaneBOffloadUnavailable(f"lane B {op}: pod unavailable "
                                          f"({type(e).__name__}: {e})") from e
        finally:
            try:
                self._ssh(host, f"rm -rf {q(run_dir)}", 30)
            except Exception:  # noqa: BLE001
                log.warning("lane B %s: cleanup of %s on %s failed", op, run_dir,
                            getattr(host, "name", host))

    def probe(self, pointer: str, contexts: list, affinity_windows: int, target: str):
        """The pod's :func:`expert_lane.probe_target_routing` of ``pointer``."""
        doc = self._call("probe", pointer, {"target": str(target),
                                            "affinity_windows": int(affinity_windows)},
                         list(contexts))
        try:
            return decode_probe(doc)
        except (KeyError, TypeError, ValueError) as e:
            raise LaneBOffloadUnavailable(f"lane B probe: malformed pod reply ({e})") from e

    def lineage(self, pointer: str, binding: dict) -> int:
        """The pod's :func:`lineage.read_measurement` of ``pointer``. A bad
        measurement raises ``ValueError`` (as the local read did)."""
        doc = self._call("lineage", pointer, {"binding": binding}, None)
        if isinstance(doc, dict) and "error" in doc:
            raise ValueError(str(doc["error"]))
        n = doc.get("tokens_seen") if isinstance(doc, dict) else None
        if type(n) is not int:
            raise LaneBOffloadUnavailable("lane B lineage: malformed pod reply")
        return n


    def evaluate(self, pointer: str, windows: list, *, num_samples: int, contract,
                 quantile_head: bool, quantile_batch: int) -> list:
        """The pod's :func:`validator.evaluator.evaluate_checkpoint` of
        ``pointer`` on ``windows`` (device: :func:`pod_device`), decoded to
        the same ``WindowScore`` list."""
        doc = self._call("eval", pointer, {}, None, payload={
            "windows": list(windows), "num_samples": int(num_samples), "contract": contract,
            "quantile_head": bool(quantile_head), "quantile_batch": int(quantile_batch),
            "device": pod_device()}, timeout_s=EVAL_TIMEOUT_S)
        try:
            return decode_scores(doc)
        except (KeyError, TypeError, ValueError) as e:
            raise LaneBOffloadUnavailable(f"lane B eval: malformed pod reply ({e})") from e

    def merge_check(self, king_pointer: str, cand_pointer: str, selection: dict,
                    strict: bool) -> str | None:
        """The pod's :func:`validator.expert_lane.merge_check_dirs` of the two
        checkpoints: None (mergeable) or the refusal reason."""
        doc = self._call("merge_check", king_pointer, {"pointers": [king_pointer, cand_pointer]},
                         None, payload={"selection": selection, "strict": bool(strict)})
        if not isinstance(doc, dict) or "merge_error" not in doc:
            raise LaneBOffloadUnavailable("lane B merge check: malformed pod reply")
        err = doc["merge_error"]
        if err is not None and not isinstance(err, str):
            raise LaneBOffloadUnavailable("lane B merge check: malformed pod reply")
        return err

    def merge_verify(self, base_pointer: str, merges: list, target_pointer: str, *,
                     strict: bool) -> dict:
        """The pod's :func:`validator.expert_lane.merged_rebuild_check`: is
        ``target`` = ``base`` + ``merges``, tensor for tensor? Exactly one of
        ``{"equal": bool}``, ``{"merge_error": str}``, ``{"value_error": str}``
        (the validator's merged-init / acceptance verification)."""
        doc = self._call("merge_verify", base_pointer,
                         {"base": base_pointer, "merges": list(merges),
                          "target": target_pointer, "strict": bool(strict)},
                         None, timeout_s=MODEL_TIMEOUT_S)
        if isinstance(doc, dict) and len(doc) == 1:
            ((k, v),) = doc.items()
            if (k == "equal" and type(v) is bool) or (
                    k in ("merge_error", "value_error") and isinstance(v, str)):
                return {k: v}
        raise LaneBOffloadUnavailable("lane B merge verify: malformed pod reply")

    # ── settlement / era init (pod → Hub; only pointers come back) ───────────

    def _model_call(self, op: str, pointer: str, request: dict) -> dict:
        doc = self._call(op, pointer, request, None, timeout_s=MODEL_TIMEOUT_S)
        if not isinstance(doc, dict):
            raise LaneBOffloadUnavailable(f"lane B {op}: malformed pod reply")
        if "merge_error" in doc:
            from .lane_b_model import MergeRefused

            if not isinstance(doc["merge_error"], str):
                raise LaneBOffloadUnavailable(f"lane B {op}: malformed pod reply")
            raise MergeRefused(doc["merge_error"])
        return doc

    @staticmethod
    def _pushed(doc: dict, op: str) -> tuple[str, str | None]:
        ptr, key = doc.get("pointer"), doc.get("content_key")
        if not isinstance(ptr, str) or not ptr or (key is not None and not isinstance(key, str)):
            raise LaneBOffloadUnavailable(f"lane B {op}: malformed pod reply")
        return ptr, key

    def rebase(self, *, king_pointer: str, base_pointer: str, leg_pointer: str,
               selection: dict, refuse_nonfinite: bool, limits: bool, push: dict,
               bench_settings: dict | None) -> dict:
        """The pod's :meth:`LaneBModel.rebase` (pushed from the pod to the Hub
        as ``push``): ``{"actuals", "pointer" (None = limits record missing),
        "content_key"}``. A refused rebase raises ``MergeRefused``."""
        doc = self._model_call("rebase", king_pointer, {
            "king": king_pointer, "base": base_pointer, "leg": leg_pointer,
            "selection": {str(k): [int(x) for x in v] for k, v in selection.items()},
            "refuse_nonfinite": bool(refuse_nonfinite), "limits": bool(limits),
            "push": push, "bench_settings": bench_settings})
        actuals = doc.get("actuals")
        if not isinstance(actuals, dict):
            raise LaneBOffloadUnavailable("lane B rebase: malformed pod reply")
        if doc.get("pointer") is None:
            return {"actuals": actuals, "pointer": None, "content_key": None}
        ptr, key = self._pushed(doc, "rebase")
        return {"actuals": actuals, "pointer": ptr, "content_key": key}

    def merge_finite(self, init_pointer: str, merges: list[dict]) -> list[bool]:
        """The pod's :meth:`LaneBModel.finite` (one bool per merge)."""
        doc = self._model_call("merge_finite", init_pointer, {"merges": list(merges)})
        got = doc.get("finite")
        if not isinstance(got, list) or len(got) != len(merges) \
                or any(type(x) is not bool for x in got):
            raise LaneBOffloadUnavailable("lane B merge_finite: malformed pod reply")
        return got

    def merge_stage(self, init_pointer: str, merges: list[dict], *, strict: bool, push: dict,
                    bench_settings: dict | None) -> tuple[str, str | None]:
        """The pod's :meth:`LaneBModel.stage`: ``(pointer, content_key)``."""
        doc = self._model_call("merge_stage", init_pointer, {
            "merges": list(merges), "strict": bool(strict), "push": push,
            "bench_settings": bench_settings})
        return self._pushed(doc, "merge_stage")

    def merge_build(self, init_pointer: str, chosen: list[dict], *, strict: bool, push: dict,
                    era_index: int) -> str:
        """The pod's :meth:`LaneBModel.build`: the merged init's pointer."""
        doc = self._model_call("merge_build", init_pointer, {
            "merges": list(chosen), "strict": bool(strict), "push": push,
            "era_index": int(era_index)})
        return self._pushed(doc, "merge_build")[0]

    def bench_stage(self, pointer: str, settings: dict) -> tuple[object, str, str]:
        """``(host, content_key, pinned dir)``: ``pointer`` fetched on the pod,
        its bench content key (:func:`lane_b_bench.content_key`, the box's
        old key) and its bench files hard-linked into a pinned dir on that
        host for the sidecar. The caller removes the pin (``unpin``)."""
        host = self.host()
        pin = f"{self.root_of(host)}/pinned/{uuid.uuid4().hex}"
        doc = self._call("bench_stage", pointer, {"settings": settings, "pin": pin}, None,
                         timeout_s=MODEL_TIMEOUT_S, host=host)
        key = doc.get("content_key") if isinstance(doc, dict) else None
        if not isinstance(key, str) or not key or doc.get("pin") != pin:
            self.unpin(host, pin)
            raise LaneBOffloadUnavailable("lane B bench_stage: malformed pod reply")
        return host, key, pin

    def unpin(self, host, pin: str) -> None:
        if f"/{OFFLOAD_DIR}/pinned/" not in pin:
            return
        try:
            self._ssh(host, f"rm -rf {shlex.quote(pin)}", 30)
        except Exception:  # noqa: BLE001 — the pod prunes stale pins itself
            log.warning("lane B: removing the bench pin %s on %s failed", pin,
                        getattr(host, "name", host))


class PodCkpt:
    """A checkpoint that exists only on the Hub / the eval pod: its pointer
    and (when the pod computed it) its bench content key — the bench handle
    of :class:`PodLaneBModel`."""

    __slots__ = ("pointer", "content_key")

    def __init__(self, pointer: str, content_key: str | None):
        self.pointer, self.content_key = pointer, content_key

    def __repr__(self) -> str:
        return f"PodCkpt({self.pointer!r}, {self.content_key!r})"


class PodLaneBModel:
    """The driver's model object on the eval pod (the trainer's production
    path; :class:`cascade.trainer.lane_b_model.LaneBModel` interface). Every
    checkpoint read, rebase, merge and push is a pod step; the box holds only
    pointers. ``push_target(repo) -> {"full", "hf_repo", "registry_url",
    "namespace"}`` names where the pod pushes; ``bench`` is the lane-B bench
    (pointer path) or None (no rows: settlement fails closed, as before)."""

    def __init__(self, runner: LaneBPodRunner, bench, push_target: Callable[[str], dict]):
        self.runner, self._bench, self._push = runner, bench, push_target

    def _settings(self) -> dict | None:
        return getattr(self._bench, "settings", None) if self._bench is not None else None

    def rows(self, pointer: str) -> dict | None:
        if self._bench is None:
            log.error("lane B bench: no --lane-b-bench-hosts configured — lane-B public rows "
                      "are never benched on the trainer box; settlement fails closed")
            return None
        return self._bench.rows_remote(pointer)

    def bench(self, handle) -> dict | None:
        if handle is None:
            return None
        if self._bench is None:
            return self.rows(handle.pointer)
        return self._bench.rows_remote(handle.pointer, key=handle.content_key)

    def rebase(self, *, king_pointer, base_pointer, leg_pointer, selection, out_dir, repo,
               refuse_nonfinite, limits):
        from .lane_b_model import RebaseResult

        got = self.runner.rebase(king_pointer=king_pointer, base_pointer=base_pointer,
                                 leg_pointer=leg_pointer, selection=selection,
                                 refuse_nonfinite=refuse_nonfinite, limits=limits,
                                 push=self._push(repo), bench_settings=self._settings())
        if got["pointer"] is None:
            return RebaseResult(dict(got["actuals"]), None, None)
        return RebaseResult(dict(got["actuals"]), got["pointer"],
                            PodCkpt(got["pointer"], got["content_key"]))

    def finite(self, init_pointer, merges):
        return self.runner.merge_finite(init_pointer, merges)

    def stage(self, init_pointer, merges, *, out_dir, repo, strict):
        ptr, key = self.runner.merge_stage(init_pointer, merges, strict=strict,
                                           push=self._push(repo),
                                           bench_settings=self._settings())
        return ptr, PodCkpt(ptr, key)

    def build(self, init_pointer, chosen, *, out_dir, repo, strict, era_index):
        return self.runner.merge_build(init_pointer, chosen, strict=strict,
                                       push=self._push(repo), era_index=era_index)


EVAL_TIMEOUT_S = 3 * 3600
POD_DEVICE_ENV = "CASCADE_LANE_B_POD_DEVICE"
SCORES_SCHEMA = "lane-b-window-scores/1"


def pod_device() -> str:
    """The pod-side eval device: ``cpu`` (the device the box path used, so the
    scores are identical) unless ``CASCADE_LANE_B_POD_DEVICE`` says otherwise."""
    return os.environ.get(POD_DEVICE_ENV, "").strip() or "cpu"


def _num(x) -> float:
    return float(x)


def encode_scores(scores: list) -> dict:
    """``WindowScore`` list as JSON: floats by ``float.hex`` (exact, NaN/inf
    safe), arrays as raw bytes."""
    rows = []
    for s in scores:
        rows.append({"series_id": str(s.series_id), "mase": _num(s.mase).hex(),
                     "qloss_per_q": _enc_array(np.asarray(s.qloss_per_q)),
                     "abs_target": _num(s.abs_target).hex(),
                     "quantile_levels": [_num(q).hex() for q in s.quantile_levels],
                     "channel": int(s.channel), "domain": str(s.domain),
                     "source": None if s.source is None else str(s.source)})
    return {"schema": SCORES_SCHEMA, "scores": rows}


def decode_scores(doc: dict) -> list:
    from ..eval.scoring import WindowScore

    if not isinstance(doc, dict) or doc.get("schema") != SCORES_SCHEMA:
        raise ValueError("lane B scores reply: unknown schema")
    out = []
    for r in doc["scores"]:
        src = r["source"]
        if src is not None and not isinstance(src, str):
            raise ValueError("lane B scores reply: bad source")
        out.append(WindowScore(
            series_id=str(r["series_id"]), mase=float.fromhex(r["mase"]),
            qloss_per_q=_dec_array(r["qloss_per_q"]), abs_target=float.fromhex(r["abs_target"]),
            quantile_levels=tuple(float.fromhex(q) for q in r["quantile_levels"]),
            channel=int(r["channel"]), domain=str(r["domain"]), source=src))
    return out


# ── pod side ─────────────────────────────────────────────────────────────────


class PodDiskShort(OSError):
    """Pod side: the cache filesystem cannot take a pull even after evicting
    every unneeded ``_lane_b_offload/ckpt`` entry. Reported to the box as
    ``offload_unavailable`` (transient: :class:`LaneBOffloadUnavailable`)."""


def _entry_name(digest_name: str, profile: str) -> str:
    # FULL keeps the old name (existing cache entries stay valid)
    return digest_name if profile == FETCH_FULL else f"{digest_name}~{profile}"


def _complete(d: Path) -> bool:
    # marker AND no chunked-v2 index left as a file (an older fetch's
    # marker over one is incomplete: refetched, which reassembles it)
    from ..shared.hippius import snapshot_complete

    return snapshot_complete(d)


def _cached(ckpt: Path, digest_name: str, profile: str) -> Path | None:
    """A complete cache entry of ``profile`` or any stronger one."""
    for prof in _PROFILES[: _PROFILES.index(profile) + 1]:
        d = ckpt / _entry_name(digest_name, prof)
        if _complete(d):
            return d
    return None


def _ensure_room(ckpt: Path, need: int, protect: set, *, what: str,
                 disk_usage: Callable | None = None) -> None:
    """Free ``need`` bytes on ``ckpt``'s filesystem: if short, evict LRU
    entries of THIS cache dir (``_lane_b_offload/ckpt``) that are not in
    ``protect`` (the step's own checkpoints), stale fetch temp dirs first;
    still short ⇒ :class:`PodDiskShort`. Never touches anything else."""
    import shutil

    du = disk_usage or shutil.disk_usage
    ckpt.mkdir(parents=True, exist_ok=True)
    free = int(du(str(ckpt)).free)
    if free >= need:
        return
    now = time.time()
    held = {Path(p).resolve() for p in protect}

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return now

    stale_tmp, entries = [], []
    for p in ckpt.iterdir():
        if not p.is_dir() or p.is_symlink() or p.resolve() in held \
                or p.resolve().parent != ckpt.resolve():
            continue
        if p.name.startswith("."):
            if ".fetch-" in p.name and now - mtime(p) > FETCH_TMP_STALE_S:
                stale_tmp.append(p)
            continue
        entries.append(p)
    for p in stale_tmp + sorted(entries, key=mtime):
        log.warning("lane B pod cache: evicting %s for %s (free %.1f GB < need %.1f GB)",
                    p.name, what, free / 1e9, need / 1e9)
        shutil.rmtree(p, ignore_errors=True)
        free = int(du(str(ckpt)).free)
        if free >= need:
            return
    msg = (f"pod disk short for {what}: {free / 1e9:.1f} GB free on the lane B cache "
           f"filesystem < {need / 1e9:.1f} GB needed (pull + headroom) after evicting every "
           f"unneeded {OFFLOAD_DIR}/ckpt entry")
    log.error("lane B: %s", msg)
    raise PodDiskShort(msg)


def _pod_fetch(pointer: str, root: Path, keep: set | None = None, *,
               profile: str = FETCH_FULL, disk: dict | None = None,
               size_fn: Callable | None = None, disk_usage: Callable | None = None) -> Path:
    """``pointer`` in this module's pod cache, fetching only ``profile``'s
    files (see :data:`LINEAGE_FILES`). ``keep`` (one pod step's own
    fetches) is never evicted: a merge step holds more than :data:`CKPT_KEEP`
    checkpoints at once (steps are serialised by the heavy flock). Before a
    pull the disk guard (:func:`_ensure_room`) checks the cache filesystem
    for the pull's size plus ``disk["headroom_bytes"]``."""
    import shutil

    from ..shared import hippius
    from ..shared.manifest import parse_trained_pointer

    if profile not in _PROFILE_PATTERNS:
        raise ValueError(f"unknown fetch profile {profile!r}")
    ref = parse_trained_pointer(pointer) or pointer
    digest_name = hippius.HubRef.parse(ref).digest.replace(":", "-")
    ckpt = root / "ckpt"
    held = set(keep or ())
    dest = _cached(ckpt, digest_name, profile)
    if dest is None:
        dest = ckpt / _entry_name(digest_name, profile)
        allow, ignore = _PROFILE_PATTERNS[profile]
        disk = disk or {}
        headroom = int(disk.get("headroom_bytes", DISK_HEADROOM_BYTES))
        size = (size_fn or hippius.hub_snapshot_bytes)(
            ref, allow_patterns=allow, ignore_patterns=ignore)
        if size is None:
            size = int(disk.get("size_fallback_bytes", SIZE_FALLBACK_BYTES))
            log.warning("lane B pod cache: size of %s unknown — guarding with %.1f GB",
                        digest_name[-16:], size / 1e9)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)     # an incomplete leftover
        _ensure_room(ckpt, int(size) + headroom, held, what=f"{digest_name[-16:]} ({profile})",
                     disk_usage=disk_usage)
        hippius.fetch_from_hub(ref, dest, allow_patterns=allow, ignore_patterns=ignore)
        if profile == FETCH_FULL:
            # the lighter copies of the same digest are now redundant
            for prof in _PROFILES[1:]:
                p = ckpt / _entry_name(digest_name, prof)
                if p.is_dir() and p not in held:
                    shutil.rmtree(p, ignore_errors=True)
    os.utime(dest)
    if keep is not None:
        keep.add(dest)
    # bounded: evict only this module's own cache entries (never hidden temp dirs)
    entries = sorted((p for p in ckpt.iterdir()
                      if p.is_dir() and not p.name.startswith(".") and p != dest
                      and p not in held),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for p in entries[CKPT_KEEP - 1:]:
        shutil.rmtree(p, ignore_errors=True)
    return dest


def _pod_prune_code(root: Path, keep: Path) -> None:
    import shutil

    code = root / "code"
    entries = sorted((p for p in code.iterdir() if p.is_dir() and ".part-" not in p.name
                      and p != keep), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in entries[CODE_KEEP - 1:]:
        shutil.rmtree(p, ignore_errors=True)


def _pod_push(push: dict) -> Callable[[Path, str], str]:
    """The pod's Hub push for one step: exactly ``RunnerLaneBOps.upload``
    (``upload_dir_to_hub_or_hf`` to ``push["full"]``, HF fallback
    ``push["hf_repo"]``), credentials from the stdin-sourced env."""
    def upload(local: Path, repo: str) -> str:
        from ..shared.hippius import HubConfig, upload_dir_to_hub_or_hf
        from ..shared.manifest import format_trained_pointer

        full = str(push["full"])
        if full.rsplit("/", 1)[-1] != str(repo):
            raise ValueError(f"push target {full!r} does not name repo {repo!r}")
        hub = HubConfig(registry_url=str(push.get("registry_url") or HubConfig.registry_url),
                        namespace=str(push.get("namespace") or HubConfig.namespace))
        up = upload_dir_to_hub_or_hf(Path(local), full, hub,
                                     hf_repo=push.get("hf_repo") or None)
        return format_trained_pointer(up.ref.immutable_ref)

    return upload


def _content_key(d: Path, settings: dict | None) -> str | None:
    if settings is None:
        return None
    from .lane_b_bench import content_key

    return content_key(Path(d), settings)


def _pin_bench_files(src: Path, pin: Path) -> None:
    """Hard-link (copy across filesystems) the bench files into ``pin``: a
    later cache eviction of ``src`` cannot pull them from under a sweep."""
    import shutil

    from ..validator.eval_offload import bench_upload_files

    files = bench_upload_files(src)
    if not files:
        raise FileNotFoundError(f"{src}: no bench files")
    part = pin.with_name(pin.name + ".part")
    part.mkdir(parents=True, exist_ok=False)
    for f in files:
        try:
            os.link(f, part / f.name)
        except OSError:
            shutil.copy2(f, part / f.name)
    part.replace(pin)


def _pod_prune_pins(root: Path) -> None:
    import shutil

    pins = root / "pinned"
    if not pins.is_dir():
        return
    now = time.time()
    for p in pins.iterdir():
        try:
            if now - p.stat().st_mtime > PIN_STALE_S:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _pod_model_op(op: str, req: dict, run_dir: Path, root: Path) -> dict:
    """The settlement / era-init / bench-stage steps (pod side)."""
    from .lane_b_model import LaneBModel, MergeRefused

    keep: set = set()
    disk = req.get("disk") or {}

    def fetch(p: str) -> Path:
        # every read but merge_build's init needs weights + config/code only
        return _pod_fetch(p, root, keep=keep, profile=FETCH_WEIGHTS, disk=disk)

    def fetch_full(p: str) -> Path:
        return _pod_fetch(p, root, keep=keep, profile=FETCH_FULL, disk=disk)

    push = _pod_push(req["push"]) if op in UPLOAD_OPS else None

    def no_push(local, repo):
        raise RuntimeError(f"lane B {op}: this step does not push")

    model = LaneBModel(fetch, push or no_push, fetch_full=fetch_full)
    settings = req.get("bench_settings")
    if op == "rebase":
        sel = {int(k): [int(x) for x in v] for k, v in req["selection"].items()}
        repo = str(req["push"]["full"]).rsplit("/", 1)[-1]
        try:
            res = model.rebase(king_pointer=req["king"], base_pointer=req["base"],
                               leg_pointer=req["leg"], selection=sel,
                               out_dir=run_dir / "rebased", repo=repo,
                               refuse_nonfinite=bool(req["refuse_nonfinite"]),
                               limits=bool(req["limits"]))
        except MergeRefused as e:
            return {"merge_error": str(e)}
        if res.pointer is None:
            return {"actuals": dict(res.actuals), "pointer": None}
        return {"actuals": dict(res.actuals), "pointer": res.pointer,
                "content_key": _content_key(res.bench, settings)}
    if op == "merge_finite":
        return {"finite": [bool(x) for x in model.finite(req["pointer"], req["merges"])]}
    if op == "merge_stage":
        repo = str(req["push"]["full"]).rsplit("/", 1)[-1]
        try:
            ptr, stage = model.stage(req["pointer"], req["merges"], out_dir=run_dir / "stage",
                                     repo=repo, strict=bool(req["strict"]))
        except MergeRefused as e:
            return {"merge_error": str(e)}
        return {"pointer": ptr, "content_key": _content_key(stage, settings)}
    if op == "merge_build":
        from .moe_model import MergeError

        repo = str(req["push"]["full"]).rsplit("/", 1)[-1]
        try:
            ptr = model.build(req["pointer"], req["merges"], out_dir=run_dir / "merged",
                              repo=repo, strict=bool(req["strict"]),
                              era_index=int(req["era_index"]))
        except MergeError as e:
            return {"merge_error": str(e)}
        return {"pointer": ptr}
    if op == "bench_stage":
        d = fetch(req["pointer"])
        key = _content_key(d, req["settings"])
        pin = Path(req["pin"])
        if pin.parent != root / "pinned":
            raise SystemExit("bench pin outside this module's dir")
        _pin_bench_files(d, pin)
        return {"content_key": key, "pin": str(pin)}
    raise SystemExit(f"unknown op {op!r}")


MODEL_OPS = frozenset({"rebase", "merge_finite", "merge_stage", "merge_build", "bench_stage"})


def pod_main(argv: list[str] | None = None) -> int:
    """``python -m cascade.trainer.lane_b_offload <op> <run_dir> <root>`` (pod)."""
    import sys

    op, run_dir, root = (argv if argv is not None else sys.argv[1:])[:3]
    run_dir, root = Path(run_dir), Path(root)
    req = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    if req.get("op") != op:
        raise SystemExit("request op mismatch")
    import contextlib

    with contextlib.suppress(OSError):
        _pod_prune_code(root, Path(__file__).resolve().parents[2])
    with contextlib.suppress(OSError):
        _pod_prune_pins(root)
    try:
        out = _pod_op(op, req, run_dir, root)
    except PodDiskShort as e:
        out = {"offload_unavailable": str(e)}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(run_dir / "result.json")
    return 0


# what the non-model ops read of their checkpoints (see LINEAGE_FILES)
OP_PROFILE = {"probe": FETCH_WEIGHTS, "lineage": FETCH_META, "eval": FETCH_WEIGHTS,
              "merge_check": FETCH_WEIGHTS}


def _pod_op(op: str, req: dict, run_dir: Path, root: Path) -> dict:
    from types import SimpleNamespace

    if op in MODEL_OPS:
        return _pod_model_op(op, req, run_dir, root)
    if op == "merge_verify":
        from ..validator.expert_lane import merged_rebuild_check

        keep: set = set()
        return merged_rebuild_check(
            lambda p: _pod_fetch(p, root, keep=keep, profile=FETCH_WEIGHTS,
                                 disk=req.get("disk")),
            req["base"], list(req["merges"]), req["target"], strict=bool(req["strict"]))
    if op not in OP_PROFILE:
        raise SystemExit(f"unknown op {op!r}")
    pointers = list(req.get("pointers") or [req["pointer"]])
    keep: set = set()
    dirs = [_pod_fetch(p, root, keep=keep, profile=OP_PROFILE[op], disk=req.get("disk"))
            for p in pointers]
    ckpt = dirs[0]
    payload = None
    if (run_dir / "payload.pkl").is_file():
        import pickle

        payload = pickle.loads((run_dir / "payload.pkl").read_bytes())  # noqa: S301 — from our box
    if op == "probe":
        from .expert_lane import probe_target_routing

        probe = probe_target_routing(ckpt, read_contexts(run_dir / "contexts.npz"),
                                     SimpleNamespace(affinity_windows=int(req["affinity_windows"])),
                                     req["target"])
        out = encode_probe(probe)
    elif op == "lineage":
        from .lineage import read_measurement

        try:
            out = {"tokens_seen": read_measurement(ckpt, req["binding"])}
        except (OSError, ValueError, KeyError, TypeError) as e:
            out = {"error": f"{type(e).__name__}: {e}"}
    elif op == "eval":
        from ..validator.evaluator import evaluate_checkpoint

        out = encode_scores(evaluate_checkpoint(
            ckpt, payload["windows"], num_samples=payload["num_samples"],
            device=payload["device"], contract=payload["contract"],
            quantile_head=payload["quantile_head"], quantile_batch=payload["quantile_batch"]))
    elif op == "merge_check":
        from ..validator.expert_lane import merge_check_dirs

        out = {"merge_error": merge_check_dirs(dirs[0], dirs[1], payload["selection"],
                                               strict=payload["strict"])}
    else:
        raise SystemExit(f"unknown op {op!r}")
    return out


if __name__ == "__main__":
    raise SystemExit(pod_main())


def pod_reply_json(stdout: str):
    """The worker's ``result.json``, which the pod command ``cat``s LAST.

    2026-11-27: the 19:27 lane B settlement rebase failed "pod reply is not
    JSON" because ``hippius_hub.file_download`` prints ``Downloading <file>
    (parallel)...`` to stdout while the step pulls an uncached checkpoint,
    ahead of the JSON. Parse the whole reply, else its last non-empty line
    (``json.dumps`` writes ``result.json`` on one line); raise ValueError when
    neither is JSON."""
    text = stdout or ""
    try:
        return json.loads(text)
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return json.loads(line)
    raise ValueError("empty pod reply")
