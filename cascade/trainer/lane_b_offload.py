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
                 timeout_s: int = PROBE_TIMEOUT_S, bundle: Callable[[], tuple] | None = None):
        self._host_fn = host_fn
        self._run = run
        self._scp = scp
        self.timeout_s = int(timeout_s)
        self._bundle = bundle or code_bundle

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
                "lane B: no eval pod configured/listed (--lane-b-bench-hosts) — model work "
                "never runs on the trainer box; retried next tick")
        return host

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
              payload: object = None, timeout_s: int | None = None) -> dict:
        import tempfile

        host = self.host()
        deadline = time.monotonic() + int(timeout_s or self.timeout_s)
        q = shlex.quote
        root = f"{host.workdir}/{OFFLOAD_DIR}"
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
                    {**request, "op": op, "pointer": pointer}), encoding="utf-8")
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
                fwd = {k: os.environ[k] for k in _FORWARD_ENV if os.environ.get(k)}
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
                    return json.loads(proc.stdout)
                except ValueError as e:
                    raise LaneBOffloadUnavailable(f"lane B {op}: pod reply is not JSON") from e
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


def _pod_fetch(pointer: str, root: Path) -> Path:
    from ..shared.hippius import HubRef, fetch_from_hub
    from ..shared.manifest import parse_trained_pointer

    ref = parse_trained_pointer(pointer) or pointer
    dest = root / "ckpt" / HubRef.parse(ref).digest.replace(":", "-")
    fetch_from_hub(ref, dest)
    os.utime(dest)
    # bounded: evict only this module's own cache entries (never hidden temp dirs)
    entries = sorted((p for p in (root / "ckpt").iterdir()
                      if p.is_dir() and not p.name.startswith(".") and p != dest),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    import shutil

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


def pod_main(argv: list[str] | None = None) -> int:
    """``python -m cascade.trainer.lane_b_offload <op> <run_dir> <root>`` (pod)."""
    import sys
    from types import SimpleNamespace

    op, run_dir, root = (argv if argv is not None else sys.argv[1:])[:3]
    run_dir, root = Path(run_dir), Path(root)
    req = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    if req.get("op") != op:
        raise SystemExit("request op mismatch")
    import contextlib

    with contextlib.suppress(OSError):
        _pod_prune_code(root, Path(__file__).resolve().parents[2])
    pointers = list(req.get("pointers") or [req["pointer"]])
    dirs = [_pod_fetch(p, root) for p in pointers]
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
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(run_dir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(pod_main())
