"""MarketCast GPU offload (DEC-CA-0064): the MAIN validator's MarketCast
forecasts on a cheap elastic pod instead of its own ``--device``.

* **Lease.** ``--mkt-eval-hosts PATH`` names the hosts file the provisioner's
  ``[provisioner.mkt]`` stage publishes. While the validator has rows to
  compute it keeps ``PATH.request`` touched (:class:`MktLease`); the
  provisioner rents a pod when that request is live, publishes it to
  ``PATH``, and tears it down once the request is gone. No pod within the
  wait ⇒ the round is simply not published yet (retried at the next
  settlement — never an infra deferral, never a miner's loss).
* **Session.** One ssh process per publish run runs
  :mod:`cascade.validator.mkt_worker`; checkpoints (the pinned base, each
  fine-tune) are uploaded once into a content-addressed cache on the pod and
  sha256-verified there before every load. The cache lives on the pod's
  large data mount (2026-11-27, as lane B's #982): the host's ``cache_root``
  roots it, else ``<workdir>/_mkt_offload`` is linked onto ``/workspace``
  (:mod:`cascade.shared.pod_data_root`); a disk-short upload names the
  alternative mount. :class:`RemoteForecaster` stands
  in for :class:`~cascade.eval.timestral_adapter.TimestralQuantileForecaster`
  (``quantile_levels`` + ``forecast_batch``), so the rows code is unchanged.
  Contexts and covariates cross; window targets, losses, the verdict, the
  report and the wallet stay on the validator.
* **Judge (W-MKTOFF, 2026-11-28).** The window cut, the covariate rows, the
  scoring and the reference tests of every entry run ON THE POD
  (:class:`PodJudge` → the worker's ``judge_*`` ops): the 10:54 OOM was the
  box building ~16 GB of covariate rows for one ``mkt:market`` entry. The
  snapshot is uploaded once per digest (``snap-<digest>`` in the cache) and
  the pod's view of it is bound to the box's by digest + meta sha256; the
  ``[marketcast]`` table crosses as JSON and is bound by sha256; each entry
  comes back as its rows + its sidecar bytes (sha256-checked). The report,
  its signature and the wallet stay on the validator.
* **Errors.** A transport failure is :class:`MktOffloadError` (a
  ``ConnectionError`` — infra to :func:`~cascade.validator.marketcast.
  infra_like`); the worker's own exceptions come back classified (a
  ``ValueError`` stays one, an infra-shaped one is infra, anything else a
  ``RuntimeError`` the rows code treats like a local forward error).
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import logging
import os
import selectors
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from ..shared.pod_data_root import (
    alternative_mounts_command,
    alternatives_text,
    base_dir,
    parse_alternatives,
)
from .mkt_worker import (
    _LEN,
    FRAME_MAX,
    SCHEMA,
    MktRemoteError,
    decode_frame,
    encode_frame,
    mc_from_wire,
    mc_to_wire,
    rows_from_wire,
    rows_to_wire,
    snapshot_meta_sha256,
    wire_sha256,
)

log = logging.getLogger("cascade.validator.mkt_offload")

OFFLOAD_DIR = "_mkt_offload"
CACHE_DIR = f"{OFFLOAD_DIR}/cache"
CACHE_KEEP = 8
CKPT_FILES = ("config.json", "model.safetensors")
CALL_TIMEOUT_S = 600.0
UPLOAD_TIMEOUT_S = 1800.0
TOUCH_EVERY_S = 60.0
# free bytes an upload needs beyond the checkpoint itself
UPLOAD_HEADROOM_BYTES = 1 << 30
# snapshots in the pod cache are ``snap-<digest>``: never pruned as checkpoints,
# the newest SNAP_KEEP kept
SNAP_PREFIX = "snap-"
SNAP_KEEP = 2
SNAP_MARKER = ".cascade_snapshot"


class MktOffloadError(ConnectionError):
    """The MarketCast pod or its transport failed (infrastructure)."""


def request_path_for(hosts_path, owner: str = "") -> Path:
    """The request file the validator keeps touched while it needs the pod:
    ``<hosts>.request`` for MarketCast, ``<hosts>.<owner>.request`` for any
    other lane-C competition sharing the same pod (the provisioner keeps the
    pod while ANY of them is fresh, so one releasing never tears it down
    under another)."""
    return Path(str(hosts_path) + (f".{owner}" if owner else "") + ".request")


def request_paths(hosts_path) -> list[Path]:
    """Every request file of ``hosts_path`` (all owners)."""
    h = Path(hosts_path)
    return sorted({request_path_for(h), *h.parent.glob(glob.escape(h.name) + ".*.request")})


# ── the lease (validator ↔ provisioner, through two local files) ─────────────


class MktLease:
    """Keep ``<hosts>.request`` touched while rows are being computed; wait for
    the provisioner to publish a host into ``hosts``."""

    def __init__(self, hosts_path, *, host_fn=None, clock=time.monotonic, sleep=time.sleep,
                 owner: str = ""):
        from .eval_offload import make_eval_host_fn

        self.hosts_path = Path(hosts_path)
        self.request = request_path_for(self.hosts_path, owner)
        self.host_fn = host_fn or make_eval_host_fn(self.hosts_path, local_fallback=False)
        self.clock, self.sleep = clock, sleep
        self._touched = -1e18

    def touch(self, rid: str = "", *, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self._touched < TOUCH_EVERY_S:
            return
        self.request.parent.mkdir(parents=True, exist_ok=True)
        # per thread: the #908 heartbeat thread and the round loop both touch
        tmp = self.request.with_name(
            self.request.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
        tmp.write_text(f'{{"round_id": "{rid}", "pid": {os.getpid()}}}\n', encoding="utf-8")
        tmp.replace(self.request)
        self._touched = now

    def acquire(self, rid: str, wait_s: float, poll_s: float = 15.0):
        """The published host (``RemoteHost``), or None after ``wait_s``."""
        deadline = self.clock() + float(wait_s)
        self.touch(rid, force=True)
        while True:
            host = self.host_fn()
            if host is not None:
                return host
            if self.clock() >= deadline:
                return None
            self.touch(rid)
            self.sleep(poll_s)

    @contextlib.contextmanager
    def heartbeat(self, rid: str = "", *, every_s: float = TOUCH_EVERY_S):
        """Keep the request fresh from a daemon thread while the body runs: a
        single round can compute for longer than the provisioner's
        ``mkt_request_stale_s``, and a lease touched only between rounds was
        torn down mid-forecast (#908)."""
        stop = threading.Event()

        def beat() -> None:
            while not stop.wait(float(every_s)):
                with contextlib.suppress(OSError):
                    self.touch(rid, force=True)

        t = threading.Thread(target=beat, name="mkt-lease-heartbeat", daemon=True)
        t.start()
        try:
            yield self
        finally:
            stop.set()
            t.join(timeout=5.0)

    def release(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.request.unlink()


# ── the session ──────────────────────────────────────────────────────────────


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hashes(d: Path) -> dict:
    return {n: _file_sha256(Path(d) / n) for n in CKPT_FILES}


class RemoteForecaster:
    """A Timestral forecaster that runs on the pod (the same two members the
    rows code reads)."""

    def __init__(self, session: MktSession, model: str, levels: tuple[float, ...]):
        self.session, self.model = session, str(model)
        self.quantile_levels = tuple(float(x) for x in levels)

    def forecast_batch(self, contexts, horizon: int, *, past_covariates=None,
                       future_covariates=None) -> np.ndarray:
        c = np.asarray(contexts, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError(f"contexts must be (B, T); got {c.shape}")
        _h, arr = self.session.call(
            {"op": "forecast", "model": self.model, "horizon": int(horizon)},
            {"ctx": c,
             "pc": None if past_covariates is None else np.asarray(past_covariates, np.float64),
             "kc": None if future_covariates is None else np.asarray(future_covariates,
                                                                     np.float64)})
        q = arr.get("q")
        if q is None or q.shape[:2] != (c.shape[0], int(horizon)) \
                or q.shape[2] != len(self.quantile_levels):
            raise MktOffloadError(f"forecast reply shape {None if q is None else q.shape} "
                                  f"!= ({c.shape[0]}, {int(horizon)}, "
                                  f"{len(self.quantile_levels)})")
        return np.asarray(q, dtype=np.float64)

    def forecast_quantiles(self, context, horizon: int, *, past_covariates=None,
                           future_covariates=None) -> np.ndarray:
        pc = None if past_covariates is None else np.asarray(past_covariates)[None]
        kc = None if future_covariates is None else np.asarray(future_covariates)[None]
        return self.forecast_batch(np.asarray(context, dtype=np.float64)[None], horizon,
                                   past_covariates=pc, future_covariates=kc)[0]


def cache_root_of(host) -> str:
    """The pod's MarketCast checkpoint cache (the worker's ``--root``):
    ``<cache_root>/_mkt_offload/cache`` when the host names a ``cache_root``,
    else ``<workdir>/_mkt_offload/cache`` (linked onto the data mount by
    :meth:`MktSession.start`)."""
    return f"{base_dir(host)}/{CACHE_DIR}"


def cache_setup_command(cache_root: str, key: str) -> str:
    """Make the cache, prune it to the most recently used entries plus
    ``key`` (and stale partials), then print ``key``'s sha256sums if cached,
    the data mounts on another filesystem (``CASCADE_ALT=``) and the cache
    filesystem's free bytes (``CASCADE_FREE=``)."""
    from .eval_offload import _FREE_TAG

    q = shlex.quote
    files = " ".join(q(f) for f in CKPT_FILES)
    prune = (f"cd {q(cache_root)} && "
             "find . -maxdepth 1 -name '*.part-*' -mmin +360 -exec rm -rf {} + ; "
             f"ls -1t | grep -v -e '[.]part-' -e {q('^' + SNAP_PREFIX)} -e {q('^' + key + '$')} "
             f"| tail -n +{CACHE_KEEP} | xargs -r rm -rf --")
    return (f"umask 077 && mkdir -p {q(cache_root)} && {{ ( {prune} ) || true; }} && "
            f"{{ cd {q(cache_root + '/' + key)} 2>/dev/null && touch . && "
            f"sha256sum -- {files} 2>/dev/null || true; }}; "
            f"{alternative_mounts_command(cache_root)}; "
            f"echo {_FREE_TAG}$(df -PB1 {q(cache_root)} 2>/dev/null | awk 'NR==2{{print $4}}')")


class MktSession:
    """One live :mod:`~cascade.validator.mkt_worker` on ``host`` over ssh."""

    def __init__(self, host, *, run=None, popen=subprocess.Popen,
                 call_timeout_s: float = CALL_TIMEOUT_S, device: str = "cuda"):
        from ..trainer.remote import run_ssh

        self.host, self.run, self.popen = host, run or run_ssh, popen
        self.call_timeout_s, self.device = float(call_timeout_s), str(device)
        self.proc = None
        self._err = None
        self._models: dict[str, RemoteForecaster] = {}
        self.calls = 0
        self.hello: dict = {}

    # transport

    def start(self) -> dict:
        from ..trainer.remote import build_ssh_argv
        from .eval_offload import _placement

        h = self.host
        env = ("OMP_NUM_THREADS=2 "
               + (f"CUDA_VISIBLE_DEVICES={shlex.quote(h.cuda_device)} "
                  if getattr(h, "cuda_device", None) else ""))
        # 2026-11-27: the checkpoint cache sits on the pod's large data mount
        # (the root overlay may be 43 GB beside an empty /workspace): placed
        # BEFORE the worker resolves its --root, so the worker's containment
        # check runs on the real location
        root = cache_root_of(h)
        # retro P4: the pod's cascade/ is this release before the worker starts
        from ..shared.release_fp import ReleaseSyncFailed, ReleaseUnreachable, ensure_release

        try:
            ensure_release(h, role="marketcast")
        except (ReleaseSyncFailed, ReleaseUnreachable) as e:
            raise MktOffloadError(f"{h.name}: {e}") from e
        cmd = (f"{_placement(h, f'{base_dir(h)}/{OFFLOAD_DIR}')}"
               f"mkdir -p {shlex.quote(root)} && "
               f"cd {shlex.quote(h.workdir)} && {env}exec {shlex.quote(h.remote_python)} -m "
               f"cascade.validator.mkt_worker --serve --device {shlex.quote(self.device)} "
               f"--root {shlex.quote(root)}")
        self._err = tempfile.TemporaryFile()  # noqa: SIM115 — lives as long as the session
        try:
            self.proc = self.popen(build_ssh_argv(h, cmd), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=self._err)
        except OSError as e:
            raise MktOffloadError(f"ssh to {h.name} failed to start: {e}") from e
        hello, _ = self.call({"op": "hello"}, timeout_s=180.0)
        self.hello = dict(hello)
        if hello.get("schema") != SCHEMA:
            raise MktOffloadError(f"{h.name}: worker schema {hello.get('schema')!r} != {SCHEMA!r}")
        log.info("marketcast offload: worker up on %s (%s %s)", h.name,
                 hello.get("device", "?"), hello.get("gpu", ""))
        return hello

    def _stderr_tail(self) -> str:
        if self._err is None:
            return ""
        with contextlib.suppress(Exception):
            self._err.seek(0)
            return self._err.read()[-600:].decode("utf-8", "replace")
        return ""

    def _read_exact(self, n: int, deadline: float) -> bytes:
        fd = self.proc.stdout.fileno()
        sel = selectors.DefaultSelector()
        sel.register(fd, selectors.EVENT_READ)
        buf = bytearray()
        try:
            while len(buf) < n:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MktOffloadError(f"{self.host.name}: worker reply timed out")
                if not sel.select(left):
                    continue
                chunk = os.read(fd, min(1 << 20, n - len(buf)))
                if not chunk:
                    raise MktOffloadError(f"{self.host.name}: worker closed the session "
                                          f"({self._stderr_tail()!r})")
                buf += chunk
        finally:
            sel.close()
        return bytes(buf)

    def call(self, header: dict, arrays: dict | None = None, *,
             timeout_s: float | None = None, exact: bool = False) -> tuple[dict, dict]:
        """One request. ``exact`` (the judge ops): a non-infra error comes
        back as the type the pod raised — ``MktForwardError`` and
        ``ValueError`` themselves, anything else :class:`MktRemoteError`
        carrying the type name — so the rows record what the in-process path
        would have recorded."""
        if self.proc is None:
            raise MktOffloadError("marketcast offload session not started")
        deadline = time.monotonic() + float(timeout_s or self.call_timeout_s)
        try:
            self.proc.stdin.write(encode_frame(header, arrays))
            self.proc.stdin.flush()
            (n,) = _LEN.unpack(self._read_exact(_LEN.size, deadline))
            if not 0 < n <= FRAME_MAX:
                raise MktOffloadError(f"{self.host.name}: reply length {n} out of range")
            reply, rarr = decode_frame(self._read_exact(n, deadline))
        except MktOffloadError:
            raise
        except (OSError, ValueError, KeyError) as e:
            raise MktOffloadError(f"{self.host.name}: session broken: {type(e).__name__}: "
                                  f"{e}") from e
        self.calls += 1
        if reply.get("ok"):
            return reply, rarr
        err, msg = str(reply.get("error", "Error")), str(reply.get("msg", ""))
        if reply.get("infra"):
            raise MktOffloadError(f"{self.host.name}: {err}: {msg}")
        if err == "ValueError":
            raise ValueError(msg)
        if exact:
            if err == "MktForwardError":
                from .marketcast import MktForwardError

                raise MktForwardError(msg)
            raise MktRemoteError(err, msg)
        raise RuntimeError(f"{err}: {msg}")

    def close(self) -> None:
        p, self.proc = self.proc, None
        if p is not None:
            with contextlib.suppress(Exception):
                p.stdin.write(encode_frame({"op": "bye"}))
                p.stdin.flush()
                p.stdin.close()
            try:
                p.wait(timeout=20)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    p.kill()
        if self._err is not None:
            with contextlib.suppress(Exception):
                self._err.close()
            self._err = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # checkpoints

    def upload(self, local_dir) -> tuple[str, str]:
        """``(dir relative to the worker's --root, weights sha256)`` of
        ``local_dir`` in the pod's content-addressed cache (uploaded only
        when no sha256-identical copy is there). A cache filesystem without
        room for it fails before the copy, naming the alternative mount."""
        from ..trainer.remote import build_ssh_argv
        from .eval_offload import (
            bench_free_bytes,
            build_scp_argv,
            parse_sha256sum,
            private_cache_key,
        )

        d = Path(local_dir)
        hashes = checkpoint_hashes(d)
        key = private_cache_key(hashes)
        root = cache_root_of(self.host)
        rel = key

        alt = ""

        def step(argv, what, timeout):
            proc = self.run(argv, timeout)
            if proc.returncode:
                raise MktOffloadError(f"{self.host.name}: {what} failed (exit "
                                      f"{proc.returncode}): {(proc.stderr or '')[-300:]}{alt}")
            return proc

        got = step(build_ssh_argv(self.host, cache_setup_command(root, key)), "cache setup", 120)
        if parse_sha256sum(got.stdout) == hashes:
            return rel, hashes["model.safetensors"]
        need = sum((d / n).stat().st_size for n in CKPT_FILES) + UPLOAD_HEADROOM_BYTES
        alt = alternatives_text(parse_alternatives(got.stdout), need)
        free = bench_free_bytes(got.stdout)
        if free is not None and free < need:
            msg = (f"{self.host.name}: pod disk short for checkpoint upload: "
                   f"{free / 1e9:.1f} GB free on the MarketCast cache filesystem ({root}) "
                   f"< {need / 1e9:.1f} GB needed{alt}")
            log.error("marketcast offload: %s", msg)
            raise MktOffloadError(msg)
        part = f"{root}/{key}.part-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        step(build_ssh_argv(self.host, f"umask 077 && mkdir -p {shlex.quote(part)}"),
             "cache part", 120)
        step(build_scp_argv(self.host, [str(d / n) for n in CKPT_FILES], part),
             "checkpoint upload", UPLOAD_TIMEOUT_S)
        step(build_ssh_argv(self.host, f"rm -rf {shlex.quote(root + '/' + key)} && "
                            f"mv {shlex.quote(part)} {shlex.quote(root + '/' + key)}"),
             "cache publish", 120)
        log.info("marketcast offload: uploaded %s to %s in %.0fs", key[:16], self.host.name,
                 time.monotonic() - started)
        return rel, hashes["model.safetensors"]

    def forecaster(self, local_dir, *, key: str | None = None,
                   family: str = "timestral") -> RemoteForecaster:
        """The pod-side forecaster of checkpoint ``local_dir`` (cached per
        ``key`` — default the weights' sha256 — for this session), loaded by
        the worker's ``family`` loader (:data:`cascade.validator.mkt_worker.LOADERS`)."""
        rel, sha = self.upload(local_dir)
        model = str(key or sha)
        hit = self._models.get(model)
        if hit is not None:
            return hit
        head = {"op": "load", "model": model, "dir": rel, "sha256": sha}
        if family != "timestral":
            head["family"] = str(family)
        reply, _ = self.call(head)
        fc = RemoteForecaster(self, model, tuple(reply.get("levels", ())))
        self._models[model] = fc
        return fc

    # the judge

    @property
    def has_judge(self) -> bool:
        return "judge" in tuple(self.hello.get("ops") or ())

    def upload_snapshot(self, snap) -> str:
        """``snap``'s directory in the pod cache as ``snap-<digest>`` (dir
        relative to the worker's --root), uploaded only when no copy marked
        with this digest is there; a cache filesystem without room fails
        before the copy. The pod re-verifies every file's sha256 on load."""
        from ..trainer.remote import build_ssh_argv
        from .eval_offload import _FREE_TAG, bench_free_bytes, build_scp_argv

        q = shlex.quote
        digest = str(snap.digest)
        key = f"{SNAP_PREFIX}{digest[:48]}"
        root = cache_root_of(self.host)
        local = Path(snap.root)

        def step(argv, what, timeout):
            proc = self.run(argv, timeout)
            if proc.returncode:
                raise MktOffloadError(f"{self.host.name}: {what} failed (exit "
                                      f"{proc.returncode}): {(proc.stderr or '')[-300:]}")
            return proc

        setup = (f"umask 077 && mkdir -p {q(root)} && {{ ( cd {q(root)} && "
                 f"ls -1td {q(SNAP_PREFIX)}* 2>/dev/null | grep -v -e '[.]part-' "
                 f"-e {q('^' + key + '$')} | tail -n +{SNAP_KEEP} | xargs -r rm -rf -- ) "
                 f"|| true; }} && {{ cat {q(root + '/' + key + '/' + SNAP_MARKER)} "
                 f"2>/dev/null && touch {q(root + '/' + key)} || true; }}; "
                 f"echo; echo {_FREE_TAG}$(df -PB1 {q(root)} 2>/dev/null | awk 'NR==2{{print $4}}')")
        got = step(build_ssh_argv(self.host, setup), "snapshot cache setup", 120)
        if digest in (got.stdout or "").split():
            return key
        size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
        need = size + UPLOAD_HEADROOM_BYTES
        free = bench_free_bytes(got.stdout)
        if free is not None and free < need:
            msg = (f"{self.host.name}: pod disk short for the snapshot upload: "
                   f"{free / 1e9:.1f} GB free on the MarketCast cache filesystem ({root}) "
                   f"< {need / 1e9:.1f} GB needed")
            log.error("marketcast offload: %s", msg)
            raise MktOffloadError(msg)
        part = f"{root}/{key}.part-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        step(build_scp_argv(self.host, str(local), part), "snapshot upload", UPLOAD_TIMEOUT_S)
        step(build_ssh_argv(self.host, f"echo {q(digest)} > {q(part + '/' + SNAP_MARKER)} && "
                            f"rm -rf {q(root + '/' + key)} && mv {q(part)} {q(root + '/' + key)}"),
             "snapshot publish", 120)
        log.info("marketcast offload: uploaded snapshot %s (%.0f MB) to %s in %.0fs",
                 digest[:16], size / 1e6, self.host.name, time.monotonic() - started)
        return key


class PodJudge:
    """The judge of one round on the pod: the same three functions the
    in-process path calls (:func:`~cascade.validator.marketcast.
    compute_entry_rows`, ``near_dup_record``, ``holder_record``), with the
    same results and the same exceptions, computed by the worker. The box
    fetches, checks and uploads every checkpoint the pod may need
    (``load_ft``) and keeps nothing heavier than the rows and the sidecars it
    publishes."""

    def __init__(self, session: MktSession, base: RemoteForecaster, load_ft):
        self.sess, self.base, self._load = session, base, load_ft
        self.snapshot = ""
        self.round_id = ""
        self._specs: dict = {}
        self.pod_peak_rss_mb = -1

    def begin_round(self, snap, mc, round_id: str) -> None:
        """Upload + bind the snapshot and the config; reset the round's
        caches on the pod (the box's ``ft_cache`` is per round too). Any
        failure is the pod's (:class:`MktOffloadError`: the round waits)."""
        try:
            self._begin_round(snap, mc, round_id)
        except MktOffloadError:
            raise
        except Exception as e:  # noqa: BLE001 — a pod that cannot hold the round
            raise MktOffloadError(f"{self.sess.host.name}: judge setup failed: "
                                  f"{type(e).__name__}: {e}") from e

    def _begin_round(self, snap, mc, round_id: str) -> None:
        text = mc_to_wire(mc)
        if mc_from_wire(text) != mc:
            raise MktOffloadError("the [marketcast] table does not survive its wire form; "
                                  "refusing to judge on the pod")
        rel = self.sess.upload_snapshot(snap)
        got, _ = self.sess.call({"op": "judge_snapshot", "dir": rel, "digest": str(snap.digest),
                                 "schema": str(getattr(mc, "pool_schema", "") or ""),
                                 "select": [list(x) for x in getattr(mc, "pool_select", ())],
                                 "channels": [list(x) for x in getattr(mc, "pool_channels", ())]},
                                timeout_s=UPLOAD_TIMEOUT_S)
        if got.get("digest") != snap.digest or \
                got.get("meta_sha256") != snapshot_meta_sha256(snap.meta):
            raise MktOffloadError(f"{self.sess.host.name}: the pod's snapshot view "
                                  f"{str(got.get('digest'))[:16]} is not the box's "
                                  f"{str(snap.digest)[:16]}")
        got, _ = self.sess.call({"op": "judge_round", "mc": text})
        if got.get("mc_sha256") != wire_sha256(text):
            raise MktOffloadError(f"{self.sess.host.name}: the pod parsed another "
                                  "[marketcast] table")
        self.snapshot, self.round_id, self._specs = str(snap.digest), str(round_id), {}

    def _spec(self, pointer: str, *, ref: bool = False) -> dict:
        """What ``load_ft(pointer)`` gave on the box, as the pod replays it
        at the point the in-process path called it. A loaded model is kept
        for the round; a fault or an error is re-tried on every call, as the
        in-process path did. ``ref``: a reference's load is inside the
        reference test's own ``except`` (an unloadable reference — even a
        failed upload — is skipped), a fine-tune's offload failure is the
        pod's (the round waits)."""
        from .marketcast import MktFault

        hit = self._specs.get(pointer)
        if hit is not None:
            return hit
        try:
            fc, covs = self._load(pointer)
            if not isinstance(fc, RemoteForecaster) or fc.session is not self.sess:
                raise MktOffloadError(f"fine-tune {pointer[:40]} is not loaded on the pod")
        except MktFault as e:
            return {"fault": str(e)}
        except MktOffloadError as e:
            if not ref:
                raise
            return {"error": type(e).__name__, "msg": str(e)}
        except Exception as e:  # noqa: BLE001 — replayed on the pod where load_ft ran
            return {"error": type(e).__name__, "msg": str(e)}
        sp = {"model": fc.model, "covariates": list(covs)}
        self._specs[pointer] = sp
        return sp

    def _call(self, header: dict) -> tuple[dict, bytes | None]:
        import hashlib

        got, arr = self.sess.call({**header, "snapshot": self.snapshot,
                                   "round_id": self.round_id}, exact=True)
        blob = None
        if "blob" in arr:
            blob = arr["blob"].astype(np.uint8, copy=False).tobytes()
            if hashlib.sha256(blob).hexdigest() != got.get("blob_sha256"):
                raise MktOffloadError(f"{self.sess.host.name}: a judge reply's sidecar does "
                                      "not match its sha256")
        self.pod_peak_rss_mb = int(got.get("pod_peak_rss_mb", -1))
        return got, blob

    def compute_entry_rows(self, entry, *, phase: str, after: float, draw: dict | None,
                           round_id: str):
        """``(rows, sidecar bytes | None)`` exactly as
        :func:`~cascade.validator.marketcast.compute_entry_rows`."""
        sp = self._spec(entry.trained_pointer)
        got, blob = self._call({
            "op": "judge_rows", "phase": str(phase), "after": float(after),
            "draw": dict(draw or {}), "base": self.base.model,
            "entry": {"miner_hotkey": entry.miner_hotkey,
                      "trained_pointer": entry.trained_pointer, "target": entry.target},
            "specs": {entry.trained_pointer: sp}})
        log.info("marketcast round=%s: %s rows judged on %s (%d windows; pod peak RSS %d MB)",
                 round_id, entry.miner_hotkey[:12], self.sess.host.name,
                 int(got.get("windows", 0)), self.pod_peak_rss_mb)
        return rows_from_wire(got["rows"]), blob

    def near_dup_record(self, rows, refs) -> tuple[dict, bytes | None]:
        if not refs or not rows.windows:
            return {}, None
        specs = {str(p): self._spec(str(p), ref=True) for _h, p, _c in refs}
        got, blob = self._call({"op": "judge_near_dup", "rows": rows_to_wire(rows),
                                "refs": [list(r) for r in refs], "specs": specs})
        return dict(got.get("near_dup") or {}), blob

    def holder_record(self, rows, holder):
        if holder is None or not rows.windows or holder[1] == rows.trained_pointer:
            return rows, None
        got, blob = self._call({"op": "judge_holder", "rows": rows_to_wire(rows),
                                "holder": list(holder),
                                "specs": {str(holder[1]): self._spec(str(holder[1]), ref=True)}})
        return rows_from_wire(got["rows"]), blob


# ── qualification ────────────────────────────────────────────────────────────


def qualify(host, base_dir, *, n: int = 48, context: int = 512, horizon: int = 60,
            rtol: float = 0.002, atol: float = 0.001, seed: int = 0, session=None) -> dict:
    """Forecast synthetic random-walk contexts with the pinned base locally
    (CPU) and on ``host``; the replies must agree within the audit's
    re-forecast tolerance (``rtol`` / ``atol``). Returns a JSON-able record;
    ``ok`` False must block enabling the offload."""
    from ..eval.timestral_adapter import TimestralQuantileForecaster

    rng = np.random.default_rng(seed)
    ctx = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, size=(n, context)), axis=1))
    local = TimestralQuantileForecaster.from_dir(base_dir, device="cpu")
    t0 = time.monotonic()
    want = local.forecast_batch(ctx, horizon)
    t_local = time.monotonic() - t0
    sess = session or MktSession(host)
    own = session is None
    if own:
        sess.start()
    try:
        fc = sess.forecaster(base_dir, key="base")
        t0 = time.monotonic()
        got = fc.forecast_batch(ctx, horizon)
        t_remote = time.monotonic() - t0
    finally:
        if own:
            sess.close()
    err = np.abs(got - want)
    tol = atol + rtol * np.abs(want)
    worst = float(np.max(err / tol))
    return {"ok": bool(np.all(err <= tol)), "windows": n, "horizon": horizon,
            "worst_fraction_of_tolerance": worst, "levels_match":
            tuple(fc.quantile_levels) == tuple(local.quantile_levels),
            "local_cpu_s": round(t_local, 2), "remote_s": round(t_remote, 2)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover — operator CLI
    import argparse
    import json

    from ..shared.config import load_chain_config
    from .eval_offload import make_eval_host_fn

    ap = argparse.ArgumentParser(prog="cascade.validator.mkt_offload",
                                 description="Qualify a MarketCast offload pod.")
    ap.add_argument("hosts", type=Path, help="hosts.toml naming the pod")
    ap.add_argument("--chain-toml", default="chain.toml")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO")
    host = make_eval_host_fn(args.hosts, local_fallback=False)()
    if host is None:
        print(f"no host in {args.hosts}")
        return 2
    from .marketcast import resolve_pinned_base_dir

    base_dir = resolve_pinned_base_dir(load_chain_config(args.chain_toml).marketcast,
                                       cache_dir=args.cache_dir)
    rec = qualify(host, base_dir)
    print(json.dumps(rec, indent=1))
    return 0 if rec["ok"] and rec["levels_match"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
