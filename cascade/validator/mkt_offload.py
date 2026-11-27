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
from .mkt_worker import _LEN, FRAME_MAX, SCHEMA, decode_frame, encode_frame

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
             f"ls -1t | grep -v -e '[.]part-' -e {q('^' + key + '$')} "
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
             timeout_s: float | None = None) -> tuple[dict, dict]:
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
