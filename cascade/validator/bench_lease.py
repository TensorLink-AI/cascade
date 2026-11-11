"""Elastic bench host for the main validator's DEC-CA-0063 sweeps (OPERATOR-LOCAL).

The main validator needs a GPU only when a challenger clears the private pool
or the king check is due — a few hours a day. A standing 96 GB pod bills the
other ~20 hours. A :class:`BenchLease` rents a pod when a sweep is needed,
publishes it to the validator's ``--eval-hosts`` file (re-read per eval by
:func:`~cascade.validator.eval_offload.make_eval_host_fn`), and terminates it
once it has been idle for ``idle_release_minutes``.

Miner safety is the design constraint, so the lease never turns "no pod yet"
into a verdict:

* while a pod is being rented, booted or bootstrapped the settlement HOLDS
  (:class:`~cascade.validator.validator_bench.ValidatorBenchHold`): nothing is
  judged, nothing is published, every other validator waits exactly as it
  already waits for the report;
* the hold is bounded by ``max_wait_hours`` (load-checked to sit well inside
  ``[v2] bench_report_timeout_settlements``, after which the trainer would
  record the report "absent"). Past it, a challenger is published
  ``budget_deferred`` — never swept, never a loss, carried by the trainer
  into a later settlement and judged again — and a king / init is infra;
* a pod that dies mid-sweep is re-rented and the settlement holds again
  (inside the same bound) instead of the sweep being published infra;
* with a lease configured the sweep never falls back to this box's own
  device (the 7 GB validator box cannot run it);
* every swept row is tagged with the host's hardware fingerprint (GPU model,
  driver, sidecar torch). A cached king swept on other hardware is a cache
  MISS (re-swept on the current pod), and the gate's existing revision check
  refuses to compare rows from two different fingerprints.

The lease is in-process state persisted to ``state_path`` (write-ahead
before every rent) so a restarted validator finds and reuses — or
terminates — the pod it already holds instead of renting a second one.
Static mode (no ``--bench-lease``) is byte-identical to before.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

log = logging.getLogger("cascade.validator.bench_lease")

# Pod name prefix (lium ``--name``); the trainer's inventory sees it as ours.
LEASE_POD_PREFIX = "cascade-benchlease"
# Revision key the hardware fingerprint is stamped under in swept rows.
HW_KEY = "hw"
# Hosts lost under a sweep (without a good sweep between) before the next
# loss stops granting a fresh bounded wait.
MAX_HOST_LOSSES = 2
PHASE_IDLE = "idle"
PHASE_RENTING = "renting"
PHASE_READY = "ready"
_PHASES = (PHASE_IDLE, PHASE_RENTING, PHASE_READY)


class BenchLeaseConfigError(ValueError):
    """The ``--bench-lease`` file is unusable (the validator refuses to start)."""


@dataclass(frozen=True)
class BenchLeaseConfig:
    """``bench_lease.toml`` — operator-local, never consensus.

    ``gpu_types`` is the rent ladder in Lium ``gpu_type`` names; the cheapest
    listed offer across it wins. Every type must be one the sweep has been
    QUALIFIED on (one measured sweep inside ``[v2] bench_sweep_hours``) —
    the fingerprint keeps mixed hardware out of a gate, but an unqualified
    slow card would still time sweeps out as infra.
    """

    hosts_path: Path
    state_path: Path
    image: str
    ssh_pubkey_path: Path
    gpu_types: tuple[str, ...]
    key_path: str = ""
    min_vram_gb: int = 32
    min_cuda: float = 13.0                 # listed max_cuda_version (driver >= 580)
    min_driver: int = 580                  # re-checked on the booted pod
    max_price_per_hour: float = 2.0
    max_wait_hours: float = 2.0
    idle_release_minutes: float = 20.0
    ttl_hours: float = 12.0                # provider-side auto-terminate backstop
    ready_timeout_minutes: float = 20.0
    bootstrap_timeout_minutes: float = 60.0
    retry_minutes: float = 10.0
    # Local script piped to the pod's ``bash -s`` (the rented image predates
    # it); "" ⇒ run ``bootstrap_command`` as-is on the pod instead.
    bootstrap_script: str = "deploy/bench_pod_bootstrap.sh"
    bootstrap_command: str = "bash -s"
    user: str = "root"
    workdir: str = "/root/cascade"
    remote_python: str = "/root/cascade/.venv/bin/python"
    sidecar_python: str = "/root/cascade/benchmarks/.venv/bin/python"
    ssh_options: tuple[str, ...] = ("StrictHostKeyChecking=accept-new",
                                    "ServerAliveInterval=30", "ServerAliveCountMax=20")


def load_bench_lease_config(path: Path | str, *, epoch_blocks: int = 900,
                            block_seconds: float = 12.0,
                            report_timeout_settlements: int = 8,
                            sweep_hours: float = 2.0) -> BenchLeaseConfig:
    """Parse and load-check ``path``. ``max_wait_hours`` plus two sweeps must
    fit inside the trainer's "absent" deadline (``report_timeout_settlements``
    settlements) with one settlement of slack — otherwise a slow rent could let
    the trainer record the report absent while we still hold, which is
    exactly the miner harm the lease exists to prevent."""
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise BenchLeaseConfigError(f"bench lease {p}: unreadable ({e})") from e
    t = raw.get("bench_lease")
    if not isinstance(t, dict):
        raise BenchLeaseConfigError(f"bench lease {p}: missing [bench_lease] table")
    known = {f for f in BenchLeaseConfig.__dataclass_fields__}
    unknown = sorted(set(t) - known)
    if unknown:
        raise BenchLeaseConfigError(f"bench lease {p}: unknown keys {unknown}")
    for k in ("hosts_path", "state_path", "image", "ssh_pubkey_path", "gpu_types"):
        if not t.get(k):
            raise BenchLeaseConfigError(f"bench lease {p}: {k} is required")
    types = t["gpu_types"]
    if (not isinstance(types, list) or not types
            or not all(isinstance(s, str) and s.strip() for s in types)):
        raise BenchLeaseConfigError(f"bench lease {p}: gpu_types must be a non-empty "
                                    "list of Lium gpu_type names")
    image = str(t["image"])
    if "@sha256:" not in image:
        raise BenchLeaseConfigError(f"bench lease {p}: image must be digest-pinned "
                                    "(repo@sha256:…)")
    try:
        cfg = BenchLeaseConfig(
            hosts_path=Path(t["hosts_path"]).expanduser(),
            state_path=Path(t["state_path"]).expanduser(),
            image=image,
            ssh_pubkey_path=Path(t["ssh_pubkey_path"]).expanduser(),
            gpu_types=tuple(s.strip() for s in types),
            key_path=str(t.get("key_path", "")),
            min_vram_gb=int(t.get("min_vram_gb", 32)),
            min_cuda=float(t.get("min_cuda", 13.0)),
            min_driver=int(t.get("min_driver", 580)),
            max_price_per_hour=float(t.get("max_price_per_hour", 2.0)),
            max_wait_hours=float(t.get("max_wait_hours", 2.0)),
            idle_release_minutes=float(t.get("idle_release_minutes", 20.0)),
            ttl_hours=float(t.get("ttl_hours", 12.0)),
            ready_timeout_minutes=float(t.get("ready_timeout_minutes", 20.0)),
            bootstrap_timeout_minutes=float(t.get("bootstrap_timeout_minutes", 60.0)),
            retry_minutes=float(t.get("retry_minutes", 10.0)),
            bootstrap_script=str(t.get("bootstrap_script",
                                       BenchLeaseConfig.bootstrap_script)),
            bootstrap_command=str(t.get("bootstrap_command",
                                        BenchLeaseConfig.bootstrap_command)),
            user=str(t.get("user", "root")),
            workdir=str(t.get("workdir", "/root/cascade")),
            remote_python=str(t.get("remote_python", BenchLeaseConfig.remote_python)),
            sidecar_python=str(t.get("sidecar_python", BenchLeaseConfig.sidecar_python)),
            ssh_options=tuple(str(o) for o in t.get("ssh_options",
                                                    BenchLeaseConfig.ssh_options)),
        )
    except (TypeError, ValueError) as e:
        raise BenchLeaseConfigError(f"bench lease {p}: {e}") from e
    check_bench_lease_config(cfg, epoch_blocks=epoch_blocks, block_seconds=block_seconds,
                             report_timeout_settlements=report_timeout_settlements,
                             sweep_hours=sweep_hours, where=str(p))
    return cfg


def check_bench_lease_config(cfg: BenchLeaseConfig, *, epoch_blocks: int,
                             block_seconds: float, report_timeout_settlements: int,
                             sweep_hours: float, where: str = "bench lease") -> None:
    for name in ("max_price_per_hour", "max_wait_hours", "idle_release_minutes",
                 "ttl_hours", "ready_timeout_minutes", "bootstrap_timeout_minutes",
                 "retry_minutes"):
        if not float(getattr(cfg, name)) > 0:
            raise BenchLeaseConfigError(f"{where}: {name} must be > 0")
    settlement_h = epoch_blocks * block_seconds / 3600.0
    deadline_h = max(1, int(report_timeout_settlements)) * settlement_h
    worst_h = cfg.max_wait_hours + 2 * float(sweep_hours) + settlement_h
    if worst_h >= deadline_h:
        raise BenchLeaseConfigError(
            f"{where}: max_wait_hours {cfg.max_wait_hours:g} + 2 sweeps "
            f"({2 * float(sweep_hours):g} h) + one settlement ({settlement_h:g} h) = "
            f"{worst_h:g} h must stay inside the trainer's absent deadline "
            f"({report_timeout_settlements} settlements = {deadline_h:g} h)")
    if cfg.ttl_hours < cfg.max_wait_hours + 2 * float(sweep_hours):
        raise BenchLeaseConfigError(
            f"{where}: ttl_hours {cfg.ttl_hours:g} is shorter than a held wait plus two "
            f"sweeps ({cfg.max_wait_hours + 2 * float(sweep_hours):g} h) — the provider "
            "would kill the pod mid-sweep")


# ── persisted state ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LeaseState:
    phase: str = PHASE_IDLE
    pod_id: str = ""                 # provider handle (lium pod name)
    gpu_type: str = ""
    ip: str = ""
    port: int = 0
    hw: str = ""                     # fingerprint of the READY host
    requested_at: float = 0.0        # wall time the current need began (0 = none)
    last_used: float = 0.0           # wall time a sweep last asked for the host
    next_try_at: float = 0.0         # rent backoff
    attempts: int = 0                # rents in the current need
    losses: int = 0                  # hosts lost under a sweep since the last good one
    last_error: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, text: str) -> LeaseState:
        d = json.loads(text)
        if not isinstance(d, dict):
            raise ValueError("lease state is not an object")
        known = cls.__dataclass_fields__
        st = cls(**{k: v for k, v in d.items() if k in known})
        if st.phase not in _PHASES:
            raise ValueError(f"unknown lease phase {st.phase!r}")
        return st


def hosts_toml(cfg: BenchLeaseConfig, *, name: str, ip: str, port: int) -> str:
    """The ``--eval-hosts`` entry for the leased pod (``stage = "any"`` so the
    validator's resolver picks it)."""
    q = json.dumps   # a JSON string literal is a valid TOML basic string here
    lines = [
        "# Written by the validator's bench lease (cascade.validator.bench_lease).",
        "# Rewritten / cleared as the leased pod comes and goes — do not hand-edit.",
        "[[host]]",
        f"name = {q(name)}",
        f"host = {q(ip)}",
        f"port = {int(port)}",
        f"user = {q(cfg.user)}",
        f"remote_python = {q(cfg.remote_python)}",
        f"workdir = {q(cfg.workdir)}",
        'stage = "any"',
    ]
    if cfg.key_path:
        lines.append(f"key_path = {q(cfg.key_path)}")
    if cfg.ssh_options:
        lines.append("ssh_options = [" + ", ".join(q(o) for o in cfg.ssh_options) + "]")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


# Probe run on the pod after bootstrap: GPU model + driver + sidecar torch. The
# output (one JSON line) IS the fingerprint's source.
_PROBE = (
    "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1; "
    "{py} -c 'import torch,json;print(json.dumps([torch.__version__, torch.version.cuda]))'"
)


def fingerprint(probe_stdout: str) -> str:
    """``"<gpu name>|driver <major>|torch <ver>"`` from the probe's output.
    Driver MAJOR only: a point release does not change kernels; a new major
    (or another GPU model, or another torch build) does."""
    lines = [ln.strip() for ln in probe_stdout.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"bench host probe output unusable: {probe_stdout[-200:]!r}")
    name, _, driver = (s.strip() for s in lines[0].partition(","))
    torch_ver, cuda = json.loads(lines[-1])
    if not name or not driver or not torch_ver:
        raise ValueError(f"bench host probe output incomplete: {lines!r}")
    return f"{name}|driver {driver.split('.')[0]}|torch {torch_ver} cuda {cuda}"


def _default_ssh(cfg: BenchLeaseConfig, ip: str, port: int, command: str,
                 timeout: float, input: str | None = None) -> subprocess.CompletedProcess:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-p", str(int(port))]
    for o in cfg.ssh_options:
        argv += ["-o", o]
    if cfg.key_path:
        argv += ["-i", os.path.expanduser(cfg.key_path)]
    argv += [f"{cfg.user}@{ip}", command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          check=False, input=input)


# ── the lease ─────────────────────────────────────────────────────────────────


@dataclass
class BenchLease:
    """One leased bench pod at a time. Thread-safe: the validator loop calls
    :meth:`need` / :meth:`tick`; the rent itself runs on a daemon worker so a
    15–40 min rent + bootstrap never blocks the loop (weights keep landing).

    ``provider`` follows :class:`cascade.provision.core.Provider` (+ an
    optional ``sku_of``). ``ssh(ip, port, command, timeout)`` returns a
    ``CompletedProcess``. ``clock`` is wall time. ``spawn(fn)`` starts the
    rent worker (tests run it inline)."""

    cfg: BenchLeaseConfig
    provider: Any
    ssh: Callable[..., subprocess.CompletedProcess] | None = None
    clock: Callable[[], float] = time.time
    spawn: Callable[[Callable[[], None]], None] | None = None
    state: LeaseState = field(default_factory=LeaseState)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _busy: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.ssh is None:
            self.ssh = lambda ip, port, cmd, timeout, input=None: _default_ssh(
                self.cfg, ip, port, cmd, timeout, input)
        if self.spawn is None:
            self.spawn = lambda fn: threading.Thread(target=fn, name="bench-lease-rent",
                                                     daemon=True).start()
        self._load()

    # ── persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        p = self.cfg.state_path
        if not p.is_file():
            return
        try:
            st = LeaseState.from_json(p.read_text())
        except (OSError, ValueError, TypeError) as e:
            log.error("bench lease state %s unreadable (%s); starting idle — check the "
                      "provider for a stray %s* pod", p, e, LEASE_POD_PREFIX)
            return
        if st.phase == PHASE_RENTING:
            # A rent was in flight when we died: the pod (if any) may exist but
            # was never bootstrapped/published. Terminate it and re-rent.
            if st.pod_id:
                self._terminate(st.pod_id, "rent interrupted by a validator restart")
            st = replace(st, phase=PHASE_IDLE, pod_id="", ip="", port=0, hw="",
                         gpu_type="")
        self.state = st
        if st.phase == PHASE_READY:
            if not self.reachable():
                # Died (or its TTL fired) while the validator was down.
                self._terminate(st.pod_id, "unreachable at validator start")
                self._clear_hosts()
                self.state = replace(st, phase=PHASE_IDLE, pod_id="", ip="", port=0,
                                     hw="", gpu_type="")
                self._save()
                return
            log.info("bench lease: resuming pod %s (%s) at %s:%d", st.pod_id, st.hw,
                     st.ip, st.port)
            self._publish()

    def _save(self) -> None:
        try:
            _atomic_write(self.cfg.state_path, self.state.to_json())
        except OSError as e:
            log.error("bench lease: could not persist %s (%s)", self.cfg.state_path, e)

    def _set(self, **kw: Any) -> None:
        self.state = replace(self.state, **kw)
        self._save()

    # ── what the validator calls ────────────────────────────────────────────

    def ready(self) -> bool:
        return self.state.phase == PHASE_READY

    def hw(self) -> str:
        return self.state.hw if self.ready() else ""

    def waited_seconds(self) -> float:
        r = self.state.requested_at
        return max(0.0, self.clock() - r) if r else 0.0

    def expired(self) -> bool:
        """The current need has waited past ``max_wait_hours`` without a pod."""
        return (not self.ready() and bool(self.state.requested_at)
                and self.waited_seconds() >= self.cfg.max_wait_hours * 3600.0)

    def status(self) -> str:
        s = self.state
        bits = [s.phase]
        if s.requested_at:
            bits.append(f"waited {self.waited_seconds() / 60:.0f} min of "
                        f"{self.cfg.max_wait_hours * 60:.0f}")
        if s.attempts:
            bits.append(f"{s.attempts} rent attempt(s)")
        if s.last_error:
            bits.append(f"last: {s.last_error[:160]}")
        return "; ".join(bits)

    def need(self) -> bool:
        """A sweep needs the host NOW. True ⇒ the host is ready (published in
        the eval-hosts file); False ⇒ a rent is (now) in progress — the caller
        holds, or, once :meth:`expired`, defers. Never blocks."""
        with self._lock:
            now = self.clock()
            if self.state.phase == PHASE_READY:
                self._set(last_used=now, requested_at=0.0, attempts=0, last_error="")
                return True
            if not self.state.requested_at:
                self._set(requested_at=now, attempts=0, last_error="")
                log.warning("bench lease: a sweep needs a bench host — renting (holds up to "
                            "%.1f h; never a verdict while it waits)", self.cfg.max_wait_hours)
            self._maybe_start_rent(now)
            return False

    def end_need(self) -> None:
        """The need that expired has been resolved by a deferral: the next
        need starts a fresh wait (a rent already in flight keeps going)."""
        with self._lock:
            if self.state.requested_at and not self.ready():
                self._set(requested_at=0.0, attempts=0)

    def host_lost(self, why: str) -> None:
        """The ready host failed under a sweep (unreachable). Drop it and
        re-rent; the caller holds the settlement (the wait clock restarts
        from now — the lost sweep was not the miner's)."""
        with self._lock:
            s = self.state
            if s.phase != PHASE_READY:
                return
            log.error("bench lease: host %s (%s) LOST under a sweep (%s) — re-renting",
                      s.pod_id, s.ip, why)
            self._terminate(s.pod_id, f"lost: {why}")
            self._clear_hosts()
            now = self.clock()
            losses = s.losses + 1
            # A fresh bounded wait for the replacement — but only once: a
            # second loss before any sweep succeeds starts the wait already
            # spent, so a flaky provider can never hold the fleet past the
            # trainer's absent deadline (the clearer is deferred instead).
            start = now if losses < MAX_HOST_LOSSES else now - self.cfg.max_wait_hours * 3600.0
            self._set(phase=PHASE_IDLE, pod_id="", ip="", port=0, hw="", gpu_type="",
                      requested_at=start, attempts=0, losses=losses,
                      last_error=f"lost: {why}"[:300], next_try_at=0.0)
            self._maybe_start_rent(now)

    def sweep_ok(self) -> None:
        """A sweep on the leased host succeeded: the loss budget resets."""
        with self._lock:
            if self.state.losses:
                self._set(losses=0)

    def reachable(self) -> bool:
        s = self.state
        if s.phase != PHASE_READY:
            return False
        try:
            return self.ssh(s.ip, s.port, "true", 30).returncode == 0
        except Exception:  # noqa: BLE001 — unreachable is the answer
            return False

    def tick(self) -> None:
        """Once per validator poll: release an idle pod; keep a pending rent
        retrying after its backoff."""
        with self._lock:
            now = self.clock()
            s = self.state
            if s.phase == PHASE_READY and not s.requested_at:
                idle = now - (s.last_used or now)
                if idle >= self.cfg.idle_release_minutes * 60.0:
                    self._release_locked(f"idle {idle / 60:.0f} min")
            elif s.phase == PHASE_IDLE and s.requested_at:
                self._maybe_start_rent(now)

    def release(self, why: str = "operator release") -> None:
        with self._lock:
            if self.state.phase == PHASE_READY:
                self._release_locked(why)

    # ── internals ───────────────────────────────────────────────────────────

    def _release_locked(self, why: str) -> None:
        s = self.state
        # Never terminate a pod with a sweep still running on it (on-pod check).
        try:
            busy = self.ssh(s.ip, s.port, "pgrep -f cascade_benchmark >/dev/null", 30)
            if busy.returncode == 0:
                log.warning("bench lease: NOT releasing %s — a benchmark process is still "
                            "running on it", s.pod_id)
                self._set(last_used=self.clock())
                return
        except Exception as e:  # noqa: BLE001 — unreachable pod: nothing runs there
            log.info("bench lease: %s unreachable at release (%s)", s.pod_id, e)
        self._clear_hosts()
        self._terminate(s.pod_id, why)
        self._set(phase=PHASE_IDLE, pod_id="", ip="", port=0, hw="", gpu_type="",
                  requested_at=0.0, attempts=0)
        log.info("bench lease: released %s (%s)", s.pod_id, why)

    def _maybe_start_rent(self, now: float) -> None:
        if self._busy or self.state.phase != PHASE_IDLE or now < self.state.next_try_at:
            return
        self._busy = True
        name = f"{LEASE_POD_PREFIX}-{int(now)}"
        # write-ahead: a crash after launch finds the name and terminates it
        self._set(phase=PHASE_RENTING, pod_id=name, attempts=self.state.attempts + 1)

        def work() -> None:
            try:
                self._rent(name)
            finally:
                with self._lock:
                    self._busy = False

        self.spawn(work)

    def _fail_rent(self, name: str, why: str) -> None:
        log.error("bench lease: rent %s failed (%s); retrying in %.0f min", name, why,
                  self.cfg.retry_minutes)
        self._terminate(name, why)
        with self._lock:
            self._set(phase=PHASE_IDLE, pod_id="", ip="", port=0, hw="", gpu_type="",
                      last_error=why[:300],
                      next_try_at=self.clock() + self.cfg.retry_minutes * 60.0)

    def _rent(self, name: str) -> None:  # noqa: C901 — one linear rent pipeline
        from ..provision.core import LaunchSpec

        cfg = self.cfg
        try:
            pubkey = cfg.ssh_pubkey_path.read_text().strip()
        except OSError as e:
            self._fail_rent(name, f"ssh pubkey unreadable: {e}")
            return
        try:
            offers = self.provider.list_offers(list(cfg.gpu_types), gpus=1)
        except Exception as e:  # noqa: BLE001
            self._fail_rent(name, f"offer listing failed: {type(e).__name__}: {e}")
            return
        fit = [(sku, e) for sku, e in offers if _offer_fits(cfg, e)]
        if not fit:
            self._fail_rent(name, f"no {'/'.join(cfg.gpu_types)} offer with >= "
                                  f"{cfg.min_vram_gb} GB, CUDA >= {cfg.min_cuda:g}, "
                                  f"<= ${cfg.max_price_per_hour:.2f}/h")
            return
        from ..provision.core import executor_price_per_hour

        fit.sort(key=lambda o: executor_price_per_hour(o[1]) or 0.0)
        sku = fit[0][0]
        # launch() re-lists the type; keep every non-fitting executor of it
        # (too little VRAM, an old driver, over the cap) out of the pick.
        unfit = tuple(str(e.get("id")) for s, e in offers
                      if s == sku and not _offer_fits(cfg, e))
        spec = LaunchSpec(sku=sku, count=1, image=cfg.image, ssh_pubkey=pubkey,
                          name_prefix=name, gpus_per_pod=1, exclude_ids=unfit,
                          ttl_seconds=cfg.ttl_hours * 3600.0)
        try:
            handles = self.provider.launch(spec)
        except Exception as e:  # noqa: BLE001
            self._fail_rent(name, f"launch failed: {type(e).__name__}: {e}")
            return
        pod = handles[0] if handles else name
        with self._lock:
            self._set(pod_id=pod)
        try:
            if not self.provider.wait_ready(pod, timeout=cfg.ready_timeout_minutes * 60.0):
                self._fail_rent(pod, "pod never became ready")
                return
            addr = self.provider.get_ip(pod)
        except Exception as e:  # noqa: BLE001
            self._fail_rent(pod, f"readiness failed: {type(e).__name__}: {e}")
            return
        if addr is None:
            self._fail_rent(pod, "pod has no address")
            return
        ip, port = str(addr.ip), int(addr.ssh_port)
        try:
            script = Path(cfg.bootstrap_script).read_text() if cfg.bootstrap_script else None
            boot = self.ssh(ip, port, cfg.bootstrap_command,
                            cfg.bootstrap_timeout_minutes * 60.0, input=script)
            if boot.returncode != 0:
                self._fail_rent(pod, f"bootstrap rc={boot.returncode}: "
                                     f"{(boot.stderr or boot.stdout or '')[-200:]}")
                return
            probe = self.ssh(ip, port, _PROBE.format(py=shlex.quote(cfg.sidecar_python)), 120)
            if probe.returncode != 0:
                self._fail_rent(pod, f"probe rc={probe.returncode}")
                return
            hw = fingerprint(probe.stdout)
            major = int(hw.split("|driver ")[1].split("|")[0])
            if major < cfg.min_driver:
                self._fail_rent(pod, f"driver {major} < {cfg.min_driver} ({hw})")
                return
        except Exception as e:  # noqa: BLE001 — ssh timeout / bad probe
            self._fail_rent(pod, f"bootstrap/probe failed: {type(e).__name__}: {e}")
            return
        gpu_type = sku
        sku_of = getattr(self.provider, "sku_of", None)
        if callable(sku_of):
            with contextlib.suppress(Exception):
                gpu_type = str(sku_of(pod) or sku)
        with self._lock:
            self._set(phase=PHASE_READY, ip=ip, port=port, hw=hw, gpu_type=gpu_type,
                      last_used=self.clock(), last_error="", next_try_at=0.0)
            self._publish()
        log.info("bench lease: pod %s READY (%s, %s) at %s:%d after %.0f min", pod,
                 gpu_type, hw, ip, port, self.waited_seconds() / 60.0)

    def _publish(self) -> None:
        s = self.state
        _atomic_write(self.cfg.hosts_path,
                      hosts_toml(self.cfg, name=s.pod_id, ip=s.ip, port=s.port))

    def _clear_hosts(self) -> None:
        try:
            _atomic_write(self.cfg.hosts_path,
                          "# bench lease: no pod right now (cleared)\n")
        except OSError as e:
            log.error("bench lease: could not clear %s (%s)", self.cfg.hosts_path, e)

    def _terminate(self, pod: str, why: str) -> None:
        if not pod:
            return
        try:
            self.provider.terminate(pod)
            log.info("bench lease: terminated %s (%s)", pod, why)
        except Exception as e:  # noqa: BLE001 — the TTL is the backstop
            log.error("bench lease: terminate %s FAILED (%s) — ttl %.0f h backstops it; "
                      "check the provider", pod, e, self.cfg.ttl_hours)


def _offer_fits(cfg: BenchLeaseConfig, executor: dict) -> bool:
    from ..provision.core import executor_price_per_hour

    try:
        vram = float(executor.get("vram_gb") or 0)
        cuda = float(executor.get("max_cuda_version") or 0)
    except (TypeError, ValueError):
        return False
    price = executor_price_per_hour(executor)
    return (vram >= cfg.min_vram_gb and cuda >= cfg.min_cuda and price is not None
            and price <= cfg.max_price_per_hour)


def build_bench_lease(path: Path | str, cfg: Any, *, eval_hosts: Path | str | None,
                      is_main: bool, provider: Any = None) -> BenchLease:
    """The validator CLI's ``--bench-lease``: load-check against the deployed
    chain config, refuse every unsafe combination, and build the lease.

    Refuses (``BenchLeaseConfigError``) unless this is the MAIN validator (the
    only one that sweeps — a lease anywhere else would rent pods nobody uses)
    and ``--eval-hosts`` names exactly the lease's ``hosts_path`` (the file the
    sweep resolves its host from)."""
    from ..shared.config.chain_config import BLOCK_SECONDS

    if not is_main:
        raise BenchLeaseConfigError(
            "--bench-lease is for the MAIN validator only (hotkey == [netconfig] "
            "owner_hotkey); every other validator replays the signed rows and never sweeps")
    v2 = getattr(cfg, "v2", None)
    lc = load_bench_lease_config(
        path, epoch_blocks=int(cfg.round.epoch_blocks), block_seconds=float(BLOCK_SECONDS),
        report_timeout_settlements=int(getattr(v2, "bench_report_timeout_settlements", 8)),
        sweep_hours=float(getattr(v2, "bench_sweep_hours", 2.0)))
    if eval_hosts is None or Path(eval_hosts).expanduser().resolve() != lc.hosts_path.resolve():
        raise BenchLeaseConfigError(
            f"--bench-lease writes {lc.hosts_path}; pass the same file as --eval-hosts "
            f"(got {eval_hosts})")
    if provider is None:
        from ..provision.core import LiumProvider

        # The operator's own Lium account (the CLI's auth). The price guard is
        # enforced by the provider too, not only by the lease's offer filter.
        provider = LiumProvider(max_price_per_hour=lc.max_price_per_hour)
    return BenchLease(cfg=lc, provider=provider)
