"""Where gauntlet jobs run: this machine, or rented Lium pods under a spend cap.

Both executors take a job spec (:mod:`.jobs`) and return its result document.
Failures come back as ``{"error": …, "candidate_fault": bool}``, never raised:
the judge requeues infrastructure faults and rejects only candidate faults.

The Lium executor
-----------------

* rents the trainer-worker image (``[compute] image``, default the contract's
  pinned ``funded_pod_image``) through :class:`cascade.provision.core.LiumProvider`,
  whose listing already refuses executors above ``max_price_per_hour``;
* streams this checkout's ``cascade/`` package + ``chain*.toml`` over the image's
  editable install (tar over SSH: the image has sshd and tar, no rsync), stages
  each job's input files, runs ``python -m cascade.miner.harness.jobs`` over SSH
  and pulls the result and outputs back;
* keeps a WRITE-AHEAD spend ledger (``spend.json``): every pod is recorded
  before it is rented and billed at the price CAP (an upper bound) from launch
  to teardown. No pod is rented when today's accrued spend plus one job at the
  cap would pass ``daily_usd_cap``; at the cap, idle pods are torn down;
* tears down pods idle for ``idle_minutes`` and, on start, every pod carrying
  its prefix that the ledger does not own (a crashed run's leftovers).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("cascade.miner.harness.executor")

REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_ROOT = "/root/cascade"
REMOTE_JOBS = "/root/gauntlet"
REMOTE_PY = f"{REMOTE_ROOT}/.venv/bin/python"
REMOTE_HARNESS_TOML = f"{REMOTE_ROOT}/harness-chain.toml"


def _infra_error(msg: str) -> dict:
    return {"error": msg, "candidate_fault": False}


class LocalExecutor:
    """Jobs as subprocesses on this machine, ``max_parallel`` at a time (one GPU
    each when several are visible)."""

    def __init__(self, workdir: Path, *, max_parallel: int = 1, device: str = "auto",
                 chain_toml: Path | None = None, timeout: float = 8 * 3600.0,
                 sandbox: bool = False,
                 runner: Callable[[list[str], dict, float], int] | None = None) -> None:
        self.workdir = Path(workdir)
        self.capacity = max_parallel
        self.device = device
        self.chain_toml = chain_toml
        self.timeout = timeout
        self.sandbox = sandbox
        self._slots = list(range(max_parallel))
        self._lock = threading.Condition()
        self._runner = runner or self._subprocess

    @staticmethod
    def _subprocess(argv: list[str], env: dict, timeout: float) -> int:
        return subprocess.run(argv, env=env, timeout=timeout, check=False).returncode

    def run(self, spec: dict, *, job_id: str | None = None, est_hours: float = 1.0) -> dict:
        job_id = job_id or uuid.uuid4().hex[:12]
        jd = self.workdir / "jobs" / job_id
        jd.mkdir(parents=True, exist_ok=True)
        spec = json.loads(json.dumps(spec))
        params = spec.setdefault("params", {})
        params.setdefault("device", self.device)
        params.setdefault("cache_dir", str(self.workdir / "cache"))
        params.setdefault("sandbox", self.sandbox)
        if self.chain_toml:
            params.setdefault("chain_toml", str(self.chain_toml))
        (jd / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        result = jd / "result.json"
        with self._lock:
            while not self._slots:
                self._lock.wait()
            slot = self._slots.pop(0)
        try:
            env = dict(os.environ)
            if self.capacity > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            argv = [sys.executable, "-m", "cascade.miner.harness.jobs",
                    str(jd / "spec.json"), str(result)]
            try:
                self._runner(argv, env, self.timeout)
            except subprocess.TimeoutExpired:
                return _infra_error(f"job timed out after {self.timeout:.0f}s")
        finally:
            with self._lock:
                self._slots.append(slot)
                self._lock.notify()
        if not result.is_file():
            return _infra_error(f"job {job_id} wrote no result (see {jd})")
        return json.loads(result.read_text(encoding="utf-8"))

    def reap(self) -> None:
        pass

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# spend ledger                                                                 #
# --------------------------------------------------------------------------- #

def _utc_day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d")


class SpendLedger:
    """Write-ahead record of every pod, billed at ``price_per_hour`` (the cap)."""

    def __init__(self, path: Path, *, now: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._now = now
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"pods": {}}

    def _save(self, doc: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def open(self, name: str, price_per_hour: float) -> None:
        with self._lock:
            doc = self._load()
            doc["pods"][name] = {"price": float(price_per_hour), "start": self._now(),
                                 "end": None}
            self._save(doc)

    def set_price(self, name: str, price_per_hour: float) -> None:
        """Replace the cap-priced write-ahead entry with the listed price."""
        with self._lock:
            doc = self._load()
            if name in doc["pods"]:
                doc["pods"][name]["price"] = float(price_per_hour)
                self._save(doc)

    def close(self, name: str) -> None:
        with self._lock:
            doc = self._load()
            if name in doc["pods"] and doc["pods"][name]["end"] is None:
                doc["pods"][name]["end"] = self._now()
                self._save(doc)

    def live(self) -> list[str]:
        return [n for n, p in self._load()["pods"].items() if p["end"] is None]

    def spent_today(self) -> float:
        """USD accrued since 00:00 UTC (pods still open accrue to now)."""
        now = self._now()
        day0 = dt.datetime.fromtimestamp(now, dt.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp()
        total = 0.0
        for p in self._load()["pods"].values():
            start, end = max(p["start"], day0), p["end"] if p["end"] is not None else now
            if end > start:
                total += (end - start) / 3600.0 * p["price"]
        return total

    def spent_total(self) -> float:
        now = self._now()
        return sum(((p["end"] if p["end"] is not None else now) - p["start"]) / 3600.0
                   * p["price"] for p in self._load()["pods"].values())


# --------------------------------------------------------------------------- #
# lium                                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class _Pod:
    name: str
    ip: str
    port: int
    busy: bool = False
    last_used: float = field(default_factory=time.time)
    code_synced: bool = False
    image: str = ""
    reserved: float = 0.0             # cap reservation of the job it is running


SshFn = Callable[[str, int, str, float], subprocess.CompletedProcess]


class TarSsh:
    """File transfer as tar (or cat) streamed over the pod's SSH.

    The worker image ships sshd and tar but no rsync, the same reason the
    trainer's bench hook streams tar: nothing beyond the image is assumed."""

    def __init__(self, ssh_base: Callable[[int], list[str]], timeout: float = 3600.0) -> None:
        self._base, self.timeout = ssh_base, timeout

    def _remote(self, pod, cmd: str) -> list[str]:
        return [*self._base(pod.port), f"root@{pod.ip}", cmd]

    def push(self, pod, src: Path, dst: str) -> bool:
        q = shlex.quote(dst)
        if src.is_dir():
            tar = subprocess.Popen(["tar", "-C", str(src), "--exclude=__pycache__", "-cf", "-",
                                    "."], stdout=subprocess.PIPE)
            r = subprocess.run(self._remote(pod, f"rm -rf {q} && mkdir -p {q} && "
                                                 f"tar -C {q} -xf -"),
                               stdin=tar.stdout, capture_output=True, timeout=self.timeout,
                               check=False)
            tar.stdout.close()
            return tar.wait() == 0 and r.returncode == 0
        parent = shlex.quote(str(Path(dst).parent))
        with open(src, "rb") as f:
            r = subprocess.run(self._remote(pod, f"mkdir -p {parent} && cat > {q}"), stdin=f,
                               capture_output=True, timeout=self.timeout, check=False)
        return r.returncode == 0

    def pull_file(self, pod, src: str, dst: Path) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f:
            r = subprocess.run(self._remote(pod, f"cat {shlex.quote(src)}"), stdout=f,
                               stderr=subprocess.PIPE, timeout=self.timeout, check=False)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        return True

    def pull_dir(self, pod, src: str, dst: Path) -> bool:
        dst.mkdir(parents=True, exist_ok=True)
        ssh = subprocess.Popen(self._remote(pod, f"tar -C {shlex.quote(src)} -cf - ."),
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        r = subprocess.run(["tar", "-C", str(dst), "-xf", "-"], stdin=ssh.stdout,
                           capture_output=True, timeout=self.timeout, check=False)
        ssh.stdout.close()
        return ssh.wait() == 0 and r.returncode == 0


_RENT_REFUSED = __import__("re").compile(r'"status_code":\s*4\d\d|\bError:', __import__("re").I)


class BudgetExhausted(RuntimeError):
    pass


class LiumExecutor:
    def __init__(self, workdir: Path, cfg, *, provider=None, ssh: SshFn | None = None,
                 transfer=None, chain_toml: Path | None = None,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """``cfg`` is the ``[compute]`` section (:class:`ComputeConfig`);
        ``chain_toml`` (the harness's) is shipped to every pod so jobs run
        under the same config as the judge."""
        self.workdir = Path(workdir)
        self.cfg = cfg
        self.chain_toml = Path(chain_toml) if chain_toml else None
        # Worst-case cost of jobs in flight: reserved against the caps so that
        # parallel jobs can never jointly overshoot them.
        self._committed = 0.0
        self.capacity = cfg.max_parallel
        if cfg.daily_usd_cap <= 0 or cfg.max_price_per_hour <= 0:
            raise ValueError("lium executor needs daily_usd_cap > 0 and max_price_per_hour > 0")
        self.ledger = SpendLedger(self.workdir / "spend.json", now=now)
        self._now, self._sleep = now, sleep
        self.key = Path(cfg.ssh_key).expanduser()
        self.known_hosts = self.workdir / "known_hosts"
        self._ssh = ssh or self._ssh_default
        self.transfer = transfer or TarSsh(self._ssh_base)
        self._pods: dict[str, _Pod] = {}
        # Machines that failed to boot or pass health: never rented again by
        # this executor (the listing is deterministic, so a retry would pick the
        # same lemon).
        self._bad_executors: set[str] = set()
        self._cv = threading.Condition()
        self._seq = 0
        if provider is None:
            from ...provision.core import LiumProvider
            provider = LiumProvider(max_price_per_hour=cfg.max_price_per_hour)
        self.provider = provider
        self._image = cfg.image or self._contract_image()
        self.reconcile()

    # -- plumbing ----------------------------------------------------------
    def _contract_image(self) -> str:
        from ...shared.config import load_chain_config
        img = str(getattr(load_chain_config(self.chain_toml).round, "funded_pod_image", "")
                  or "")
        if not img:
            raise ValueError("set [compute] image (a digest-pinned cascade-worker image)")
        return img

    def _ssh_base(self, port: int) -> list[str]:
        return ["ssh", "-i", str(self.key), "-p", str(port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "ConnectTimeout=20"]

    def _ssh_default(self, ip: str, port: int, cmd: str, timeout: float):
        return subprocess.run([*self._ssh_base(port), f"root@{ip}", cmd],
                              capture_output=True, text=True, timeout=timeout, check=False)

    # -- pods --------------------------------------------------------------
    def reconcile(self) -> None:
        """Tear down any pod with our prefix that this executor does not hold —
        a previous run that crashed before teardown must never keep billing."""
        try:
            tagged = self.provider.list_tagged(self.cfg.pod_prefix)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot list pods to reconcile: %s", e)
            tagged = []
        for name in set(tagged) | set(self.ledger.live()):
            if name not in self._pods:
                log.warning("tearing down orphan pod %s", name)
                self._terminate(name)

    def _terminate(self, name: str) -> None:
        """Tear a pod down and BELIEVE THE LISTING, not the call: a teardown the
        CLI did not perform (observed: lium 0.9.1 exits 2 on `rm` without --yes,
        which the provider logs as "already terminated") keeps the pod in the
        ledger, still billing, and the reaper retries it."""
        from ...provision.funded import terminate_verified

        try:
            gone = terminate_verified(self.provider, name)
        except Exception as e:  # noqa: BLE001
            log.error("terminate %s failed: %s (check the Lium console)", name, e)
            gone = False
        self._pods.pop(name, None)
        if gone:
            self.ledger.close(name)
        else:
            log.error("pod %s is STILL RUNNING after teardown; it stays in the spend "
                      "ledger and is retried by the reaper", name)

    def _mark_bad(self, name: str) -> None:
        machine = getattr(self.provider, "machine_of", lambda n: None)(name)
        if machine:
            self._bad_executors.add(str(machine))
            log.warning("excluding machine %s after pod %s failed", machine, name)

    def _job_hours(self, est_hours: float) -> float:
        """The longest a job may run: its estimate with slack, never above
        job_timeout. The SSH call enforces it, so the cap's reservation holds."""
        return min(self.cfg.job_timeout / 3600.0, est_hours * 1.25 + 0.5)

    def _job_cost_bound(self, est_hours: float) -> float:
        return self._job_hours(est_hours) * self.cfg.max_price_per_hour

    def _launch(self) -> _Pod:
        from ...provision.core import LaunchSpec

        pub = Path(str(self.key) + ".pub")
        if not pub.is_file():
            raise RuntimeError(f"no SSH public key at {pub} (ssh-keygen -t ed25519 -f {self.key})")
        self._seq += 1
        prefix = f"{self.cfg.pod_prefix}-{int(self._now())}-{self._seq}"
        spec = LaunchSpec(sku=self.cfg.sku, count=1, image=self._image,
                          ssh_pubkey=pub.read_text().strip(), name_prefix=prefix,
                          sku_choices=tuple(self.cfg.sku_choices),
                          exclude_ids=tuple(sorted(self._bad_executors)))
        name = f"{prefix}-0"
        self.ledger.open(name, self.cfg.max_price_per_hour)     # write-ahead
        try:
            names = self.provider.launch(spec)
            name = names[0] if names else name
            price = getattr(self.provider, "price_of", lambda n: None)(name)
            if price is not None and 0 < price <= self.cfg.max_price_per_hour:
                self.ledger.set_price(name, price)        # bill what it really costs
            self._wait_ready(name)
            addr = self.provider.get_ip(name)
            if addr is None:
                raise RuntimeError(f"pod {name} has no address")
        except Exception:
            self._mark_bad(name)
            self._terminate(name)
            raise
        pod = _Pod(name, addr.ip, int(addr.ssh_port), last_used=self._now(), image=self._image)
        # A real CUDA op through the image's pinned torch: a GPU the build does
        # not support (e.g. Blackwell under cu124) fails here, at boot, not
        # inside a paid job.
        r = self._ssh(pod.ip, pod.port,
                      f"nvidia-smi -L && {REMOTE_PY} -c 'import torch; "
                      f"torch.ones(8, device=\"cuda\").sum().item()'", 300.0)
        if r.returncode != 0:
            self._mark_bad(name)
            self._terminate(name)
            raise RuntimeError(f"pod {name} failed its health check: {r.stderr[-300:]}")
        log.info("pod %s up at %s:%d", name, pod.ip, pod.port)
        return pod

    def _wait_ready(self, name: str) -> None:
        """Wait for the pod, but fail fast when Lium already refused the rental
        (``lium up`` exits with a 4xx — e.g. "another rental is in progress on
        this node") instead of sitting out the whole boot timeout."""
        tail = getattr(self.provider, "_up_log_tail", None)
        for _ in range(max(1, math.ceil(self.cfg.boot_timeout / 60.0))):
            if self.provider.wait_ready(name, timeout=60.0):
                return
            text = tail(name) if callable(tail) else ""
            if text and _RENT_REFUSED.search(text):
                raise RuntimeError(f"lium refused the rental of {name}: "
                                   f"{' '.join(text.split())[-200:]}")
        raise RuntimeError(f"pod {name} not ready after {self.cfg.boot_timeout:.0f}s")

    def _acquire(self, est_hours: float) -> _Pod:
        bound = self._job_cost_bound(est_hours)
        with self._cv:
            while True:
                # The cap gates EVERY job, a warm pod included (a job keeps its
                # pod alive, and a live pod bills), and counts the worst case of
                # every job already running, so parallel jobs cannot overshoot.
                committed = self._committed + bound
                spent = self.ledger.spent_today()
                if spent + committed > self.cfg.daily_usd_cap:
                    raise BudgetExhausted(
                        f"daily cap ${self.cfg.daily_usd_cap:.2f}: ${spent:.2f} accrued + "
                        f"${self._committed:.2f} reserved by running jobs + this job's "
                        f"${bound:.2f}")
                cap = float(getattr(self.cfg, "total_usd_cap", 0.0) or 0.0)
                total = self.ledger.spent_total()
                if cap and total + committed > cap:
                    raise BudgetExhausted(
                        f"total cap ${cap:.2f}: ${total:.2f} accrued + ${self._committed:.2f} "
                        f"reserved + this job's ${bound:.2f}")
                idle = [p for p in self._pods.values() if not p.busy]
                if idle:
                    pod = idle[0]
                    pod.busy, pod.reserved = True, bound
                    self._committed += bound
                    return pod
                if len(self._pods) < self.capacity:
                    self._committed += bound          # reserve before renting
                    break
                self._cv.wait(timeout=60.0)
        try:
            pod = self._launch()
        except Exception:
            with self._cv:
                self._committed -= bound
                self._cv.notify_all()
            raise
        pod.busy, pod.reserved = True, bound
        with self._cv:
            self._pods[pod.name] = pod
        return pod

    def set_image(self, image: str) -> None:
        """Rent ``image`` from now on (the round contract's worker image). Idle
        pods on another image are retired at once, busy ones when released."""
        with self._cv:
            if not image or image == self._image:
                return
            log.info("worker image -> %s", image)
            self._image = image
            for pod in list(self._pods.values()):
                if not pod.busy and pod.image != image:
                    self._terminate(pod.name)

    @property
    def image(self) -> str:
        return self._image

    def _release(self, pod: _Pod, *, dead: bool = False) -> None:
        with self._cv:
            self._committed = max(0.0, self._committed - pod.reserved)
            pod.reserved = 0.0
            if dead or pod.image != self._image:
                self._terminate(pod.name)
            else:
                pod.busy, pod.last_used = False, self._now()
            self._cv.notify_all()

    def reap(self) -> None:
        """Tear down idle pods past ``idle_minutes``, every idle pod at the cap,
        and retry any pod a failed teardown left running (still in the ledger,
        no longer held)."""
        for name in self.ledger.live():
            if name not in self._pods:
                self._terminate(name)
        cap = float(getattr(self.cfg, "total_usd_cap", 0.0) or 0.0)
        at_cap = (self.ledger.spent_today() >= self.cfg.daily_usd_cap
                  or bool(cap and self.ledger.spent_total() >= cap))
        with self._cv:
            for pod in list(self._pods.values()):
                idle_for = self._now() - pod.last_used
                if not pod.busy and (at_cap or idle_for > self.cfg.idle_minutes * 60):
                    log.info("tearing down %s (%s)", pod.name,
                             "daily cap reached" if at_cap else f"idle {idle_for/60:.0f} min")
                    self._terminate(pod.name)

    def close(self) -> None:
        with self._cv:
            for name in list(self._pods):
                self._terminate(name)

    # -- jobs --------------------------------------------------------------
    def _sync_code(self, pod: _Pod) -> bool:
        """This checkout's ``cascade/`` + ``chain*.toml`` over the image's
        editable install (once per pod)."""
        if pod.code_synced:
            return True
        ok = self.transfer.push(pod, REPO_ROOT / "cascade", f"{REMOTE_ROOT}/cascade")
        for toml in ("chain.toml", "chain.testnet.toml"):
            if ok and (REPO_ROOT / toml).is_file():
                ok = self.transfer.push(pod, REPO_ROOT / toml, f"{REMOTE_ROOT}/{toml}")
        if ok and self.chain_toml is not None:
            ok = self.transfer.push(pod, self.chain_toml, REMOTE_HARNESS_TOML)
        pod.code_synced = ok
        return ok

    def run(self, spec: dict, *, job_id: str | None = None, est_hours: float = 1.0) -> dict:
        job_id = job_id or uuid.uuid4().hex[:12]
        try:
            pod = self._acquire(est_hours)
        except BudgetExhausted as e:
            return {"error": str(e), "candidate_fault": False, "budget_exhausted": True}
        except Exception as e:  # noqa: BLE001
            return _infra_error(f"no pod: {e}")
        dead = False
        try:
            return self._run_on(pod, spec, job_id, est_hours)
        except (subprocess.TimeoutExpired, OSError) as e:
            dead = True
            return _infra_error(f"pod {pod.name}: {e}")
        finally:
            self._release(pod, dead=dead)

    def _run_on(self, pod: _Pod, spec: dict, job_id: str, est_hours: float = 1.0) -> dict:
        if not self._sync_code(pod):
            raise OSError("code sync failed")
        rj = f"{REMOTE_JOBS}/{job_id}"
        self._ssh(pod.ip, pod.port, f"mkdir -p {rj}/in {rj}/out", 60.0)
        remote = json.loads(json.dumps(spec))
        for k, v in spec.get("inputs", {}).items():
            src = Path(v)
            if k in ("snapshot", "pool"):
                # Pools are cached on the pod by folder name (a revealed folder
                # is immutable): ~100 MB, sent once per pod.
                dst = f"{REMOTE_JOBS}/cache/{src.name}"
                have = self._ssh(pod.ip, pod.port,
                                 f"test -f {shlex.quote(dst)}/metadata.json", 60.0)
                if have.returncode != 0 and not self.transfer.push(pod, src, dst):
                    raise OSError(f"staging {k} failed")
            else:
                dst = f"{rj}/in/{k}"
                if not self.transfer.push(pod, src, dst):
                    raise OSError(f"staging {k} failed")
            remote["inputs"][k] = dst
        for k in spec.get("outputs", {}):
            remote["outputs"][k] = f"{rj}/out/{k}"
        params = remote.setdefault("params", {})
        params.setdefault("device", "cuda")
        params.setdefault("cache_dir", f"{REMOTE_JOBS}/cache")
        if self.chain_toml is not None:
            params.setdefault("chain_toml", REMOTE_HARNESS_TOML)
        local_spec = self.workdir / "jobs" / job_id / "spec.remote.json"
        local_spec.parent.mkdir(parents=True, exist_ok=True)
        local_spec.write_text(json.dumps(remote), encoding="utf-8")
        if not self.transfer.push(pod, local_spec, f"{rj}/spec.json"):
            raise OSError("staging spec failed")
        cmd = (f"cd {REMOTE_ROOT} && CUBLAS_WORKSPACE_CONFIG=:4096:8 {REMOTE_PY} -m "
               f"cascade.miner.harness.jobs {rj}/spec.json {rj}/result.json "
               f"> {rj}/job.log 2>&1; true")
        self._ssh(pod.ip, pod.port, cmd, self._job_hours(est_hours) * 3600.0)
        local_result = self.workdir / "jobs" / job_id / "result.json"
        self.transfer.pull_file(pod, f"{rj}/job.log", local_result.with_name("job.log"))
        if not self.transfer.pull_file(pod, f"{rj}/result.json", local_result):
            return _infra_error(f"job {job_id} on {pod.name} wrote no result")
        for k, v in spec.get("outputs", {}).items():
            if not self.transfer.pull_dir(pod, f"{rj}/out/{k}", Path(v)):
                log.warning("job %s: output %s not retrieved", job_id, k)
        self._ssh(pod.ip, pod.port, f"rm -rf {rj}", 60.0)
        return json.loads(local_result.read_text(encoding="utf-8"))


def make_executor(hcfg, workdir: Path):
    c = hcfg.compute
    if c.executor == "lium":
        return LiumExecutor(workdir, c, chain_toml=hcfg.chain_toml)
    return LocalExecutor(workdir, max_parallel=c.max_parallel, device=c.device,
                         chain_toml=hcfg.chain_toml, timeout=c.job_timeout,
                         sandbox=c.local_sandbox)
