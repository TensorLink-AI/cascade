"""The orchestrator's generation stage (DEC-CA-0053): one leg's corpus, made
on a rented CPU VM and handed to the GPU pod as download links.

For a leg whose contract trains on ``shards``:

1. rent the cheapest CPU VM the configured providers offer at the pinned
   shape (``[generation] vcpus`` under ``max_price_per_hour``);
2. bootstrap it with the same script that provisions bare GPU pods
   (``deploy/pod_bootstrap.sh``: source tree + the pinned ``uv sync``);
3. copy the generator repo and a set of pre-signed upload links, and run
   ``cascade-generate`` (sandboxed children, one per vCPU) — it uploads every
   shard and then ``corpus.json`` through the links;
4. tag the corpus in the dataset (``meta.json`` + index row);
5. terminate the VM, whatever happened.

A leg whose corpus is already in the dataset (same round id and hotkey, a
complete, matching ``corpus.json``) reuses it instead of renting a VM
(:meth:`GenerationStage.existing`). Every VM is written to a crash ledger
(``ledger_path``) before its launch and dropped only after a successful
terminate; :func:`reap_generation_vms` terminates what a crash or a failed
terminate left behind.

The VM never holds a storage credential (only per-object upload links) and
the GPU pod later gets only ``corpus.json`` plus per-shard download links
(:meth:`GenerationStage.pod_files`). A generator fault is the miner's
(:class:`GenerationFault`); everything else — no capacity, a VM that never
boots, a failed upload — is infrastructure (:class:`GenerationInfraError`)
and the leg requeues.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..provision.cpu import CpuProvider, select_cpu_offer
from ..shared.atomic import path_lock, write_text_atomic
from ..trainer.remote import RemoteHost, build_scp_argv, build_ssh_argv, run_ssh
from .dataset import _row, leg_prefix, read_dataset_index, register_corpus
from .shards import INDEX_NAME, URLS_NAME
from .transfer import download_plan, shard_names, upload_plan

log = logging.getLogger("cascade.generation.stage")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXIT_GENERATOR = 3                  # cascade-generate: the generator faulted


class GenerationFault(RuntimeError):
    """The miner's generator faulted on the CPU tier (their shot)."""

    returncode = 3


class GenerationInfraError(RuntimeError):
    """The tier could not produce the corpus for reasons that are not the
    miner's (capacity, boot, bootstrap, transfer). The leg requeues."""


@dataclass
class StageResult:
    index: dict
    base: str                       # the corpus's key prefix in the store
    provider: str
    shape: str
    row: dict


VM_NAME_PREFIX = "cgen-"


def _vm_name(hotkey: str, round_id: str) -> str:
    raw = f"{VM_NAME_PREFIX}{hotkey[:10]}-{str(round_id)[-8:]}".lower()
    return re.sub(r"[^a-z0-9-]", "-", raw)[:32]


# ── the VM crash ledger ─────────────────────────────────────────────────────
#
# One JSON list of {provider, vm_id, name, created} rows. A row is written
# BEFORE the launch (vm_id "" — the intent) and completed when the launch
# returns, so a crash anywhere leaves a row; it is removed only after a
# successful terminate. Rows of stages live in THIS process (``_LIVE``) are
# never reaped; while any launch is mid-flight (``_LAUNCHING``) the listing
# pass is skipped, since its VM may be up before its id is on the ledger.

_LIVE: set[str] = set()                   # names owned by a running stage
_LAUNCHING: set[str] = set()              # names whose launch has not returned
_LIVE_GUARD = threading.Lock()


def _ledger_load(path: Path) -> list[dict]:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as e:  # noqa: BLE001 — do not overwrite orphan billing evidence
        raise GenerationInfraError(f"generation VM ledger {path} unreadable: {e}") from e
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise GenerationInfraError(f"generation VM ledger {path} is malformed")
    return rows


def _ledger_update(path: Path, fn: Callable[[list[dict]], list[dict]]) -> None:
    with path_lock(path):
        write_text_atomic(path, json.dumps(fn(_ledger_load(path)), indent=1))


def _ledger_put(path: Path | None, row: dict) -> None:
    if path is None:
        return

    def update(rows):
        if not row.get("vm_id") and any(r.get("name") == row["name"] for r in rows):
            # The previous launch may still be billing (including when its
            # response was lost). Never erase its evidence or launch a twin.
            raise GenerationInfraError("prior CPU launch needs reconciliation")
        return [r for r in rows if r.get("name") != row["name"]] + [row]

    try:
        _ledger_update(path, update)
    except Exception as e:  # noqa: BLE001 — logged loudly; the reaper's listing remains
        raise GenerationInfraError(
            f"generation VM ledger write failed for {row.get('name')}: {e}") from e


def _ledger_drop(path: Path | None, name: str) -> None:
    if path is None:
        return
    try:
        _ledger_update(path, lambda rows: [r for r in rows if r.get("name") != name])
    except Exception as e:  # noqa: BLE001
        log.error("generation VM ledger drop failed for %s: %s", name, e)


def reap_generation_vms(ledger_path: Path | str, providers: Sequence[CpuProvider]) -> list[str]:
    """Terminate generation VMs no live stage owns; returns the VM ids stopped.

    1. Every ledger row not owned by a stage running in this process: its VM
       is terminated and the row dropped (a row whose provider is no longer
       configured, or an intent row with no id that no listing can resolve,
       stays recorded for manual reconciliation).
    2. Providers that can list (``list_tagged``: Shadeform) are swept for
       ``cgen-`` VMs that are on no ledger row. Targon resolves intent rows
       with an exact-name lookup, without an organization-wide sweep.
    """
    path = Path(ledger_path)
    by_name = {p.name: p for p in providers}
    with _LIVE_GUARD:
        launching = bool(_LAUNCHING)
    reaped: list[str] = []
    rows = _ledger_load(path)
    for row in rows:
        name, vm = str(row.get("name", "")), str(row.get("vm_id", "") or "")
        with _LIVE_GUARD:
            if name in _LIVE:
                continue
        prov = by_name.get(str(row.get("provider", "")))
        if prov is None or not vm:
            ids: list[str] = []
            lister = (getattr(prov, "find_by_name", None)
                      or getattr(prov, "list_tagged", None)) if prov is not None else None
            if lister is not None:
                try:
                    ids = list(lister(name))
                except Exception as e:  # noqa: BLE001 — retried next reap
                    log.warning("generation reaper: %s listing failed (%s)", prov.name, e)
                    continue
            if prov is None or (not vm and lister is None):
                log.error("generation reaper: cannot resolve VM %s (%s, provider %r) — "
                          "stop it by hand if it exists", name, vm or "no id",
                          row.get("provider"))
                continue
            if not ids:
                # Empty listings can be eventually consistent after a timed-out
                # create. Never erase the intent and permit a duplicate rent.
                log.warning("generation reaper: %s not yet visible; intent retained", name)
                continue
            vm_ids = ids
        else:
            vm_ids = [vm]
        # A stage can start while provider lookup is in flight. Its durable
        # intent may have appeared after the initial ledger/live snapshot.
        with _LIVE_GUARD:
            if name in _LIVE:
                continue
        try:
            for v in vm_ids:
                prov.terminate(v)
                reaped.append(v)
        except Exception as e:  # noqa: BLE001 — the row stays for the next reap
            log.error("generation reaper: terminate of %s VM %s failed (%s) — kept on "
                      "the ledger", prov.name, name, e)
            continue
        _ledger_drop(path, name)
        log.warning("generation reaper: terminated orphan %s VM %s (%s)", prov.name, name,
                    ", ".join(vm_ids) or "none listed")
    if launching:
        return reaped
    # Ledgered VMs (live or kept for a retry) and the ones just stopped (a
    # listing can lag a delete) are never swept again here.
    known = {str(r.get("vm_id")) for r in _ledger_load(path) if r.get("vm_id")} | set(reaped)
    for prov in providers:
        lister = getattr(prov, "list_tagged", None)
        if lister is None:
            continue
        try:
            ids = [v for v in lister(VM_NAME_PREFIX) if v not in known]
        except Exception as e:  # noqa: BLE001
            log.warning("generation reaper: %s listing failed (%s)", prov.name, e)
            continue
        with _LIVE_GUARD:
            if _LAUNCHING:
                # Prefix listings expose ids, not names. A newly started VM
                # can be visible before its id reaches the ledger; defer this
                # sweep instead of deleting a generation that is running.
                return reaped
        # Launches that finished DURING the listing now have a durable id.
        # Exclude those too, rather than using only the pre-listing snapshot.
        known.update(str(r["vm_id"]) for r in _ledger_load(path) if r.get("vm_id"))
        for v in ids:
            if v in known:
                continue
            try:
                prov.terminate(v)
                reaped.append(v)
                log.warning("generation reaper: terminated unledgered %s VM %s",
                            prov.name, v)
            except Exception as e:  # noqa: BLE001
                log.error("generation reaper: terminate of unledgered %s VM %s failed: %s",
                          prov.name, v, e)
    return reaped


def _last_json(text: str) -> dict:
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


@dataclass
class GenerationStage:
    cfg: object                                 # ChainConfig
    store: object                               # S3Store for the corpus bucket (presigns)
    providers: Sequence[CpuProvider]
    ssh_key_path: str
    ssh_pubkey: str
    vm_image: str = ""                          # provider VM image (Targon needs one)
    workdir: str = "cascade"
    bootstrap_script: Path = REPO_ROOT / "deploy" / "pod_bootstrap.sh"
    ready_timeout_s: float = 1800.0
    bootstrap_timeout_s: float = 1800.0
    # Slack past the generation window for bootstrap-free work: the final
    # digest pass and the upload of every shard.
    upload_slack_s: float = 3600.0
    runner: Callable = run_ssh
    shell: Callable = field(default=subprocess.run)
    # The VM crash ledger (see reap_generation_vms); None ⇒ unledgered (tests).
    ledger_path: Path | None = None
    # CPU drought: how long to keep re-probing the market before the leg
    # fails as infra (0 ⇒ fail on the first empty probe), and the first
    # re-probe delay (doubling, capped at five minutes).
    capacity_wait_s: float = 0.0
    capacity_poll_s: float = 60.0
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    # ── the stage ────────────────────────────────────────────────────────────

    def generate(self, *, round_id: str, hotkey: str, uid: int | None, era: int | None,
                 gen_ref: str, repo_dir: Path | str, seed: int,
                 priority: bool = False) -> StageResult:
        from ..trainer.admission import FairSlots

        if not self.cfg.generation.enabled:
            raise GenerationInfraError("[generation] enabled = false")
        reused = self.existing(round_id=round_id, hotkey=hotkey, gen_ref=gen_ref, seed=seed)
        if reused is not None:
            return reused
        gate = self.__dict__.setdefault("_slots", FairSlots(clock=self.clock))
        deadline = self.clock() + self.capacity_wait_s if self.capacity_wait_s > 0 else None
        with gate.slot(lambda: self.cfg.generation.max_parallel_jobs,
                       deadline=deadline, priority=priority) as acquired:
            if not acquired:
                raise GenerationInfraError("CPU generation admission is full — requeued")
            return self._generate_admitted(round_id=round_id, hotkey=hotkey, uid=uid, era=era,
                                           gen_ref=gen_ref, repo_dir=repo_dir, seed=seed)

    def _generate_admitted(self, *, round_id, hotkey, uid, era, gen_ref, repo_dir, seed):
        gcfg = self.cfg.generation
        if not gcfg.enabled:
            raise GenerationInfraError("[generation] enabled = false")
        reused = self.existing(round_id=round_id, hotkey=hotkey, gen_ref=gen_ref, seed=seed)
        if reused is not None:
            return reused
        pick = self._wait_for_offer(gcfg)
        provider, shape = pick
        name = _vm_name(hotkey, round_id)
        with _LIVE_GUARD:
            if name in _LIVE:
                raise GenerationInfraError("CPU generation for this identity is already running")
            _LIVE.add(name)
            _LAUNCHING.add(name)
        try:
            return self._generate_on(provider, shape, name, round_id=round_id, hotkey=hotkey,
                                     uid=uid, era=era, gen_ref=gen_ref, repo_dir=repo_dir,
                                     seed=seed)
        finally:
            with _LIVE_GUARD:
                _LIVE.discard(name)
                _LAUNCHING.discard(name)

    def _wait_for_offer(self, gcfg):
        """The CPU offer to rent, re-probing a sold-out market with backoff for
        up to ``capacity_wait_s`` — a drought is the operator tier's, never a
        reason to fail the leg at once (it fails as infra only after that)."""
        deadline = self.clock() + float(self.capacity_wait_s)
        delay = max(1.0, float(self.capacity_poll_s))
        while True:
            pick = select_cpu_offer(self.providers, gcfg)
            if pick is not None:
                return pick
            left = deadline - self.clock()
            if left <= 0:
                raise GenerationInfraError(
                    f"no CPU capacity: {gcfg.vcpus} vCPU ≤ ${gcfg.max_price_per_hour:.2f}/h "
                    f"on {', '.join(gcfg.providers)}"
                    + (f" (waited {self.capacity_wait_s:.0f}s)" if self.capacity_wait_s else ""))
            log.warning("generation: no CPU capacity (%d vCPU ≤ $%.2f/h) — re-probing in "
                        "%.0fs", gcfg.vcpus, gcfg.max_price_per_hour, min(delay, left))
            self.sleep(min(delay, left))
            delay = min(delay * 2, 300.0)

    def _generate_on(self, provider, shape, name: str, *, round_id: str, hotkey: str,
                     uid: int | None, era: int | None, gen_ref: str, repo_dir: Path | str,
                     seed: int) -> StageResult:
        gcfg = self.cfg.generation
        row = {"provider": provider.name, "vm_id": "", "name": name,
               "created": time.time(), "round_id": str(round_id), "hotkey": hotkey}
        _ledger_put(self.ledger_path, row)             # write-ahead: the intent
        try:
            vm = provider.launch(shape, name=name, ssh_pubkey=self.ssh_pubkey,
                                 image=self.vm_image)
        except Exception as e:  # noqa: BLE001 — a market miss is infra
            # The row stays: a launch that raised may still have created a
            # VM; the reaper resolves it (or logs it for a manual stop).
            raise GenerationInfraError(f"{provider.name}: launch failed: {e}") from e
        try:
            _ledger_put(self.ledger_path, {**row, "vm_id": str(vm)})
            with _LIVE_GUARD:
                _LAUNCHING.discard(name)
            log.info("generation %s round=%s: %s %s (%d vCPU, $%.2f/h) → %s", hotkey[:12],
                     round_id, provider.name, shape.type_id, shape.vcpus, shape.price_per_hour, vm)
            host = self._ready_host(provider, vm, name)
            self._bootstrap(host)
            base = leg_prefix(round_id, hotkey, gcfg.storage_prefix)
            summary = self._run(host, repo_dir, seed=seed, gen_ref=gen_ref,
                                urls=upload_plan(self.store, base, gcfg))
            index = self._uploaded_index(base, summary)
            row = register_corpus(self.store, index, round_id=round_id, hotkey=hotkey,
                                  uid=uid, era=era, provider=provider.name,
                                  shape=shape.type_id, prefix=gcfg.storage_prefix)
            log.info("generation %s round=%s: corpus %s — %d series, %d points, stop=%s",
                     hotkey[:12], round_id, index["corpus_digest"][:12], index["n_series"],
                     index["points"], index.get("stop_reason", ""))
            return StageResult(index=index, base=base, provider=provider.name,
                               shape=shape.type_id, row=row)
        finally:
            try:
                provider.terminate(vm)
                _ledger_drop(self.ledger_path, name)
            except Exception as e:  # noqa: BLE001 — logged loudly; billing keeps running
                log.error("generation %s: TERMINATE FAILED for %s VM %s (%s) — kept on the "
                          "VM ledger for the reaper%s", hotkey[:12], provider.name, vm, e,
                          "" if self.ledger_path else "; stop it by hand")

    def existing(self, *, round_id: str, hotkey: str, gen_ref: str, seed: int
                 ) -> StageResult | None:
        """This leg's corpus when it is already in the dataset — its index row
        (written only after every shard and ``corpus.json`` landed) plus a
        ``corpus.json`` whose digest, generator ref and seed match — so a
        requeued leg or a restarted trainer never generates it twice. Any
        doubt (unreadable, mismatched, expired) ⇒ ``None``: generate anew."""
        gcfg = self.cfg.generation
        try:
            row = _row(read_dataset_index(self.store, gcfg.storage_prefix), round_id, hotkey)
            if row is None or row.get("expired"):
                return None
            base = leg_prefix(round_id, hotkey, gcfg.storage_prefix)
            index = self._uploaded_index(base, {"corpus_digest": row.get("corpus_digest")})
        except Exception as e:  # noqa: BLE001 — never block a leg on the probe
            log.info("generation %s round=%s: no reusable corpus (%s)", hotkey[:12],
                     round_id, e)
            return None
        if (str(index.get("generator_ref", "")) != str(gen_ref)
                or int(index.get("seed", -1)) != int(seed)):
            return None
        # A corpus built before the width policy changed must not be reused
        # for a new leg and only discovered as incompatible on a paid GPU.
        channels = index.get("channels_histogram", {})
        if self.cfg.generator.min_channels > 1:
            try:
                if not channels or any(int(count) <= 0 or not (
                        self.cfg.generator.min_channels <= int(c) <= self.cfg.generator.max_channels)
                        for c, count in channels.items()):
                    return None
            except (ValueError, TypeError, AttributeError):
                return None
        log.info("generation %s round=%s: reusing the uploaded corpus %s (%d series)",
                 hotkey[:12], round_id, str(index["corpus_digest"])[:12],
                 int(index.get("n_series", 0)))
        return StageResult(index=index, base=base, provider=str(row.get("provider", "")),
                           shape=str(row.get("shape", "")), row=row)

    def pod_files(self, result: StageResult) -> dict[str, str]:
        """What the GPU pod receives: the index and one download link per shard."""
        return {
            INDEX_NAME: json.dumps(result.index, sort_keys=True),
            URLS_NAME: json.dumps(download_plan(self.store, result.base, result.index),
                                  sort_keys=True),
        }

    # ── steps ────────────────────────────────────────────────────────────────

    def _ready_host(self, provider: CpuProvider, vm: str, name: str) -> RemoteHost:
        try:
            ready = provider.wait_ready(vm, timeout=self.ready_timeout_s)
            addr = provider.get_ip(vm) if ready else None
        except Exception as e:  # noqa: BLE001
            raise GenerationInfraError(f"{provider.name} VM {vm} never came up: {e}") from e
        if addr is None:
            raise GenerationInfraError(f"{provider.name} VM {vm} not ready within "
                                       f"{self.ready_timeout_s:.0f}s")
        return RemoteHost(name=name, host=addr.ip, port=addr.ssh_port, user=provider.ssh_user,
                          key_path=self.ssh_key_path, workdir=self.workdir,
                          remote_python=f"{self.workdir}/.venv/bin/python")

    def _bootstrap(self, host: RemoteHost) -> None:
        env = dict(os.environ)
        env.update({"POD_IP": host.host, "POD_PORT": str(host.port), "POD_USER": host.user,
                    "POD_KEY": str(Path(self.ssh_key_path).expanduser()),
                    "POD_STAGE": "generation", "POD_WORKDIR": host.workdir})
        try:
            proc = self.shell(["bash", str(self.bootstrap_script)], env=env,
                              timeout=self.bootstrap_timeout_s, capture_output=True, text=True)
        except subprocess.TimeoutExpired as e:
            raise GenerationInfraError(f"bootstrap of {host.host} timed out") from e
        if proc.returncode != 0:
            raise GenerationInfraError(
                f"bootstrap of {host.host} failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[-400:]}")

    def _copy(self, host: RemoteHost, local: str, remote: str, *, recursive: bool = False,
              timeout: float = 900.0) -> None:
        argv = build_scp_argv(host, local, remote)
        if recursive:
            argv.insert(1, "-r")
        proc = self.shell(argv, timeout=timeout, capture_output=True, text=True)
        if proc.returncode != 0:
            raise GenerationInfraError(f"copy to {host.host}:{remote} failed: "
                                       f"{(proc.stderr or '')[-300:]}")

    def _run(self, host: RemoteHost, repo_dir: Path | str, *, seed: int, gen_ref: str,
             urls: dict[str, str]) -> dict:
        gen = f"{host.workdir}/_gen"
        self.runner(build_ssh_argv(host, f"rm -rf {shlex.quote(gen)} && "
                                         f"mkdir -p {shlex.quote(gen)}"), 120)
        self._copy(host, str(Path(repo_dir)), f"{gen}/repo", recursive=True)
        with tempfile.TemporaryDirectory() as tmp:
            # The links travel as a file, never on a command line (ps-visible).
            p = Path(tmp) / URLS_NAME
            p.write_text(json.dumps(urls), encoding="utf-8")
            self._copy(host, str(p), f"{gen}/{URLS_NAME}")
        cmd = (f"cd {shlex.quote(host.workdir)} && .venv/bin/cascade-generate "
               f"_gen/repo _gen/out --seed {int(seed)} "
               f"--generator-ref {shlex.quote(gen_ref)} "
               f"--upload-urls _gen/{URLS_NAME} --json")
        timeout = int(self.cfg.generation.window_seconds + self.upload_slack_s)
        try:
            proc = self.runner(build_ssh_argv(host, cmd), timeout)
        except subprocess.TimeoutExpired as e:
            raise GenerationInfraError(f"cascade-generate on {host.host} timed out") from e
        summary = _last_json(proc.stdout)
        if proc.returncode == EXIT_GENERATOR:
            faults = summary.get("faults") or []
            why = faults[0].get("error", "") if faults else (proc.stderr or "")[-300:]
            raise GenerationFault(f"miner submission rejected: generator fault: {why}")
        if proc.returncode != 0 or not summary.get("ok"):
            raise GenerationInfraError(
                f"cascade-generate on {host.host} failed (rc={proc.returncode}): "
                f"{summary.get('error') or (proc.stderr or '')[-400:]}")
        return summary

    def _uploaded_index(self, base: str, summary: dict) -> dict:
        try:
            index = json.loads(self.store.get_text(base + INDEX_NAME))
        except Exception as e:  # noqa: BLE001
            raise GenerationInfraError(f"uploaded corpus index unreadable: {e}") from e
        if index.get("corpus_digest") != summary.get("corpus_digest"):
            raise GenerationInfraError("uploaded corpus.json does not match the run's digest")
        # The pod joins these names onto its corpus dir: only the planned ones.
        allowed = set(shard_names(self.cfg.generation))
        bad = [sh.get("name") for c in index.get("children", [])
               for sh in c.get("shards", []) if sh.get("name") not in allowed]
        if bad:
            raise GenerationFault(f"miner submission rejected: corpus index names "
                                  f"unplanned shards: {bad[:3]}")
        return index
