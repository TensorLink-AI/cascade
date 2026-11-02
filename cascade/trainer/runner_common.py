"""Shared value types and small contract/keyword helpers for the trainer runner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..shared.atomic import path_lock, write_text_atomic

log = logging.getLogger("cascade.trainer")


@dataclass(frozen=True)
class ResolvedGenerator:
    hotkey: str
    uid: int
    ref: str           # generator's Hippius Hub reference (repo@digest)
    reveal_block: int = 0  # block the pointer became publicly readable at


def funded_queue_of(runner, *, lane="a"):
    """Fresh file-backed lane queue, resolved against the same work root and TTL."""
    if runner.cfg.round.funded_mode == "off":
        return None
    from ..funding.queue import (
        FundedQueue,
        LaneBQueue,
        LaneCQueue,
        lane_b_queue_path,
        lane_c_queue_path,
    )

    path = Path(runner.cfg.round.funded_queue_path)
    if not path.is_absolute():
        path = Path(runner.work_root) / path
    if lane == "a":
        queue_type = FundedQueue
    elif lane == "b":
        queue_type, path = LaneBQueue, lane_b_queue_path(path)
    elif lane == "c":
        queue_type, path = LaneCQueue, lane_c_queue_path(path)
    else:
        raise ValueError(f"unknown funding lane: {lane}")
    return queue_type(path, entry_ttl_seconds=runner.cfg.round.funded_entry_ttl_hours * 3600.0,
                      operator_billed_retry_seconds=float(getattr(
                          runner.cfg.round, "operator_billed_retry_seconds", 60) or 0),
                      operator_fault_cap=int(getattr(runner.cfg.round, "operator_fault_cap", 3)
                                             or 0),
                      operator_pause_resume_seconds=float(getattr(
                          runner.cfg.round, "operator_pause_resume_seconds", 21600) or 0))


def _load_burns(path: Path) -> dict[str, int]:
    """The persisted burn set: hotkey → block it was burned at (best-effort).
    A legacy list (the old lifetime burn) loads at block -1, which no era
    matches — so those hotkeys are free again."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        log.warning("submissions db %s unreadable (%s); starting from empty", path, e)
        return {}
    if isinstance(raw, dict):
        return {str(h): int(b) for h, b in raw.items()}
    return {str(h): -1 for h in raw}


def _load_seen_hotkeys(path: Path) -> set[str]:
    """Every hotkey in the burn set, whatever era it was burned in."""
    return set(_load_burns(path))


def _save_seen_hotkeys(path: Path, seen) -> None:
    """Persist the burn set (best-effort — anti-spam must never abort a round).
    Atomic (tmp + rename): a crash mid-write never leaves a torn set, which
    :func:`_load_burns` would read as EMPTY — un-burning everyone."""
    burns = seen if isinstance(seen, dict) else {str(h): -1 for h in seen}
    try:
        write_text_atomic(path, json.dumps(dict(sorted(burns.items()))))
    except Exception as e:  # noqa: BLE001
        log.warning("could not persist submissions db to %s: %s", path, e)


def _add_seen_hotkeys(path: Path, hotkeys, block: int = -1) -> None:
    """Record ``hotkeys`` as burned at ``block`` (the latest burn wins): load
    + merge + save under the path's lock, so the tick thread and leg threads
    burning at once never drop each other's additions."""
    with path_lock(path):
        burns = _load_burns(path)
        burns.update({str(h): int(block) for h in hotkeys})
        _save_seen_hotkeys(path, burns)


# ── executor claims (never re-picked by a sibling while in use) ─────────────
#
# ``_funded_claimed_execs`` (a set: every rent path reads it as the exclude
# list) is paired with ``_funded_claim_meta`` (machine → (claimed_at,
# instance_id)). A claim is released when the pod rented on it is confirmed
# torn down; a claim older than EXEC_CLAIM_MAX_AGE_SECONDS is dropped as a
# backstop (a teardown path that never released it, or an executor excluded
# for booting a stale image / failing the bench floor — host quarantine keeps
# covering those). Longer than any pod lives: a leg is ≤ a 5 h wall plus rent,
# generation and bench, and the JIT king pod lives one 12 h era.
EXEC_CLAIM_MAX_AGE_SECONDS = 24 * 3600.0


def _claim_meta(runner) -> dict:
    meta = getattr(runner, "_funded_claim_meta", None)
    if meta is None:
        meta = {}
        runner._funded_claim_meta = meta
    return meta


def _exec_lock(runner):
    lock = getattr(runner, "_funded_exec_lock", None)
    return lock if lock is not None else threading.Lock()


def claim_executor(runner, machine_id: str, instance_id: str = "", *,
                   now: float | None = None) -> None:
    """Claim ``machine_id`` for the pod ``instance_id`` rented on it."""
    if not machine_id:
        return
    with _exec_lock(runner):
        runner._funded_claimed_execs.add(machine_id)
        _claim_meta(runner)[machine_id] = (time.time() if now is None else float(now),
                                           str(instance_id or ""))


def release_executor_of(runner, instance_id: str) -> list[str]:
    """Release every claim held for pod ``instance_id`` (it is gone); returns
    the machines released. A claim recorded without an instance is kept."""
    if not instance_id:
        return []
    with _exec_lock(runner):
        meta = _claim_meta(runner)
        gone = [m for m, (_t, iid) in meta.items() if iid == str(instance_id)]
        for m in gone:
            meta.pop(m, None)
            runner._funded_claimed_execs.discard(m)
    return gone


def live_claimed_executors(runner, *, now: float | None = None) -> tuple[str, ...]:
    """The claimed executors (sorted), aged-out claims dropped first."""
    claimed = getattr(runner, "_funded_claimed_execs", None)
    if claimed is None:
        return ()
    t = time.time() if now is None else float(now)
    with _exec_lock(runner):
        meta = _claim_meta(runner)
        stale = [m for m, (at, _iid) in meta.items()
                 if t - at > EXEC_CLAIM_MAX_AGE_SECONDS]
        for m in stale:
            meta.pop(m, None)
            claimed.discard(m)
        if stale:
            log.info("executor claims aged out (> %.0fh): %s",
                     EXEC_CLAIM_MAX_AGE_SECONDS / 3600, ", ".join(sorted(stale)))
        return tuple(sorted(claimed))


def _v2_scheduled(cfg) -> bool:
    return bool(int(getattr(getattr(cfg, "training", None), "v2_from_block", 0) or 0))


def era_contracts(cfg, era) -> list:
    """The throne contracts of ``era`` (resolved at its start block —
    DEC-CA-0050 v2: the MoE from the v2 block). Without a v2 schedule,
    exactly :meth:`ChainConfig.throne_contracts`."""
    if not _v2_scheduled(cfg):
        return cfg.throne_contracts()
    return cfg.throne_contracts_at(int(era.start_block))


def settlement_contracts(cfg, block: int) -> list:
    """The throne contracts of the settlement published at ``block``: those of
    the era that settlement belongs to (a settlement at boundary ``B`` belongs
    to the era containing ``B − 1``)."""
    if not _v2_scheduled(cfg):
        return cfg.throne_contracts()
    from ..shared.config import effective_epoch_blocks
    from ..shared.era import settlement_contract_block

    eb = int(effective_epoch_blocks(cfg.round, int(block)))
    return cfg.throne_contracts_at(
        settlement_contract_block(cfg.round, int(block) // eb * eb))


def router_kw(router_slow: bool) -> dict:
    """``router_slow=True`` as a dispatch keyword only when set (DEC-CA-0050
    slow router): every other leg dispatches with the pre-v2 signature."""
    return {"router_slow": True} if router_slow else {}


def expert_kw(expert_selection: dict | None) -> dict:
    """``expert_selection`` as a keyword only when set (DEC-CA-0050 expert
    legs): every other leg calls trainers and hooks with the exact pre-lane-B
    signature, so third-party ``BaseTrainer``s need no change."""
    return {} if expert_selection is None else {"expert_selection": expert_selection}


# ── harvested checkpoints on the orchestrator's disk ────────────────────────
#
# work_root/_funded_harvest/<base_seed>/<hotkey>/ holds a harvested checkpoint
# from the pull until its upload is verified; nothing reads it after (benches
# and validators fetch the uploaded ref), so a successful upload deletes it.
# An upload that fails after its retries leaves the copy plus a sibling
# ``<hotkey>.pending.json`` record: the requeued leg (same era seeds, ref,
# size, init) re-uploads it instead of retraining. Anything left behind
# (tamper evidence, a crash, a record whose era passed) ages out after
# HARVEST_RETENTION_SECONDS.
HARVEST_DIRNAME = "_funded_harvest"
HARVEST_RETENTION_SECONDS = 3 * 86400.0


def harvest_job_id(receipt, suffix: str) -> str:
    """Separate replicas, lanes, sources and init/contract generations on disk."""
    binding = {name: getattr(receipt, name, None) for name in
               ("miner_hotkey", "role", "size", "gen_ref", "init_pointer", "training_contract_digest")}
    binding["suffix"] = suffix
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:24]


def harvest_dir(work_root: Path, base_seed, hotkey: str, *, job_id: str = "") -> Path:
    tag = f"{hotkey}--{job_id}" if job_id else hotkey
    return Path(work_root) / HARVEST_DIRNAME / f"{base_seed}" / tag


def _pending_upload_path(work_root: Path, base_seed, hotkey: str, job_id: str = "") -> Path:
    return harvest_dir(work_root, base_seed, hotkey, job_id=job_id).with_suffix(".pending.json")


def write_pending_upload(work_root: Path, base_seed, hotkey: str, record: dict) -> None:
    """Remember a verified harvest whose upload failed (best-effort)."""
    try:
        path = _pending_upload_path(work_root, base_seed, hotkey, record.get("harvest_job_id", ""))
        write_text_atomic(path, json.dumps(record, sort_keys=True))
    except Exception as e:  # noqa: BLE001 — the requeue then simply retrains
        log.warning("could not record the pending upload of %s: %s", hotkey[:12], e)


def read_pending_upload(work_root: Path, base_seed, hotkey: str, **binding) -> dict | None:
    """Read the unique matching job; legacy unscoped records remain readable."""
    old = _pending_upload_path(work_root, base_seed, hotkey)
    paths = [old, *sorted(old.parent.glob(f"{hotkey}--*.pending.json"))]
    matches = []
    for p in paths:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as e:
            log.warning("pending upload record %s unreadable (%s); ignored", p, type(e).__name__)
            continue
        if not isinstance(rec, dict) or not (
                isinstance(rec.get("r2"), dict)          # r2 transport: the copy is on R2
                or Path(str(rec.get("dest", ""))).is_dir()):
            continue
        if any(rec.get(key) != value for key, value in binding.items()):
            continue
        matches.append(rec)
    return matches[0] if len(matches) == 1 else None


def clear_harvest(work_root: Path, base_seed, hotkey: str, *, job_id: str = "") -> None:
    """Remove this job only; another replica's upload/recovery is independent."""
    import shutil

    shutil.rmtree(harvest_dir(work_root, base_seed, hotkey, job_id=job_id), ignore_errors=True)
    _pending_upload_path(work_root, base_seed, hotkey, job_id).unlink(missing_ok=True)


SHADOW_EMA_DIRNAME = "_shadow_ema"


def shadow_ema_dir(work_root: Path, base_seed, hotkey: str, *, job_id: str = "") -> Path:
    """Where ``[round] funded_keep_shadow_ema`` keeps one leg's power-EMA
    comparison files on the box (outside the harvest dir, so neither the
    upload nor :func:`clear_harvest` touches them)."""
    tag = f"{hotkey}--{job_id}" if job_id else hotkey
    return Path(work_root) / SHADOW_EMA_DIRNAME / f"{base_seed}" / tag


def set_aside_shadow_ema(cfg, work_root: Path, dest: Path, base_seed, hotkey: str, *,
                         job_id: str = "") -> list[str]:
    """After the harvested copy at ``dest`` is verified: take the power-EMA
    comparison files (``r2_transfer.shadow_ema_file``) out of it — moved to
    :func:`shadow_ema_dir` when ``[round] funded_keep_shadow_ema`` is on, else
    deleted (the default: no extra weights on the box disk). Returns the
    names handled."""
    import os

    from .r2_transfer import shadow_ema_file

    d = Path(dest)
    if not d.is_dir():
        return []
    names = sorted(p.name for p in d.iterdir() if p.is_file() and shadow_ema_file(p.name))
    if not names:
        return []
    if bool(getattr(getattr(cfg, "round", None), "funded_keep_shadow_ema", False)):
        target = shadow_ema_dir(work_root, base_seed, hotkey, job_id=job_id)
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            os.replace(d / name, target / name)
        log.info("harvest: power-EMA comparison files kept in %s (never published): %s",
                 target, ", ".join(names))
    else:
        for name in names:
            (d / name).unlink(missing_ok=True)
    return names


def r2_transport_on(cfg) -> bool:
    """``[round] funded_harvest_transport = "r2"``: harvested checkpoints go
    pod → Cloudflare R2 → Hippius and never land on this box
    (:mod:`cascade.trainer.r2_transfer`)."""
    return str(getattr(getattr(cfg, "round", None), "funded_harvest_transport", "ssh")) == "r2"


# ── trainer disk guard (2026-10-31: concurrent harvests filled the disk) ────
#
# Resolved through this module attribute so tests swap in a fake filesystem.
DISK_USAGE = None   # None ⇒ shutil.disk_usage


def _free_under(path: Path) -> int:
    import shutil

    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return int((DISK_USAGE or shutil.disk_usage)(p).free)


def active_harvests(runner) -> set[str]:
    """Resolved harvest dirs a harvest in THIS process is writing now (the
    space manager never reclaims them)."""
    return set(runner.__dict__.get("_active_harvest_dirs") or ())


def reclaim_box_space(runner, target: Path | str, *, need_bytes: int = 0,
                      floor_bytes: int | None = None, why: str = ""):
    """The ONE space manager (cascade.shared.disk_space) on behalf of the
    trainer: evict reclaimable data (eval cache beyond the validator's
    protection set, fetch caches, settled harvest leftovers) so ``target``'s
    filesystem has ``floor_bytes + need_bytes`` free. Never raises."""
    from ..shared import disk_space

    try:
        return disk_space.reclaim(
            runner.cfg, target, need_bytes=need_bytes, floor_bytes=floor_bytes,
            work_root=runner.work_root, active_harvests=lambda: active_harvests(runner),
            why=why)
    except Exception as e:  # noqa: BLE001 — space management never sinks a leg
        log.warning("disk: reclaim on %s failed (%s)", target, e)
        return None


def probe_pod_live(live, pod_id: str, *, tries: int = 3, sleep=None,
                   delay: float = 5.0):
    """``(address | None, definitive)``: ask ``live(pod_id)``; a probe that
    ERRORS (a provider blip) is retried up to ``tries`` times and never reads
    as "gone" (review F1 residual). A clean "not live" answer is definitive."""
    sleep = sleep or time.sleep
    for i in range(max(1, tries)):
        try:
            addr = live(pod_id)
        except Exception as e:  # noqa: BLE001 — a blip: retried, never "gone"
            log.warning("pod liveness probe of %s failed (%s) — try %d/%d", pod_id,
                        str(e)[:120], i + 1, tries)
            if i + 1 < tries:
                sleep(delay)
            continue
        return addr, True
    return None, False


def _leg_hotkey(runner) -> str | None:
    """The hotkey of the leg this thread is running (its progress binding)."""
    binding = getattr(runner.__dict__.get("_leg_local_obj"), "funded_progress", None)
    try:
        return str(binding[1]) if binding else None
    except Exception:  # noqa: BLE001
        return None


# Phases of an in-flight leg that holds a pod whose checkpoint is still to
# land here (preparation / GPU waits hold none; judged legs are harvested).
_POD_PHASES = frozenset({"booting", "training", "harvesting"})


def _legs_holding_pods(runner) -> int:
    """Live legs that will still harvest a checkpoint to this box: in-flight
    entries (lanes A/B/C) in a pod-holding phase, the calling leg excluded
    (review F10 — the ledger alone also counts stale other-seed rows, the
    king's and this leg's own write-ahead row)."""
    me = _leg_hotkey(runner)
    n = 0
    for lane in ("a", "b", "c"):
        try:
            q = funded_queue_of(runner, lane=lane)
        except Exception:  # noqa: BLE001 — a lane without a queue
            continue
        if q is None:
            continue
        try:
            n += sum(1 for e in q.entries() if e.status == "in_flight"
                     and e.phase in _POD_PHASES and e.hotkey != me)
        except Exception:  # noqa: BLE001 — unreadable queue: count none
            continue
    return n


def disk_guard_refusal(runner) -> str:
    """Why a NEW funded/operator leg pod must not be rented now ("" = rent).

    In order (2026-11-02):

    1. the box-health halt (:mod:`cascade.trainer.box_health`) — an our-side
       fault halts EVERY new rent until its clear condition passes (an
       own-key king leg: only a disk halt);
    2. the pre-rent canary — write + fsync + rename + delete in the harvest
       dir; a failure trips the halt;
    3. the disk reservation — free space on the harvest filesystem must hold
       the expected checkpoint of every live leg holding a pod plus this one
       (``[round] expected_checkpoint_gb`` each; 0 with the r2 transport) on
       top of ``[round] min_free_disk_gb`` (GiB; 0 = the reservation off),
       measured AFTER the space manager evicted whatever is reclaimable; a
       shortfall refuses this rent only (no halt).

    Only for fresh rents: a re-attached or adopted pod is a live leg and
    always keeps going. An unmeasurable disk never blocks a rent."""
    from ..shared.disk_space import GIB
    from .box_health import LOCAL_DISK, BoxHealth, harvest_canary, rent_halt_refusal
    from .money_guard import alert, clear_alert

    if getattr(runner, "work_root", None) is None:
        return ""                                   # no box state to judge (a bare test double)
    # an own-key KING leg is held only by a real disk halt (review N4)
    king = bool(getattr(runner.__dict__.get("_leg_local_obj"), "funded_is_king", False))
    halted = rent_halt_refusal(runner, king=king)
    if halted:
        return f"operator disk: {halted} — no new pod rented (requeued, never burned)"
    who = _leg_hotkey(runner) or "box"
    canary = harvest_canary(runner)
    if canary:
        BoxHealth.for_runner(runner).record(LOCAL_DISK, who, canary)
        return f"operator disk: {canary} — no new pod rented (requeued, never burned)"
    rnd = getattr(runner.cfg, "round", None)
    gb = float(getattr(rnd, "min_free_disk_gb", 0.0) or 0.0)
    if gb <= 0:
        return ""
    per = 0.0 if r2_transport_on(runner.cfg) else float(
        getattr(rnd, "expected_checkpoint_gb", 0.0) or 0.0)
    holding = _legs_holding_pods(runner)
    need = int(per * GIB) * (holding + 1)
    floor = int(gb * GIB)
    target = Path(runner.work_root) / HARVEST_DIRNAME
    try:
        free = _free_under(target)
    except OSError as e:
        log.warning("disk guard: free space under %s unreadable (%s); not blocking",
                    runner.work_root, e)
        return ""
    if free < floor + need:
        reclaim_box_space(runner, target, need_bytes=need, floor_bytes=floor,
                          why="pre-rent disk reservation")
        with contextlib.suppress(OSError):
            free = _free_under(target)
    if free >= floor + need:
        clear_alert(runner, "operator_disk", "box")
        return ""
    msg = (f"operator disk: trainer disk low: {free / GIB:.1f} GiB free under {runner.work_root} "
           f"< {need / GIB:.0f} GiB for {holding + 1} checkpoint(s) of {per:g} GB + [round] "
           f"min_free_disk_gb={gb:g} (after evicting reclaimable data) — no new pod rented "
           "(requeued, never burned)")
    # A shortfall refuses THIS rent only (legs may legitimately hold the
    # space) — never a halt: only a real disk error or a failed canary
    # trips the box-health breaker (review N2).
    alert(runner, "operator_disk", "box", msg)
    return msg


def prune_harvest_dirs(work_root: Path, *, now: float | None = None,
                       max_age_seconds: float = HARVEST_RETENTION_SECONDS) -> int:
    """Delete harvest leftovers older than ``max_age_seconds``; returns how many."""
    import shutil

    root = Path(work_root) / HARVEST_DIRNAME
    if not root.is_dir():
        return 0
    t = time.time() if now is None else float(now)
    gone = 0
    for seed_dir in root.iterdir():
        if not seed_dir.is_dir():
            continue
        for item in seed_dir.iterdir():
            try:
                if t - item.stat().st_mtime <= max_age_seconds:
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                gone += 1
            except OSError:
                continue
        with contextlib.suppress(OSError):
            seed_dir.rmdir()                       # only when now empty
    return gone
