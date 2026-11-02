"""System-wide our-side fault breaker: "box health" (2026-11-02).

That day, while the checkpoint volume was full, every finished leg failed its
harvest with ENOSPC, and the requeued leg re-rented on the MINER's key and
retrained from scratch. Each rerun failed again for the same reason: about
$640 of miners' money and five hours before anyone noticed. A per-leg view
can never see that the BOX is broken, so this module keeps one persisted view:

* every our-side fault on a leg (:func:`our_side_fault_class`: local disk/IO,
  a harvest our side could not finish, our storage) is recorded in
  ``<work_root>/box_health.json`` with one critical alert;
* ANY local-disk fault, or the same class on ≥ 2 legs within
  ``[round] box_fault_window_seconds`` (2 h), HALTS every new GPU rent:
  miner keys and operator, new legs and retries, and — for a disk fault —
  the king's JIT pod too (its checkpoint could not be saved either), loudly;
* the halt clears only when the automatic health check passes (a write +
  fsync + rename + delete canary in the harvest dir AND free space for every
  queued and in-flight leg's checkpoint plus the reserve) or on the owner's
  ``cascade-trainer resume-rents``;
* legs already training keep running; waiting legs requeue unburned as
  ``operator_disk`` (backed off, never counted against any cap);
* a leg hit by an our-side fault AFTER it trained (:func:`note_leg_fault`)
  never retrains blindly: its kept pod is re-harvested, and a retrain after
  the finished run was lost needs the owner's per-leg approval even on the
  miner's key (``kind="retrain_after_loss"``), with one alert pricing what
  the miner already lost.

Trainer-only; nothing here reaches a validator.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("cascade.trainer.box_health")

BOX_HEALTH_FILE = "box_health.json"
LOCAL_DISK, HARVEST, STORAGE = "local_disk", "harvest", "storage"
CHECK_EVERY_SECONDS = 30.0
# In-process latch (review N3): a halt this process set holds even when its
# own state write failed (a full disk can refuse box_health.json itself).
_LATCH: dict[str, dict] = {}


def our_side_fault_class(exc: BaseException | None) -> str | None:
    """The box-health class of a leg failure on OUR side, or ``None``."""
    from ..shared.hippius import StorageError
    from .faults import StorageFault
    from .harvest_guard import HarvestDiskFault, HarvestKeepPod, local_disk_error

    if exc is None:
        return None
    if local_disk_error(exc) is not None:
        return LOCAL_DISK
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        # A HarvestDiskFault without a real disk error in its chain is a room
        # shortfall (legs legitimately holding space): a harvest-class fault,
        # never a LOCAL_DISK halt that would hold the king (review N2).
        if isinstance(e, (HarvestDiskFault, HarvestKeepPod)):
            return HARVEST
        if isinstance(e, (StorageFault, StorageError)):
            return STORAGE
        e = e.__cause__ or e.__context__
    return None


class BoxHealth:
    """``<work_root>/box_health.json``: recent our-side faults, the rent halt,
    and the per-leg fault records (flock'd, reload-before-act)."""

    def __init__(self, path: Path | str, *, window_seconds: float = 7200.0,
                 clock=time.time) -> None:
        self.path = Path(path)
        self.window_seconds = float(window_seconds)
        self.clock = clock

    @classmethod
    def for_runner(cls, runner) -> BoxHealth:
        rnd = getattr(getattr(runner, "cfg", None), "round", None)
        return cls(Path(runner.work_root) / BOX_HEALTH_FILE, window_seconds=float(
            getattr(rnd, "box_fault_window_seconds", 7200) or 7200))

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(self.path.suffix + ".lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield self._load()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _load(self) -> dict:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"events": [], "halt": None, "legs": {}}
        except (OSError, ValueError):
            # Unreadable health state: fail CLOSED — halted until checked.
            return {"events": [], "legs": {},
                    "halt": {"cls": LOCAL_DISK, "since": self.clock(),
                             "reason": "box health state unreadable"}}
        doc.setdefault("events", [])
        doc.setdefault("halt", None)
        doc.setdefault("legs", {})
        return doc

    def _save(self, doc: dict) -> None:
        from ..shared.atomic import write_text_atomic

        try:
            write_text_atomic(self.path, json.dumps(doc, sort_keys=True))
        except OSError as e:     # a full disk can refuse this very write
            log.critical("box health: could not persist (%s) — the halt holds in memory", e)

    def state(self) -> dict:
        with self._locked() as doc:
            return doc

    def halt(self) -> dict | None:
        return self.state().get("halt") or _LATCH.get(str(self.path))

    def record(self, cls: str, hotkey: str, detail: str) -> dict | None:
        """Record one our-side fault; returns the halt when it is (now) set."""
        now = self.clock()
        with self._locked() as doc:
            ev = [e for e in doc["events"] if now - float(e.get("at", 0)) <= self.window_seconds]
            ev.append({"cls": cls, "hotkey": hotkey, "at": now, "detail": detail[:300]})
            doc["events"] = ev[-200:]
            legs = {e["hotkey"] for e in ev if e["cls"] == cls}
            new = None
            if doc.get("halt") is None and (cls == LOCAL_DISK or len(legs) >= 2):
                new = doc["halt"] = {
                    "cls": cls, "since": now,
                    "reason": (f"{cls} fault on {hotkey[:12]}: {detail[:200]}" if cls == LOCAL_DISK
                               else f"{cls} faults on {len(legs)} legs within "
                                    f"{self.window_seconds / 3600:.1f} h")}
            if new is not None:
                _LATCH[str(self.path)] = new
            self._save(doc)
            halt = doc.get("halt")
        if new is not None:
            log.critical("operator alert: rents_halted — ALL new GPU rents halted (miner keys "
                         "and operator%s): %s. Legs already training keep running; the halt "
                         "clears when the box health check passes or on `cascade-trainer "
                         "resume-rents`", ", king included" if cls == LOCAL_DISK else "",
                         new["reason"])
        return halt

    def clear(self, by: str) -> bool:
        with self._locked() as doc:
            was = doc.get("halt") or _LATCH.pop(str(self.path), None)
            _LATCH.pop(str(self.path), None)
            if was is None:
                return False
            doc["halt"] = None
            doc["events"] = []
            doc["last_clear"] = {"at": self.clock(), "by": by, "was": was}
            self._save(doc)
        log.critical("operator alert: rents_resumed — GPU rents resume (%s); was: %s", by,
                     was.get("reason", ""))
        return True

    # per-leg fault records (no blind retrain)

    def note_leg(self, hotkey: str, rec: dict) -> None:
        with self._locked() as doc:
            doc["legs"][hotkey] = {**(doc["legs"].get(hotkey) or {}), **rec,
                                   "at": self.clock()}
            self._save(doc)

    def leg(self, hotkey: str) -> dict | None:
        return (self.state().get("legs") or {}).get(hotkey)

    def forget_leg(self, hotkey: str) -> None:
        with self._locked() as doc:
            if doc["legs"].pop(hotkey, None) is not None:
                self._save(doc)


# ── the health check (auto-clear) and the pre-rent canary ───────────────────


def harvest_canary(runner) -> str:
    """Write + fsync + rename + delete a small file in the harvest dir; ""
    when the box can save a checkpoint there, else why not."""
    import uuid

    from .runner_common import HARVEST_DIRNAME

    root = Path(runner.work_root) / HARVEST_DIRNAME
    # unique per call: concurrent leg threads run their canaries at once
    tag = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    tmp = root / f".canary-{tag}.tmp"
    done = root / f".canary-{tag}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, b"cascade harvest canary\n" * 16)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, done)
        dfd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        done.unlink()
        return ""
    except OSError as e:
        for p in (tmp, done):
            with contextlib.suppress(OSError):
                p.unlink()
        return f"harvest-dir canary failed: {e}"


def storage_probe(runner) -> str:
    """"" when OUR storage takes a write: put + get + delete of a small canary
    key on the manifest store (the clear condition of a harvest/storage halt —
    a disk check says nothing about them; review N3)."""
    import uuid

    probe = getattr(runner, "_storage_probe", None)
    if callable(probe):
        return str(probe() or "")
    store_fn = getattr(runner, "manifest_store", None)
    if not callable(store_fn):
        return "no storage probe available — clear with `cascade-trainer resume-rents`"
    key = f"health/box-canary-{uuid.uuid4().hex[:12]}"
    try:
        store = store_fn()
        store.put_bytes(key, b"cascade box-health storage canary\n")
        if store.get_bytes(key) != b"cascade box-health storage canary\n":
            return "storage canary read back different bytes"
        delete = getattr(store, "delete", None)
        if callable(delete):
            with contextlib.suppress(Exception):
                delete(key)
        return ""
    except Exception as e:  # noqa: BLE001 — any failure keeps the halt
        return f"storage canary failed: {str(e)[:200]}"


def health_check(runner, cls: str = LOCAL_DISK) -> str:
    """"" when the box is healthy enough to resume rents. A disk halt needs the
    canary AND free space for every live leg holding a pod plus one more, on
    top of the ``min_free_disk_gb`` reserve (the same sizing as the pre-rent
    reservation; review N2). A harvest/storage halt needs the storage probe."""
    from ..shared.disk_space import GIB
    from .runner_common import (
        HARVEST_DIRNAME,
        _free_under,
        _legs_holding_pods,
        r2_transport_on,
        reclaim_box_space,
    )

    if cls in (HARVEST, STORAGE):
        return storage_probe(runner)
    why = harvest_canary(runner)
    if why:
        return why
    rnd = runner.cfg.round
    per = 0.0 if r2_transport_on(runner.cfg) else float(
        getattr(rnd, "expected_checkpoint_gb", 0.0) or 0.0)
    need = int(per * GIB) * (_legs_holding_pods(runner) + 1)
    floor = int(float(getattr(rnd, "min_free_disk_gb", 0.0) or 0.0) * GIB)
    target = Path(runner.work_root) / HARVEST_DIRNAME
    try:
        free = _free_under(target)
        if free < need + floor:
            reclaim_box_space(runner, target, need_bytes=need, floor_bytes=floor,
                              why="box health check")
            free = _free_under(target)
    except OSError as e:
        return f"free space unreadable: {e}"
    if free < need + floor:
        return (f"{free / GIB:.1f} GiB free < {(need + floor) / GIB:.1f} GiB for every live "
                "leg's checkpoint plus one more and the reserve")
    return ""


def rent_halt_refusal(runner, *, king: bool = False) -> str:
    """Why NO new GPU pod may be rented now ("" = go). While halted the
    health check runs (at most every 30 s) and clears the halt itself. A
    king rent is held only by a disk halt (the other classes leave it)."""
    bh = BoxHealth.for_runner(runner)
    h = bh.halt()
    if not h:
        return ""
    if king and h.get("cls") != LOCAL_DISK:
        return ""
    last = runner.__dict__.get("_box_health_checked")
    now = time.monotonic()
    if last is None or now - last[0] >= CHECK_EVERY_SECONDS:
        why = health_check(runner, h.get("cls", LOCAL_DISK))
        runner.__dict__["_box_health_checked"] = (now, why)
    else:
        why = last[1]
    if not why:
        bh.clear("automatic health check passed")
        return ""
    # (the watchdog's one critical rents_halted alert reads the halt file)
    if king:
        log.critical("king rent HALTED by the box health breaker (%s) — the king's checkpoint "
                     "could not be saved either", h.get("reason", ""))
    return f"GPU rents halted by the box health breaker ({h.get('reason', '')}; {why})"


def record_leg_fault(runner, hotkey: str, exc: BaseException, *, trained: bool,
                     pod_id: str = "", payer: bool = False, kept: bool = False,
                     ref: str = "", extra: dict | None = None) -> str | None:
    """Record an our-side fault of a leg (no-op for any other failure):
    the box-health event (may halt rents), one critical alert, and the leg's
    record that stops a blind retrain. Returns the class."""
    cls = our_side_fault_class(exc)
    if cls is None:
        return None
    from .money_guard import alert

    bh = BoxHealth.for_runner(runner)
    bh.record(cls, hotkey, str(exc))
    alert(runner, "our_side_fault", hotkey[:16],
          f"{cls} fault on {hotkey[:12]}'s leg{' after training' if trained else ''}: "
          f"{str(exc)[:200]}", severity="critical")
    if trained:
        bh.note_leg(hotkey, {"cls": cls, "pod_id": pod_id, "payer": payer, "kept": kept,
                             "trained": True, "ref": ref, **(extra or {})})
    return cls


def miner_lost_usd(runner) -> float:
    """Worst-case miner money lost on one finished run: the price ceiling ×
    the leg's expected wall."""
    rnd = runner.cfg.round
    rate = float(getattr(rnd, "funded_max_price_per_hour", 0.0) or 0.0)
    try:
        hours = float(runner._leg_wall_seconds()) / 3600.0
    except Exception:  # noqa: BLE001
        hours = 0.0
    return rate * hours


__all__ = ["BoxHealth", "HARVEST", "LOCAL_DISK", "STORAGE", "harvest_canary", "health_check",
           "miner_lost_usd", "our_side_fault_class", "record_leg_fault", "rent_halt_refusal"]
