"""The persistent SHORT-POD blacklist (money safety II, owner 2026-11-03).

A Lium executor listed as an 8-GPU layout that exposes fewer GPUs once
rented (``short pod: 4 GPU(s) visible, rented as 8xH200``) was re-rented
across legs, accounts and days: 29 short-pod rejects since Oct 25, the same
executor (3bb98239…, "golden-lion-db", always 4 of 8 GPUs) four times. The
old exclusion lived for one dispatch only.

Here every short-pod reject blocks the executor id AND the machine identity
(the pod's host IP from the post-rent check) for :data:`FIRST_STRIKE_SECONDS`
(72 h); a second strike within :data:`STRIKE_MEMORY_SECONDS` blocks it for
:data:`REPEAT_STRIKE_SECONDS` (7 d). The state is one flock'd JSON file
under the trainer's work root (``_train_work/short_pod_blacklist.json``,
:func:`set_path` — set by the trainer before every rent), so it holds for
ALL legs, payer and operator accounts, and restarts, and it is filtered in
``LiumProvider._list_executors`` BEFORE any rent. ``cascade-trainer
rent-status`` shows it; ``block-executor`` / ``unblock-executor`` edit it.

Lium's ``lium ls --format json`` carries no free-GPU field, so a listing
cannot reveal a short machine before the rent: the strike after the first
short rent is what keeps it out from then on.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("cascade.provision")

FILE_NAME = "short_pod_blacklist.json"
PATH_ENV = "CASCADE_SHORT_POD_FILE"
FIRST_STRIKE_SECONDS = 72 * 3600.0
REPEAT_STRIKE_SECONDS = 7 * 86400.0
STRIKE_MEMORY_SECONDS = 30 * 86400.0
# A re-strike of the same key this soon is the SAME reject seen twice (two
# call sites of one post-rent check) — never a second strike.
RESTRIKE_IGNORE_SECONDS = 600.0
# One Lium NAT IP fronts several executors: an IP is blocked only once short
# rents hit at least this many DISTINCT executor ids behind it.
IP_DISTINCT_EXECUTORS = 2

_PATH: Path | None = None
_LOCK = threading.Lock()


def set_path(path: Path | str | None) -> None:
    """Point the process-wide blacklist at ``path`` (the trainer's work root)."""
    global _PATH
    _PATH = Path(path) if path else None


def path() -> Path:
    if _PATH is not None:
        return _PATH
    return Path(os.environ.get(PATH_ENV, f"~/.cache/cascade/{FILE_NAME}")).expanduser()


@contextlib.contextmanager
def _locked(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(p.with_suffix(p.suffix + ".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _load(p: Path) -> dict:
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return {str(k): v for k, v in doc.items() if isinstance(v, dict)} if isinstance(doc, dict) \
        else {}


def _save(p: Path, doc: dict) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1), encoding="utf-8")
    tmp.replace(p)


def _key(kind: str, ident: str) -> str:
    return f"{kind}:{ident}"


def strike(*, executor_id: str = "", ip: str = "", reason: str = "",
           now: float | None = None, file: Path | None = None) -> dict[str, float]:
    """Record one short-pod strike against ``executor_id`` and/or ``ip``;
    returns ``{key: until}``. A second strike within the strike memory
    blocks for :data:`REPEAT_STRIKE_SECONDS`. Never raises."""
    t = time.time() if now is None else float(now)
    p = file or path()
    out: dict[str, float] = {}
    try:
        with _locked(p):
            doc = _load(p)
            eid = str(executor_id or "").strip()
            for kind, ident in (("executor", eid), ("ip", ip)):
                ident = str(ident or "").strip()
                if not ident:
                    continue
                k = _key(kind, ident)
                prev = doc.get(k) or {}
                execs = sorted(set(prev.get("executors") or ()) | ({eid} if eid else set()))
                if prev and t - float(prev.get("last_strike") or 0) < RESTRIKE_IGNORE_SECONDS:
                    # the same reject seen twice: no second strike (review N1)
                    doc[k] = {**prev, "executors": execs} if kind == "ip" else prev
                    out[k] = float(prev.get("until") or 0)
                    continue
                recent = t - float(prev.get("last_strike") or 0) <= STRIKE_MEMORY_SECONDS
                strikes = (int(prev.get("strikes") or 0) if recent else 0) + 1
                until = t + (REPEAT_STRIKE_SECONDS if strikes >= 2 else FIRST_STRIKE_SECONDS)
                doc[k] = {"kind": kind, "id": ident, "strikes": strikes, "last_strike": t,
                          "until": max(until, float(prev.get("until") or 0)),
                          "reason": str(reason)[:200]}
                if kind == "ip":
                    doc[k]["executors"] = execs
                out[k] = doc[k]["until"]
            doc = {k: v for k, v in doc.items()
                   if float(v.get("until") or 0) > t
                   or t - float(v.get("last_strike") or 0) <= STRIKE_MEMORY_SECONDS}
            _save(p, doc)
        for k, until in out.items():
            log.warning("short-pod blacklist: %s blocked for %.0f h (%s)", k,
                        (until - t) / 3600.0, reason[:160])
    except OSError as e:
        log.warning("short-pod blacklist not recorded (%s)", e)
    return out


def block(ident: str, *, kind: str = "executor", seconds: float = FIRST_STRIKE_SECONDS,
          reason: str = "blocked by the operator", now: float | None = None,
          file: Path | None = None) -> None:
    """The operator's hand block (``cascade-trainer block-executor``)."""
    t = time.time() if now is None else float(now)
    p = file or path()
    with _locked(p):
        doc = _load(p)
        prev = doc.get(_key(kind, ident)) or {}
        doc[_key(kind, ident)] = {"kind": kind, "id": ident,
                                  "strikes": int(prev.get("strikes") or 0) + 1,
                                  "last_strike": t, "until": t + float(seconds),
                                  "reason": reason[:200], "manual": True}
        _save(p, doc)


def unblock(ident: str, *, file: Path | None = None) -> int:
    """Lift every block of ``ident`` (executor id or IP); returns how many."""
    p = file or path()
    with _locked(p):
        doc = _load(p)
        gone = [k for k, v in doc.items() if v.get("id") == ident or k == ident]
        for k in gone:
            doc.pop(k, None)
        if gone:
            _save(p, doc)
    return len(gone)


def _in_force(v: dict, t: float) -> bool:
    if float(v.get("until") or 0) <= t:
        return False
    if v.get("kind") == "ip" and not v.get("manual"):
        # a shared NAT IP: blocked only after short rents on >= 2 DISTINCT
        # executors behind it (review N2); otherwise the executor ids only
        return len(set(v.get("executors") or ())) >= IP_DISTINCT_EXECUTORS
    return True


def active(*, now: float | None = None, file: Path | None = None) -> dict[str, dict]:
    """``{key: record}`` of every block in force (an IP record only once it
    covers >= ``IP_DISTINCT_EXECUTORS`` distinct executors, or by hand)."""
    t = time.time() if now is None else float(now)
    return {k: v for k, v in _load(file or path()).items() if _in_force(v, t)}


def blocked_ids(*, now: float | None = None, file: Path | None = None) -> tuple[set, set]:
    """``(executor ids, ips)`` blocked now."""
    a = active(now=now, file=file)
    return ({v["id"] for v in a.values() if v.get("kind") == "executor"},
            {v["id"] for v in a.values() if v.get("kind") == "ip"})


__all__ = ["FIRST_STRIKE_SECONDS", "REPEAT_STRIKE_SECONDS", "active", "block", "blocked_ids",
           "path", "set_path", "strike", "unblock"]
