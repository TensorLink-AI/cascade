"""Lium executor → host identity, learned from our own rentals (2026-11-01).

Lium dropped ``executor_ip_address`` from ``GET /executors`` (0/172 records
carried it at 22:20 UTC), so the host quarantine / ``funded_host_blocklist``
could no longer map a marketplace executor to its host IP — and the rc16
fail-closed rule then skipped EVERY executor: no rentals at all.

What the API still gives, measured read-only on the live listing (173
executors):

* ``machine_name`` is the GPU MODEL ("NVIDIA H200": 23 distinct values over
  173 executors) — NOT a per-host identity. Executors sharing a NAT IP carry
  different machine_names (146.120.227.147 = an RTX 4090 and an RTX 5070 Ti of
  one miner). It is recorded for the log only, never matched on.
* ``specs.gpu.details[].uuid`` — hardware GPU UUIDs, unique per executor
  (626 UUIDs, 0 shared) and stable across a re-registration of the same
  cards under a new executor id: the HARDWARE identity.
* ``miner_hotkey`` + ``location`` (geo-IP of the executor's address:
  ``location.source = "ip-api,ipinfo,ipapi"``) — the SITE identity. Every
  multi-id NAT host in our own rental history that is still listed
  (146.120.227.147 ×2, 91.224.44.81 ×2) shares one (hotkey, geo) site, so it
  is the closest surviving proxy for "same IP".

So we persist, per executor id we ever rented, the IPs its pods came up on
(the ``lium up → executor`` record + the ready pod's SSH IP) and its hardware
and site identities from ``/executors``. A host IP that is quarantined or
blocklisted then marks as bad every executor ever seen at it — and every
listed executor sharing a GPU UUID or a (miner hotkey, geo) site with one of
those. Unknown executors (no learned identity, no match, not API-flagged) are
rentable: fail open on unknown, fail closed on known-bad. The derived blocks
follow the live quarantine view: an IP dropped from the blocklist (or a
quarantine that expires) un-blocks its executors.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

LIUM_IDENTITY_FILE_ENV = "CASCADE_LIUM_IDENTITY_FILE"
# An executor marked bad directly (a lemon pod whose host IP never resolved).
LIUM_BAD_EXECUTOR_SECONDS = 3 * 3600.0
_MAX_IPS = 16


def lium_identity_path() -> Path:
    return Path(os.environ.get(
        LIUM_IDENTITY_FILE_ENV, "~/.cache/cascade/lium_executor_identity.json")).expanduser()


def _empty() -> dict:
    return {"executors": {}, "bad": {}}


def load_identities(path: Path | None = None) -> dict:
    """``{"executors": {id: {...}}, "bad": {id: {"until", "reason"}}}``; empty on
    a missing/corrupt file (best-effort memory, never a crash)."""
    path = path or lium_identity_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return _empty()
    if not isinstance(raw, dict):
        return _empty()
    ex = raw.get("executors") if isinstance(raw.get("executors"), dict) else {}
    bad = raw.get("bad") if isinstance(raw.get("bad"), dict) else {}
    out = _empty()
    for k, v in ex.items():
        if isinstance(v, dict):
            out["executors"][str(k)] = {
                "ips": [str(i) for i in (v.get("ips") or []) if str(i).strip()],
                "miner_hotkey": str(v.get("miner_hotkey") or ""),
                "machine_name": str(v.get("machine_name") or ""),
                "site": str(v.get("site") or ""),
                "gpu_uuids": [str(u) for u in (v.get("gpu_uuids") or []) if str(u).strip()],
                "updated": float(v.get("updated") or 0.0),
            }
    for k, v in bad.items():
        if isinstance(v, dict) and "until" in v:
            try:
                out["bad"][str(k)] = {"until": float(v["until"]),
                                      "reason": str(v.get("reason") or "")}
            except (TypeError, ValueError):
                continue
    return out


def _write(d: dict, path: Path) -> None:
    from ..shared.atomic import write_text_atomic

    write_text_atomic(path, json.dumps(d, sort_keys=True))


def site_of(record: dict | None) -> str:
    """``miner_hotkey|country|region|city|zip`` of an ``/executors`` record ("" when
    the hotkey or the city is missing — a partial geo must not lump sites)."""
    if not isinstance(record, dict):
        return ""
    hk = str(record.get("miner_hotkey") or "").strip()
    loc = record.get("location") if isinstance(record.get("location"), dict) else {}
    city = str(loc.get("city") or "").strip()
    if not hk or not city:
        return ""
    return "|".join((hk, str(loc.get("country_code") or "").strip(),
                     str(loc.get("region") or "").strip(), city,
                     str(loc.get("zip") or "").strip()))


def gpu_uuids_of(record: dict | None) -> list[str]:
    if not isinstance(record, dict):
        return []
    specs = record.get("specs") if isinstance(record.get("specs"), dict) else {}
    gpu = specs.get("gpu") if isinstance(specs.get("gpu"), dict) else {}
    out = []
    for g in gpu.get("details") or []:
        u = g.get("uuid") if isinstance(g, dict) else None
        if isinstance(u, str) and u.strip():
            out.append(u.strip())
    return sorted(set(out))


def identity_of(record: dict | None) -> dict:
    """The persisted identity fields of an ``/executors`` record."""
    if not isinstance(record, dict):
        return {}
    return {"miner_hotkey": str(record.get("miner_hotkey") or ""),
            "machine_name": str(record.get("machine_name") or ""),
            "site": site_of(record), "gpu_uuids": gpu_uuids_of(record)}


def learn_executor(executor_id: str, *, ip: str = "", record: dict | None = None,
                   path: Path | None = None, now: float | None = None) -> bool:
    """Remember that ``executor_id`` served a pod at ``ip`` and/or carries the
    identity of ``record``. Merges (IPs accumulate; identity fields are only
    overwritten by non-empty values). Best-effort; True when the file changed."""
    executor_id = str(executor_id or "").strip()
    ip = str(ip or "").strip()
    ident = identity_of(record)
    if not executor_id or not (ip or any(ident.values())):
        return False
    path = path or lium_identity_path()
    now = time.time() if now is None else now
    try:
        from ..shared.atomic import path_lock

        with path_lock(path):
            d = load_identities(path)
            cur = d["executors"].get(executor_id) or {
                "ips": [], "miner_hotkey": "", "machine_name": "", "site": "",
                "gpu_uuids": [], "updated": 0.0}
            new = dict(cur)
            if ip and ip not in cur["ips"]:
                new["ips"] = (cur["ips"] + [ip])[-_MAX_IPS:]
            for k in ("miner_hotkey", "machine_name", "site"):
                if ident.get(k):
                    new[k] = ident[k]
            if ident.get("gpu_uuids"):
                new["gpu_uuids"] = sorted(set(cur["gpu_uuids"]) | set(ident["gpu_uuids"]))
            if {k: v for k, v in new.items() if k != "updated"} == \
                    {k: v for k, v in cur.items() if k != "updated"} and executor_id in d["executors"]:
                return False
            new["updated"] = now
            d["executors"][executor_id] = new
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(d, path)
        if ip:
            log.info("lium: learned executor %s → host %s", executor_id[:8], ip)
        return True
    except OSError as e:
        log.warning("lium: could not persist the identity of executor %s: %s", executor_id, e)
        return False


def mark_executor_bad(executor_id: str, reason: str, *,
                      seconds: float = LIUM_BAD_EXECUTOR_SECONDS,
                      path: Path | None = None, now: float | None = None) -> None:
    """Block ``executor_id`` (and its learned GPU/site identity) for ``seconds``
    — the case where its host IP is unknown so the IP quarantine cannot."""
    executor_id = str(executor_id or "").strip()
    if not executor_id:
        return
    path = path or lium_identity_path()
    now = time.time() if now is None else now
    try:
        from ..shared.atomic import path_lock

        with path_lock(path):
            d = load_identities(path)
            d["bad"] = {k: v for k, v in d["bad"].items() if v["until"] > now}
            prev = d["bad"].get(executor_id)
            until = max(now + float(seconds), prev["until"] if prev else 0.0)
            d["bad"][executor_id] = {"until": until, "reason": str(reason)[:300]}
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(d, path)
        log.warning("lium: executor %s marked bad for %.1f h — %s", executor_id,
                    (until - now) / 3600, reason)
    except OSError as e:
        log.warning("lium: could not mark executor %s bad: %s", executor_id, e)


@dataclass(frozen=True)
class BlockedIdentities:
    """Everything a listing must skip: ``{key: why}`` per identity kind."""

    ids: dict = field(default_factory=dict)
    gpus: dict = field(default_factory=dict)
    sites: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.ids or self.gpus or self.sites)


def blocked_identities(bad_ips, *, path: Path | None = None,
                       now: float | None = None, extra_bad_ids: dict | None = None,
                       records: dict | None = None) -> BlockedIdentities:
    """Executors (and their GPU / site identities) that are known-bad: every
    learned executor ever seen at an IP in ``bad_ips`` ({ip: reason} or a set),
    every executor marked bad directly and not yet expired, and
    ``extra_bad_ids`` ({id: why}, e.g. hand-pinned cooldowns). A bad executor
    with no stored identity takes it from its live ``records`` entry."""
    bad_ips = dict(bad_ips) if isinstance(bad_ips, dict) else {i: "" for i in (bad_ips or ())}
    now = time.time() if now is None else now
    d = load_identities(path)
    ids: dict[str, str] = {}
    for eid, rec in d["executors"].items():
        hit = next((ip for ip in rec["ips"] if ip in bad_ips), "")
        if hit:
            ids[eid] = f"seen at blocked host {hit}"
    for eid, b in d["bad"].items():
        if b["until"] > now:
            ids.setdefault(eid, f"marked bad: {b['reason']}")
    for eid, why in (extra_bad_ids or {}).items():
        ids.setdefault(str(eid), str(why))
    gpus: dict[str, str] = {}
    sites: dict[str, str] = {}
    for eid, why in ids.items():
        rec = dict(d["executors"].get(eid) or {})
        live = identity_of((records or {}).get(eid))
        for k in ("site", "gpu_uuids"):
            if live.get(k) and not rec.get(k):
                rec[k] = live[k]
        if not rec:
            continue
        for u in rec.get("gpu_uuids") or []:
            gpus.setdefault(u, f"GPU of bad executor {eid[:8]} ({why})")
        if rec.get("site"):
            sites.setdefault(rec["site"], f"site of bad executor {eid[:8]} ({why})")
    return BlockedIdentities(ids=ids, gpus=gpus, sites=sites)


def identity_block_reason(executor_id: str, record: dict | None,
                          blocked: BlockedIdentities) -> str:
    """Why this executor is known-bad ("" = not known-bad)."""
    eid = str(executor_id or "")
    if eid in blocked.ids:
        return f"executor {eid[:8]} {blocked.ids[eid]}"
    for u in gpu_uuids_of(record):
        if u in blocked.gpus:
            return blocked.gpus[u]
    site = site_of(record)
    if site and site in blocked.sites:
        return blocked.sites[site]
    return ""


# ``/executors`` health flags. Only an EXPLICIT bad value skips: ``active`` and
# ``is_responsive`` were None on all 173 records (2026-11-01) — unknown is not
# a fault. ``reliability_score`` stays informational (no threshold in use).
_FLAGS_TRUE_IS_BAD = ("is_slow_machine", "frequent_rent_failures")
_FLAGS_FALSE_IS_BAD = ("disk_health_ok", "is_responsive", "active")


def health_flags(record: dict | None) -> list[str]:
    """Names of the API health flags that disqualify this executor."""
    if not isinstance(record, dict):
        return []
    out = [k for k in _FLAGS_TRUE_IS_BAD if record.get(k) is True]
    out += [f"{k}=false" for k in _FLAGS_FALSE_IS_BAD if record.get(k) is False]
    return out
