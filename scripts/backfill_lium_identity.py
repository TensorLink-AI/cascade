#!/usr/bin/env python3
"""One-time backfill of the Lium executor → host identity store.

Lium stopped listing executor IPs (2026-11-01), so the host quarantine /
``funded_host_blocklist`` steers by the identity store
(``cascade.provision.lium_identity``) that the provider now fills on every
ready pod. This script seeds it from what the box already recorded:

* ``trainer.log``: each ``lium up → executor <id> as <name>`` paired with the
  next ``pod <name> ready … at <ip>:<port>`` / ``is still LIVE at`` line;
* the executor cooldown file: ids pinned far in the future (a hand-made
  permanent block) are marked bad until that time;
* optionally (``--fetch-api``) ``GET /executors`` (key from ``LIUM_API_KEY``,
  never printed) to capture the GPU UUIDs / site of learned executors that are
  still listed.

``funded_pods.json`` carries no executor id or IP, so it cannot help. Dry run
by default (prints what it would learn); ``--write`` merges into the store.

    python scripts/backfill_lium_identity.py --log /root/cascade/trainer.log [--fetch-api] [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UP_RE = re.compile(r"lium up → executor ([0-9a-fA-F-]{8,}) as (\S+)")
READY_RE = re.compile(
    r"pod (\S+) (?:ready|is still LIVE)\b.*? at (\d{1,3}(?:\.\d{1,3}){3}):\d+")
PIN_MARGIN_SECONDS = 86400.0


def parse_log_pairs(lines) -> dict[str, list[str]]:
    """``{executor id: [ip, …]}`` from trainer.log lines (a pod name maps to the
    executor of its MOST RECENT ``lium up``; names are reused across retries)."""
    last: dict[str, str] = {}
    out: dict[str, list[str]] = {}
    for line in lines:
        m = UP_RE.search(line)
        if m:
            last[m.group(2)] = m.group(1)
            continue
        m = READY_RE.search(line)
        if not m:
            continue
        executor = last.get(m.group(1))
        if executor:
            ips = out.setdefault(executor, [])
            if m.group(2) not in ips:
                ips.append(m.group(2))
    return out


def pinned_cooldowns(cooldowns: dict, *, now: float) -> dict[str, float]:
    """Cooldown entries stamped far in the FUTURE — hand-pinned permanent blocks."""
    out = {}
    for k, v in (cooldowns or {}).items():
        try:
            if float(v) > now + PIN_MARGIN_SECONDS:
                out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _fetch_records() -> dict[str, dict]:
    import requests

    key = os.environ.get("LIUM_API_KEY", "")
    if not key:
        raise SystemExit("--fetch-api needs LIUM_API_KEY in the environment")
    from cascade.provision.core import LIUM_API_BASE

    r = requests.get(f"{LIUM_API_BASE}/executors", headers={"X-API-KEY": key}, timeout=30)
    r.raise_for_status()
    return {str(e["id"]): e for e in r.json() or [] if isinstance(e, dict) and e.get("id")}


def backfill(pairs: dict[str, list[str]], pins: dict[str, float], records: dict[str, dict],
             *, write: bool, path: Path | None = None, now: float | None = None) -> dict:
    """Merge ``pairs`` / ``pins`` / ``records`` into the store (``write``) and
    return a summary."""
    from cascade.provision.lium_identity import learn_executor, mark_executor_bad

    now = time.time() if now is None else now
    if write:
        for executor, ips in pairs.items():
            for ip in ips:
                learn_executor(executor, ip=ip, record=records.get(executor), path=path, now=now)
        for executor, until in pins.items():
            if records.get(executor):
                learn_executor(executor, record=records[executor], path=path, now=now)
            mark_executor_bad(executor, "pinned in the executor cooldown file (backfill)",
                              seconds=until - now, path=path, now=now)
    return {"executors": len(pairs), "ips": len({i for v in pairs.values() for i in v}),
            "pinned_bad": sorted(pins), "with_api_record": sorted(
                e for e in set(pairs) | set(pins) if e in records)}


def main(argv=None) -> int:
    from cascade.provision.core import lium_cooldown_path, quarantined_hosts

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, type=Path, help="trainer.log to scan")
    ap.add_argument("--cooldown-file", type=Path, default=None)
    ap.add_argument("--fetch-api", action="store_true")
    ap.add_argument("--write", action="store_true", help="merge into the store (else dry run)")
    a = ap.parse_args(argv)
    with a.log.open(errors="ignore") as fh:
        pairs = parse_log_pairs(fh)
    cd_path = a.cooldown_file or lium_cooldown_path()
    try:
        cooldowns = json.loads(cd_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cooldowns = {}
    now = time.time()
    pins = pinned_cooldowns(cooldowns, now=now)
    records = _fetch_records() if a.fetch_api else {}
    summary = backfill(pairs, pins, records, write=a.write, now=now)
    bad_ips = quarantined_hosts()
    for ip in sorted(bad_ips):
        ids = sorted(e for e, ips in pairs.items() if ip in ips)
        print(f"blocked host {ip}: executors {ids or '(none seen in the log)'}")
    print(json.dumps(summary, indent=1))
    print("WROTE the identity store" if a.write else "dry run — nothing written (--write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
