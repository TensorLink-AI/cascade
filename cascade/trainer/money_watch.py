"""Periodic money watch (money safety II, the 2026-11-03 money-leak audit).

Run from the operator-health sampler (:class:`cascade.trainer.operator_health.
OperatorMonitor`), never from a leg path, each part on its own cadence and
each best-effort (a failure logs and never sinks the sampler):

* :func:`money_guard.runway_tick` — the operator runway alert and the
  "no new operator rents" state (cleared by itself on recovery);
* :func:`king_reaper_tick` — the enforcing half of the king-pod TTL (Lium
  cannot extend a TTL): a king BACKUP / stale-era king pod that is not the
  king host in use, with no live worker and no unharvested finished run on
  ``king_idle_reap_checks`` consecutive checks, is terminated (verified)
  after a final on-pod check, with one alert;
* :func:`inventory_tick` — every ``operator_inventory_interval_seconds`` the
  OPERATOR account's ``lium ps`` is diffed against an allowlist; one alert
  per unknown pod with its name and $/h. It NEVER deletes anything;
* :func:`refresh_payer_ttls` — a payer's vaulted key (and its leases) is
  re-stamped while that payer has funded-ledger rows or open create
  intents, so the key that must stop a pod never ages out first.

Trainer-only: nothing here changes what validators compute.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

log = logging.getLogger("cascade.trainer.money")


def money_periodic(runner, balance: float | None, *, now: float | None = None,
                   background: bool = True) -> None:
    """Every part below, each guarded."""
    from .money_guard import runway_tick

    t = time.time() if now is None else float(now)
    _run("runway", lambda: runway_tick(runner, balance))
    _run("payer TTL refresh", lambda: refresh_payer_ttls(runner))
    _run("stuck teardowns", lambda: stuck_teardown_tick(runner, now=t))

    # The reaper (ssh probes) and the inventory (a provider listing) may be
    # slow: one background pass at a time, never holding the sampler.
    def slow():
        _run("king reaper", lambda: king_reaper_tick(runner, now=t))
        _run("inventory", lambda: inventory_tick(runner, now=t))
        _run("receipt watch", lambda: rejected_receipt_tick(runner, now=t))

    if background:
        import threading

        prev = runner.__dict__.get("_money_watch_thread")
        if prev is not None and prev.is_alive():
            return
        th = threading.Thread(target=slow, name="money-watch", daemon=True)
        runner.__dict__["_money_watch_thread"] = th
        th.start()
    else:
        slow()


def _run(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — a watch never sinks the sampler
        log.warning("money watch: %s pass failed (%s)", name, e)


def _due(runner, key: str, interval: float, now: float) -> bool:
    if interval <= 0:
        return False
    last = float(runner.__dict__.get(key, 0.0) or 0.0)
    if now - last < interval:
        return False
    runner.__dict__[key] = now
    return True


# ── the idle-king reaper ─────────────────────────────────────────────────────


def _king_host_in_use(runner) -> str:
    host = runner.__dict__.get("_funded_king_host")
    return str(getattr(host, "allocation_id", "") or "") if host is not None else ""


def _current_king_prefixes(runner) -> tuple[str, ...]:
    from .lane_pool import king_pod_name_prefix

    sched = runner.__dict__.get("_rolling_sched")
    state = getattr(sched, "state", None)
    out = []
    for era in (getattr(state, "current", None), getattr(state, "next", None)):
        seed = getattr(era, "base_seed", None)
        if seed is not None:
            out.append(king_pod_name_prefix(runner.cfg.subnet.netuid, str(seed)) + "-")
    return tuple(out)


def reap_candidates(runner) -> list:
    """Operator ledger rows the reaper may judge: king / backup pods that are
    NOT the king host in use — every backup ("-replica-") and every king pod
    of an era that is neither current nor next. The in-use king pod is never
    a candidate (the era's benches run on it); its backstop is the create-
    time TTL (``king_pod_ttl_margin_seconds``)."""
    in_use = _king_host_in_use(runner)
    live_eras = _current_king_prefixes(runner)
    out = []
    for pod in runner._load_funded_ledger():
        pid = str(getattr(pod, "instance_id", "") or "")
        if getattr(pod, "payer_hotkey", "") or "-funded-king" not in pid or pid == in_use:
            continue
        backup = "-replica-" in pid
        current = any(pid.startswith(p) for p in live_eras)
        if backup or not current:
            out.append(pod)
    return out


def king_reaper_tick(runner, *, now: float | None = None) -> list[str]:
    """One reaper pass (at most every ``king_idle_check_seconds``); returns
    the pod ids terminated. A pod counts as idle on a check when the sweep's
    own per-pod guard would release it (no live worker, no unharvested
    finished run, reachable — :meth:`_sweep_keeps_pod`); a keep resets its
    count. At ``king_idle_reap_checks`` idle checks in a row, a FINAL on-pod
    check runs and only then the verified teardown, with one alert."""
    from .money_guard import alert

    rnd = runner.cfg.round
    checks = int(getattr(rnd, "king_idle_reap_checks", 0) or 0)
    t = time.time() if now is None else float(now)
    if checks <= 0 or not _due(runner, "_king_reap_at",
                               float(getattr(rnd, "king_idle_check_seconds", 900) or 0), t):
        return []
    counts: dict = runner.__dict__.setdefault("_king_idle_counts", {})
    seen, reaped = set(), []
    for pod in reap_candidates(runner):
        pid = pod.instance_id
        seen.add(pid)
        adapter = runner._operator_pod_provider(pod)
        if runner._sweep_keeps_pod(adapter, pid, what="king/backup pod (idle reaper)"):
            counts.pop(pid, None)
            continue
        counts[pid] = int(counts.get(pid, 0)) + 1
        if counts[pid] < checks:
            log.warning("king reaper: %s idle on %d/%d check(s)", pid, counts[pid], checks)
            continue
        # the final on-pod check, right before the teardown
        if runner._sweep_keeps_pod(adapter, pid, what="king/backup pod (final reaper check)"):
            counts.pop(pid, None)
            continue
        from .money_guard import OperatorSpend, ceiling_rate

        row = OperatorSpend.for_runner(runner).row_for(pid) or {}
        rate = float(row.get("rate") or ceiling_rate(runner))
        runner._teardown_operator_pod(pod)
        counts.pop(pid, None)
        reaped.append(pid)
        alert(runner, "king_pod_reaped", pid[-24:],
              f"idle king/backup pod {pid} (≈${rate:.2f}/h, no worker on {checks} checks) "
              "terminated by the reaper", severity="warning")
        log.error("king reaper: %s idle on %d checks — terminated (≈$%.2f/h)", pid, checks, rate)
    for pid in [k for k in counts if k not in seen]:
        counts.pop(pid, None)
    return reaped


# ── the operator-account inventory ──────────────────────────────────────────


def _host_file_ips(paths) -> tuple[set[str], set[str]]:
    """``(ips, names)`` of every ``[[host]]`` in the listed TOML host files."""
    import tomllib

    ips, names = set(), set()
    for raw in paths or ():
        p = Path(raw).expanduser()
        try:
            doc = tomllib.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("inventory: host file %s unreadable (%s)", p, e)
            continue
        for h in doc.get("host", []) or []:
            if isinstance(h, dict):
                if h.get("host"):
                    ips.add(str(h["host"]))
                if h.get("name"):
                    names.add(str(h["name"]))
    return ips, names


def unknown_operator_pods(runner, rows: list[dict]) -> list[dict]:
    """The rows of the OPERATOR account's ``lium ps`` that nothing accounts
    for: not on the funded ledger, not a king pod / this deployment's funded
    name, not in ``operator_inventory_allow``, not at a host IP (or name) of
    the listed host files (eval_hosts.toml, hosts.toml lanes)."""
    from ..provision.core import executor_price_per_hour, lium_pod_address

    rnd = runner.cfg.round
    allow = set(getattr(rnd, "operator_inventory_allow", ()) or ())
    allow |= {name for name, _r in getattr(rnd, "operator_standing_pods_usd_per_hour", ()) or ()}
    ips, names = _host_file_ips(getattr(rnd, "operator_inventory_host_files", ()) or ())
    allow |= names
    try:
        ledger = {p.instance_id for p in runner._load_funded_ledger()}
    except Exception:  # noqa: BLE001 — unknown ledger: report nothing (never a false alarm)
        return []
    prefix = f"cascade-n{int(runner.cfg.subnet.netuid)}-"
    out = []
    for row in rows:
        ids = {str(row.get(k) or "") for k in ("name", "huid", "id")} - {""}
        if ids & (ledger | allow):
            continue
        if any(i.startswith(prefix) and "-funded-" in i for i in ids):
            continue                      # this deployment's own funded names (the sweep's)
        addr = lium_pod_address(row) if isinstance(row, dict) else None
        if addr is not None and addr.ip in ips:
            continue
        price = None
        for cand in (row, row.get("executor") or {}, row.get("machine") or {}):
            if isinstance(cand, dict):
                price = executor_price_per_hour(cand)
                if price:
                    break
        out.append({"name": str(row.get("name") or row.get("huid") or row.get("id") or "?"),
                    "status": str(row.get("status") or "?"),
                    "ip": getattr(addr, "ip", "") if addr is not None else "",
                    "usd_per_hour": price})
    return out


def inventory_tick(runner, *, now: float | None = None, lister=None) -> list[dict]:
    """One inventory pass (every ``operator_inventory_interval_seconds``):
    one alert per unknown pod on the operator account. Never deletes."""
    from .money_guard import alert, clear_alert

    t = time.time() if now is None else float(now)
    every = float(getattr(runner.cfg.round, "operator_inventory_interval_seconds", 0) or 0)
    if not _due(runner, "_inventory_at", every, t):
        return []
    if lister is None:
        from ..provision.core import LiumProvider

        lister = LiumProvider()._list_pods
    unknown = unknown_operator_pods(runner, list(lister()))
    known_alerts = runner.__dict__.setdefault("_inventory_alerted", set())
    names = set()
    for u in unknown:
        names.add(u["name"])
        rate = f"${u['usd_per_hour']:.2f}/h" if u["usd_per_hour"] else "$?/h"
        alert(runner, "operator_unknown_pod", u["name"][:32],
              f"UNKNOWN pod on the OPERATOR account: {u['name']} ({u['status']}, {rate}"
              + (f", {u['ip']}" if u["ip"] else "") + ") — on no ledger, lane, eval-host file "
              "or allowlist; check it (never auto-deleted)", severity="warning",
              usd_per_hour=u["usd_per_hour"])
        if u["name"] not in known_alerts:
            log.error("inventory: unknown operator pod %s (%s, %s)", u["name"], u["status"], rate)
    for gone in known_alerts - names:
        clear_alert(runner, "operator_unknown_pod", gone[:32])
    runner.__dict__["_inventory_alerted"] = names
    return unknown


# ── stuck teardowns (audit L) ────────────────────────────────────────────────

TEARDOWN_FILE = "teardown_failures.json"
STUCK_TEARDOWN_SECONDS = 15 * 60.0


def note_teardown(runner, pod_id: str, *, ok: bool, now: float | None = None) -> None:
    """Record (``ok`` False) or clear a failed teardown of ``pod_id``; the
    first failure's time is kept. Never raises."""
    import json

    from ..shared.atomic import write_text_atomic

    try:
        t = time.time() if now is None else float(now)
        path = Path(runner.work_root) / TEARDOWN_FILE
        try:
            recs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recs = {}
        before = dict(recs)
        if ok:
            recs.pop(str(pod_id), None)
        else:
            recs.setdefault(str(pod_id), t)
        if recs != before:
            write_text_atomic(path, json.dumps(recs, sort_keys=True))
        if ok:
            from .money_guard import clear_alert

            clear_alert(runner, "teardown_stuck", str(pod_id)[-32:])
    except Exception as e:  # noqa: BLE001
        log.debug("teardown record skipped (%s)", e)


def stuck_teardown_tick(runner, *, now: float | None = None) -> list[str]:
    """One alert per ledger pod whose teardown has been failing for longer
    than ``STUCK_TEARDOWN_SECONDS``, priced (the $ accrued since the first
    failure at its booked rate, else the price ceiling)."""
    import json

    from .money_guard import OperatorSpend, alert, ceiling_rate

    t = time.time() if now is None else float(now)
    try:
        recs = json.loads((Path(runner.work_root) / TEARDOWN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    on_ledger = {p.instance_id: p for p in runner._load_funded_ledger()}
    out = []
    for pid, since in sorted(recs.items()):
        if pid not in on_ledger:
            note_teardown(runner, pid, ok=True)        # gone from the ledger: settled
            continue
        if t - float(since) < STUCK_TEARDOWN_SECONDS:
            continue
        row = OperatorSpend.for_runner(runner).row_for(pid) or {}
        rate = float(row.get("rate") or ceiling_rate(runner))
        accrued = rate * (t - float(since)) / 3600.0
        payer = on_ledger[pid].payer_hotkey
        alert(runner, "teardown_stuck", pid[-32:],
              f"teardown of {pid} failing for {(t - float(since)) / 60:.0f} min "
              f"({'payer ' + payer[:12] if payer else 'OPERATOR'} account) — ≈${accrued:.2f} "
              "accrued and still billing; stop it by hand (bad key / CLI auth?)",
              severity="critical", accrued_usd=round(accrued, 2))
        out.append(pid)
    return out


# ── our own validator's rejected receipts (audit D) ─────────────────────────

PUBLISHED_FILE = "published_settlements.json"


def note_published_settlement(runner, manifest) -> None:
    """Remember a settlement we published (round id → its entry hotkeys) so
    the receipt watch can name the legs a rejection hit. Never raises."""
    import json

    from ..shared.atomic import write_text_atomic

    try:
        path = Path(runner.work_root) / PUBLISHED_FILE
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {}
        doc[str(manifest.round_id)] = {
            "at": time.time(),
            "hotkeys": sorted({str(getattr(e, "miner_hotkey", "") or "")
                               for e in getattr(manifest, "entries", ())} - {""})}
        for k in sorted(doc, key=lambda k: doc[k].get("at", 0))[:-64]:
            doc.pop(k, None)
        write_text_atomic(path, json.dumps(doc, sort_keys=True))
    except Exception as e:  # noqa: BLE001
        log.debug("published-settlement record skipped (%s)", e)


def rejected_receipt_tick(runner, *, now: float | None = None, reader=None) -> dict | None:
    """Read our own validator's latest receipt (``[round]
    receipt_watch_validator_hotkey``; "" = off). A REJECTED receipt for a
    settlement WE published raises one critical alert naming the reason and
    the legs it carried (read-only: no validator code or state is touched;
    the requeue of those legs is the owner's call). Every 10 min at most."""
    import json

    from .money_guard import alert

    hk = str(getattr(runner.cfg.round, "receipt_watch_validator_hotkey", "") or "")
    t = time.time() if now is None else float(now)
    if not hk or not _due(runner, "_receipt_watch_at", 600.0, t):
        return None
    if reader is None:
        from ..shared.hippius import receipt_latest_key

        store = runner.manifest_store()

        def reader():
            return store.get_text(receipt_latest_key(hk))
    try:
        doc = json.loads(reader())
    except Exception as e:  # noqa: BLE001 — no receipt yet / storage blip
        log.debug("receipt watch: latest receipt unreadable (%s)", e)
        return None
    rid, status = str(doc.get("round_id", "")), str(doc.get("status", ""))
    if status != "rejected":
        return None
    try:
        ours = json.loads((Path(runner.work_root) / PUBLISHED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ours = {}
    if rid not in ours:
        return None
    legs = ours[rid].get("hotkeys") or []
    alert(runner, "settlement_rejected", rid[:24],
          f"our validator REJECTED settlement {rid}: {str(doc.get('reject_reason'))[:200]} — "
          f"{len(legs)} leg(s) carried ({', '.join(h[:12] for h in legs[:8])}); their "
          "results did not count: requeue them unburned (owner decision)",
          severity="critical", hotkeys=legs)
    return {"round_id": rid, "reason": doc.get("reject_reason"), "hotkeys": legs}


# ── payer vault TTLs ─────────────────────────────────────────────────────────


def _all_intents(vault) -> list[dict]:
    """Every open create intent in the vault (any scope)."""
    import json

    if getattr(vault, "dir", None) is None:
        return list(getattr(vault, "_create_intents", {}).values())
    out = []
    for p in (Path(vault.dir) / "create-intents").glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def refresh_payer_ttls(runner) -> int:
    """Re-stamp the vaulted key (and the leases) of every payer that still
    has a funded-ledger row or an open create intent: the key that must stop
    (or resolve) a pod never ages out before it. Returns the count."""
    vault = runner._payer_vault() if callable(getattr(runner, "_payer_vault", None)) else None
    if vault is None:
        return 0
    from ..funding.providers import _lease_vault, create_intents, lease_records

    hotkeys = {p.payer_hotkey for p in runner._load_funded_ledger() if p.payer_hotkey}
    leases = {token: hk for token, hk, _c in lease_records(vault)}
    tokens = set()
    for p in runner._load_funded_ledger():
        if p.payer_credential_id:
            tokens.add(p.payer_credential_id)
    scopes = {p.instance_id.removesuffix("-0") for p in runner._load_funded_ledger()
              if p.payer_hotkey}
    rows = [r for scope in scopes for r in create_intents(vault, scope)] + _all_intents(vault)
    for row in rows:
        tok = str(row.get("token") or "")
        tokens.add(tok)
        if tok in leases:
            hotkeys.add(leases[tok])
    n = 0
    for hk in sorted(hotkeys):
        try:
            n += bool(vault.refresh(hk))
        except Exception as e:  # noqa: BLE001
            log.warning("payer TTL refresh of %s failed (%s)", hk[:12], e)
    lv = _lease_vault(vault)
    for tok in sorted(t for t in tokens if t):
        try:
            lv.refresh(tok)
        except Exception as e:  # noqa: BLE001
            log.debug("lease TTL refresh failed (%s)", e)
    return n


__all__ = ["inventory_tick", "king_reaper_tick", "money_periodic", "note_published_settlement",
           "note_teardown", "rejected_receipt_tick",
           "reap_candidates", "stuck_teardown_tick",
           "refresh_payer_ttls", "unknown_operator_pods"]
