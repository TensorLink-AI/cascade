"""Independent operator watchdog: durable alerts, public health and optional delivery.

No wallet/API-key values or exact account balances enter alerts. Delivery is
at-least-once with stable event IDs; receivers can deduplicate a timed-out POST.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from ..shared.atomic import write_text_atomic
from .gpu_health import health_for

log = logging.getLogger(__name__)


def lium_balance():
    """Documented GET /users/me; missing data is unknown, never zero."""
    import math

    import requests

    from ..provision.core import LIUM_API_BASE

    key = os.environ.get("LIUM_API_KEY")
    if not key:
        return None
    response = requests.get(
        f"{LIUM_API_BASE}/users/me", headers={"X-API-KEY": key}, timeout=(3, 10)
    )
    response.raise_for_status()
    value = response.json().get("balance")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def inspect_health(
    *,
    now,
    current_block,
    chain_timestamp,
    era,
    last_settled_boundary,
    settlement_boundary,
    queues,
    balances,
    cfg,
    gpu_blocked=None,
):
    """Pure condition evaluator; diagnostic ages never change settlement identity."""
    conditions = []

    def add(code, scope, **data):
        conditions.append({"code": code, "scope": scope, **data})

    for provider, balance in balances.items():
        if balance is None:
            add("balance_unknown", provider)
        elif balance < cfg.ops_low_balance_usd:
            add("low_balance", provider, threshold_usd=cfg.ops_low_balance_usd)
    if chain_timestamp is None and cfg.ops_clock_drift_seconds > 0:
        add("chain_timestamp_unknown", "chain")
    elif (
        chain_timestamp is not None
        and cfg.ops_clock_drift_seconds > 0
        and abs(now - chain_timestamp) > cfg.ops_clock_drift_seconds
    ):
        add("chain_clock_drift", "chain", drift_seconds=round(now - chain_timestamp, 1))
    if (
        era is not None
        and max(
            int(getattr(era, "king_leg_failures_total", 0)),
            int(getattr(era, "king_leg_failures", 0)),
        )
        >= 2
    ):
        add(
            "king_repeated_failure",
            "king",
            count=max(
                int(getattr(era, "king_leg_failures_total", 0)),
                int(getattr(era, "king_leg_failures", 0)),
            ),
        )
    if (
        era is not None
        and getattr(era, "king_hotkey", "")
        and not getattr(era, "king_entry", None)
        and last_settled_boundary < settlement_boundary
    ):
        # Hold age is measured by the watchdog across observations, not inferred
        # from a fixed twelve-second projection of an unhealthy chain.
        add("settlement_waiting_for_king", "settlement", boundary=settlement_boundary)
    if (
        era is not None
        and getattr(era, "king_entry", None)
        and last_settled_boundary < settlement_boundary
    ):
        add("settlement_held", "settlement", boundary=settlement_boundary)
    waiting = {}
    for lane, entries in queues.items():
        starts = [
            float(e.funded_at)
            for e in entries
            if e.status in {"queued", "pending"}
            or e.status == "in_flight"
            and getattr(e, "phase", "") in {"waiting_admission", "waiting_gpu", "waiting_operator"}
        ]
        age = max(0.0, now - min(starts)) if starts else 0.0
        waiting[lane] = {"count": len(starts), "oldest_wait_seconds": round(age, 1)}
        if age >= cfg.ops_oldest_wait_seconds and starts:
            add("queue_backlog", lane, oldest_wait_seconds=round(age, 1), count=len(starts))
    for sku, until in (gpu_blocked or {}).items():
        add("gpu_type_cooldown", sku, until=until)
    return {
        "as_of": now,
        "block": current_block,
        "conditions": conditions,
        "queues": waiting,
        "king_ready": bool(era is not None and getattr(era, "king_entry", None)),
        "gpu_blocked": gpu_blocked or {},
    }


class OperatorMonitor:
    def __init__(
        self, runner, client, *, clock=time.time, balance_reader=lium_balance, deliver=None
    ):
        self.runner, self.client, self.clock = runner, client, clock
        self.balance_reader, self.deliver = balance_reader, deliver
        self.path = Path(runner.work_root) / "operator_health.json"
        self.stop_event = threading.Event()
        self.thread = None
        self._record_lock = threading.RLock()
        self._sample_thread = None
        self._sample_started = 0.0

    def _load(self):
        if not self.path.exists():
            return {"active": {}, "outbox": []}
        value = json.loads(self.path.read_text())
        if not isinstance(value.get("active"), dict) or not isinstance(value.get("outbox"), list):
            raise ValueError("operator alert state is invalid")
        return value

    def record(self, snapshot):
        with self._record_lock:
            return self._record(snapshot)

    def _record(self, snapshot):
        state, now = self._load(), snapshot["as_of"]
        seen = set()
        for condition in snapshot["conditions"]:
            key = f"{condition['code']}:{condition['scope']}"
            seen.add(key)
            previous = state["active"].get(key)
            first = previous["first"] if previous else now
            state["active"][key] = {"first": first, "last": now, "condition": condition}
            hold = condition["code"] in {"settlement_waiting_for_king", "settlement_held"}
            delay = self.runner.cfg.round.ops_settlement_hold_seconds if hold else 0
            if now - first < delay:
                continue
            if previous is None or not previous.get("emitted"):
                self._enqueue(state, condition, first, "active")
                state["active"][key]["emitted"] = True
            else:
                state["active"][key]["emitted"] = True
        for key in list(state["active"]):
            if key not in seen:
                old = state["active"].pop(key)
                if old.get("emitted"):
                    self._enqueue(state, old["condition"], now, "resolved")
        # Bound disk growth without dropping undelivered active alerts in favor
        # of a verbose successful-poll history.
        pending = [e for e in state["outbox"] if not e["delivered"]]
        delivered = [e for e in state["outbox"] if e["delivered"]]
        state["outbox"] = delivered[-64:] + pending
        state["snapshot"] = snapshot
        write_text_atomic(self.path, json.dumps(state, sort_keys=True), backup=True)
        self.runner.__dict__["_operations_status"] = {
            **snapshot,
            "alerts": [v["condition"] for v in state["active"].values() if v.get("emitted")],
        }
        self._deliver(state)
        return state

    def _enqueue(self, state, condition, at, status):
        event = {"condition": condition, "at": at, "status": status}
        event["id"] = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        key = (condition["code"], condition["scope"])
        pending = [
            e
            for e in state["outbox"]
            if not e["delivered"]
            and (e["event"]["condition"]["code"], e["event"]["condition"]["scope"]) == key
        ]
        # Keep the original failure followed by its latest resolution. An
        # outage cannot grow the queue through repeated active/resolved cycles.
        if len(pending) >= 2:
            state["outbox"].remove(pending[-1])
        state["outbox"].append({"event": event, "delivered": False})
        log.warning("operator alert: %s %s (%s)", condition["code"], condition["scope"], status)

    def _deliver(self, state):
        name = getattr(self.runner.cfg.round, "ops_alert_webhook_env", "")
        url = os.environ.get(name, "") if name else ""
        sender = self.deliver
        if sender is None and url:
            parsed = urlparse(url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError("operator alert delivery requires a credential-free HTTPS URL")
            import requests

            def sender(event):
                response = requests.post(url, json=event, timeout=(3, 10), allow_redirects=False)
                response.raise_for_status()
                if not 200 <= response.status_code < 300:
                    raise RuntimeError("operator alert receiver did not accept the event")

        if sender is None:
            return
        sent = 0
        for item in state["outbox"]:
            if item["delivered"]:
                continue
            try:
                sender(item["event"])
            except Exception:
                log.warning("operator alert delivery failed; event retained for retry")
                break
            item["delivered"] = True
            sent += 1
            write_text_atomic(self.path, json.dumps(state, sort_keys=True), backup=True)
            if sent >= 3:
                break

    def sample(self):
        from ..shared.config import effective_epoch_blocks
        from .runner_common import funded_queue_of

        runner, now = self.runner, self.clock()
        from ..provision.operator_pool import OperatorPool, enabled

        if enabled(runner):
            OperatorPool(runner).reap_expired()
        # This daemon is independent of the settlement loop. A dead RPC/API
        # cannot hold a king's allocation or delay another lane.
        block = self.client.current_block()
        timestamp = None
        get_time = getattr(self.client, "block_timestamp", None)
        if get_time is not None:
            with contextlib.suppress(Exception):
                timestamp = get_time(block)
        balances = {}
        for provider in runner.cfg.round.ops_operator_providers:
            balances[provider] = None
            if provider == "lium":
                with contextlib.suppress(Exception):
                    balances[provider] = self.balance_reader()
        sched = runner.__dict__.get("_rolling_sched")
        state = getattr(sched, "state", None)
        queues = {}
        for lane in ("a", "b", "c"):
            q = funded_queue_of(runner, lane=lane)
            queues[lane] = q.entries() if q is not None else []
        eb = effective_epoch_blocks(runner.cfg.round, block)
        snap = inspect_health(
            now=now,
            current_block=block,
            chain_timestamp=timestamp,
            era=getattr(state, "current", None),
            last_settled_boundary=int(getattr(state, "last_settled_boundary", 0)),
            settlement_boundary=(int(block) // eb) * eb,
            queues=queues,
            balances=balances,
            cfg=runner.cfg.round,
            gpu_blocked=health_for(runner).blocked(),
        )
        snap["sample_duration_seconds"] = round(max(0, self.clock() - now), 1)
        if snap["sample_duration_seconds"] > max(
            30, runner.cfg.round.ops_alert_interval_seconds * 2
        ):
            snap["stale"] = True
            snap["observed_as_of"] = now
            snap["conditions"].append({"code": "watchdog_stalled", "scope": "watchdog"})
        return self.record(snap)

    def _unavailable(self, code):
        # A failed sampler must not turn previously observed problems into
        # resolved alerts or make an old queue snapshot look freshly sampled.
        with self._record_lock:
            old = self._load().get("snapshot", {})
            conditions = [
                c
                for c in old.get("conditions", [])
                if c["code"] not in {"watchdog_stalled", "watchdog_unavailable"}
            ]
            self._record(
                {
                    **old,
                    "as_of": self.clock(),
                    "observed_as_of": old.get("observed_as_of", old.get("as_of")),
                    "stale": True,
                    "conditions": conditions + [{"code": code, "scope": "watchdog"}],
                }
            )

    def poll(self):
        """At most one outstanding sampler, even when an RPC never returns."""
        if self._sample_thread is not None and self._sample_thread.is_alive():
            if self.clock() - self._sample_started > max(
                30, self.runner.cfg.round.ops_alert_interval_seconds * 2
            ):
                self._unavailable("watchdog_stalled")
            return
        self._sample_started = self.clock()

        def sample_once():
            try:
                self.sample()
            except Exception:
                log.warning("operator watchdog sample unavailable")
                with contextlib.suppress(Exception):
                    self._unavailable("watchdog_unavailable")

        self._sample_thread = threading.Thread(
            target=sample_once, name="cascade-operator-health-sample", daemon=True
        )
        self._sample_thread.start()

    def start(self):
        interval = self.runner.cfg.round.ops_alert_interval_seconds
        if self.thread is not None or interval <= 0:
            return

        def run():
            while not self.stop_event.is_set():
                try:
                    self.poll()
                except Exception:
                    log.warning("operator watchdog state unavailable")
                self.stop_event.wait(max(5, interval))

        self.thread = threading.Thread(target=run, name="cascade-operator-watchdog", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
