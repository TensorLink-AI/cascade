"""Persistent, idempotent GPU-type circuit breaker, shared across A/B/C and king work."""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

from ..funding.faults import classify_rent_failure
from ..shared.atomic import path_lock, write_text_atomic


class GpuHealth:
    def __init__(self, path, *, threshold=2, window=3600, cooldown=10800, clock=time.time):
        self.path = Path(path)
        self.threshold, self.window, self.cooldown = threshold, window, cooldown
        self.clock = clock

    def _read(self):
        if not self.path.exists():
            return {"events": [], "blocked": {}}
        value = json.loads(self.path.read_text())
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("events"), list)
            or not isinstance(value.get("blocked"), dict)
        ):
            raise ValueError("GPU health state is corrupt; restore before allocating")
        return value

    def record(self, sku, incident, *, kind, reason=""):
        if (
            not self.threshold
            or not sku
            or not incident
            or kind not in {"boot", "lost", "slow", "stale"}
        ):
            return False
        if classify_rent_failure(reason) in {"auth", "rate_limited", "no_capacity"}:
            return False
        now = self.clock()
        with path_lock(self.path):
            state = self._read()
            events = [e for e in state["events"] if now - e["at"] <= self.window]
            if any(e["incident"] == incident for e in events):
                return False
            events.append({"sku": sku, "incident": incident, "kind": kind, "at": now})
            blocked = {s: until for s, until in state["blocked"].items() if until > now}
            if sum(e["sku"] == sku for e in events) >= self.threshold:
                blocked[sku] = max(blocked.get(sku, 0), now + self.cooldown)
            write_text_atomic(self.path, json.dumps({"events": events, "blocked": blocked}))
            return sku in blocked

    def blocked(self):
        with path_lock(self.path):
            return {
                s: until for s, until in self._read()["blocked"].items() if until > self.clock()
            }

    def allowed(self, skus):
        blocked = self.blocked() if self.threshold else {}
        return tuple(s for s in skus if s not in blocked)


class GuardedProvider:
    def __init__(self, provider, health):
        self.provider, self.health = provider, health

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def available(self, sku, count, **kwargs):
        return bool(self.health.allowed((sku,))) and self.provider.available(sku, count, **kwargs)

    def capacity(self, sku, **kwargs):
        if not self.health.allowed((sku,)):
            return 0
        method = getattr(self.provider, "capacity", None)
        return method(sku, **kwargs) if method else int(self.provider.available(sku, 1, **kwargs))

    def launch(self, spec):
        from ..provision.core import ProvisionError

        fit = self.health.allowed(spec.sku_choices or (spec.sku,))
        if not fit:
            raise ProvisionError("no_capacity: approved GPU types are temporarily quarantined")
        self._launch_sku = fit[0]
        ids = self.provider.launch(
            replace(spec, sku=fit[0], sku_choices=fit if len(fit) > 1 else ())
        )
        self._pod_skus = getattr(self, "_pod_skus", {})
        self._pod_incidents = getattr(self, "_pod_incidents", {})
        for pod in ids:
            self._pod_incidents[pod] = f"{self.name}:{pod}:{uuid.uuid4().hex}"
            getter = getattr(self.provider, "sku_of", None)
            try:
                self._pod_skus[pod] = (getter(pod) if getter else "") or fit[0]
            except Exception:
                # Return the accepted IDs so the caller journals/cleans them.
                # A metadata failure must not turn a real create into an
                # unknown allocation or misidentify a GPU outage.
                self._pod_skus[pod] = ""
        return ids

    def wait_ready(self, pod, *, timeout):
        try:
            result = self.provider.wait_ready(pod, timeout=timeout)
        except Exception as exc:
            reason = str(exc)
            self._failure(pod, reason)
            raise
        if not result:
            # Lium can accept the CLI command then reject it for account reasons.
            tail = getattr(self.provider, "_up_log_tail", lambda _: "")(pod)
            self._failure(pod, tail)
        return result

    def incident_of(self, pod):
        if getattr(self, "incident_scope", ""):
            return self.incident_scope
        incidents = getattr(self, "_pod_incidents", {})
        if pod in incidents:
            return incidents[pod]
        identity = None
        getter = getattr(self.provider, "pod_identity", None)
        if getter is not None:
            with contextlib.suppress(Exception):
                identity = getter(pod)
        return f"{self.name}:{(identity or {}).get('id') or pod}"

    def _failure(self, pod, reason):
        sku = getattr(self, "_pod_skus", {}).get(pod, "")
        self.health.record(sku, self.incident_of(pod), kind="boot", reason=reason)


def health_for(runner):
    cfg = runner.cfg.round
    threshold = int(getattr(cfg, "ops_gpu_failure_threshold", 0))
    return GpuHealth(
        Path(runner.work_root) / "gpu_health.json",
        threshold=threshold,
        window=getattr(cfg, "ops_gpu_failure_window_seconds", 3600),
        cooldown=getattr(cfg, "ops_gpu_cooldown_seconds", 10800),
    )


def guard_provider(runner, provider):
    health = health_for(runner)
    return GuardedProvider(provider, health) if health.threshold else provider


def record_pod_failure(runner, provider, pod_id, sku, reason, *, kind):
    """Use a physical allocation identity consistently across failure callbacks."""
    health = health_for(runner)
    if not health.threshold:
        return
    incident = getattr(provider, "incident_of", None)
    if callable(incident):
        health.record(sku, incident(pod_id), kind=kind, reason=reason)
        return
    name, physical = provider.name, pod_id
    if name == "operator-pool":
        row = provider._active(pod_id)
        if row and row.get("physical_id"):
            name, physical = row["provider"], row["physical_id"]
    health.record(sku, f"{name}:{physical}", kind=kind, reason=reason)
