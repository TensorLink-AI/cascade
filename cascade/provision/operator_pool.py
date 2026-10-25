"""Capped operator rentals with durable account and create-intent bindings.

Public IDs are stable leg names. Physical IDs (including Shadeform UUIDs) and
immutable cleanup credentials stay in operator-local state. A timed-out create
is unresolved, even if a subsequent eventually consistent list is empty.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path

from ..funding.providers import PayerCredential, forget_lease, lease_credentials, pod_credential
from ..funding.vault import PayerKeyVault, load_seal_key
from ..shared.atomic import path_lock, write_text_atomic
from .core import ProvisionError
from .funded import apply_cpu_blocklist, apply_price_caps, provider_for_key, terminate_verified


class SpendBook:
    def __init__(self, path, *, clock=time.time):
        self.path, self.clock = Path(path), clock

    @contextlib.contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with path_lock(self.path), open(self.path.with_suffix(".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {"allocations": []}
            if not isinstance(state, dict) or not isinstance(state.get("allocations"), list):
                raise ValueError("operator spending journal is corrupt; allocation refused")
            yield state
            write_text_atomic(self.path, json.dumps(state, sort_keys=True))

    def reserve(self, alias, *, provider, credential_id, amount, seconds, leg_cap, daily_cap):
        if any(not math.isfinite(v) or v <= 0 for v in (amount, seconds, leg_cap, daily_cap)):
            raise ProvisionError(
                "operator rental requires explicit positive finite spending limits"
            )
        now = self.clock()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        with self.transaction() as state:
            rows = state["allocations"]
            if any(r["alias"] == alias and r["status"] != "closed" for r in rows):
                raise ProvisionError("operator allocation is unresolved; reconcile before renting")
            if (
                sum(r["reserved_usd"] for r in rows if r["alias"] == alias) + amount
                > leg_cap + 1e-9
            ):
                raise ProvisionError("operator leg spending cap reached")
            if sum(r["reserved_usd"] for r in rows if r["day"] == day) + amount > daily_cap + 1e-9:
                raise ProvisionError("operator daily spending cap reached")
            row = {
                "alias": alias,
                "attempt": len(rows) + 1,
                "day": day,
                "provider": provider,
                "credential_id": credential_id,
                "reserved_usd": amount,
                "reserved_at": now,
                "expires_at": now + seconds,
                "status": "creating",
                "physical_id": "",
            }
            rows.append(row)
            return dict(row)

    def update(self, attempt, **changes):
        with self.transaction() as state:
            row = next(r for r in state["allocations"] if r["attempt"] == attempt)
            row.update(changes)
            return dict(row)

    def rows(self):
        with self.transaction() as state:
            return [dict(r) for r in state["allocations"]]


class OperatorPool:
    name = "operator-pool"

    def __init__(self, runner, *, factory=provider_for_key, clock=time.time):
        self.runner, self.factory, self.clock = runner, factory, clock
        root = Path(runner.work_root)
        self.book = SpendBook(root / "operator_spending.json", clock=clock)
        self.vault = PayerKeyVault(
            dir=root / "operator_credentials",
            seal_key=load_seal_key(),
            ttl_seconds=90 * 86400,
            clock=clock,
        )
        self.vault.hydrate()

    def _adapter(self, credential):
        from ..trainer.gpu_health import guard_provider

        provider = apply_cpu_blocklist(
            self.factory(credential.provider, credential.api_key),
            self.runner.cfg.round.funded_cpu_blocklist,
        )
        provider = apply_price_caps(provider, **self.runner._funded_price_caps())
        return guard_provider(self.runner, provider)

    def _credential(self, row):
        from types import SimpleNamespace

        value = pod_credential(
            self.vault,
            SimpleNamespace(
                payer_credential_id=row["credential_id"],
                payer_hotkey="operator",
                provider=row["provider"],
            ),
        )
        if value is None:
            raise ProvisionError(
                "original operator cleanup credential unavailable; allocation refused"
            )
        return value

    def _active(self, alias):
        rows = [r for r in self.book.rows() if r["alias"] == alias and r["status"] != "closed"]
        if len(rows) > 1:
            raise ProvisionError("multiple operator allocations for one leg; reconcile")
        return rows[0] if rows else None

    def _resolve(self, alias):
        row = self._active(alias)
        if row is None:
            return None
        adapter = self._adapter(self._credential(row))
        if hasattr(adapter, "incident_of"):
            adapter.incident_scope = f"operator:{row['attempt']}"
        if not row["physical_id"]:
            # Only this original account may resolve an uncertain create.
            found = adapter.list_tagged(alias.removesuffix("-0"))
            if len(found) != 1:
                raise ProvisionError("operator create outcome unresolved; no second rental")
            row = self.book.update(row["attempt"], physical_id=found[0], status="allocated")
        return row, adapter

    def _accounts(self):
        skip = getattr(getattr(self.runner, "_leg_local", None), "operator_exclude_providers", ())
        for name in self.runner.cfg.round.ops_operator_providers:
            if name in skip:
                continue
            key = os.environ.get(f"{name.upper()}_API_KEY", "").strip()
            if key:
                credential = PayerCredential(name, key)
                yield credential, self._adapter(credential)

    def capacity(self, sku, exclude_ids=()):
        total = 0
        for credential, adapter in self._accounts():
            try:
                kw = {"exclude_ids": exclude_ids} if credential.provider == "lium" else {}
                total += int(adapter.capacity(sku, **kw))
            except Exception:
                continue
        return total

    def available(self, sku, count, **kwargs):
        return self.capacity(sku, kwargs.get("exclude_ids", ())) >= count

    def launch(self, spec):
        if spec.count != 1:
            raise ProvisionError("operator pool supports one allocation per leg")
        alias = f"{spec.name_prefix}-0"
        existing = self._resolve(alias)
        if existing:
            return [alias]
        # Scan every configured account before any create. Read failures cannot
        # be interpreted as proof that an earlier rental does not exist.
        accounts = []
        for credential, adapter in self._accounts():
            try:
                found = adapter.list_tagged(spec.name_prefix)
            except Exception:
                # No create has been attempted for this job, and the durable
                # intent scan above ruled out an unresolved allocation. An
                # unused failing account may be skipped. Legacy allocations
                # from before the pool migration still require reconciliation.
                legacy = getattr(self.runner, "_load_funded_ledger", lambda: [])()
                if any(
                    not p.payer_hotkey
                    and p.provider != self.name
                    and p.instance_id.startswith(spec.name_prefix)
                    for p in legacy
                ):
                    raise ProvisionError("legacy operator allocation cannot be inspected") from None
                continue
            if found:
                raise ProvisionError(
                    "unregistered operator allocation exists; reconcile before renting"
                )
            accounts.append((credential, adapter))
        rnd = self.runner.cfg.round
        hourly = float(rnd.funded_max_price_per_hour)
        if not math.isfinite(hourly) or hourly <= 0:
            raise ProvisionError("operator pool requires a positive hourly price ceiling")
        end = getattr(getattr(self.runner, "_leg_local", None), "end_wall", None)
        seconds = max(
            self.runner._leg_wall_seconds() + rnd.funded_ready_timeout_seconds + 900,
            float(end or self.clock()) - self.clock(),
        )
        leg_cap = rnd.ops_operator_leg_cap_usd
        if "-backup-" in alias:
            leg_cap = min(leg_cap, rnd.ops_king_backup_cap_usd)
        for credential, adapter in accounts:
            choices = tuple(
                s
                for s in spec.sku_choices or (spec.sku,)
                if adapter.available(
                    s,
                    1,
                    gpus=spec.gpus_per_pod,
                    **({"exclude_ids": spec.exclude_ids} if credential.provider == "lium" else {}),
                )
            )
            if not choices:
                continue
            token = lease_credentials(
                self.vault, "operator", credential, scope=alias + ":" + uuid.uuid4().hex
            )
            try:
                row = self.book.reserve(
                    alias,
                    provider=credential.provider,
                    credential_id=token,
                    amount=hourly * seconds / 3600,
                    seconds=seconds,
                    leg_cap=leg_cap,
                    daily_cap=rnd.ops_operator_daily_cap_usd,
                )
            except Exception:
                forget_lease(self.vault, token)
                raise
            if hasattr(adapter, "incident_of"):
                adapter.incident_scope = f"operator:{row['attempt']}"
            # Persist intent before the paid call. Exceptions retain both the
            # debit and cleanup credential; never fall through to another API.
            ids = adapter.launch(
                replace(spec, sku=choices[0], sku_choices=choices if len(choices) > 1 else ())
            )
            if len(ids) != 1:
                raise ProvisionError("operator create returned unexpected allocations; reconcile")
            self.book.update(row["attempt"], physical_id=ids[0], status="allocated")
            return [alias]
        raise ProvisionError("no_capacity on operator providers within configured limits")

    def incident_of(self, alias):
        row = self._active(alias)
        return f"operator:{row['attempt']}" if row else f"operator:{alias}"

    def _call(self, alias, method, *args, **kwargs):
        resolved = self._resolve(alias)
        if resolved is None:
            raise ProvisionError("operator allocation is not registered")
        row, adapter = resolved
        return getattr(adapter, method)(row["physical_id"], *args, **kwargs)

    def wait_ready(self, alias, *, timeout):
        return self._call(alias, "wait_ready", timeout=timeout)

    def get_ip(self, alias):
        return self._call(alias, "get_ip")

    def live_pod_address(self, alias):
        if self._active(alias) is None:
            return None
        return self._call(alias, "live_pod_address")

    def sku_of(self, alias):
        return self._call(alias, "sku_of")

    def machine_of(self, alias):
        return self._call(alias, "machine_of")

    def host_of_pod(self, alias):
        row, adapter = self._resolve(alias)
        method = getattr(adapter, "host_of_pod", None)
        return method(row["physical_id"]) if method else ""

    def list_tagged(self, prefix):
        live = []
        for row in self.book.rows():
            if row["status"] == "closed" or not row["alias"].startswith(prefix):
                continue
            resolved = self._resolve(row["alias"])
            entry, adapter = resolved
            # Physical IDs are opaque; list the original scope, then compare IDs.
            if entry["physical_id"] in adapter.list_tagged(entry["alias"].removesuffix("-0")):
                live.append(entry["alias"])
        return live

    def terminate(self, alias):
        resolved = self._resolve(alias)
        if resolved is None:
            return
        row, adapter = resolved
        if not terminate_verified(adapter, row["physical_id"]):
            raise ProvisionError("operator allocation teardown not confirmed")
        self.book.update(row["attempt"], status="closed")
        forget_lease(self.vault, row["credential_id"])

    def reap_expired(self):
        """Best effort; unresolved creates and failed deletions stay journaled."""
        for row in self.book.rows():
            if row["status"] != "closed" and row["expires_at"] <= self.clock():
                with contextlib.suppress(Exception):
                    self.terminate(row["alias"])


def enabled(runner):
    cfg = runner.cfg.round
    return bool(
        cfg.ops_operator_providers != ("lium",)
        or cfg.ops_operator_leg_cap_usd
        or cfg.ops_operator_daily_cap_usd
        or cfg.ops_king_backup_after_seconds
        or cfg.ops_paid_fallback_after_seconds
    )


def operator_provider(runner, legacy_factory):
    if enabled(runner):
        return OperatorPool(runner)
    from ..trainer.gpu_health import guard_provider

    return guard_provider(runner, legacy_factory())
