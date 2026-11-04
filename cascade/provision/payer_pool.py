"""Select a miner's provider before a rental, and retain its credential for cleanup.

Selection is read-only. After launch begins, this function never falls through
to another provider: a timeout does not prove that the first create was rejected.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from ..funding.faults import classify_rent_failure
from ..funding.providers import (
    begin_create,
    create_intents,
    decode_credentials,
    lease_credentials,
    lease_records,
    resolve_create,
)
from .core import PriceCapUnavailable, ProvisionError

log = logging.getLogger("cascade.provision.payer_pool")


def rent_from_pool(options, rent_one):
    from .funded import (
        FundedRentResult,
        apply_cpu_blocklist,
        apply_price_caps,
        funded_pod_name,
        provider_for_key,
    )

    options = dict(options)
    raw = options.pop("api_key")
    vault = options.pop("vault")
    selected_callback = options.pop("on_provider_selected")
    gpu_health = options.pop("gpu_health", None)
    hotkey = options["hotkey"]
    credentials = decode_credentials(raw)
    name = funded_pod_name(options["round_id"], hotkey, options["netuid"])
    if vault is None:
        return FundedRentResult(
            hotkey,
            False,
            error="provider portfolio needs a persistent payer vault",
            error_class="infra",
            burn_attempt=False,
        )
    all_credentials = list(credentials)
    try:
        # Retired credentials remain usable only to find/adopt/stop an existing
        # rental. They are never candidates for another paid create.
        historical = [c for _, _, c in lease_records(vault, hotkey)]
        all_credentials = list(dict.fromkeys((*credentials, *historical)))
        adapters = []
        existing = []
        for credential in all_credentials:
            adapter = apply_cpu_blocklist(
                provider_for_key(credential.provider, credential.api_key), options["cpu_blocklist"]
            )
            adapter = apply_price_caps(adapter, **(options["price_caps"] or {}))
            if gpu_health is not None and gpu_health.threshold:
                from ..trainer.gpu_health import GuardedProvider

                adapter = GuardedProvider(adapter, gpu_health)
            # If an account cannot be queried, do not assume its prior create
            # failed and spend on a different account.
            # This listing is also the teardown-key check (2026-10-31
            # 'Invalid API key'): every verified teardown starts with it, so
            # a key that cannot list cannot stop what it would rent.
            try:
                pods = adapter.list_tagged(name)
            except Exception as exc:
                from .funded import _log_key_cannot_stop

                _log_key_cannot_stop(hotkey, f"{name}-0 ({credential.provider})", exc,
                                     credential.api_key)
                raise
            adapters.append((credential, adapter))
            if len(pods) > 1:
                raise ProvisionError(
                    "multiple allocations exist for this leg; reconcile before another rental"
                )
            if pods:
                existing.append((credential, adapter))
        if len(existing) > 1:
            raise ProvisionError(
                "multiple providers have this leg; reconcile before another rental"
            )
        chosen = existing[0] if existing else None
        if chosen is None and create_intents(vault, name):
            return FundedRentResult(
                hotkey, False, error="create_unresolved: prior rental is not resolved; no new rental",
                error_class="infra", burn_attempt=False, leaked_pod=name + "-0")
        if chosen is None:
            pricing_errors = []
            for credential, adapter in adapters:
                if credential not in credentials:
                    continue
                try:
                    available = any(adapter.available(
                        sku,
                        1,
                        gpus=options["gpus_per_pod"],
                        **(
                            {"exclude_ids": options["exclude_ids"]}
                            if credential.provider in ("lium", "vast") and options["exclude_ids"]
                            else {}
                        ),
                    )
                        for sku in options["skus"] or (options["sku"],))
                except PriceCapUnavailable as exc:
                    pricing_errors.append(str(exc))
                    continue
                if available:
                    chosen = credential, adapter
                    break
        if chosen is None:
            if pricing_errors:
                return FundedRentResult(hotkey, False, error="; ".join(pricing_errors),
                                        error_class="price_cap_unavailable", burn_attempt=False)
            return FundedRentResult(
                hotkey,
                False,
                error="no_capacity on the supplied providers under the GPU/price limits",
                error_class="no_capacity",
                burn_attempt=False,
            )
        credential, adapter = chosen
        token = lease_credentials(vault, hotkey, credential, scope=name)
        if selected_callback is not None:
            selected_callback(credential.provider, token)
        if existing:
            from .funded import terminate_verified

            pod_id = adapter.list_tagged(name)[0]
            callback = options.get("on_allocated")
            if callback is not None:
                callback("", pod_id)
            try:
                ready = adapter.wait_ready(pod_id, timeout=options["ready_timeout"])
            except Exception:
                return FundedRentResult(
                    hotkey,
                    False,
                    error="existing allocation cannot be inspected; no new rental",
                    error_class="infra",
                    burn_attempt=False,
                    leaked_pod=pod_id,
                )
            if not ready:
                try:
                    gone = terminate_verified(adapter, pod_id)
                except Exception:
                    gone = False
                if gone:
                    from ..funding.providers import forget_lease

                    forget_lease(vault, token)
                return FundedRentResult(
                    hotkey,
                    False,
                    error="existing allocation did not become ready; no new rental",
                    error_class="infra",
                    burn_attempt=False,
                    leaked_pod="" if gone else pod_id,
                )
        # Exactly one provider crosses the paid boundary, even if it times out.
        result = rent_one(
            api_key=credential.api_key,
            provider_factory=lambda key: adapter,
            existing_only=bool(existing),
            on_create_started=(None if existing else lambda: begin_create(
                vault, token, scope=name, provider=credential.provider)),
            **options,
        )
        if not result.ok and not result.leaked_pod:
            from ..funding.providers import forget_lease

            forget_lease(vault, token)
        if result.pod is not None:
            resolve_create(vault, token)
            result = replace(result, pod=replace(result.pod, payer_credential_id=token))
        return result
    except Exception as exc:
        message = str(exc)
        for credential in all_credentials:
            message = message.replace(credential.api_key, "<redacted>")
        return FundedRentResult(
            hotkey,
            False,
            error=message[:500],
            error_class=classify_rent_failure(message),
            burn_attempt=False,
        )


def reconcile_payer_accounts(owned, vault, *, netuid, provider_factory, hotkeys, exclude,
                             keep=None):
    import logging

    from ..funding.providers import decode_credentials, lease_records
    from .funded import (
        lium_provider_for_key,
        payer_pod_pattern,
        provider_for_key,
        terminate_verified,
    )

    log = logging.getLogger(__name__)
    owned_ids = {(p.provider, p.instance_id) for p in owned if p.stage == "funded"}
    only, skip = (None if hotkeys is None else set(hotkeys)), set(exclude)
    accounts = []
    for hotkey in vault.hotkeys():
        accounts.extend((hotkey, c) for c in decode_credentials(vault.get(hotkey) or ""))
    accounts.extend((hk, c) for _, hk, c in lease_records(vault))
    killed = []
    for hotkey, credential in dict.fromkeys(accounts):
        if hotkey in skip or (only is not None and hotkey not in only):
            continue
        try:
            adapter = (
                provider_factory(credential.api_key)
                if provider_factory is not lium_provider_for_key
                else provider_for_key(credential.provider, credential.api_key)
            )
            named = getattr(adapter, "list_named", None)
            if named is not None:
                pods = named("cascade-")
            else:
                lister = getattr(adapter, "list_tagged", None)
                if lister is None:
                    continue
                pods = {pid: pid for pid in lister("cascade-")}
            pattern = payer_pod_pattern(hotkey, netuid)
            for pod_id, name in pods.items():
                if pattern.match(name) and (credential.provider, pod_id) not in owned_ids:
                    if keep is not None and keep(adapter, pod_id, hotkey):
                        continue
                    if terminate_verified(adapter, pod_id):
                        killed.append(pod_id)
                    else:
                        log.error(
                            "funded reconcile: %s pod %s still bills payer %s",
                            credential.provider,
                            pod_id,
                            hotkey,
                        )
            from ..funding.providers import credential_scopes, forget_lease

            for token, scope in list(credential_scopes(vault, hotkey, credential)):
                if not scope or not pattern.match(scope):
                    continue
                remaining = [
                    pid
                    for pid, name in pods.items()
                    if name.startswith(scope) and pid not in killed
                ]
                observed_gone = any(name.startswith(scope) and pid in killed
                                    for pid, name in pods.items())
                if not remaining and (observed_gone or not create_intents(vault, scope)):
                    forget_lease(vault, token)
        except Exception:
            # Exceptions may include an HTTP request carrying authentication;
            # report identities only, never headers or credential-bearing URLs.
            log.warning(
                "funded reconcile: cannot inspect/stop %s allocations for %s",
                credential.provider,
                hotkey,
            )
    return killed


def payer_capacity(raw, skus, price_caps, exclude_ids=(), gpu_health=None):
    """Read-only availability on the supplied accounts; never use operator keys."""
    from .funded import apply_price_caps, provider_for_key

    for credential in decode_credentials(raw):
        adapter = apply_price_caps(
            provider_for_key(credential.provider, credential.api_key), **price_caps
        )
        if gpu_health is not None and gpu_health.threshold:
            from ..trainer.gpu_health import GuardedProvider

            adapter = GuardedProvider(adapter, gpu_health)
        try:
            available = any(
            adapter.available(
                sku,
                1,
                gpus=1,
                **(
                    {"exclude_ids": exclude_ids}
                    if credential.provider in ("lium", "vast") and exclude_ids
                    else {}
                ),
            )
            for sku in skus
            )
        except PriceCapUnavailable:
            continue
        if available:
            return 1
    return 0


class PayerAccountSuspect(Exception):
    """The payer's ACCOUNT, not the market, is what lists nothing.

    Raised by :class:`PayerCapacityWatch` out of a capacity probe, so the
    funded wait ends with a miner-readable verdict instead of polling
    ``no_capacity`` until the latest safe start and re-dispatching every era
    (2026-10-31 → 11-04: one key saw an empty ``lium ls`` for five days
    while two other payers rented the same types). The text never carries
    key material: the provider adapters scrub it before it reaches here."""

    def __init__(self, kind: str, *, skus=(), control: int | None = None,
                 zero_polls: int = 0, span_seconds: float = 0.0, detail: str = ""):
        self.kind = str(kind)               # "empty_listing" | "auth"
        self.skus = tuple(skus)
        self.control = control
        self.zero_polls = int(zero_polls)
        self.span_seconds = float(span_seconds)
        self.detail = str(detail or "")
        super().__init__(self.describe())

    def describe(self) -> str:
        types = "/".join(self.skus) or "the requested GPU types"
        if self.kind == "auth":
            return (f"your GPU provider refused the key when listing {types}: "
                    f"{self.detail[-300:]}")
        return (f"your GPU provider account lists no {types} executor under your price caps "
                f"({self.zero_polls} polls over {self.span_seconds / 60.0:.0f} min) while the "
                f"marketplace shows {self.control} on the operator's listing under the same caps")


@dataclass
class PayerCapacityWatch:
    """A funded leg's capacity probe that tells an EMPTY ACCOUNT from an
    empty MARKET.

    ``payer_probe`` counts the payer's own listing (``payer_capacity``);
    ``control_probe`` is the same read-only listing on the OPERATOR's key
    under the payer's price caps and GPU-health guard. After
    ``zero_polls_before_control`` consecutive payer zeros the control runs
    (at most every ``control_interval_seconds``); ``control_hits_needed``
    consecutive controls that show capacity — while every payer poll in
    between still saw nothing — raise :class:`PayerAccountSuspect`. A control
    that is empty too resets the count (a real drought stays ``no_capacity``).
    A payer probe whose error classifies ``auth`` raises at once; any other
    probe failure is logged and answered "unknown" (None), never swallowed."""

    payer_probe: Callable[[tuple[str, ...]], int | None]
    control_probe: Callable[[tuple[str, ...]], int | None]
    now: Callable[[], float] = time.time
    zero_polls_before_control: int = 3
    control_interval_seconds: float = 300.0
    control_hits_needed: int = 2
    zero_polls: int = field(default=0, init=False)
    first_zero_at: float = field(default=0.0, init=False)
    control_hits: int = field(default=0, init=False)
    last_control_at: float = field(default=0.0, init=False)
    last_control: int | None = field(default=None, init=False)
    verdict: PayerAccountSuspect | None = field(default=None, init=False)

    def __call__(self, skus) -> int | None:
        skus = tuple(skus)
        try:
            n = self.payer_probe(skus)
        except PayerAccountSuspect:
            raise
        except Exception as e:  # noqa: BLE001 — classified and logged, never swallowed
            text = str(e)
            if classify_rent_failure(text) == "auth":
                self.verdict = PayerAccountSuspect("auth", skus=skus, detail=text)
                raise self.verdict from None
            log.warning("payer capacity probe for %s failed (%s: %s) — no answer this poll",
                        "/".join(skus), type(e).__name__, text[-200:])
            return None
        if n is None:
            return None
        if n:
            self.zero_polls, self.control_hits, self.first_zero_at = 0, 0, 0.0
            return n
        t = float(self.now())
        if not self.zero_polls:
            self.first_zero_at = t
        self.zero_polls += 1
        if (self.zero_polls >= int(self.zero_polls_before_control)
                and t - self.last_control_at >= float(self.control_interval_seconds)):
            self.last_control_at = t
            try:
                c = self.control_probe(skus)
            except Exception as e:  # noqa: BLE001 — an unreadable control never convicts
                log.warning("operator control listing for %s failed (%s: %s) — not counted",
                            "/".join(skus), type(e).__name__, str(e)[-200:])
                c = None
            self.last_control = c
            if c:
                self.control_hits += 1
                log.info("payer listing empty for %s on %d poll(s) while the operator's listing "
                         "shows %d under the same caps (%d/%d controls)", "/".join(skus),
                         self.zero_polls, c, self.control_hits, self.control_hits_needed)
            else:
                self.control_hits = 0
            if self.control_hits >= int(self.control_hits_needed):
                self.verdict = PayerAccountSuspect(
                    "empty_listing", skus=skus, control=c, zero_polls=self.zero_polls,
                    span_seconds=t - self.first_zero_at)
                raise self.verdict
        return 0
