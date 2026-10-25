"""Select a miner's provider before a rental, and retain its credential for cleanup.

Selection is read-only. After launch begins, this function never falls through
to another provider: a timeout does not prove that the first create was rejected.
"""

from __future__ import annotations

from dataclasses import replace

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
            pods = adapter.list_tagged(name)
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


def reconcile_payer_accounts(owned, vault, *, netuid, provider_factory, hotkeys, exclude):
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
