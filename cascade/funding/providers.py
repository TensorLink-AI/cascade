"""Ordered rolling-lane payer credentials and immutable rental credential references.

Only environment variable *names* enter CLI arguments. Credential bundles are
signed, sent over TLS, and stored inside the existing local sealed payer vault.
Keys never enter public queues. Rental ledgers contain provider names and
opaque lease references only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from ..shared.atomic import path_lock, write_text_atomic

ROLLING_PROVIDERS = ("lium", "shadeform", "vast", "runpod")
CREDENTIALS_HEADER = "X-Payer-Credentials"
PREFIX = "cascade-payer-v1:"


def spending_controls():
    """Discovery only: provider capabilities, never proof of a key's budget."""
    return {
        "lium": {"hourly_price_cap": True, "key_budget": "documented_daily_monthly_lifetime",
                 "key_budget_verified": False, "docs": "https://docs.lium.io/pod-users/api-keys#budgets"},
        "shadeform": {"hourly_price_cap": True, "key_budget": "not_confirmed",
                      "instance_spend_threshold": True, "key_budget_verified": False,
                      "docs": "https://docs.shadeform.ai/api-reference/instances/instances-create"},
        "vast": {"hourly_price_cap": True, "key_budget": "not_confirmed", "key_budget_verified": False,
                 "docs": "https://docs.vast.ai/guides/reference/api-keys"},
        "runpod": {"hourly_price_cap": False, "key_budget": "not_confirmed", "key_budget_verified": False,
                   "requires_explicit_uncapped_gpu_choice": True,
                   "docs": "https://docs.runpod.io/get-started/credentials"},
    }


@dataclass(frozen=True)
class PayerCredential:
    provider: str
    api_key: str = field(repr=False)


def decode_credentials(value: str) -> tuple[PayerCredential, ...]:
    if not value:
        return ()
    if not value.startswith(PREFIX):
        return (PayerCredential("lium", value),)
    try:
        rows = json.loads(value[len(PREFIX) :])
        if not isinstance(rows, list) or not 1 <= len(rows) <= len(ROLLING_PROVIDERS):
            raise ValueError
        result = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"provider", "api_key"}:
                raise ValueError
            provider, key = row["provider"], row["api_key"]
            if provider not in ROLLING_PROVIDERS or provider in seen:
                raise ValueError
            if not isinstance(key, str) or not key.strip() or len(key) > 2048:
                raise ValueError
            seen.add(provider)
            result.append(PayerCredential(provider, key))
        return tuple(result)
    except (ValueError, TypeError, KeyError):
        raise ValueError(
            "invalid payer credentials; use one nonempty key per supported provider"
        ) from None


def encode_credentials(credentials) -> str:
    value = PREFIX + json.dumps(
        [{"provider": c.provider, "api_key": c.api_key} for c in credentials],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    decode_credentials(value)
    return value


def credentials_from_env(args) -> str:
    specs = getattr(args, "provider_key_env", None) or []
    if not specs:
        return (os.environ.get(args.lium_key_env) or "").strip()
    credentials = []
    for spec in specs:
        provider, sep, name = spec.partition("=")
        if not sep or provider not in ROLLING_PROVIDERS or not name.isidentifier():
            raise ValueError("--provider-key-env must be PROVIDER=ENV_NAME with PROVIDER one of "
                             + ", ".join(ROLLING_PROVIDERS))
        value = (os.environ.get(name) or "").strip()
        if not value:
            raise ValueError(f"no API key in ${name}; export it first")
        credentials.append(PayerCredential(provider, value))
    return encode_credentials(credentials)


def credential_headers(value: str) -> dict[str, str]:
    return (
        {(CREDENTIALS_HEADER if value.startswith(PREFIX) else "X-Lium-Api-Key"): value}
        if value
        else {}
    )


def header_credentials(headers) -> str:
    bundle = headers.get(CREDENTIALS_HEADER) or ""
    legacy = (headers.get("X-Lium-Api-Key") or "").strip()
    if bundle and legacy:
        raise ValueError("send either payer credentials or a Lium key, not both")
    if bundle:
        if not bundle.startswith(PREFIX) or len(bundle) > 16384:
            raise ValueError("invalid payer credentials header")
        decode_credentials(bundle)
    return bundle or legacy


def add_provider_args(parser) -> None:
    parser.add_argument(
        "--provider-key-env",
        action="append",
        metavar="PROVIDER=ENV_NAME",
        help="Payer key environment variable; repeat in fallback order for lium, shadeform, vast and runpod. Overrides --lium-key-env; never pass key values. Provider budgets are configured separately.",
    )


def _lease_vault(vault):
    from .vault import PayerKeyVault

    result = getattr(vault, "_provider_lease_vault", None)
    if result is None:
        result = PayerKeyVault(
            dir=vault.dir / "leases" if vault.dir else None,
            seal_key=vault.seal_key,
            ttl_seconds=vault.ttl_seconds,
            clock=vault.clock,
        )
        result.hydrate()
        vault._provider_lease_vault = result
    return result


def lease_credentials(vault, hotkey: str, credential: PayerCredential, *, scope: str = "") -> str:
    token = hashlib.sha256(
        f"{hotkey}:{scope}:{credential.provider}:{credential.api_key}".encode()
    ).hexdigest()
    value = json.dumps(
        {
            "hotkey": hotkey,
            "provider": credential.provider,
            "api_key": credential.api_key,
            "scope": scope,
        }
    )
    _lease_vault(vault).insert(token, value)
    return token


def lease_records(vault, hotkey: str | None = None):
    leases = _lease_vault(vault)
    leases.hydrate()
    for token in leases.hotkeys():
        raw = leases.get_shared(token)
        if not raw:
            continue
        row = json.loads(raw)
        if hotkey is None or row["hotkey"] == hotkey:
            yield token, row["hotkey"], PayerCredential(row["provider"], row["api_key"])


def pod_credential(vault, pod) -> PayerCredential | None:
    token = getattr(pod, "payer_credential_id", "")
    if token:
        leases = _lease_vault(vault)
        raw = leases.get_shared(token)
        if not raw:
            return None
        row = json.loads(raw)
        if row["hotkey"] != pod.payer_hotkey or row["provider"] != pod.provider:
            return None
        leases.refresh(token)
        return PayerCredential(row["provider"], row["api_key"])
    raw = vault.get_shared(pod.payer_hotkey) or ""
    return next((c for c in decode_credentials(raw) if c.provider == pod.provider), None)


def forget_lease(vault, token: str) -> None:
    if token:
        resolve_create(vault, token)
        _lease_vault(vault).remove(token)


def create_intents(vault, scope: str) -> list[dict]:
    """Non-secret create intents outlive credential TTLs and process restarts."""
    if vault.dir is None:
        rows = getattr(vault, "_create_intents", {}).values()
    else:
        directory = vault.dir / "create-intents"
        rows = [json.loads(p.read_text()) for p in directory.glob("*.json")]
    return [row for row in rows if row["scope"] == scope]


def begin_create(vault, token: str, *, scope: str, provider: str) -> None:
    row = {"token": token, "scope": scope, "provider": provider,
           "state": "create_unresolved"}
    if vault.dir is None:
        if not hasattr(vault, "_create_intents"):
            vault._create_intents = {}
        vault._create_intents[token] = row
    else:
        path = vault.dir / "create-intents" / f"{token}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path_lock(path):
            write_text_atomic(path, json.dumps(row))


def resolve_create(vault, token: str) -> None:
    if vault.dir is None:
        getattr(vault, "_create_intents", {}).pop(token, None)
    elif token:
        path = vault.dir / "create-intents" / f"{token}.json"
        with path_lock(path):
            path.unlink(missing_ok=True)


def credential_scopes(vault, hotkey, credential):
    for token, owner, entry in lease_records(vault, hotkey):
        if owner == hotkey and entry == credential:
            row = json.loads(_lease_vault(vault).get_shared(token))
            yield token, str(row.get("scope") or "")
