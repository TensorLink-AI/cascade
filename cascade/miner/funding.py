"""Shared signing and pre-send checks for miner-funded lane requests.

Lane modules retain their target validation, endpoints and result envelopes.
This module owns the common key/transport/consent/wallet order and wire signing.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .output import USAGE


@dataclass(frozen=True)
class FundingRequest:
    base: str
    hotkey: str
    sign_fn: Callable = field(repr=False)
    api_key: str = field(repr=False)
    label: str
    choice: tuple[str, bool, float | None]


def prepare_funding_request(args, out, *, funding: bool, confirmation: str,
                            key_hint: str = "key as an argument") -> FundingRequest | int:
    """Validate before touching a wallet or sending a request; return a CLI exit on refusal."""
    from .cli import (
        _cli_label,
        _intake_transport_ok,
        cli_gpu_choice,
        load_signing_wallet,
        needs_confirmation,
    )

    label = _cli_label(args) if funding else ""
    if label is None:
        return USAGE
    choice = cli_gpu_choice(args) if funding else ("", False, None)
    if choice is None:
        return USAGE
    api_key = ""
    if funding:
        from ..funding.providers import credentials_from_env

        try:
            api_key = credentials_from_env(args)
        except ValueError as exc:
            return out.fail("bad_provider_credentials", str(exc), USAGE)
        if not api_key:
            return out.fail("missing_api_key", f"no Lium API key in ${args.lium_key_env} — "
                            f"export it first (never pass the {key_hint})", USAGE)
    base = args.intake_url.rstrip("/")
    if not _intake_transport_ok(base):
        return out.fail("insecure_intake", "refusing to send your provider keys over plain http "
                        "to a non-local intake; use https", USAGE)
    if funding:
        refused = needs_confirmation(args, out, confirmation)
        if refused is not None:
            return refused
    wallet = load_signing_wallet(args, out)
    if wallet is None:
        return USAGE
    hotkey, sign_fn = wallet
    return FundingRequest(base, hotkey, sign_fn, api_key, label, choice)


def build_intake_headers(action, hotkey, ref, api_key, sign_fn, *, now=None,
                         label="", key_digest=None, gpu="", allow_fallback=False,
                         max_price=None, fund_action="fund", signed_ref=None,
                         target_headers=None):
    """Keep each lane's canonical identity while sharing v2/v3 header construction.

    Withdrawals send neither keys nor GPU choices. A direct credential deposit
    may supply its provider-specific digest without sending a Lium key header.
    Cosmetic labels are intentionally outside the signed identity.
    """
    from ..funding.intake import LABEL_HEADER, canonical_fund_message
    from .cli import gpu_headers

    ts = str(int((now or time.time)()))
    sends_key = action == fund_action and bool(api_key)
    if key_digest is None:
        key_digest = hashlib.sha256(api_key.encode()).hexdigest() if sends_key else "-"
    extra, segment = gpu_headers(gpu if action == fund_action else "", allow_fallback, max_price)
    headers = {
        "X-Miner-Hotkey": hotkey,
        "X-Commit-Ref": ref,
        **(target_headers or {}),
        "X-Timestamp": ts,
        "X-Signature": sign_fn(canonical_fund_message(
            action, hotkey, ref if signed_ref is None else signed_ref, ts, key_digest, segment)).hex(),
        **extra,
    }
    if sends_key:
        from ..funding.providers import credential_headers

        headers.update(credential_headers(api_key))
    if label:
        headers[LABEL_HEADER] = label
    return headers
