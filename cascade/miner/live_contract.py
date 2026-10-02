"""Score under the contract the live trainer actually runs.

The operator's deployed config can differ from this checkout's chain.toml
(observed 2026-10-02: every round since block ~9.17M billed ``points+mv20`` on
worker image v0.13.0 while the repo still read ``series_points`` / v0.9.0). A
local score under the wrong billing gives a multichannel generator the wrong
token budget, so ``cascade score`` / ``mine`` / ``ralph`` default to the live
contract: the latest published manifest's signed ``contract_body``, accepted
only when it hashes to the manifest's ``contract_digest``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

log = logging.getLogger("cascade.miner.live_contract")


def with_live_contract(cfg, *, fetch=None):
    """``(cfg', info)``: ``cfg`` with its training contract replaced by the live
    one. Falls back to ``cfg`` unchanged (``info["source"] = "local"``) when the
    manifest is unreachable, carries no body, or fails its digest check."""
    from ..shared.manifest import load_manifest
    from .replay import round_contract

    try:
        doc = fetch() if fetch is not None else _latest_receipt_manifest(cfg)
        if not doc:
            raise ValueError("no published receipt manifest")
        manifest = load_manifest(json.dumps(doc))
        if not isinstance(manifest.contract_body, dict):
            raise ValueError("latest manifest carries no contract_body")
        contract, match, over = round_contract(cfg, manifest, None)
    except Exception as e:  # noqa: BLE001 — offline / old manifest: score locally, say so
        log.warning("live contract unavailable (%s); scoring under the local chain.toml", e)
        return cfg, {"source": "local", "reason": str(e)}
    if not match:
        log.warning("latest manifest's contract_body does not match its digest; "
                    "scoring under the local chain.toml")
        return cfg, {"source": "local", "reason": "contract digest mismatch"}
    return (replace(cfg, training=contract),
            {"source": "live", "round_id": manifest.round_id,
             "overrides": {k: str(v) for k, v in over.items()}})


def _latest_receipt_manifest(cfg) -> dict | None:
    """The manifest embedded in the anchor validator's latest public receipt
    (manifests themselves are not public-read; receipts are)."""
    from ..audit.main import _fetch_text
    from ..shared.hippius import receipt_latest_key

    anchor = str(getattr(cfg.manifest, "validator_hotkey", "") or "")
    doc = json.loads(_fetch_text(cfg, receipt_latest_key(anchor)))
    return doc.get("manifest") if isinstance(doc, dict) else None


def describe(info: dict) -> str:
    if info.get("source") != "live":
        return f"contract: LOCAL chain.toml ({info.get('reason', 'requested')})"
    over = info.get("overrides") or {}
    diff = ", ".join(f"{k}={v}" for k, v in over.items()) or "same as chain.toml"
    return f"contract: live (round {info.get('round_id')}): {diff}"
