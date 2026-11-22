"""The signed lane D report, schema ``lane-d-report/1`` (LANE_D_PLAN §1.5,
§14.3; W7a deepens the schema).

The report is a plain JSON dict. Its signature is the main validator's sr25519
signature over the canonical body (every field except ``signature``,
``sort_keys``, compact separators, no NaN), the MarketCast report pattern
(``cascade/shared/marketcast_report.py``) verified through
``cascade.shared.sr25519``. ``rows_sha256`` is the sha256 of the canonical
serialisation of ``rows`` (A9): anyone holding the rows can prove they are
the ones the verdict used.

Light imports only: this module sits on the validator package's import path.
"""

from __future__ import annotations

import copy
import hashlib
import json

REPORT_SCHEMA = "lane-d-report/1"


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def rows_sha256(rows: list) -> str:
    """sha256 hex of the canonical serialisation of ``rows`` (in order)."""
    return hashlib.sha256(canonical_json(list(rows))).hexdigest()


def canonical_body(report: dict) -> bytes:
    """The signed bytes: the report without its ``signature``."""
    body = {k: v for k, v in report.items() if k != "signature"}
    return canonical_json(body)


def sign_report(report: dict, signer) -> dict:
    """A copy of ``report`` signed by ``signer`` (a keypair, or a wallet whose
    ``hotkey`` is one). ``signer`` is stamped as the report's ``signer``."""
    key = getattr(signer, "hotkey", signer)
    out = copy.deepcopy(report)
    out.pop("signature", None)
    out["signer"] = str(key.ss58_address)
    sig = key.sign(canonical_body(out))
    out["signature"] = sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    return out


def verify_report(report: dict, signer_ss58: str) -> bool:
    """True iff ``report`` carries ``signer_ss58``'s valid signature over its
    canonical body. Raises ``SignatureVerifierUnavailable`` when the SDK
    cannot verify (never a silent False)."""
    from ..shared.sr25519 import verify_hotkey

    sig = report.get("signature") if isinstance(report, dict) else None
    if not sig or not signer_ss58 or report.get("signer") != signer_ss58:
        return False
    try:
        body = canonical_body(report)
    except (TypeError, ValueError):
        return False
    return verify_hotkey(signer_ss58, body, str(sig))


__all__ = ["REPORT_SCHEMA", "canonical_body", "canonical_json", "rows_sha256", "sign_report",
           "verify_report"]
