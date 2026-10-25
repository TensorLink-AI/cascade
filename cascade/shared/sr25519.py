"""sr25519 signature verification against an ss58 hotkey (bittensor 11.x).

Every signed artifact (manifest, receipt, bench report, promotion record,
expert manifest, intake request) verifies through :func:`verify_hotkey`, so
the SDK import lives in exactly one place.

Two outcomes are kept strictly apart (DEC-CA-0051):

* **signature invalid** — wrong key, tampered body, malformed hex, bad
  address: :func:`verify_hotkey` returns ``False``.
* **cannot verify** — the SDK is missing, too old, or broken:
  :class:`SignatureVerifierUnavailable` is RAISED. A verifier that silently
  answered ``False`` here would reject every valid signature (10.5 → 11 moved
  ``bittensor.Keypair`` to ``bittensor.sp_core.Keypair``; a caught
  ImportError once turned that into a silent reject-everything).

Signatures produced by 10.5 (``bittensor_wallet``) verify here byte-for-byte,
and 11.x signatures verify under 10.5 — both are schnorrkel sr25519 with the
"substrate" signing context.
"""

from __future__ import annotations

from typing import Any


class SignatureVerifierUnavailable(RuntimeError):
    """The sr25519 verifier cannot run (bittensor SDK missing or broken) —
    NOT a verdict on any signature."""


def keypair_class() -> Any:
    """``bittensor.sp_core.Keypair``, or :class:`SignatureVerifierUnavailable`."""
    try:
        from bittensor.sp_core import Keypair  # type: ignore
    except Exception as e:  # noqa: BLE001 — ImportError, a broken native ext, …
        raise SignatureVerifierUnavailable(
            "bittensor>=11 (bittensor.sp_core.Keypair) is required to verify sr25519 "
            f"signatures; install the [chain] extra ({type(e).__name__}: {e})"
        ) from e
    return Keypair


def verify_hotkey(ss58_address: str, message: bytes, signature_hex: str) -> bool:
    """True iff ``signature_hex`` is ``ss58_address``'s sr25519 signature over
    ``message``. False on any malformed input or mismatch; raises
    :class:`SignatureVerifierUnavailable` when verification cannot run."""
    keypair = keypair_class()
    try:
        msg = message.encode("utf-8") if isinstance(message, str) else bytes(message)
        sig = bytes.fromhex(signature_hex)
        return bool(keypair(ss58_address=ss58_address).verify(msg, sig))
    except Exception:  # noqa: BLE001 — any malformed sig/address ⇒ untrusted
        return False
