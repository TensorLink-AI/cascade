"""Domain-separated sr25519 authentication using the pinned chain SDK."""

from ..shared.direct_moe import canonical
from ..shared.sr25519 import verify_hotkey


def message(scope, signer, body):
    return canonical(
        {"protocol": "cascade-direct-signed/1", "scope": scope, "signer": signer, "body": body}
    ).encode()


def sign(body, keypair, *, scope):
    signer = keypair.ss58_address
    signature = keypair.sign(message(scope, signer, body))
    return {
        "protocol": "cascade-direct-signed/1",
        "scope": scope,
        "signer": signer,
        "body": body,
        "signature": bytes(signature).hex(),
    }


def verify(envelope, *, expected_signer, scope):
    if (
        envelope.get("protocol") != "cascade-direct-signed/1"
        or envelope.get("scope") != scope
        or envelope.get("signer") != expected_signer
    ):
        raise ValueError("signed direct document identity/scope mismatch")
    if not verify_hotkey(
        expected_signer, message(scope, expected_signer, envelope["body"]), envelope["signature"]
    ):
        raise ValueError("invalid direct document signature")
    return envelope["body"]
