"""Training manifest — the trainer→validator hand-off.

The trainer is the one component that touches GPUs: it draws each generator's
corpus, trains a fresh base model under the fixed contract, and pushes the
resulting checkpoint to the Hippius registry (IPFS). Validators never train; they
read this manifest to learn *which* trained checkpoint (CID) corresponds to
*which* miner's generator (CID), then pull and evaluate.

A manifest is a JSON document published to the owner-controlled Hippius S3
manifest bucket (``[storage] manifest_bucket``). Each :class:`TrainedEntry` is a
receipt: generator CID in, trained-model CID out, plus the digests that make the
run auditable — a second honest trainer (or a suspicious validator) can re-draw
the corpus from the pinned generator + seed and re-train to confirm the digests
match.

Trust model (v1): validators trust manifests signed by ``[manifest]
trainer_hotkey`` only. :func:`sign_manifest` signs the canonical body with the
trainer's bittensor hotkey and :func:`verify_signature` checks it against the
configured ss58 address; see OPEN_QUESTIONS.md #1 for the decentralisation path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

# The trainer's output pointer — distinct ``trained`` tag so it can never be
# confused with a miner's ``gen`` submission. Trained checkpoints live on the
# Hippius registry (IPFS), content-addressed by CID.
TRAINED_RE = re.compile(r"^metro-v1:trained:hippius:(?P<cid>[A-Za-z0-9]+)$")

MANIFEST_VERSION = 1
VALID_ROLES = ("king", "challenger")


def parse_trained_pointer(payload: str) -> str | None:
    """Return the registry ``cid`` for a trained-model pointer, else None."""
    from .hippius import is_cid

    m = TRAINED_RE.match(payload.strip())
    if not m:
        return None
    cid = m.group("cid")
    return cid if is_cid(cid) else None


def format_trained_pointer(cid: str) -> str:
    """Build a trained-model pointer from a registry CID; raises if malformed."""
    payload = f"metro-v1:trained:hippius:{cid.strip()}"
    if parse_trained_pointer(payload) is None:
        raise ValueError(f"refusing to emit malformed trained pointer: {payload!r}")
    return payload


def corpus_digest(series: Sequence[np.ndarray]) -> str:
    """Stable sha256 over a generated corpus.

    Each series is canonicalised to ``(C, L)`` (a 1-D ``(L,)`` array is promoted
    to ``(1, L)``), and the hash covers the count, every series' full ``(C, L)``
    shape, and its raw float64 bytes in yield order. Carrying the channel count
    in the digest keeps it stable as the corpus moves from univariate ``(1, L)``
    to multivariate ``(C, L)`` — a univariate and a single-channel-of-multivariate
    corpus never collide. Two trainers that draw the same corpus from the same
    pinned generator + seed get the same digest, which is what makes a training
    run auditable.
    """
    h = hashlib.sha256()
    h.update(len(series).to_bytes(8, "big"))
    for arr in series:
        a = np.ascontiguousarray(np.atleast_2d(np.asarray(arr, dtype=np.float64)))
        h.update(a.shape[0].to_bytes(8, "big"))   # channels
        h.update(a.shape[1].to_bytes(8, "big"))   # length
        h.update(a.tobytes())
    return h.hexdigest()


def contract_digest(contract: object) -> str:
    """Stable sha256 over the fields of a training contract dataclass.

    Used to assert king and challenger were trained under byte-identical terms.
    Accepts any dataclass (typically ``TrainingContractConfig``).
    """
    if hasattr(contract, "__dataclass_fields__"):
        payload = asdict(contract)  # type: ignore[arg-type]
    elif isinstance(contract, dict):
        payload = contract
    else:
        raise TypeError(f"contract_digest expects a dataclass or dict; got {type(contract)}")
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class TrainedEntry:
    """One miner's training receipt for a round.

    ``gen_cid`` is the miner's generator pointer on the Hippius registry;
    ``trained_pointer`` is the trained checkpoint's registry pointer. ``tar_digest``
    is the sha256 of the uploaded checkpoint tar — a CID-independent integrity
    check the validator re-verifies after fetch.
    """

    miner_hotkey: str
    miner_uid: int
    role: str                 # "king" | "challenger"
    gen_cid: str              # miner's generator CID on the registry
    trained_pointer: str      # metro-v1:trained:hippius:<cid>
    corpus_digest: str
    train_block: int
    tar_digest: str = ""      # sha256 of the checkpoint tar (integrity)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}; got {self.role!r}")
        if parse_trained_pointer(self.trained_pointer) is None:
            raise ValueError(f"malformed trained_pointer: {self.trained_pointer!r}")


@dataclass(frozen=True)
class TrainingManifest:
    """A round's worth of training receipts plus the shared contract context.

    ``contract_digest`` and ``base_arch_digest`` are recorded once and asserted
    equal for every entry's training run — the controlled-experiment guarantee.
    """

    round_id: str
    created_block: int
    contract_digest: str
    base_arch_digest: str
    eval_dataset: str
    entries: list[TrainedEntry] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION
    signature: str | None = None  # trainer_hotkey signature over the canonical body; TODO

    def entry_for_role(self, role: str) -> TrainedEntry | None:
        for e in self.entries:
            if e.role == role:
                return e
        return None

    def canonical_body(self) -> bytes:
        """Deterministic byte serialisation of everything except the signature.

        The signed payload. Stable key ordering so the trainer and every
        validator hash the identical bytes.
        """
        body = {
            "manifest_version": self.manifest_version,
            "round_id": self.round_id,
            "created_block": self.created_block,
            "contract_digest": self.contract_digest,
            "base_arch_digest": self.base_arch_digest,
            "eval_dataset": self.eval_dataset,
            "entries": [asdict(e) for e in self.entries],
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump_manifest(manifest: TrainingManifest) -> str:
    """Serialise a manifest (including signature) to a JSON string."""
    body = json.loads(manifest.canonical_body().decode("utf-8"))
    body["signature"] = manifest.signature
    return json.dumps(body, indent=2, sort_keys=True)


def load_manifest(text: str) -> TrainingManifest:
    """Parse a manifest JSON string. Raises ``ValueError`` on schema problems."""
    obj = json.loads(text)
    version = int(obj.get("manifest_version", 0))
    if version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest_version {version}; need {MANIFEST_VERSION}")
    entries = [
        TrainedEntry(
            miner_hotkey=str(e["miner_hotkey"]),
            miner_uid=int(e["miner_uid"]),
            role=str(e["role"]),
            gen_cid=str(e["gen_cid"]),
            trained_pointer=str(e["trained_pointer"]),
            corpus_digest=str(e["corpus_digest"]),
            train_block=int(e["train_block"]),
            tar_digest=str(e.get("tar_digest", "")),
        )
        for e in obj["entries"]
    ]
    return TrainingManifest(
        round_id=str(obj["round_id"]),
        created_block=int(obj["created_block"]),
        contract_digest=str(obj["contract_digest"]),
        base_arch_digest=str(obj["base_arch_digest"]),
        eval_dataset=str(obj["eval_dataset"]),
        entries=entries,
        manifest_version=version,
        signature=obj.get("signature"),
    )


def sign_manifest(manifest: TrainingManifest, wallet: object) -> TrainingManifest:
    """Sign ``canonical_body()`` with the trainer's bittensor hotkey.

    ``wallet`` is a ``bittensor.wallet`` (or anything exposing ``.hotkey`` with a
    ``.sign(bytes) -> bytes``). The hex signature is stored on a copy of the
    manifest. Validators verify it with :func:`verify_signature` against the
    configured ``[manifest] trainer_hotkey`` ss58 address.
    """
    from dataclasses import replace

    hotkey = getattr(wallet, "hotkey", wallet)
    try:
        sig = hotkey.sign(manifest.canonical_body())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"manifest_signing_failed: {type(e).__name__}: {e}") from e
    return replace(manifest, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_signature(manifest: TrainingManifest, trainer_hotkey: str) -> bool:
    """Verify the manifest was signed by ``trainer_hotkey`` (an ss58 address).

    Recreates the signer's public key from the ss58 address and checks the hex
    signature over :meth:`TrainingManifest.canonical_body`. Returns False on a
    missing signature, an address/signature mismatch, or any verification error.
    Requires ``bittensor`` (the trust check only runs in the validator, which
    already depends on it); if it is unavailable this raises so the caller does
    not silently accept an unverified manifest.
    """
    if not manifest.signature or not trainer_hotkey:
        return False
    try:
        from bittensor import Keypair  # type: ignore
    except ImportError as e:  # pragma: no cover - validator has bittensor
        raise RuntimeError(
            "bittensor required to verify manifest signatures; install the [chain] extra"
        ) from e
    try:
        kp = Keypair(ss58_address=trainer_hotkey)
        return bool(kp.verify(manifest.canonical_body(), bytes.fromhex(manifest.signature)))
    except Exception:  # noqa: BLE001 — any malformed sig/address ⇒ untrusted
        return False
