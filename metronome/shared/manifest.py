"""Training manifest — the trainer→validator hand-off.

The trainer is the one component that touches GPUs: it draws each generator's
corpus, trains a fresh base model under the fixed contract, and pushes the
resulting checkpoint to HuggingFace. Validators never train; they read this
manifest to learn *which* trained checkpoint corresponds to *which* miner's
generator at *which* revision, then pull and evaluate.

A manifest is a JSON document published to an owner-controlled HF dataset repo
(``[manifest] hf_dataset_repo``). Each :class:`TrainedEntry` is a receipt:
generator pointer in, trained-model pointer out, plus the digests that make the
run auditable — a second honest trainer (or a suspicious validator) can re-draw
the corpus from the pinned generator + seed and re-train to confirm the digests
match.

Trust model (v1): validators trust manifests signed by ``[manifest]
trainer_hotkey`` only. Signature verification is a TODO boundary
(:func:`verify_signature`); see OPEN_QUESTIONS.md #1 for the decentralisation
path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

# The trainer's output pointer — distinct ``trained`` tag so it can never be
# confused with a miner's ``gen`` submission.
TRAINED_RE = re.compile(
    r"^metro-v1:trained:hf:(?P<repo>[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*)"
    r"@(?P<sha>[A-Fa-f0-9]{40})$"
)

MANIFEST_VERSION = 1
VALID_ROLES = ("king", "challenger")


def parse_trained_pointer(payload: str) -> tuple[str, str] | None:
    """Return ``(repo, revision)`` for a trained-model pointer, else None."""
    m = TRAINED_RE.match(payload.strip())
    if not m:
        return None
    return m.group("repo"), m.group("sha").lower()


def format_trained_pointer(repo: str, revision: str) -> str:
    """Build a trained-model pointer; raises if it would not round-trip."""
    payload = f"metro-v1:trained:hf:{repo}@{revision.lower()}"
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
    """One miner's training receipt for a round."""

    miner_hotkey: str
    miner_uid: int
    role: str                 # "king" | "challenger"
    gen_repo: str
    gen_revision: str
    trained_pointer: str      # metro-v1:trained:hf:<repo>@<sha>
    corpus_digest: str
    train_block: int

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
            gen_repo=str(e["gen_repo"]),
            gen_revision=str(e["gen_revision"]),
            trained_pointer=str(e["trained_pointer"]),
            corpus_digest=str(e["corpus_digest"]),
            train_block=int(e["train_block"]),
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


def verify_signature(manifest: TrainingManifest, trainer_hotkey: str) -> bool:
    """Verify the manifest was signed by ``trainer_hotkey``.

    TODO: wire to ``bittensor`` keypair verification over
    :meth:`TrainingManifest.canonical_body`. Until signing is implemented this
    returns True when a signature is present and the trust model is "operator
    controls the HF dataset repo write token" — see OPEN_QUESTIONS.md #1.
    """
    # Placeholder: presence check only. Real ss58 signature verification is the
    # decentralisation milestone.
    return manifest.signature is not None
