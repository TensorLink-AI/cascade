"""Training manifest — the trainer→validator hand-off.

The trainer is the one component that touches GPUs: it draws each generator's
corpus, trains a fresh base model under the fixed contract, and pushes the
resulting checkpoint to the Hippius Hub registry. Validators never train; they
read this manifest to learn *which* trained checkpoint (``repo@digest``)
corresponds to *which* miner's generator (``repo@digest``), then pull and
evaluate.

A manifest is a JSON document published to the owner-controlled Hippius S3
manifest bucket (``[storage] manifest_bucket``). Each :class:`TrainedEntry` is a
receipt: generator ref in, trained-model ref out, plus the digests that make the
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
# Hippius Hub registry, pinned by ``repo@digest``.
TRAINED_RE = re.compile(r"^metro-v1:trained:hippius:(?P<ref>.+)$")

MANIFEST_VERSION = 2
VALID_ROLES = ("king", "challenger")


def parse_trained_pointer(payload: str) -> str | None:
    """Return the registry ``repo@digest`` for a trained-model pointer, else None."""
    from .hippius import is_hub_ref

    m = TRAINED_RE.match(payload.strip())
    if not m:
        return None
    ref = m.group("ref").strip()
    return ref if is_hub_ref(ref) else None


def format_trained_pointer(ref: str) -> str:
    """Build a trained-model pointer from a Hub ``repo@digest``; raises if malformed."""
    payload = f"metro-v1:trained:hippius:{ref.strip()}"
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
class BenchScores:
    """Public-benchmark scores for one trained checkpoint, stamped by the trainer.

    The CRPS and MASE the checkpoint scored on each of the three public suites —
    GIFT-Eval, BOOM, and TIME — run once by the (trusted, owner-operated) trainer
    via the benchmark sidecar and published on the king's :class:`TrainedEntry`.
    Cascade (:mod:`cascade.validator.cascade`) consumes these: every validator
    reads the *same signed numbers* instead of re-running a non-bit-reproducible
    GPU sweep, so the reign checkpoint log — and the promotion it drives — is
    deterministic across validators. These feed ONLY Cascade's warm-start
    promotion; the round's dethrone verdict stays on the private eval pool.
    """

    gifteval_crps: float
    gifteval_mase: float
    boom_crps: float
    boom_mase: float
    time_crps: float
    time_mase: float


@dataclass(frozen=True)
class TrainedEntry:
    """One miner's training receipt for a round.

    ``gen_ref`` is the miner's generator pointer on the Hippius Hub
    (``repo@digest``); ``trained_pointer`` is the trained checkpoint's registry
    pointer. The OCI digest inside each ``repo@digest`` is itself the integrity
    hash — the fetch verifies the layer blobs against it — so no separate tar
    digest is carried.

    ``bench_scores`` is the trainer's signed public-benchmark scoring of this
    checkpoint (GIFT-Eval / BOOM / TIME), carried on the king entry when Cascade
    is enabled and ``None`` otherwise. It is folded into ``canonical_body`` only
    when present, so a manifest without it serialises byte-for-byte as before (no
    wire-format break, no version bump) and old signatures stay valid.
    """

    miner_hotkey: str
    miner_uid: int
    role: str                 # "king" | "challenger"
    gen_ref: str              # miner's generator repo@digest on the registry
    trained_pointer: str      # metro-v1:trained:hippius:<repo>@<digest>
    corpus_digest: str
    train_block: int
    gpu_name: str = ""        # GPU model the run used; gated for matched-hardware audit
    size: str = ""            # arch_preset this entry was trained at ("" = primary/legacy).
                              # A round carries one (king, challenger) pair PER size.
    bench_scores: BenchScores | None = None  # trainer-signed public-benchmark scores (king only)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}; got {self.role!r}")
        if parse_trained_pointer(self.trained_pointer) is None:
            raise ValueError(f"malformed trained_pointer: {self.trained_pointer!r}")


HEAT_STATUSES = ("advanced", "screened", "failed_train", "failed_screen", "duplicate")


@dataclass(frozen=True)
class HeatEntrant:
    """One challenger's standing in the heat screen — an *informational* record.

    The heat trains every eligible challenger cheaply and ranks them; only the
    top ``finalists`` advance. Those scores are otherwise thrown away (the heat
    checkpoints are discarded), so this is the miner's only window into how a
    non-finalist submission fared. It is deliberately coarse: a ``rank`` and a
    ``rel_score`` *relative to the best entrant* (``heat_score / best``, ≥ 1.0,
    where 1.0 is the best), never the raw per-window numbers — the eval pool
    rotates privately and exposing absolute scores would hand a miner a gradient
    to distribution-match it. ``rank``/``rel_score`` are None for an entrant that
    never produced a score (``failed_train`` / ``failed_screen``).
    """

    uid: int
    hotkey: str
    gen_ref: str
    status: str                    # one of HEAT_STATUSES
    rank: int | None = None        # 1-based placement among scored entrants
    rel_score: float | None = None  # heat_score / best_heat_score (≥ 1.0; 1.0 = best)

    def __post_init__(self) -> None:
        if self.status not in HEAT_STATUSES:
            raise ValueError(f"status must be one of {HEAT_STATUSES}; got {self.status!r}")


@dataclass(frozen=True)
class HeatResult:
    """The round's heat screen, as a presentational (unsigned) block.

    Rides in the manifest but is excluded from :meth:`TrainingManifest.canonical_body`
    — it is a *view* for the dashboard, not part of the signed/audited claim (an
    auditor cannot cheaply reproduce a discarded heat checkpoint). ``None`` on a
    manifest means no screen ran: the field fit within ``finalists``, or the
    round had a single eligible challenger.
    """

    screen_size: str               # arch_preset the heat screened at
    finalists: int                 # how many advanced to the final
    entrants: tuple[HeatEntrant, ...] = ()


def _entry_body(e: TrainedEntry) -> dict:
    """One entry's canonical dict. ``bench_scores`` is omitted when ``None`` so a
    manifest without it is byte-identical to a pre-``bench_scores`` manifest — the
    signed payload only grows for entries that actually carry the scores."""
    d = asdict(e)
    if d.get("bench_scores") is None:
        d.pop("bench_scores", None)
    return d


def _bench_from_json(obj: object) -> BenchScores | None:
    if not isinstance(obj, dict):
        return None
    return BenchScores(
        gifteval_crps=float(obj["gifteval_crps"]),
        gifteval_mase=float(obj["gifteval_mase"]),
        boom_crps=float(obj["boom_crps"]),
        boom_mase=float(obj["boom_mase"]),
        time_crps=float(obj["time_crps"]),
        time_mase=float(obj["time_mase"]),
    )


def _heat_to_json(heat: HeatResult | None) -> dict | None:
    if heat is None:
        return None
    return {
        "screen_size": heat.screen_size,
        "finalists": heat.finalists,
        "entrants": [asdict(e) for e in heat.entrants],
    }


def _heat_from_json(obj: object) -> HeatResult | None:
    if not isinstance(obj, dict):
        return None
    return HeatResult(
        screen_size=str(obj.get("screen_size", "")),
        finalists=int(obj.get("finalists", 0)),
        entrants=tuple(
            HeatEntrant(
                uid=int(e["uid"]),
                hotkey=str(e["hotkey"]),
                gen_ref=str(e["gen_ref"]),
                status=str(e["status"]),
                rank=(None if e.get("rank") is None else int(e["rank"])),
                rel_score=(None if e.get("rel_score") is None else float(e["rel_score"])),
            )
            for e in obj.get("entrants", ())
        ),
    )


@dataclass(frozen=True)
class TrainingManifest:
    """A round's worth of training receipts plus the shared contract context.

    ``contract_digest`` and ``base_arch_digest`` are recorded once and asserted
    equal for every entry's training run — the controlled-experiment guarantee.

    ``heat`` is an *informational* screening summary (unsigned; see
    :class:`HeatResult`): it is serialised alongside the manifest but never enters
    :meth:`canonical_body`, so adding it leaves every existing signature valid.
    """

    round_id: str
    created_block: int
    contract_digest: str
    base_arch_digest: str
    eval_dataset: str
    entries: list[TrainedEntry] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION
    heat: HeatResult | None = None
    # Eval-pool pin: the exact snapshot (bucket key or pool ref + sha256) the
    # trainer screened this round on. Signed (see canonical_body), so validators
    # can verify their own deterministic snapshot selection against a digest that
    # descends from the trainer signature they already trust — pool integrity
    # then no longer rests on the unsigned pool/index.json (see docs/EVAL_POOL.md).
    # Empty ⇒ unpinned (trainer predates the field, or no pool provenance).
    eval_pool_key: str = ""
    eval_pool_sha256: str = ""
    # Warm-start pin (Cascade, DEC-CA-0003/0004): the content-addressed
    # checkpoint pointer this round's runs at ``warm_start_size`` initialised
    # from (same ``trained_pointer`` format — the OCI digest pins the bytes),
    # instead of random init. Signed (see canonical_body), so validators verify
    # the trainer trained from the init THEIR deterministic promotion selected,
    # and cascade-audit re-derives from the pinned checkpoint. Empty ⇒ random
    # init (no promotion yet, or a pre-warm-start trainer).
    warm_start_ckpt: str = ""
    warm_start_size: str = ""
    signature: str | None = None  # trainer_hotkey signature over canonical_body()

    def entry_for_role(self, role: str) -> TrainedEntry | None:
        for e in self.entries:
            if e.role == role:
                return e
        return None

    def entries_for_role(self, role: str) -> list[TrainedEntry]:
        """All entries for ``role`` — one per trained size (the primary plus any
        ``[[training.sizes]]``). Order follows the manifest's entry order, which
        the trainer emits size-by-size."""
        return [e for e in self.entries if e.role == role]

    def sizes(self) -> list[str]:
        """Distinct size tags present, in first-seen order (e.g. the king's
        sizes). ``[""]`` for a legacy single-size manifest."""
        seen: list[str] = []
        for e in self.entries:
            if e.size not in seen:
                seen.append(e.size)
        return seen

    def canonical_body(self) -> bytes:
        """Deterministic byte serialisation of everything except the signature.

        The signed payload. Stable key ordering so the trainer and every
        validator hash the identical bytes. ``bench_scores`` is dropped from an
        entry when ``None`` (see :func:`_entry_body`), and the eval-pool pin is
        dropped when unset, so a manifest without them hashes exactly as it did
        before the fields existed — old signatures stay valid without a version
        bump. (Rollout note: once the trainer emits a pin, validators must run
        code that knows the field, or their recomputed body won't verify.)
        """
        body = {
            "manifest_version": self.manifest_version,
            "round_id": self.round_id,
            "created_block": self.created_block,
            "contract_digest": self.contract_digest,
            "base_arch_digest": self.base_arch_digest,
            "eval_dataset": self.eval_dataset,
            "entries": [_entry_body(e) for e in self.entries],
        }
        if self.eval_pool_key and self.eval_pool_sha256:
            body["eval_pool_key"] = self.eval_pool_key
            body["eval_pool_sha256"] = self.eval_pool_sha256
        # Same drop-when-unset convention: a random-init round hashes exactly as
        # it did before the warm-start fields existed.
        if self.warm_start_ckpt:
            body["warm_start_ckpt"] = self.warm_start_ckpt
            body["warm_start_size"] = self.warm_start_size
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump_manifest(manifest: TrainingManifest) -> str:
    """Serialise a manifest (including signature) to a JSON string.

    ``heat`` is attached outside the signed :meth:`~TrainingManifest.canonical_body`
    — it travels with the manifest for the dashboard but is not part of what the
    trainer signs.
    """
    body = json.loads(manifest.canonical_body().decode("utf-8"))
    body["signature"] = manifest.signature
    # Only present when a screen actually ran, so a heat-less manifest (the common
    # single-finalist round, and every manifest predating this field) serialises
    # byte-for-byte as before — no wire-format break, no version bump.
    if manifest.heat is not None:
        body["heat"] = _heat_to_json(manifest.heat)
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
            gen_ref=str(e["gen_ref"]),
            trained_pointer=str(e["trained_pointer"]),
            corpus_digest=str(e["corpus_digest"]),
            train_block=int(e["train_block"]),
            gpu_name=str(e.get("gpu_name", "")),
            size=str(e.get("size", "")),
            bench_scores=_bench_from_json(e.get("bench_scores")),
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
        heat=_heat_from_json(obj.get("heat")),
        eval_pool_key=str(obj.get("eval_pool_key", "") or ""),
        eval_pool_sha256=str(obj.get("eval_pool_sha256", "") or ""),
        warm_start_ckpt=str(obj.get("warm_start_ckpt", "") or ""),
        warm_start_size=str(obj.get("warm_start_size", "") or ""),
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
