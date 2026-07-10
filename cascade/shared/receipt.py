"""Round receipt — the validator's signed public record of one round.

A :class:`TrainingManifest` is the *trainer's* claim ("these checkpoints came
from these generators under this contract"). A :class:`RoundReceipt` is the
*validator's* countersigned record of what it then did with that claim: the
chain context it derived the round from, the manifest it gated (embedded
verbatim, trainer signature and all), the eligible participant set, the eval
slice it scored, every per-window score that fed the KOTH bootstrap, the
decision parameters, the verdict, and the weight vector it set. Everything a
third party needs to re-derive the round without trusting the owner — see
``docs/AUDIT.md`` and the ``cascade-audit`` CLI.

Receipts are published to the owner-controlled Hippius S3 manifest bucket
(``receipts/round-<id>.json`` + ``receipts/latest.json``; see
:mod:`cascade.shared.hippius`). A round the validator *rejected* still gets a
receipt (``status = "rejected"``) carrying the gate's reason, so a censored or
malformed round is publicly visible rather than silently absent.

Conventions follow :mod:`cascade.shared.manifest` exactly: frozen dataclasses,
an explicit ``receipt_version``, a canonical sorted-key JSON body, and
sign/verify over :meth:`RoundReceipt.canonical_body` with a bittensor hotkey
(here the validator's — ``validator_hotkey`` is carried inside the signed body).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from .manifest import TrainingManifest, dump_manifest, load_manifest

RECEIPT_VERSION = 2
RECEIPT_STATUSES = ("scored", "rejected")


def _none_for_nan(x: float | None) -> float | None:
    """Canonical JSON is strict (``allow_nan=False``): NaN/±inf become None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class Participant:
    """One eligible entrant: a pre-cutoff on-chain commitment, resolved.

    ``commit_block`` is the block the pointer was revealed at; the audit checks
    every participant committed strictly before the epoch boundary (the
    submission deadline).
    """

    hotkey: str
    uid: int
    gen_ref: str          # generator repo@digest from the commitment payload
    commit_block: int


@dataclass(frozen=True)
class WindowScoreRecord:
    """A JSON-safe :class:`cascade.eval.scoring.WindowScore`.

    Carries the exact bootstrap inputs — per-quantile pinball sums, the
    ``abs_target`` denominator companion, MASE, and the window's cluster key
    (``source``, the upstream feed id) — so an auditor can feed the recorded
    scores back into ``evaluate_round`` and reproduce the verdict bit-for-bit
    without a GPU. ``source`` drives the cluster bootstrap; without it the
    recomputed LCB would not match a verdict judged on a source-labeled pool.
    """

    series_id: str
    channel: int
    mase: float
    qloss_per_q: tuple[float, ...]
    abs_target: float
    quantile_levels: tuple[float, ...]
    source: str | None = None

    @classmethod
    def from_score(cls, ws) -> WindowScoreRecord:
        """From a :class:`cascade.eval.scoring.WindowScore`."""
        return cls(
            series_id=str(ws.series_id),
            channel=int(ws.channel),
            mase=float(ws.mase),
            qloss_per_q=tuple(float(q) for q in ws.qloss_per_q),
            abs_target=float(ws.abs_target),
            quantile_levels=tuple(float(q) for q in ws.quantile_levels),
            source=(str(ws.source) if ws.source else None),
        )

    def to_score(self):
        """Back to a :class:`cascade.eval.scoring.WindowScore` (numpy import is
        local so the schema stays importable without the eval stack)."""
        import numpy as np

        from ..eval.scoring import WindowScore

        return WindowScore(
            series_id=self.series_id,
            mase=self.mase,
            qloss_per_q=np.asarray(self.qloss_per_q, dtype=np.float64),
            abs_target=self.abs_target,
            quantile_levels=self.quantile_levels,
            source=self.source,
            channel=self.channel,
        )


@dataclass(frozen=True)
class EntryScores:
    """Every per-window score one trained entry earned, in scoring order.

    One record per manifest entry the validator evaluated — i.e. per
    (role, size). The order of ``scores`` is the exact order the scorer emitted
    (window order × channel), which is what keeps king and challenger paired
    when an auditor re-pools them across sizes.
    """

    role: str
    size: str
    hotkey: str
    uid: int
    scores: tuple[WindowScoreRecord, ...]


@dataclass(frozen=True)
class EvalContext:
    """Which held-out windows the round was scored on.

    ``pool_ref`` names the pool source (a Hub ``repo@digest`` for a static pool,
    or the snapshot S3 key for a bucket-published pool); ``pool_digest`` is its
    integrity hash (the OCI digest, or the snapshot tar sha256). ``window_ids``
    is the rotated slice actually scored, in selection order — re-deriving the
    slice from ``base_seed`` over the same pool must reproduce it.
    """

    pool_ref: str
    pool_digest: str
    window_ids: tuple[str, ...]
    n_windows: int
    num_samples: int


@dataclass(frozen=True)
class VerdictRecord:
    """The KOTH decision and the state transition it caused.

    ``params`` is the full ``KothParams`` (asdict), ``bootstrap_seed`` the seed
    fed to ``evaluate_round``, and ``king_tenure_rounds`` the king's tenure at
    decision time (it sets the margin under a warmup schedule) — together with
    the recorded scores they make the verdict a pure recomputation. ``lcb`` is
    None when the round was inconclusive (NaN has no strict-JSON form).
    """

    params: dict
    bootstrap_seed: str
    king_tenure_rounds: int
    lcb: float | None
    margin: float
    challenger_wins_round: bool
    inconclusive: bool
    n_windows: int
    king_geomean: float | None
    chal_geomean: float | None
    gift_lcb: float | None
    gift_gate_passed: bool | None
    dethroned: bool
    note: str
    king_hotkey: str | None     # the throne AFTER this round
    king_uid: int | None

    @classmethod
    def from_round(
        cls, result, transition, *, params, bootstrap_seed, king_tenure_rounds: int = 0
    ) -> VerdictRecord:
        """From an ``eval.koth.RoundResult`` + ``validator.state.StateTransition``."""
        return cls(
            params=dict(asdict(params)),
            bootstrap_seed=str(bootstrap_seed),
            king_tenure_rounds=int(king_tenure_rounds),
            lcb=_none_for_nan(result.lcb),
            margin=float(result.margin),
            challenger_wins_round=bool(result.challenger_wins_round),
            inconclusive=bool(result.inconclusive),
            n_windows=int(result.n_windows),
            king_geomean=_none_for_nan(result.king_geomean),
            chal_geomean=_none_for_nan(result.chal_geomean),
            gift_lcb=_none_for_nan(result.gift_lcb),
            gift_gate_passed=result.gift_gate_passed,
            dethroned=bool(transition.dethroned),
            note=str(transition.note),
            king_hotkey=transition.state.king_hotkey,
            king_uid=transition.state.king_uid,
        )


@dataclass(frozen=True)
class RoundReceipt:
    """One round's validator-signed public record. See the module docstring.

    ``status = "scored"`` carries the full eval + verdict + weights;
    ``status = "rejected"`` carries ``reject_reason`` (the ``check_manifest``
    gate's string) with the eval fields empty. Either way the manifest is
    embedded verbatim (its own signature included), so the trainer's claim and
    the validator's judgement of it travel together.
    """

    round_id: str
    status: str                              # "scored" | "rejected"
    # chain context the round is a pure function of
    epoch_start_block: int                   # the epoch boundary (submission cutoff)
    epoch_block_hash: str                    # chain block hash at that boundary
    base_seed: int                           # block_seed(epoch_start_block)
    generation_seed: int                     # RoundSeeds.derive(base_seed).generation_seed
    training_seed: int                       # RoundSeeds.derive(base_seed).training_seed
    # the trainer's claim, verbatim (parsed manifest JSON incl. its signature)
    manifest: dict
    # who was eligible (pre-cutoff resolved commitments); empty when the chain
    # was unreachable at receipt time (the audit then WARNs rather than passes)
    participants: tuple[Participant, ...] = ()
    # what was scored and decided (None/empty on a rejected round)
    eval_context: EvalContext | None = None
    entry_scores: tuple[EntryScores, ...] = ()
    verdict: VerdictRecord | None = None
    reward_uids: tuple[int, ...] = ()
    weights: tuple[float, ...] = ()          # the equal-share vector set on chain
    reject_reason: str | None = None
    validator_hotkey: str = ""               # ss58 of the signer (inside the signed body)
    receipt_version: int = RECEIPT_VERSION
    signature: str | None = None             # validator-hotkey signature over canonical_body

    def __post_init__(self) -> None:
        if self.status not in RECEIPT_STATUSES:
            raise ValueError(f"status must be one of {RECEIPT_STATUSES}; got {self.status!r}")
        if self.status == "rejected" and not self.reject_reason:
            raise ValueError("a rejected receipt must carry reject_reason")

    def load_embedded_manifest(self) -> TrainingManifest:
        """Parse the verbatim-embedded manifest back into a TrainingManifest."""
        return load_manifest(json.dumps(self.manifest))

    def canonical_body(self) -> bytes:
        """Deterministic byte serialisation of everything except the signature.

        The signed payload. Sorted keys, no NaN (strict JSON), so the validator
        and every auditor hash identical bytes.
        """
        body = {
            "receipt_version": self.receipt_version,
            "round_id": self.round_id,
            "status": self.status,
            "epoch_start_block": self.epoch_start_block,
            "epoch_block_hash": self.epoch_block_hash,
            "base_seed": self.base_seed,
            "generation_seed": self.generation_seed,
            "training_seed": self.training_seed,
            "manifest": self.manifest,
            "participants": [asdict(p) for p in self.participants],
            "eval_context": asdict(self.eval_context) if self.eval_context else None,
            "entry_scores": [asdict(e) for e in self.entry_scores],
            "verdict": asdict(self.verdict) if self.verdict else None,
            "reward_uids": list(self.reward_uids),
            "weights": list(self.weights),
            "reject_reason": self.reject_reason,
            "validator_hotkey": self.validator_hotkey,
        }
        return json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")


def build_receipt(
    *,
    round_id: str,
    status: str,
    epoch_start_block: int,
    epoch_block_hash: str,
    base_seed: int,
    seeds,
    manifest: TrainingManifest,
    validator_hotkey: str = "",
    participants: tuple[Participant, ...] = (),
    eval_context: EvalContext | None = None,
    entry_scores: tuple[EntryScores, ...] = (),
    verdict: VerdictRecord | None = None,
    reward_uids: tuple[int, ...] = (),
    weights: tuple[float, ...] = (),
    reject_reason: str | None = None,
) -> RoundReceipt:
    """Assemble a receipt from live-loop objects (``seeds`` is a ``RoundSeeds``).

    The manifest is embedded verbatim — the parsed JSON of ``dump_manifest``, so
    its signature travels inside the receipt body and the trainer's exact signed
    bytes are recoverable.
    """
    return RoundReceipt(
        round_id=str(round_id),
        status=status,
        epoch_start_block=int(epoch_start_block),
        epoch_block_hash=str(epoch_block_hash),
        base_seed=int(base_seed),
        generation_seed=int(seeds.generation_seed),
        training_seed=int(seeds.training_seed),
        manifest=json.loads(dump_manifest(manifest)),
        participants=tuple(participants),
        eval_context=eval_context,
        entry_scores=tuple(entry_scores),
        verdict=verdict,
        reward_uids=tuple(int(u) for u in reward_uids),
        weights=tuple(float(w) for w in weights),
        reject_reason=reject_reason,
        validator_hotkey=validator_hotkey,
    )


def dump_receipt(receipt: RoundReceipt) -> str:
    """Serialise a receipt (including signature) to a JSON string."""
    body = json.loads(receipt.canonical_body().decode("utf-8"))
    body["signature"] = receipt.signature
    return json.dumps(body, indent=2, sort_keys=True)


def load_receipt(text: str) -> RoundReceipt:
    """Parse a receipt JSON string. Raises ``ValueError`` on schema problems."""
    obj = json.loads(text)
    version = int(obj.get("receipt_version", 0))
    if version != RECEIPT_VERSION:
        raise ValueError(f"unsupported receipt_version {version}; need {RECEIPT_VERSION}")
    ec = obj.get("eval_context")
    verdict = obj.get("verdict")
    return RoundReceipt(
        round_id=str(obj["round_id"]),
        status=str(obj["status"]),
        epoch_start_block=int(obj["epoch_start_block"]),
        epoch_block_hash=str(obj["epoch_block_hash"]),
        base_seed=int(obj["base_seed"]),
        generation_seed=int(obj["generation_seed"]),
        training_seed=int(obj["training_seed"]),
        manifest=dict(obj["manifest"]),
        participants=tuple(
            Participant(
                hotkey=str(p["hotkey"]),
                uid=int(p["uid"]),
                gen_ref=str(p["gen_ref"]),
                commit_block=int(p["commit_block"]),
            )
            for p in obj.get("participants", ())
        ),
        eval_context=EvalContext(
            pool_ref=str(ec["pool_ref"]),
            pool_digest=str(ec["pool_digest"]),
            window_ids=tuple(str(w) for w in ec["window_ids"]),
            n_windows=int(ec["n_windows"]),
            num_samples=int(ec["num_samples"]),
        ) if ec else None,
        entry_scores=tuple(
            EntryScores(
                role=str(e["role"]),
                size=str(e["size"]),
                hotkey=str(e["hotkey"]),
                uid=int(e["uid"]),
                scores=tuple(
                    WindowScoreRecord(
                        series_id=str(s["series_id"]),
                        channel=int(s["channel"]),
                        mase=float(s["mase"]),
                        qloss_per_q=tuple(float(q) for q in s["qloss_per_q"]),
                        abs_target=float(s["abs_target"]),
                        quantile_levels=tuple(float(q) for q in s["quantile_levels"]),
                        source=(str(s["source"]) if s.get("source") else None),
                    )
                    for s in e["scores"]
                ),
            )
            for e in obj.get("entry_scores", ())
        ),
        verdict=VerdictRecord(
            params=dict(verdict["params"]),
            bootstrap_seed=str(verdict["bootstrap_seed"]),
            king_tenure_rounds=int(verdict["king_tenure_rounds"]),
            lcb=(None if verdict["lcb"] is None else float(verdict["lcb"])),
            margin=float(verdict["margin"]),
            challenger_wins_round=bool(verdict["challenger_wins_round"]),
            inconclusive=bool(verdict["inconclusive"]),
            n_windows=int(verdict["n_windows"]),
            king_geomean=(None if verdict["king_geomean"] is None
                          else float(verdict["king_geomean"])),
            chal_geomean=(None if verdict["chal_geomean"] is None
                          else float(verdict["chal_geomean"])),
            gift_lcb=(None if verdict.get("gift_lcb") is None else float(verdict["gift_lcb"])),
            gift_gate_passed=verdict.get("gift_gate_passed"),
            dethroned=bool(verdict["dethroned"]),
            note=str(verdict["note"]),
            king_hotkey=verdict.get("king_hotkey"),
            king_uid=(None if verdict.get("king_uid") is None else int(verdict["king_uid"])),
        ) if verdict else None,
        reward_uids=tuple(int(u) for u in obj.get("reward_uids", ())),
        weights=tuple(float(w) for w in obj.get("weights", ())),
        reject_reason=obj.get("reject_reason"),
        validator_hotkey=str(obj.get("validator_hotkey", "")),
        receipt_version=version,
        signature=obj.get("signature"),
    )


def summarize_receipt(receipt: RoundReceipt) -> dict:
    """A compact, presentational summary of one receipt for ``receipts/index.json``.

    Pure (stdlib only), so it stays importable without the eval/chain stacks. It
    pulls the king/challenger identities from the signed ``entry_scores``
    (role → hotkey/uid), their generator refs from the embedded manifest, and the
    KOTH headline numbers from the verdict — everything the dashboard needs to
    render the reigns/rounds tables *without* fetching every per-round receipt.

    This is a *view*, not part of the audit trail: the signed per-round receipt
    remains the source of truth, and the index carries a ``receipt_key`` pointer
    back to it (added by :func:`cascade.shared.hippius.update_receipt_index`).
    """
    entries = receipt.manifest.get("entries", []) if isinstance(receipt.manifest, dict) else []

    def _gen_ref(role: str) -> str | None:
        for e in entries:
            if isinstance(e, dict) and e.get("role") == role:
                return e.get("gen_ref")
        return None

    def _scorer(role: str):
        for es in receipt.entry_scores:
            if es.role == role:
                return es
        return None

    king_es = _scorer("king")
    chal_es = _scorer("challenger")
    v = receipt.verdict

    # Distinct eval-window clusters (upstream feeds) behind the verdict —
    # derived from the SIGNED per-window ``source`` keys, so it is trustworthy
    # (an auditor recomputes the same value) without bloating the signed body.
    # This is the breadth the cluster bootstrap actually resampled; the raw
    # window count overstates the evidence when many windows share a feed.
    n_clusters = None
    if king_es is not None:
        keys = {s.series_id if s.source in (None, "") else s.source for s in king_es.scores}
        n_clusters = len(keys) if king_es.scores else 0

    sizes: list[str] = []
    for e in entries:
        if isinstance(e, dict):
            s = e.get("size", "")
            if s not in sizes:
                sizes.append(s)

    # Heat screen summary (informational; rides in the embedded manifest). Just
    # the counts the rounds table needs — the full per-entrant standings live on
    # the round receipt (receipts/latest.json) that the dashboard reads directly.
    heat = receipt.manifest.get("heat") if isinstance(receipt.manifest, dict) else None
    heat_summary = None
    if isinstance(heat, dict):
        ents = heat.get("entrants") or []
        heat_summary = {
            "screen_size": heat.get("screen_size", ""),
            "finalists": heat.get("finalists"),
            "n_entrants": len(ents),
            "n_advanced": sum(1 for e in ents if isinstance(e, dict)
                              and e.get("status") == "advanced"),
        }

    return {
        "round_id": receipt.round_id,
        "status": receipt.status,
        "epoch_start_block": receipt.epoch_start_block,
        # king/challenger of THIS round (from the signed per-entry scores)
        "king_hotkey": king_es.hotkey if king_es else None,
        "king_uid": king_es.uid if king_es else None,
        "king_gen_ref": _gen_ref("king"),
        "chal_hotkey": chal_es.hotkey if chal_es else None,
        "chal_uid": chal_es.uid if chal_es else None,
        "chal_gen_ref": _gen_ref("challenger"),
        "sizes": sizes,
        "n_participants": len(receipt.participants),
        # verdict headline
        "n_windows": (receipt.eval_context.n_windows if receipt.eval_context
                      else (v.n_windows if v else None)),
        "n_clusters": n_clusters,
        "king_geomean": v.king_geomean if v else None,
        "chal_geomean": v.chal_geomean if v else None,
        "lcb": v.lcb if v else None,
        "margin": v.margin if v else None,
        "challenger_wins_round": v.challenger_wins_round if v else None,
        "inconclusive": v.inconclusive if v else None,
        "dethroned": v.dethroned if v else None,
        "gift_gate_passed": v.gift_gate_passed if v else None,
        # the throne AFTER this round (winner), and the payout set
        "post_round_king_hotkey": v.king_hotkey if v else None,
        "post_round_king_uid": v.king_uid if v else None,
        "reward_uids": list(receipt.reward_uids),
        "n_rewarded": len(receipt.reward_uids),
        "heat": heat_summary,
        "reject_reason": receipt.reject_reason,
        "validator_hotkey": receipt.validator_hotkey or None,
    }


def sign_receipt(receipt: RoundReceipt, wallet: object) -> RoundReceipt:
    """Sign ``canonical_body()`` with the validator's bittensor hotkey.

    Mirrors :func:`cascade.shared.manifest.sign_manifest`: ``wallet`` is a
    ``bittensor.wallet`` (or anything exposing ``.hotkey`` with
    ``.sign(bytes) -> bytes``); the hex signature lands on a copy. The signer's
    ss58 (``validator_hotkey``) must already be set on the receipt — it is part
    of the signed body.
    """
    from dataclasses import replace

    hotkey = getattr(wallet, "hotkey", wallet)
    try:
        sig = hotkey.sign(receipt.canonical_body())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"receipt_signing_failed: {type(e).__name__}: {e}") from e
    return replace(receipt, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_receipt_signature(receipt: RoundReceipt, validator_hotkey: str) -> bool:
    """Verify the receipt was signed by ``validator_hotkey`` (an ss58 address).

    Mirrors :func:`cascade.shared.manifest.verify_signature`: False on a missing
    signature or any mismatch; raises if ``bittensor`` is unavailable so a
    caller never silently accepts an unverified receipt.
    """
    if not receipt.signature or not validator_hotkey:
        return False
    try:
        from bittensor import Keypair  # type: ignore
    except ImportError as e:  # pragma: no cover — audit warns before calling this
        raise RuntimeError(
            "bittensor required to verify receipt signatures; install the [chain] extra"
        ) from e
    try:
        kp = Keypair(ss58_address=validator_hotkey)
        return bool(kp.verify(receipt.canonical_body(), bytes.fromhex(receipt.signature)))
    except Exception:  # noqa: BLE001 — any malformed sig/address ⇒ untrusted
        return False
