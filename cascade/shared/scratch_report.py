"""Scratch-shadow bench report — DEC-CA-0014 Stage 1's telemetry artifact.

Every M-th warm-started round (``[telemetry] scratch_shadow_every_rounds``)
the trainer additionally trains the reigning king's generator FROM SCRATCH —
random init, identical contract, the round's own seeds — strictly after the
manifest publishes, benches the result on the public suites, and publishes the
numbers here: ``benchmarks/scratch/round-<id>.json`` in the manifest bucket,
beside (never inside) the round's signed :mod:`cascade.shared.bench_report`.

This document is **telemetry only**. It must never feed scoring, weights,
KOTH verdicts, warm-start promotion, or the validator's reign log — the
canonical body says so in-band (``kind = "scratch_shadow"``,
``telemetry_only = true``) so no future consumer can mistake it for a scoring
input. It is deliberately a SEPARATE artifact from the round's
:class:`~cascade.shared.bench_report.BenchReport`: validators parse that
report with strict role validation on a consensus path (promotion
provenance), so extending it would be a fleet-lockstep schema change, while a
new key namespace is invisible to every deployed consumer.

What the numbers are for: the lineage-vs-scratch gap and its trend — the
evidence DEC-CA-0014 stages everything on. The report therefore carries the
scratch checkpoint's six numbers AND (when the round's duel bench produced
them) the lineage king's six as an in-document reference, so one fetch yields
one point on both curves. ``benchmarks/scratch/index.json`` is the unsigned,
presentational roll-up of every published report for dashboards and quick
trend reads — the signed per-round report stays the source of truth.

Conventions follow :mod:`cascade.shared.bench_report` exactly: frozen
dataclass, an explicit version, a canonical sorted-key JSON body, sign/verify
with the trainer hotkey.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .hippius import StorageError
from .manifest import BenchScores

SCRATCH_REPORT_VERSION = 1
SCRATCH_INDEX_KEY = "benchmarks/scratch/index.json"


def scratch_report_key(round_id: str) -> str:
    return f"benchmarks/scratch/round-{round_id}.json"


def bench_geomean(s: BenchScores) -> float:
    """The six-number geomean (same aggregate the reign log ranks on) — the
    scalar each curve is plotted in."""
    vals = [s.gifteval_crps, s.gifteval_mase, s.boom_crps,
            s.boom_mase, s.time_crps, s.time_mase]
    return math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / len(vals))


def _scores_body(s: BenchScores | None) -> dict | None:
    if s is None:
        return None
    return {
        "gifteval_crps": s.gifteval_crps,
        "gifteval_mase": s.gifteval_mase,
        "boom_crps": s.boom_crps,
        "boom_mase": s.boom_mase,
        "time_crps": s.time_crps,
        "time_mase": s.time_mase,
    }


def _scores_from(obj: dict | None) -> BenchScores | None:
    if obj is None:
        return None
    return BenchScores(
        gifteval_crps=float(obj["gifteval_crps"]),
        gifteval_mase=float(obj["gifteval_mase"]),
        boom_crps=float(obj["boom_crps"]),
        boom_mase=float(obj["boom_mase"]),
        time_crps=float(obj["time_crps"]),
        time_mase=float(obj["time_mase"]),
    )


@dataclass(frozen=True)
class ScratchBenchReport:
    """One round's scratch-shadow result, signed by the trainer hotkey.

    ``scores`` benches the from-scratch checkpoint (``trained_pointer``, the
    king's generator ``gen_ref`` trained under the identical contract from
    random init). ``king_scores``/``king_pointer`` mirror the lineage king's
    entry from the round's signed bench report when it exists — a convenience
    copy for the gap curve, not a second source of truth. ``warm_start_ckpt``
    records the lineage init the round's REAL runs trained from, and
    ``generation`` the promotion generation, so the gap is attributable to a
    specific lineage depth.
    """

    round_id: str
    created_block: int
    gen_ref: str
    miner_hotkey: str
    miner_uid: int
    trained_pointer: str
    scores: BenchScores
    warm_start_ckpt: str
    generation: int = 0
    king_pointer: str = ""
    king_scores: BenchScores | None = None
    report_version: int = SCRATCH_REPORT_VERSION
    signature: str | None = None  # trainer_hotkey signature over canonical_body()

    def canonical_body(self) -> bytes:
        """Deterministic signed payload; ``kind``/``telemetry_only`` are IN the
        body so the shadow-never-scores label is itself signed."""
        body = {
            "report_version": self.report_version,
            "kind": "scratch_shadow",
            "telemetry_only": True,
            "round_id": self.round_id,
            "created_block": self.created_block,
            "gen_ref": self.gen_ref,
            "miner_hotkey": self.miner_hotkey,
            "miner_uid": self.miner_uid,
            "trained_pointer": self.trained_pointer,
            "scores": _scores_body(self.scores),
            "warm_start_ckpt": self.warm_start_ckpt,
            "generation": self.generation,
            "king_pointer": self.king_pointer,
            "king_scores": _scores_body(self.king_scores),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump_scratch_report(report: ScratchBenchReport) -> str:
    body = json.loads(report.canonical_body().decode("utf-8"))
    body["signature"] = report.signature
    return json.dumps(body, indent=2, sort_keys=True)


def load_scratch_report(text: str) -> ScratchBenchReport:
    """Parse a scratch-report JSON string. Raises ``ValueError`` on schema
    problems (including a document that is not labeled scratch_shadow)."""
    obj = json.loads(text)
    version = int(obj.get("report_version", 0))
    if version != SCRATCH_REPORT_VERSION:
        raise ValueError(f"unsupported report_version {version}; need {SCRATCH_REPORT_VERSION}")
    if obj.get("kind") != "scratch_shadow":
        raise ValueError(f"not a scratch_shadow report: kind={obj.get('kind')!r}")
    return ScratchBenchReport(
        round_id=str(obj["round_id"]),
        created_block=int(obj["created_block"]),
        gen_ref=str(obj["gen_ref"]),
        miner_hotkey=str(obj["miner_hotkey"]),
        miner_uid=int(obj["miner_uid"]),
        trained_pointer=str(obj["trained_pointer"]),
        scores=_scores_from(obj["scores"]),
        warm_start_ckpt=str(obj.get("warm_start_ckpt", "")),
        generation=int(obj.get("generation", 0)),
        king_pointer=str(obj.get("king_pointer", "")),
        king_scores=_scores_from(obj.get("king_scores")),
        report_version=version,
        signature=obj.get("signature"),
    )


def sign_scratch_report(report: ScratchBenchReport, wallet: object) -> ScratchBenchReport:
    """Sign ``canonical_body()`` with the trainer hotkey — the exact scheme of
    :func:`cascade.shared.bench_report.sign_bench_report`."""
    from dataclasses import replace

    hotkey = getattr(wallet, "hotkey", wallet)
    try:
        sig = hotkey.sign(report.canonical_body())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"scratch_report_signing_failed: {type(e).__name__}: {e}") from e
    return replace(report, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_scratch_report_signature(report: ScratchBenchReport, trainer_hotkey: str) -> bool:
    """Verify the report was signed by ``trainer_hotkey`` (ss58). False on a
    missing signature or mismatch; raises only when bittensor is unavailable."""
    if not report.signature or not trainer_hotkey:
        return False
    try:
        from bittensor import Keypair  # type: ignore
    except ImportError as e:  # pragma: no cover — consumers have bittensor
        raise RuntimeError(
            "bittensor required to verify scratch-report signatures; install the [chain] extra"
        ) from e
    try:
        kp = Keypair(ss58_address=trainer_hotkey)
        return bool(kp.verify(report.canonical_body(), bytes.fromhex(report.signature)))
    except Exception:  # noqa: BLE001 — any malformed sig/address ⇒ untrusted
        return False


def publish_scratch_report(store: object, report_text: str, round_id: str) -> str:
    """Write the round's scratch report to the manifest-bucket store (same
    public-read + ACL-fallback convention as the bench report) and return the
    key."""
    key = scratch_report_key(round_id)
    try:
        store.put_text(key, report_text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        store.put_text(key, report_text, content_type="application/json")
    return key


def update_scratch_index(store: object, report: ScratchBenchReport) -> str:
    """Fold one report into ``benchmarks/scratch/index.json`` — the unsigned,
    presentational trend roll-up (newest last, keyed by round). Each row is
    one point on the two curves DEC-CA-0014 stages on: the scratch geomean,
    the lineage king's geomean (when the duel bench produced it), and their
    gap. Best-effort read-modify-write; the signed per-round reports remain
    the source of truth."""
    try:
        idx = json.loads(store.get_text(SCRATCH_INDEX_KEY))
        rows = list(idx.get("rounds") or [])
    except Exception:  # noqa: BLE001 — first report, or unreadable index
        rows = []
    scratch_gm = bench_geomean(report.scores)
    king_gm = bench_geomean(report.king_scores) if report.king_scores else None
    row = {
        "round_id": report.round_id,
        "created_block": report.created_block,
        "generation": report.generation,
        "warm_start_ckpt": report.warm_start_ckpt,
        "scratch_geomean": scratch_gm,
        "king_geomean": king_gm,
        # gap > 0 ⇔ the lineage is ahead of scratch (lower is better on every
        # suite); a closing gap is Stage 2's arming signal.
        "gap": (scratch_gm - king_gm) if king_gm is not None else None,
        "scratch_scores": _scores_body(report.scores),
        "king_scores": _scores_body(report.king_scores),
    }
    rows = [r for r in rows if r.get("round_id") != report.round_id] + [row]
    rows.sort(key=lambda r: int(r.get("created_block", 0)))
    text = json.dumps({"kind": "scratch_shadow_index", "telemetry_only": True,
                       "rounds": rows}, indent=2, sort_keys=True)
    try:
        store.put_text(SCRATCH_INDEX_KEY, text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        store.put_text(SCRATCH_INDEX_KEY, text, content_type="application/json")
    return SCRATCH_INDEX_KEY
