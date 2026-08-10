"""Round bench report — the trainer's post-publish public-benchmark artifact.

When ``[scoring] cascade_enabled``, the trainer benchmarks BOTH final-duel
checkpoints — the king's AND the challenger's — on the three public suites
(GIFT-Eval / BOOM / TIME) strictly *after* the round's manifest is published,
so validators start scoring the duel the moment the manifest lands and never
wait on a benchmark sweep. The six numbers per checkpoint ride in this
separate, trainer-signed JSON document (one per round, keyed by ``round_id``)
instead of inside the manifest entry.

These numbers are telemetry + Cascade warm-start promotion input ONLY
(:mod:`cascade.validator.cascade` reads the king's set into the reign log).
They never feed scoring, weights, KOTH verdicts, or the gift gate — the
consensus eval on the private pool stays entirely with the validator. A round
whose report never appears simply contributes no bench numbers; promotion
selects over the reign rounds that have them.

Conventions follow :mod:`cascade.shared.manifest` exactly: frozen dataclasses,
an explicit ``report_version``, a canonical sorted-key JSON body, and
sign/verify over :meth:`BenchReport.canonical_body` with the trainer's
bittensor hotkey — the same trust anchor (``[manifest] trainer_hotkey``)
validators already hold for the manifest. Reports are published to the
manifest bucket (``benchmarks/round-<id>.json``) through the same store as the
manifest itself, so the R2 dual-write backup applies unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .hippius import StorageError
from .manifest import VALID_ROLES, BenchScores

BENCH_REPORT_VERSION = 1


def bench_report_key(round_id: str) -> str:
    return f"benchmarks/round-{round_id}.json"


@dataclass(frozen=True)
class BenchEntry:
    """One duel checkpoint's public-benchmark scoring.

    ``trained_pointer`` identifies the exact checkpoint (same
    ``metro-v1:trained:hippius:…`` pointer as the manifest entry it mirrors),
    which is what the validator joins the report to the manifest on — never
    the UID, which recycles. ``scores`` is the same six-number
    :class:`~cascade.shared.manifest.BenchScores` shape the manifest's in-entry
    field carried, so both sources feed one consumer unchanged.
    """

    role: str                 # "king" | "challenger"
    size: str                 # arch_preset the checkpoint was trained at
    miner_hotkey: str
    miner_uid: int
    trained_pointer: str
    scores: BenchScores

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}; got {self.role!r}")


@dataclass(frozen=True)
class BenchReport:
    """A round's public-benchmark results, signed by the trainer hotkey.

    A partial report is legitimate: benchmarks are best-effort, so a round
    where only one duel checkpoint produced a complete score set publishes
    what it has (possibly a single entry). ``created_block`` is the round's
    manifest block, carried for observability only.
    """

    round_id: str
    created_block: int
    entries: tuple[BenchEntry, ...] = ()
    report_version: int = BENCH_REPORT_VERSION
    signature: str | None = None  # trainer_hotkey signature over canonical_body()

    def entry_for(self, role: str, *, size: str | None = None) -> BenchEntry | None:
        """First entry for ``role`` (optionally at ``size``), else ``None``."""
        for e in self.entries:
            if e.role == role and (size is None or e.size == size):
                return e
        return None

    def entry_for_pointer(self, trained_pointer: str) -> BenchEntry | None:
        """The entry benching exactly this checkpoint pointer, else ``None``."""
        for e in self.entries:
            if e.trained_pointer == trained_pointer:
                return e
        return None

    def canonical_body(self) -> bytes:
        """Deterministic byte serialisation of everything except the signature —
        the signed payload, stable key ordering, mirroring
        :meth:`cascade.shared.manifest.TrainingManifest.canonical_body`."""
        body = {
            "report_version": self.report_version,
            "round_id": self.round_id,
            "created_block": self.created_block,
            "entries": [
                {
                    "role": e.role,
                    "size": e.size,
                    "miner_hotkey": e.miner_hotkey,
                    "miner_uid": e.miner_uid,
                    "trained_pointer": e.trained_pointer,
                    "scores": {
                        "gifteval_crps": e.scores.gifteval_crps,
                        "gifteval_mase": e.scores.gifteval_mase,
                        "boom_crps": e.scores.boom_crps,
                        "boom_mase": e.scores.boom_mase,
                        "time_crps": e.scores.time_crps,
                        "time_mase": e.scores.time_mase,
                    },
                }
                for e in self.entries
            ],
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dump_bench_report(report: BenchReport) -> str:
    """Serialise a report (including signature) to a JSON string."""
    body = json.loads(report.canonical_body().decode("utf-8"))
    body["signature"] = report.signature
    return json.dumps(body, indent=2, sort_keys=True)


def load_bench_report(text: str) -> BenchReport:
    """Parse a bench-report JSON string. Raises ``ValueError`` on schema problems."""
    obj = json.loads(text)
    version = int(obj.get("report_version", 0))
    if version != BENCH_REPORT_VERSION:
        raise ValueError(f"unsupported report_version {version}; need {BENCH_REPORT_VERSION}")
    entries = tuple(
        BenchEntry(
            role=str(e["role"]),
            size=str(e.get("size", "")),
            miner_hotkey=str(e["miner_hotkey"]),
            miner_uid=int(e["miner_uid"]),
            trained_pointer=str(e["trained_pointer"]),
            scores=BenchScores(
                gifteval_crps=float(e["scores"]["gifteval_crps"]),
                gifteval_mase=float(e["scores"]["gifteval_mase"]),
                boom_crps=float(e["scores"]["boom_crps"]),
                boom_mase=float(e["scores"]["boom_mase"]),
                time_crps=float(e["scores"]["time_crps"]),
                time_mase=float(e["scores"]["time_mase"]),
            ),
        )
        for e in obj["entries"]
    )
    return BenchReport(
        round_id=str(obj["round_id"]),
        created_block=int(obj["created_block"]),
        entries=entries,
        report_version=version,
        signature=obj.get("signature"),
    )


def sign_bench_report(report: BenchReport, wallet: object) -> BenchReport:
    """Sign ``canonical_body()`` with the trainer's bittensor hotkey — the exact
    scheme (and wallet duck-type) of :func:`cascade.shared.manifest.sign_manifest`."""
    from dataclasses import replace

    hotkey = getattr(wallet, "hotkey", wallet)
    try:
        sig = hotkey.sign(report.canonical_body())
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bench_report_signing_failed: {type(e).__name__}: {e}") from e
    return replace(report, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_bench_report_signature(report: BenchReport, trainer_hotkey: str) -> bool:
    """Verify the report was signed by ``trainer_hotkey`` (an ss58 address).

    Mirrors :func:`cascade.shared.manifest.verify_signature`: False on a missing
    signature or any mismatch; raises only when ``bittensor`` is unavailable so
    a caller never silently accepts an unverified report."""
    if not report.signature or not trainer_hotkey:
        return False
    try:
        from bittensor import Keypair  # type: ignore
    except ImportError as e:  # pragma: no cover - validator has bittensor
        raise RuntimeError(
            "bittensor required to verify bench-report signatures; install the [chain] extra"
        ) from e
    try:
        kp = Keypair(ss58_address=trainer_hotkey)
        return bool(kp.verify(report.canonical_body(), bytes.fromhex(report.signature)))
    except Exception:  # noqa: BLE001 — any malformed sig/address ⇒ untrusted
        return False


def publish_bench_report(store: object, report_text: str, round_id: str) -> str:
    """Write the round's bench report to the manifest-bucket store and return its
    key. The store comes from ``open_manifest_store`` at every call site, so the
    R2 dual-write (and HF failover) the manifest enjoys covers the report too.

    Published PUBLIC-READ: the manifest bucket is private by default (only the
    receipt trail, the status docs and the dashboards carry the canned ACL), and
    the stakeholder scoreboard fetches this object anonymously from the browser.
    Without the ACL the round doc is written but reads 403, which is
    indistinguishable from "never benched" on the page. Same fallback as the
    receipt publisher — a backend without canned ACLs publishes private rather
    than not at all."""
    key = bench_report_key(round_id)
    try:
        store.put_text(key, report_text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        # ACL unsupported on this backend: publish private rather than not at all.
        store.put_text(key, report_text, content_type="application/json")
    return key
