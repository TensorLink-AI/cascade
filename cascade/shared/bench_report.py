"""Round bench report — the trainer's post-publish public-benchmark artifact.

When ``[scoring] cascade_enabled``, the trainer benchmarks BOTH final-duel
checkpoints — the king's AND the challenger's — on the three public suites
(GIFT-Eval / BOOM / TIME) strictly *after* the round's manifest is published,
so validators start scoring the duel the moment the manifest lands and never
wait on a benchmark sweep. The six numbers per checkpoint ride in this
separate, trainer-signed JSON document (one per round, keyed by ``round_id``)
instead of inside the manifest entry.

Originally telemetry and promotion evidence, these reports also bind public
guard rows under DEC-CA-0054/0061. Missing required rows can hold a judgement;
they are not a successful guard. DEC-CA-0063 uses the same wire format under
``bench_v`` for main-validator reports acknowledged by later trainer manifests.
The private-pool verdict and public gate must both pass before a dethrone.

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
from .manifest import VALID_ROLES, BenchScores, bench_scores_body, bench_scores_from_json

# DEC-CA-0055: one entry per v2 report may carry role "init" — the CURRENT
# init's BenchScores (benched once per seed epoch; the random-init model of
# the era's seed in a random-init era), the reference of the king check.
BENCH_ROLES = (*VALID_ROLES, "init")

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

    role: str                 # "king" | "challenger" | "init" (DEC-CA-0055)
    size: str                 # arch_preset the checkpoint was trained at
    miner_hotkey: str
    miner_uid: int
    trained_pointer: str
    scores: BenchScores
    # DEC-CA-0054: the operator-run sweep's per-item rows (the three-suite
    # bench gate's input), published at ``rows_key`` and pinned by sha256.
    # "" on every report before the gate existed — dropped from the signed
    # body then, so archived signatures still verify.
    rows_key: str = ""
    rows_sha256: str = ""
    # DEC-CA-0063 (validator reports only, drop-when-empty): the sidecar's
    # pinned data revision per suite the sweep scored against
    # (``report["data_revisions"]``) — a gate pairing two entries swept on
    # different revisions is uncomputable (the GIFT gate's rule).
    data_revisions: dict | None = None

    def __post_init__(self) -> None:
        if self.role not in BENCH_ROLES:
            raise ValueError(f"role must be one of {BENCH_ROLES}; got {self.role!r}")


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
    # DEC-CA-0063 (validator reports only, drop-when-empty, SIGNED): the
    # checkpoints the main validator was asked for but could not sweep —
    # ``{"trained_pointer", "role", "reason"}``, sorted by pointer. A sweep
    # that failed every attempt, or one the settlement's bench budget left no
    # time for, is INFRA (never the miner's loss): distinct from an entry the
    # report simply omits.
    infra_failed: tuple = ()
    # DEC-CA-0063 items 9-10 (validator reports only, drop-when-empty, SIGNED,
    # same ``{"trained_pointer", "role", "reason"}`` shape, sorted by pointer):
    # ``budget_deferred`` — clearers the settlement's sweep cap ([v2]
    # bench_budget_hours) left no whole allowance for: never swept, never a
    # loss, re-judged at a later settlement (the trainer carries the leg into
    # the manifest that records this report). ``checkpoint_fault`` — sweeps
    # that failed on EVERY attempt for a reason the checkpoint itself causes
    # (non-finite forecasts, every config failing at scoring with no
    # environmental marker): a challenger listed here FAILS the bench gate on
    # every validator (a loss); a king / init listed here is infra.
    budget_deferred: tuple = ()
    checkpoint_fault: tuple = ()

    def infra_reason(self, trained_pointer: str) -> str | None:
        """Why the main validator could not sweep ``trained_pointer``
        (DEC-CA-0063), else ``None``."""
        return _listed_reason(self.infra_failed, trained_pointer, "infra")

    def deferred_reason(self, trained_pointer: str) -> str | None:
        """Why the settlement's sweep cap deferred ``trained_pointer``
        (DEC-CA-0063 item 9), else ``None``."""
        return _listed_reason(self.budget_deferred, trained_pointer, "budget deferred")

    def fault_reason(self, trained_pointer: str) -> str | None:
        """The checkpoint fault ``trained_pointer``'s sweeps reproduced
        (DEC-CA-0063 item 10), else ``None``."""
        return _listed_reason(self.checkpoint_fault, trained_pointer, "checkpoint fault")

    def entry_for_pointer(self, trained_pointer: str) -> BenchEntry | None:
        """The entry benching exactly this checkpoint pointer, else ``None``."""
        for e in self.entries:
            if e.trained_pointer == trained_pointer:
                return e
        return None

    def init_entry(self) -> BenchEntry | None:
        """The DEC-CA-0055 ``role == "init"`` entry (the current init's
        scores), else ``None``."""
        for e in self.entries:
            if e.role == "init":
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
            "entries": [_entry_body(e) for e in self.entries],
        }
        if self.infra_failed:
            body["infra_failed"] = [dict(f) for f in self.infra_failed]
        if self.budget_deferred:
            body["budget_deferred"] = [dict(f) for f in self.budget_deferred]
        if self.checkpoint_fault:
            body["checkpoint_fault"] = [dict(f) for f in self.checkpoint_fault]
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _listed_reason(listed: tuple, trained_pointer: str, default: str) -> str | None:
    for f in listed:
        if str(f.get("trained_pointer", "")) == trained_pointer:
            return str(f.get("reason", "") or default)
    return None


def _listed(obj: dict, key: str) -> tuple:
    return tuple(
        {"trained_pointer": str(f["trained_pointer"]), "role": str(f.get("role", "")),
         "reason": str(f.get("reason", ""))}
        for f in obj.get(key) or ())


def _entry_body(e: BenchEntry) -> dict:
    """One entry's canonical dict; the DEC-CA-0054 rows pointer only when set."""
    d = {
        "role": e.role,
        "size": e.size,
        "miner_hotkey": e.miner_hotkey,
        "miner_uid": e.miner_uid,
        "trained_pointer": e.trained_pointer,
        # the six numbers, plus FEV's only when benched (DEC-CA-0061)
        "scores": bench_scores_body(e.scores),
    }
    if e.rows_key:
        d["rows_key"] = e.rows_key
        d["rows_sha256"] = e.rows_sha256
    if e.data_revisions:
        d["data_revisions"] = dict(e.data_revisions)
    return d


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
            scores=bench_scores_from_json(e["scores"]),
            rows_key=str(e.get("rows_key", "")),
            rows_sha256=str(e.get("rows_sha256", "")),
            data_revisions=(dict(e["data_revisions"])
                            if isinstance(e.get("data_revisions"), dict)
                            and e["data_revisions"] else None),
        )
        for e in obj["entries"]
    )
    return BenchReport(
        round_id=str(obj["round_id"]),
        created_block=int(obj["created_block"]),
        entries=entries,
        report_version=version,
        signature=obj.get("signature"),
        infra_failed=_listed(obj, "infra_failed"),
        budget_deferred=_listed(obj, "budget_deferred"),
        checkpoint_fault=_listed(obj, "checkpoint_fault"),
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
    from .sr25519 import verify_hotkey

    return verify_hotkey(trainer_hotkey, report.canonical_body(), report.signature)


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


def bench_rows_key(trained_pointer: str) -> str:
    """Where a checkpoint's per-item bench rows live (DEC-CA-0054)."""
    import hashlib

    return f"benchmarks/rows/{hashlib.sha256(trained_pointer.encode()).hexdigest()}.json"


def dump_bench_rows(rows: dict) -> str:
    """Canonical text of an ``item_rows_from_report`` result — the bytes the
    signed ``rows_sha256`` pins."""
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def load_bench_rows(text: str, sha256: str) -> dict:
    """Parse published rows, refusing bytes that do not hash to ``sha256``."""
    import hashlib

    if hashlib.sha256(text.encode("utf-8")).hexdigest() != sha256:
        raise ValueError("bench rows fail their signed sha256")
    rows = json.loads(text)
    if not isinstance(rows, dict):
        raise ValueError("bench rows must be a JSON object")
    return rows


# ── DEC-CA-0063: the main validator's own bench report ──────────────────────
# Same BenchReport / rows wire format as the trainer's, under a DISTINCT key
# prefix and signed by the main validator's hotkey ([netconfig] owner_hotkey),
# never the trainer's. One report per settlement whose era takes its lane-A
# rows from the validator (possibly with no entries: "nothing was swept" is a
# published fact other validators wait for, never a guess).

VALIDATOR_BENCH_PREFIX = "bench_v"
# DEC-CA-0063 item 9: how many times the trainer carries a budget-deferred
# leg into a later settlement of its era; after that it stays skipped.
VBENCH_MAX_CARRIES = 2


def validator_bench_report_key(round_id: str) -> str:
    return f"{VALIDATOR_BENCH_PREFIX}/round-{round_id}.json"


def validator_bench_rows_key(round_id: str, trained_pointer: str, rows_sha256: str = "") -> str:
    """Per-ROUND, content-addressed rows key: a re-sweep of the same
    checkpoint — at a later settlement, or by a restarted main validator
    within the round (GPU sweeps are not bit-reproducible) — lands at a
    DIFFERENT key, so it can never overwrite the bytes a published report's
    ``rows_sha256`` pins. ``rows_sha256`` = the sha256 of the rows text
    (empty = the pre-content-address layout)."""
    import hashlib

    h = hashlib.sha256(trained_pointer.encode()).hexdigest()
    if rows_sha256:
        return f"{VALIDATOR_BENCH_PREFIX}/rows/{round_id}/{h}/{rows_sha256}.json"
    return f"{VALIDATOR_BENCH_PREFIX}/rows/{round_id}/{h}.json"


def publish_validator_bench_report(store: object, report_text: str, round_id: str) -> str:
    """Write the main validator's report (public-read, private fallback —
    :func:`publish_bench_report`'s convention) and return its key."""
    key = validator_bench_report_key(round_id)
    try:
        store.put_text(key, report_text, content_type="application/json",
                       acl="public-read")
    except StorageError:
        store.put_text(key, report_text, content_type="application/json")
    return key
