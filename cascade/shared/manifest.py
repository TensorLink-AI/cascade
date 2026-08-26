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
configured ss58 address. Decentralising training is future work; the
corpus/contract digests already make every run independently reproducible.
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


def corpus_digest(series: Sequence[np.ndarray | dict]) -> str:
    """Stable sha256 over a generated corpus.

    Each values-only series is canonicalised to ``(C, L)`` (a 1-D ``(L,)``
    array is promoted to ``(1, L)``), and the hash covers the count, every
    series' full ``(C, L)`` shape, and its raw float64 bytes in yield order.
    Carrying the channel count in the digest keeps it stable as the corpus
    moves from univariate ``(1, L)`` to multivariate ``(C, L)`` — a univariate
    and a single-channel-of-multivariate corpus never collide. Two trainers
    that draw the same corpus from the same pinned generator + seed get the
    same digest, which is what makes a training run auditable.

    An EXTENDED record element (a ``{"values": …, "mask"/"roles": …}`` dict
    from an ``accepted_fields``-armed drain, DEC-CA-0020/0023/0026) hashes via
    its 0xFF-sentinel frame (:func:`cascade.interface.generator.
    record_frame_bytes`): a legacy element's bytes start with an 8-byte BE
    channel count (first byte 0x00), so the two framings can never collide,
    and every values-only corpus keeps its frozen bytes (golden-vector
    enforced) forever.
    """
    from ..interface.generator import canonicalize_record, record_frame_bytes

    h = hashlib.sha256()
    h.update(len(series).to_bytes(8, "big"))
    for arr in series:
        if isinstance(arr, dict):
            h.update(record_frame_bytes(canonicalize_record(arr)))
            continue
        a = np.ascontiguousarray(np.atleast_2d(np.asarray(arr, dtype=np.float64)))
        h.update(a.shape[0].to_bytes(8, "big"))   # channels
        h.update(a.shape[1].to_bytes(8, "big"))   # length
        h.update(a.tobytes())
    return h.hexdigest()


# Contract fields dropped from the digest payload while they hold their inert
# default — the contract-side twin of canonical_body's drop-when-unset
# convention (bench_scores, duel_rank, the eval-pool pin). This is what lets a
# release ADD a digest-bound [training] key without moving any deployed
# fleet's contract_digest: the field enters the hash only when an operator
# actually sets it, which is a deliberate, coordinated digest bump (the
# routine re-pin protocol). NEVER remove or change an entry once shipped —
# that would move digests for configs relying on the drop; the golden-vector
# test freezes the behaviour.
_DIGEST_DROP_WHEN_DEFAULT: dict[str, tuple] = {
    # DEC-CA-0020 layer 3: the accepted record-field set ([training]
    # accepted_fields). Empty = values-only (every deployed config).
    "accepted_fields": ((), []),
    # DEC-CA-0026: future-known covariate admission (roles value 2). False
    # until the EVAL_POOL exogeneity rule exists in writing.
    "allow_future_known": (False,),
    # DEC-CA-0028: the owner-published shared real corpus pin. "" = none
    # (every deployed config); setting it is the deliberate digest bump that
    # arms the shared-corpus regime.
    "real_corpus_ref": ("",),
    # DEC-CA-0029: the fork-anneal fraction (finished-form duel checkpoints —
    # the deferred "D" of wsd). 0.0 = off (every deployed config); setting it
    # is the deliberate digest bump that arms the anneal cut.
    "anneal_fraction": (0.0, 0),
    # DEC-CA-0033: the measured variance bundle. EMA finished-form artifact
    # (0.0 = off), N-seed generation mix (1 = single invocation), and the
    # warm-started re-warmup (0.0 = off) — each inert at its default on
    # every deployed config; setting one is the deliberate digest bump.
    "ema_decay": (0.0, 0),
    "gen_seed_mix": (1,),
    "rewarmup_fraction": (0.0, 0),
}


def contract_digest(contract: object) -> str:
    """Stable sha256 over the fields of a training contract dataclass.

    Used to assert king and challenger were trained under byte-identical terms.
    Accepts any dataclass (typically ``TrainingContractConfig``). Fields listed
    in :data:`_DIGEST_DROP_WHEN_DEFAULT` are omitted while they hold their
    inert default, so adding such a field to the dataclass never moves a
    deployed digest — setting it is the digest bump.
    """
    if hasattr(contract, "__dataclass_fields__"):
        payload = asdict(contract)  # type: ignore[arg-type]
    elif isinstance(contract, dict):
        payload = dict(contract)
    else:
        raise TypeError(f"contract_digest expects a dataclass or dict; got {type(contract)}")
    for key, defaults in _DIGEST_DROP_WHEN_DEFAULT.items():
        if key not in payload:
            continue
        val = payload[key]
        if isinstance(val, list):
            val = tuple(val)      # asdict/json round-trips lose tuple-ness
        if val in defaults:
            payload.pop(key)
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
                              # A round carries one king per size and, when the heat
                              # advanced a tied cohort, up to ``max_finalists``
                              # challengers per size (DEC-CA-0012).
    bench_scores: BenchScores | None = None  # trainer-signed public-benchmark scores (king only)
    # Heat placement within the advancing cohort, 0-based, best observed geomean
    # first (DEC-CA-0012). Record order only — the validator judges the WHOLE
    # cohort and crowns the best margin-clearer, so this never decides the
    # throne; it exists so every validator serialises ``entry_scores`` in one
    # order and the dashboard can show the screen's ranking. Dropped from
    # ``canonical_body`` when 0 (see :func:`_entry_body`), so a single-finalist
    # manifest hashes exactly as it did before the field existed. Ranks may be
    # non-contiguous — ``_drop_final_content_clones`` can remove one — so
    # consumers sort, never index.
    duel_rank: int = 0

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
    non-finalist submission fared. It carries a ``rank``, a ``rel_score``
    *relative to the best entrant* (``heat_score / best``, ≥ 1.0, where 1.0 is the
    best), and the raw aggregate error components ``crps`` (CRPS-family MWSQL) and
    ``mase`` (geometric-mean MASE) on the round's eval-pool slice — the same two
    components the ranking geomean is built from, so ``sqrt(crps * mase)``
    reproduces the ``heat_score`` behind ``rel_score``.

    Note: publishing the raw ``crps``/``mase`` exposes absolute error on the
    private, per-round rotated eval pool — deliberately withheld in earlier
    versions, since an absolute per-round signal can help a miner distribution-
    match the pool. It is emitted now by owner decision for miner transparency
    (owner confirmed 2026-07-26; see OPSLOG and DEC-CA-0007 if written).
    ``rank``/``rel_score``/``crps``/``mase`` are None for an entrant that never
    produced a score (``failed_train`` / ``failed_screen``).

    ``p_best`` is the shadow selection diagnostic (:mod:`cascade.eval.heat`): the
    fraction of joint-bootstrap bags in which this entrant scored best. It says
    how much of the ranking is signal — a leader sitting near ``1 / n_entrants``
    was ranked by noise. Unlike ``crps``/``mase`` it is a comparison among
    entrants, not an absolute pool score. None when the screener returned only a
    scalar (no per-window components to resample).
    """

    uid: int
    hotkey: str
    gen_ref: str
    status: str                    # one of HEAT_STATUSES
    rank: int | None = None        # 1-based placement among scored entrants
    rel_score: float | None = None  # heat_score / best_heat_score (≥ 1.0; 1.0 = best)
    p_best: float | None = None    # P(best) over the joint bootstrap; diagnostic only
    crps: float | None = None      # raw CRPS-family loss (MWSQL) on the eval pool; None if unscored
    mase: float | None = None      # raw geometric-mean MASE on the eval pool; None if unscored

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

    ``leader_lcb`` is the shadow diagnostic on how decisive the screen was: the
    paired lower confidence bound on the leader's relative improvement over the
    runner-up (the duel's statistic, with the runner-up in the king's slot).
    ``> 0`` means the screen genuinely separated first from second; ``<= 0`` means
    it did not and the two were interchangeable on this evidence. It never
    changed which entrants advanced — see :mod:`cascade.eval.heat`. None when the
    screener returned only scalars, or with a single scored entrant.
    """

    screen_size: str               # arch_preset the heat screened at
    finalists: int                 # how many advanced to the final
    entrants: tuple[HeatEntrant, ...] = ()
    leader_lcb: float | None = None   # leader-vs-runner-up paired LCB; diagnostic only
    n_windows: int | None = None      # eval windows the screen ranked on
    n_clusters: int | None = None     # distinct upstream feeds behind those windows
    # The round's warm-start init scored on the same heat slice — the null
    # baseline ([round] init_gate_mode, shadow/enforce). None = gate off or
    # no warm start; dropped from the JSON when None so pre-gate standings
    # stay byte-identical.
    init_baseline: float | None = None


def _entry_body(e: TrainedEntry) -> dict:
    """One entry's canonical dict. ``bench_scores`` is omitted when ``None`` and
    ``duel_rank`` when 0, so a manifest without them is byte-identical to a
    manifest predating those fields — the signed payload only grows for entries
    that actually carry them, which keeps every archived signature valid without
    a ``MANIFEST_VERSION`` bump."""
    d = asdict(e)
    if d.get("bench_scores") is None:
        d.pop("bench_scores", None)
    if not d.get("duel_rank"):
        d.pop("duel_rank", None)
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


def heat_to_json(heat: HeatResult | None) -> dict | None:
    """The heat block's JSON shape. Public because the trainer publishes the
    SAME shape mid-round as the standalone heat mirror
    (:mod:`cascade.shared.heat_status`) — one shape means the dashboards render
    a live heat and a settled round's heat with one code path."""
    if heat is None:
        return None
    d = {
        "screen_size": heat.screen_size,
        "finalists": heat.finalists,
        "entrants": [asdict(e) for e in heat.entrants],
        "leader_lcb": heat.leader_lcb,
        "n_windows": heat.n_windows,
        "n_clusters": heat.n_clusters,
    }
    if heat.init_baseline is not None:
        d["init_baseline"] = heat.init_baseline
    return d


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
                p_best=(None if e.get("p_best") is None else float(e["p_best"])),
                crps=(None if e.get("crps") is None else float(e["crps"])),
                mase=(None if e.get("mase") is None else float(e["mase"])),
            )
            for e in obj.get("entrants", ())
        ),
        leader_lcb=(None if obj.get("leader_lcb") is None else float(obj["leader_lcb"])),
        n_windows=(None if obj.get("n_windows") is None else int(obj["n_windows"])),
        n_clusters=(None if obj.get("n_clusters") is None else int(obj["n_clusters"])),
        init_baseline=(None if obj.get("init_baseline") is None
                       else float(obj["init_baseline"])),
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
    # Warm-start pin (Cascade, DEC-CA-0005/0004): the content-addressed
    # checkpoint pointer this round's runs at ``warm_start_size`` initialised
    # from (same ``trained_pointer`` format — the OCI digest pins the bytes),
    # instead of random init. Signed (see canonical_body), so validators verify
    # the trainer trained from the init THEIR deterministic promotion selected,
    # and cascade-audit re-derives from the pinned checkpoint. Empty ⇒ random
    # init (no promotion yet, or a pre-warm-start trainer).
    warm_start_ckpt: str = ""
    warm_start_size: str = ""
    # Realised round composition (jittered mix, DEC-TB-0003 port): domain
    # counts, effective domains, cadences, class count of the round's served
    # eval windows. Informational and UNSIGNED like ``heat`` — never enters
    # :meth:`canonical_body` (post-hoc: each round's Dirichlet draw is
    # independent, so it predicts nothing about the next round). ``None`` while
    # the mix is inactive, so pre-activation manifests serialise byte-for-byte
    # as before.
    composition: dict | None = None
    signature: str | None = None  # trainer_hotkey signature over canonical_body()

    def entry_for_role(self, role: str) -> TrainedEntry | None:
        for e in self.entries:
            if e.role == role:
                return e
        return None

    def entries_for_role(self, role: str) -> list[TrainedEntry]:
        """All entries for ``role`` — one per trained size (the primary plus any
        ``[[training.sizes]]``) per distinct miner. Order follows the manifest's
        entry order, which the trainer emits size-by-size.

        NOTE: with a multi-finalist cohort (DEC-CA-0012) the challenger role has
        one entry per (hotkey, size), so keying the result by ``size`` alone
        silently drops challengers. Use :meth:`challenger_cohort`.
        """
        return [e for e in self.entries if e.role == role]

    def challenger_cohort(self) -> list[tuple[str, dict[str, TrainedEntry]]]:
        """The advancing cohort as ``[(hotkey, {size: entry})]`` in duel order.

        Duel order is ``(duel_rank, hotkey)``: the heat's observed ranking, with
        the hotkey breaking ties so a legacy manifest (every ``duel_rank`` 0) is
        still ordered identically by every validator. This is *record* order, not
        precedence — the validator judges the whole cohort and crowns the best
        margin-clearer (DEC-CA-0012), so nothing about the throne depends on it.

        Grouping by hotkey rather than size is the whole point: one generator
        competes at every size, and the pooled decision needs all of a
        challenger's sizes together.
        """
        by_hotkey: dict[str, dict[str, TrainedEntry]] = {}
        rank: dict[str, int] = {}
        for e in self.entries:
            if e.role != "challenger":
                continue
            by_hotkey.setdefault(e.miner_hotkey, {})[e.size] = e
            # A generator carries one rank for the round; entries agree across
            # sizes, but take the lowest defensively so a malformed manifest
            # cannot make the order depend on dict insertion.
            prev = rank.get(e.miner_hotkey)
            rank[e.miner_hotkey] = e.duel_rank if prev is None else min(prev, e.duel_rank)
        return [(hk, by_hotkey[hk]) for hk in sorted(by_hotkey, key=lambda h: (rank[h], h))]

    def king_sizes(self) -> list[str]:
        """Sizes the king was trained at, in manifest order."""
        king = {e.size for e in self.entries if e.role == "king"}
        return [s for s in self.sizes() if s in king]

    def duel_cohort(
        self,
    ) -> tuple[list[tuple[str, dict[str, TrainedEntry]]], list[str]]:
        """``(cohort, paired_sizes)`` — who gets duelled, and on which sizes.

        A challenger is scored on the sizes it shares with the king; a size the
        king trained but the challenger did not is simply skipped, and the round is
        still decided on the rest. That is long-standing behaviour and it is
        preserved exactly for a single challenger.

        With a cohort (DEC-CA-0012) the crowning step compares challengers by
        observed geomean, so they must be scored on the SAME sizes or the
        comparison is meaningless. Challengers are therefore grouped by the size
        set they share with the king and only the **maximal** group is duelled:
        largest set wins, ties broken toward the earlier (primary) sizes. A
        challenger that failed to train at a size where its peers succeeded does
        not qualify to compete against them — training more never costs you a
        slot, and every survivor is comparable by construction. With one
        challenger there is exactly one group, so this reduces to the rule above.

        ``len(cohort)`` is the ``k`` that sets the duel's family-wise alpha
        (``bootstrap_alpha / k``). It lives here, on the signed manifest, so the
        validator and ``cascade-audit`` derive it from one implementation and
        cannot drift — the receipt deliberately does NOT record the adjusted alpha
        (that would fail ``check_koth_params`` against the published
        ``[scoring]``). Note it counts the ADVANCED cohort, not the number
        evaluated: an inconclusive round stops early, but the alpha it was judged
        under was already fixed by the cohort size.
        """
        king_sizes = self.king_sizes()
        if not king_sizes:
            return [], []
        idx = {s: i for i, s in enumerate(king_sizes)}
        groups: dict[tuple[str, ...], list[tuple[str, dict[str, TrainedEntry]]]] = {}
        for hk, by_size in self.challenger_cohort():
            common = tuple(s for s in king_sizes if s in by_size)
            if common:
                groups.setdefault(common, []).append((hk, by_size))
        if not groups:
            return [], []
        best = min(groups, key=lambda c: (-len(c), tuple(idx[s] for s in c)))
        return groups[best], list(best)

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
        body["heat"] = heat_to_json(manifest.heat)
    # Same pattern as ``heat``: only present when the jittered mix served the
    # round, so every earlier manifest serialises byte-for-byte as before.
    if manifest.composition is not None:
        body["composition"] = manifest.composition
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
            duel_rank=int(e.get("duel_rank", 0)),
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
        composition=obj.get("composition"),
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
