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

RECEIPT_VERSION = 3
RECEIPT_STATUSES = ("scored", "rejected")


def _none_for_nan(x: float | None) -> float | None:
    """Canonical JSON is strict (``allow_nan=False``): NaN/±inf become None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _clean_per_horizon(ph: dict | None) -> dict | None:
    """``{horizon: {king, chal, win_rate, n}}`` → strict-JSON, sorted by horizon,
    NaN-scrubbed. ``None`` when empty."""
    if not ph:
        return None
    out: dict[str, dict] = {}
    for h in sorted(ph, key=lambda x: int(x)):
        r = ph[h] or {}
        out[str(int(h))] = {
            "king": _none_for_nan(r.get("king")),
            "chal": _none_for_nan(r.get("chal")),
            "win_rate": _none_for_nan(r.get("win_rate")),
            "n": int(r.get("n", 0) or 0),
        }
    return out


def _clean_per_domain_scores(pd: dict | None) -> dict | None:
    """``{domain: {king, chal, win_rate, n}}`` → strict-JSON, sorted by domain
    name, NaN-scrubbed. ``None`` when empty. The per-domain twin of
    :func:`_clean_per_horizon` (``eval.koth.per_domain_breakdown``)."""
    if not pd:
        return None
    out: dict[str, dict] = {}
    for dom in sorted(pd, key=str):
        r = pd[dom] or {}
        out[str(dom)] = {
            "king": _none_for_nan(r.get("king")),
            "chal": _none_for_nan(r.get("chal")),
            "win_rate": _none_for_nan(r.get("win_rate")),
            "n": int(r.get("n", 0) or 0),
        }
    return out


def _clean_per_domain(pd: dict | None) -> dict | None:
    """``{domain: (win_rate, n)}`` → strict-JSON ``{domain: [win_rate|None, n]}``.

    Sorted, NaN-scrubbed, and list-valued so it round-trips byte-identically
    through ``json`` (tuples serialise as lists anyway). ``None`` when empty."""
    if not pd:
        return None
    out: dict[str, list] = {}
    for dom in sorted(pd, key=str):
        wr, n = pd[dom]
        out[str(dom)] = [_none_for_nan(wr), int(n)]
    return out


# The per-challenger shadow diagnostics a cohort receipt publishes under
# ``VerdictRecord.cohort_stats``. Every key is recomputable from the signed
# ``entry_scores`` (the audit replays them); none of them gates.
COHORT_STAT_KEYS = ("geomean", "win_rate", "wilcoxon_p", "boot_p50", "boot_p95",
                    "per_domain_win_rate")


def cohort_stats_of(result) -> dict:
    """One challenger's published diagnostics from its ``eval.koth.RoundResult``.

    Same fields the headline verdict carries for the decided challenger
    (``chal_geomean``, ``win_rate``, ``wilcoxon_p``, ``boot_p50``, ``boot_p95``,
    ``per_domain_win_rate``), NaN-scrubbed and strict-JSON shaped so the dict
    round-trips byte-identically through ``load_receipt``.
    """
    return _clean_stats({
        "geomean": getattr(result, "chal_geomean", None),
        "win_rate": getattr(result, "win_rate", None),
        "wilcoxon_p": getattr(result, "wilcoxon_p", None),
        "boot_p50": getattr(result, "boot_p50", None),
        "boot_p95": getattr(result, "boot_p95", None),
        "per_domain_win_rate": getattr(result, "per_domain_win_rate", None),
    })


def _clean_stats(stats: dict | None) -> dict:
    """Canonical shape of one ``cohort_stats`` entry: every key present (None
    when unknown), floats finite-or-None, ``per_domain_win_rate`` through
    :func:`_clean_per_domain`. Idempotent — applied on write AND on load."""
    stats = stats or {}
    out: dict = {}
    for key in COHORT_STAT_KEYS:
        if key == "per_domain_win_rate":
            out[key] = _clean_per_domain(stats.get(key))
        else:
            out[key] = _none_for_nan(stats.get(key))
    return out


def _clean_cohort_stats(by_hotkey: dict | None) -> dict | None:
    """``{challenger_hotkey: stats}`` → canonical, or None when empty."""
    if not by_hotkey:
        return None
    return {str(h): _clean_stats(s) for h, s in by_hotkey.items()}


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

    ``domain`` (DEC-CA-0050 P2) is the window's pool domain label, so an
    auditor can re-derive per-domain statistics (the lane-B domain verdict and
    its guards, the ``per_domain_win_rate`` diagnostic) from the signed scores
    alone. DROP-WHEN-DEFAULT: ``None`` (an unlabeled pool, and every receipt
    archived before the field existed) is omitted from the canonical body, so
    those receipts hash byte-for-byte as before.
    """

    series_id: str
    channel: int
    mase: float
    qloss_per_q: tuple[float, ...]
    abs_target: float
    quantile_levels: tuple[float, ...]
    source: str | None = None
    domain: str | None = None

    @classmethod
    def from_score(cls, ws, *, keep_domain: bool = False) -> WindowScoreRecord:
        """``keep_domain`` stamps the window's pool domain. The validator sets
        it ONLY while lane B is active at the settlement's block
        (``ValidatorRunner._window_records``): the deployed pool already
        carries domain labels, so stamping unconditionally would change every
        receipt's bytes the day v2 deploys. Default off = pre-v2 records."""
        """From a :class:`cascade.eval.scoring.WindowScore`."""
        return cls(
            series_id=str(ws.series_id),
            channel=int(ws.channel),
            mase=float(ws.mase),
            qloss_per_q=tuple(float(q) for q in ws.qloss_per_q),
            abs_target=float(ws.abs_target),
            quantile_levels=tuple(float(q) for q in ws.quantile_levels),
            source=(str(ws.source) if ws.source else None),
            domain=(str(ws.domain) if keep_domain and getattr(ws, "domain", "") else None),
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
            domain=self.domain or "",
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


# KothParams keys omitted from the recorded ``VerdictRecord.params`` dict while
# at their inert defaults — so a receipt for a round judged under the
# historical rules hashes byte-for-byte as before the fields existed (the
# drop-when-default convention, applied INSIDE the params dict). NEVER change a
# listed default once shipped; ``check_koth_params`` re-fills dropped keys via
# ``KothParams`` defaults when comparing against chain.toml.
_PARAMS_DROP_WHEN_DEFAULT: dict[str, object] = {
    "margin_mode": "level",
    "margin_increment_floor": 0.01,
    # Init-baseline floor: off on every archived round.
    "init_gate_mode": "off",
    "init_gate_tolerance": 0.0,
    # Multivariate window aggregation (DEC-CA-0041): per-channel on every round
    # before mv_score_from_block, including all archived rounds.
    "mv_score": False,
    # DEC-CA-0054 bench gate: off on every archived round.
    "bench_gate_mode": "off",
    "bench_gate_min_items": 10,
    # DEC-CA-0061 bench-gate rule: the DEC-CA-0054 rule on every archived round,
    # and rule v2's thresholds at their defaults ([scoring] bench_gate_*).
    "bench_gate_rule": "dec-ca-0054",
    "bench_gate_tolerance": 0.01,
    "bench_gate_min_win_rate": 0.5,
}


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
    # Shadow diagnostics — recomputable from the signed ``entry_scores`` and
    # NEVER part of the gate (the LCB decides). Persisted so the dashboard and
    # ``cascade-audit`` can explain a verdict — e.g. a great geomean with a low
    # ``win_rate`` is a rare-but-big-wins/losses signal — without recomputation.
    n_clusters: int = 0
    win_rate: float | None = None
    wilcoxon_p: float | None = None
    per_domain_win_rate: dict | None = None
    # Bootstrap spread of the relative improvement: median (p50) and 95th pct.
    # The LCB above is the same distribution's 5th pct; a big p50−lcb gap flags a
    # fragile verdict (better point estimate riding a heavy tail). Display only.
    boot_p50: float | None = None
    boot_p95: float | None = None

    # ── the cohort duel (DEC-CA-0012) ───────────────────────────────────────
    # Recorded so a third party can check the multiplicity math without trusting
    # the validator or re-deriving anything: ``cohort_k`` is the k that set the
    # per-challenger alpha, and ``cohort_lcbs`` is every duelled challenger's LCB
    # under it. With ``margin`` (already here) that is the complete input to the
    # selection rule — who cleared, and whether the crowned one was the best.
    #
    # Both are DROPPED from the canonical body at their defaults (see
    # :func:`_verdict_body`), which is what makes adding them safe; see the note
    # below. ``cohort_k = 0`` means "not a cohort round" — every single-challenger
    # round, i.e. every round before this shipped.
    cohort_k: int = 0
    cohort_lcbs: dict | None = None   # {challenger_hotkey: lcb at alpha/k}
    # Every duelled challenger's shadow diagnostics — the same set the headline
    # verdict carries for the decided challenger only (geomean, win_rate,
    # wilcoxon_p, boot_p50/p95, per-domain win rates), keyed by hotkey. Display
    # and audit material, never a gate; recomputable from ``entry_scores`` and
    # checked by ``cascade-audit``'s duel-cohort replay. Absent (dropped from the
    # signed body) on single-challenger rounds and on every receipt archived
    # before it shipped, so their signatures still verify.
    cohort_stats: dict | None = None  # {challenger_hotkey: {COHORT_STAT_KEYS}}

    # ── init-baseline floor ([scoring] init_gate_mode) ──────────────────────
    # The shared warm-start init's observed geomean on the verdict windows and
    # the floor verdict for the decided challenger. Both None on every round
    # judged with the gate off — and DROPPED from the canonical body then
    # (see ``_verdict_body``), so archived receipt signatures survive.
    init_baseline_geomean: float | None = None
    init_floor_passed: bool | None = None
    # DEC-CA-0054 bench gate outcome; None (not run) is dropped from the body.
    bench_gate_passed: bool | None = None
    # DEC-CA-0061: the rule, its three numbers (avg_ratio, worst_lcb,
    # win_rate) and per-suite ratios that judged (``BenchGateResult.detail``);
    # None (DEC-CA-0054 / not run) is dropped.
    bench_gate_detail: dict | None = None

    # ── scored horizon ladder / duel-only display fields ────────────────────
    # ``per_horizon``: the decided pair's per-rung breakdown on a ladder round
    # (``{horizon: {king, chal, win_rate, n}}``); ``cohort_geomeans`` and
    # ``cohort_per_horizon``: the same for EVERY judged challenger, keyed by
    # hotkey, so a dashboard can list the whole seated field. All None on a
    # single-horizon / pre-ladder round and DROPPED from the canonical body
    # then (``_verdict_body``), so archived signatures survive. Display only.
    per_horizon: dict | None = None
    cohort_geomeans: dict | None = None
    cohort_per_horizon: dict | None = None
    # ``per_domain`` / ``cohort_per_domain``: the same breakdown by pool DOMAIN
    # (``{domain: {king, chal, win_rate, n}}``) — the decided pair's, and every
    # judged challenger's keyed by hotkey — so a miner reads which domains they
    # beat the king in and by how much. ``per_domain_win_rate`` above is the
    # how-often half; this is the by-how-much half. None on a domain-less pool
    # and on every receipt archived before it shipped; DROPPED from the
    # canonical body then. Display only.
    per_domain: dict | None = None
    cohort_per_domain: dict | None = None

    # NOTE on adding fields here. ``asdict`` of this dataclass goes into
    # ``RoundReceipt.canonical_body`` — the SIGNED bytes — so a field that always
    # serialises re-writes every archived receipt with bytes that were never
    # signed, and the whole public audit trail fails verification. Bumping
    # ``RECEIPT_VERSION`` is not an escape: ``load_receipt`` rejects any version
    # but the current one, so archived receipts would stop loading at all.
    #
    # The ONLY safe shape is drop-when-default, via :func:`_verdict_body` — the
    # same convention ``manifest._entry_body`` uses for ``bench_scores`` /
    # ``duel_rank``. An archived receipt re-serialises byte-for-byte because its
    # new fields sit at their defaults and are omitted; the payload grows only for
    # rounds that genuinely carry the data. ``tests/fixtures/round_receipt_v3.json``
    # is the regression guard: it is a SIGNED fixture, so if a new field leaks into
    # the body its signature check fails.
    #
    # This does not extend to scoring-RULE changes. Those have no natural "absent"
    # default (every round used some rule), so recording one would grow the body
    # for archived receipts too. They stay re-derived by
    # ``cascade.audit.checks.check_verdict``, which replays under each known rule
    # and reports which reproduces the LCB.

    @classmethod
    def from_round(
        cls, result, transition, *, params, bootstrap_seed, king_tenure_rounds: int = 0,
        cohort_k: int = 0, cohort_lcbs: dict | None = None,
        cohort_geomeans: dict | None = None, cohort_per_horizon: dict | None = None,
        cohort_stats: dict | None = None, cohort_per_domain: dict | None = None,
    ) -> VerdictRecord:
        """From an ``eval.koth.RoundResult`` + ``validator.state.StateTransition``.

        ``params`` is the UNMODIFIED ``chain.toml [scoring]`` set even on a cohort
        round — ``check_koth_params`` asserts exactly that — so the per-challenger
        alpha is published as ``cohort_k`` and the resulting bounds as
        ``cohort_lcbs`` instead of by mutating ``params.bootstrap_alpha``.

        Params keys at their inert defaults are DROPPED (the canonical-body
        convention applied inside the params dict): a round judged under the
        historical rules serialises byte-for-byte as it did before the fields
        existed, and ``check_koth_params`` normalises through ``KothParams`` so
        the dropped keys compare at their defaults.
        """
        p = dict(asdict(params))
        for key, default in _PARAMS_DROP_WHEN_DEFAULT.items():
            if p.get(key) == default:
                p.pop(key, None)
        return cls(
            params=p,
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
            n_clusters=int(getattr(result, "n_clusters", 0) or 0),
            win_rate=_none_for_nan(getattr(result, "win_rate", None)),
            wilcoxon_p=_none_for_nan(getattr(result, "wilcoxon_p", None)),
            per_domain_win_rate=_clean_per_domain(getattr(result, "per_domain_win_rate", None)),
            boot_p50=_none_for_nan(getattr(result, "boot_p50", None)),
            boot_p95=_none_for_nan(getattr(result, "boot_p95", None)),
            cohort_k=int(cohort_k or 0),
            cohort_lcbs=(
                {str(h): _none_for_nan(float(v)) for h, v in cohort_lcbs.items()}
                if cohort_lcbs else None
            ),
            cohort_stats=_clean_cohort_stats(cohort_stats),
            init_baseline_geomean=_none_for_nan(
                getattr(result, "baseline_geomean", None)),
            init_floor_passed=getattr(result, "init_floor_passed", None),
            bench_gate_passed=getattr(result, "bench_gate_passed", None),
            bench_gate_detail=getattr(result, "bench_gate_detail", None),
            per_horizon=_clean_per_horizon(getattr(result, "per_horizon", None)),
            cohort_geomeans=(
                {str(h): _none_for_nan(float(v)) for h, v in cohort_geomeans.items()}
                if cohort_geomeans else None
            ),
            cohort_per_horizon=(
                {str(h): _clean_per_horizon(ph) for h, ph in cohort_per_horizon.items()
                 if _clean_per_horizon(ph)}
                or None
            ) if cohort_per_horizon else None,
            per_domain=_clean_per_domain_scores(getattr(result, "per_domain", None)),
            cohort_per_domain=(
                {str(h): _clean_per_domain_scores(pd) for h, pd in cohort_per_domain.items()
                 if _clean_per_domain_scores(pd)}
                or None
            ) if cohort_per_domain else None,
        )


def _verdict_body(v: VerdictRecord | None) -> dict | None:
    """One verdict's canonical dict, with the cohort fields omitted at their
    defaults.

    This is what makes the DEC-CA-0012 fields safe to add to a SIGNED structure: a
    single-challenger round — every round archived before the cohort duel existed —
    carries ``cohort_k = 0`` and ``cohort_lcbs = None``, both dropped, so its body
    serialises byte-for-byte as it did before the fields were declared and its
    signature still verifies. Mirrors ``manifest._entry_body``.
    """
    if v is None:
        return None
    d = asdict(v)
    if not d.get("cohort_k"):
        d.pop("cohort_k", None)
    if not d.get("cohort_lcbs"):
        d.pop("cohort_lcbs", None)
    if not d.get("cohort_stats"):
        d.pop("cohort_stats", None)
    # Init-baseline floor fields: absent ⇒ dropped, so a receipt from a
    # gate-off round serialises byte-for-byte as before the fields existed.
    if d.get("init_baseline_geomean") is None:
        d.pop("init_baseline_geomean", None)
    if d.get("init_floor_passed") is None:
        d.pop("init_floor_passed", None)
    if d.get("bench_gate_passed") is None:
        d.pop("bench_gate_passed", None)
    if d.get("bench_gate_detail") is None:
        d.pop("bench_gate_detail", None)
    for key in ("per_horizon", "cohort_geomeans", "cohort_per_horizon",
                "per_domain", "cohort_per_domain"):
        if not d.get(key):
            d.pop(key, None)
    return d


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
    # Era context (DEC-CA-0043, drop-when-default so every archived receipt
    # keeps its signed bytes): the era's start block and the seed the era's
    # TRAINING seeds derive from (``block_seed(era.seed_block)`` — the
    # previous era's start block hash). ``base_seed`` / the eval draw stay
    # the settlement boundary's; only ``generation_seed`` / ``training_seed``
    # move to the era. Absent (0) before the gate, required after — the
    # audit checks both.
    era_start_block: int = 0
    era_base_seed: int = 0
    # Stake-weighted activation (DEC-CA-0045, drop-when-default): the
    # DEC-CA-0043 rollover block this validator resolved FROM VALIDATOR
    # SIGNALS (0 = none resolved, or the rollover is typed into chain.toml).
    # Stamped on every receipt from lock-in on, so the audit replays each
    # round under the block the fleet decided, not the config it runs with.
    activation_block: int = 0
    # Dethrone bar v2 (DEC-CA-0049, drop-when-default): the
    # ``margin_v2_from_block`` this validator resolved FROM VALIDATOR SIGNALS
    # (0 = none resolved, or typed into chain.toml). Stamped from lock-in on so
    # the audit replays each verdict under the bar the fleet decided.
    margin_v2_block: int = 0
    # cascade v2 (DEC-CA-0050, drop-when-default): ``[v2] from_block`` when
    # this settlement judges a v2 era (its era starts at or after it), else 0
    # — so every pre-v2 receipt serialises byte-for-byte as before and the
    # audit replays each round under its own block's rules.
    v2_block: int = 0
    # Lane B (DEC-CA-0050, drop-when-default): this validator's verdict rows
    # for the settlement's expert manifest (None = lane B not active, so
    # every pre-lane-B receipt serialises and verifies byte-for-byte as before).
    lane_b: tuple[dict, ...] | None = None
    # Lane-B payout record (DEC-CA-0050, drop-when-default): present exactly
    # when the recorded vector is the combined lane-A + lane-B vector —
    # ``{"block", "era_blocks", "wins": [{domain, hotkey, uid, merge_block,
    # age, multiplier}]}`` (age in eras), the decay clock and every PAID
    # domain win (expired ones are unpaid, absent) — so the
    # audit rebuilds the vector exactly (each slot ≤ share/n × multiplier, the
    # unpaid share to burn). None on every pre-lane-B receipt.
    lane_b_pay: dict | None = None
    # DEC-CA-0055 (all drop-when-default, so every earlier receipt keeps its
    # signed bytes): ``prelaunch_burn_block`` — the pre-launch burn block this
    # validator resolved FROM VALIDATOR SIGNALS (0 = none, or typed);
    # ``emission`` — the emission record the vector was built under
    # (:func:`cascade.shared.emission.emission_record`: the policy, the
    # lane-A ladder state and the transition that produced it), present
    # whenever a DEC-CA-0055 rule shaped the vector; ``king_check`` — the
    # v2 king check judged at this settlement (``{"result": "pass" | "fail" |
    # "deferred", "round_id", "king_pointer", "reasons", ...}``).
    prelaunch_burn_block: int = 0
    emission: dict | None = None
    king_check: dict | None = None
    # DEC-CA-0058 (drop-when-default: only when a king check is stamped AND it
    # scored rows the receipt does not already carry): the pool rows the king
    # check fed ``evaluate_round`` that are not in ``entry_scores`` — role
    # "king" (a settlement with no duel, so no king rows) and/or role "init"
    # (the current init, when the verdict scored no ``baseline`` rows) — so
    # the audit replays the check's LCB instead of trusting its result.
    king_check_scores: tuple[EntryScores, ...] = ()
    # Lane C (DEC-CA-0056, drop-when-default): this validator's verdict rows
    # for the settlement's bound lane-C manifest (``[]`` when nothing was
    # judged), and the payout record ``{"block", "era_blocks", "share",
    # "slots", "wins": [{target, hotkey, uid, win_block, age, multiplier}]}``
    # present exactly when the vector carries lane C — so the audit rebuilds
    # the 3-lane vector. None on every pre-lane-C receipt (bytes unchanged).
    lane_c: tuple[dict, ...] | None = None
    lane_c_pay: dict | None = None
    # DEC-CA-0057 (drop-when-default, so every receipt judged without an
    # overlay keeps its exact bytes): the owner-signed network-config overlay
    # this settlement was judged under — its ``seq`` and body digest; the
    # audit fetches ``config/network/seq-<n>.json`` and checks the digest.
    netconfig_seq: int = 0
    netconfig_digest: str = ""
    # DEC-CA-0058 (drop-when-default; stamped only once ``[netconfig]
    # owner_hotkey`` is set, so every receipt before the main/satellite
    # topology keeps its exact bytes): ``manifest_sha256`` — sha256 of the
    # EXACT published ``manifests/round-<id>.json`` text this settlement
    # judged (the validator's ``last_handled_manifest_sha``);
    # ``state_prior_sha256`` — :func:`cascade.shared.settlement_state.
    # state_digest` of the settlement state BEFORE it was applied;
    # ``state_post`` — that state AFTER it (king, tenure, court, ladder,
    # lane-B/C slots and pending). A satellite checks each receipt's prior
    # digest against the previous receipt's posterior.
    manifest_sha256: str = ""
    state_prior_sha256: str = ""
    state_post: dict | None = None
    # DEC-CA-0063 (drop-when-default: only on a settlement whose lane-A bench
    # rows come from the main validator): the validator bench report the
    # bench gate / king check read — ``{"key", "sha256", "signer"}``, the
    # store key, sha256 of its EXACT published text and the main validator
    # hotkey it verifies against — so the audit (and a satellite) replays
    # from exactly those bytes. None on every trainer-sourced receipt.
    bench_report: dict | None = None
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
            "entry_scores": [_entry_scores_body(e) for e in self.entry_scores],
            "verdict": _verdict_body(self.verdict),
            "reward_uids": list(self.reward_uids),
            "weights": list(self.weights),
            "reject_reason": self.reject_reason,
            "validator_hotkey": self.validator_hotkey,
        }
        if self.era_start_block:
            body["era_start_block"] = int(self.era_start_block)
        if self.era_base_seed:
            body["era_base_seed"] = int(self.era_base_seed)
        if self.activation_block:
            body["activation_block"] = int(self.activation_block)
        if self.margin_v2_block:
            body["margin_v2_block"] = int(self.margin_v2_block)
        if self.v2_block:
            body["v2_block"] = int(self.v2_block)
        if self.lane_b is not None:
            body["lane_b"] = [dict(r) for r in self.lane_b]
        if self.lane_b_pay is not None:
            body["lane_b_pay"] = json.loads(json.dumps(self.lane_b_pay, sort_keys=True))
        if self.prelaunch_burn_block:
            body["prelaunch_burn_block"] = int(self.prelaunch_burn_block)
        if self.emission is not None:
            body["emission"] = json.loads(json.dumps(self.emission, sort_keys=True))
        if self.king_check is not None:
            body["king_check"] = json.loads(json.dumps(self.king_check, sort_keys=True))
        if self.king_check_scores:
            body["king_check_scores"] = [_entry_scores_body(e) for e in self.king_check_scores]
        if self.lane_c is not None:
            body["lane_c"] = [json.loads(json.dumps(r, sort_keys=True)) for r in self.lane_c]
        if self.lane_c_pay is not None:
            body["lane_c_pay"] = json.loads(json.dumps(self.lane_c_pay, sort_keys=True))
        if self.netconfig_seq:
            body["netconfig_seq"] = int(self.netconfig_seq)
            body["netconfig_digest"] = str(self.netconfig_digest)
        if self.manifest_sha256:
            body["manifest_sha256"] = str(self.manifest_sha256)
        if self.state_prior_sha256:
            body["state_prior_sha256"] = str(self.state_prior_sha256)
        if self.state_post is not None:
            body["state_post"] = json.loads(json.dumps(self.state_post, sort_keys=True))
        if self.bench_report is not None:
            body["bench_report"] = json.loads(json.dumps(self.bench_report, sort_keys=True))
        return json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")


def _entry_scores_body(e: EntryScores) -> dict:
    """``asdict(e)`` with each window score's ``domain`` dropped at its
    default (``None``) — the drop-when-default convention applied inside the
    per-window records, so receipts of unlabeled pools (and every archived
    receipt) keep their exact canonical bytes."""
    d = asdict(e)
    for rec in d["scores"]:
        if rec.get("domain") is None:
            rec.pop("domain", None)
    return d


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
    era_start_block: int = 0,
    era_base_seed: int = 0,
    activation_block: int = 0,
    margin_v2_block: int = 0,
    lane_b: list[dict] | None = None,
    lane_b_pay: dict | None = None,
    v2_block: int = 0,
    prelaunch_burn_block: int = 0,
    emission: dict | None = None,
    king_check: dict | None = None,
    king_check_scores: tuple[EntryScores, ...] = (),
    lane_c: list[dict] | None = None,
    lane_c_pay: dict | None = None,
    netconfig_seq: int = 0,
    netconfig_digest: str = "",
    manifest_sha256: str = "",
    state_prior_sha256: str = "",
    state_post: dict | None = None,
    bench_report: dict | None = None,
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
        era_start_block=int(era_start_block or 0),
        era_base_seed=int(era_base_seed or 0),
        activation_block=int(activation_block or 0),
        margin_v2_block=int(margin_v2_block or 0),
        v2_block=int(v2_block or 0),
        lane_b=None if lane_b is None else tuple(dict(r) for r in lane_b),
        lane_b_pay=None if lane_b_pay is None else dict(lane_b_pay),
        prelaunch_burn_block=int(prelaunch_burn_block or 0),
        emission=None if emission is None else dict(emission),
        king_check=None if king_check is None else dict(king_check),
        king_check_scores=tuple(king_check_scores),
        lane_c=None if lane_c is None else tuple(dict(r) for r in lane_c),
        lane_c_pay=None if lane_c_pay is None else dict(lane_c_pay),
        netconfig_seq=int(netconfig_seq or 0),
        netconfig_digest=str(netconfig_digest or "") if netconfig_seq else "",
        manifest_sha256=str(manifest_sha256 or ""),
        state_prior_sha256=str(state_prior_sha256 or ""),
        state_post=None if state_post is None else dict(state_post),
        bench_report=None if bench_report is None else dict(bench_report),
    )


def dump_receipt(receipt: RoundReceipt) -> str:
    """Serialise a receipt (including signature) to a JSON string."""
    body = json.loads(receipt.canonical_body().decode("utf-8"))
    body["signature"] = receipt.signature
    return json.dumps(body, indent=2, sort_keys=True)


def _received_body(obj: dict) -> bytes:
    """The signed bytes of a received receipt: its JSON minus ``signature``,
    serialised exactly as :meth:`RoundReceipt.canonical_body` serialises (the
    published JSON is ``canonical_body`` + signature, re-indented)."""
    body = {k: v for k, v in obj.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def receipt_has_unknown_fields(receipt: RoundReceipt) -> bool:
    """True when a loaded receipt carries fields this code does not model —
    written by NEWER code than is running here (e.g. a new drop-when-default
    stamp). The receipt still verifies (see :func:`verify_receipt_signature`);
    this is the operator's cue to restart on current code. False for a
    receipt that was not loaded from text."""
    received = getattr(receipt, "_received_body", None)
    return received is not None and received != receipt.canonical_body()


def load_receipt(text: str) -> RoundReceipt:
    """Parse a receipt JSON string. Raises ``ValueError`` on schema problems.

    The received body is kept on the returned receipt, so the signature is
    checked over the bytes the validator actually signed. Fields this code
    does not know (written by newer code) are ignored for parsing but stay
    covered by the signature, so an older reader never fails verification
    on a valid receipt that carries a newer stamp."""
    obj = json.loads(text)
    receipt = _receipt_from_obj(obj)
    object.__setattr__(receipt, "_received_body", _received_body(obj))
    return receipt


def _entry_scores_from_obj(e: dict) -> EntryScores:
    return EntryScores(
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
                domain=(str(s["domain"]) if s.get("domain") else None),
            )
            for s in e["scores"]
        ),
    )


def _receipt_from_obj(obj: dict) -> RoundReceipt:
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
        entry_scores=tuple(_entry_scores_from_obj(e) for e in obj.get("entry_scores", ())),
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
            n_clusters=int(verdict.get("n_clusters", 0) or 0),
            win_rate=(None if verdict.get("win_rate") is None else float(verdict["win_rate"])),
            wilcoxon_p=(None if verdict.get("wilcoxon_p") is None
                        else float(verdict["wilcoxon_p"])),
            per_domain_win_rate=_clean_per_domain(verdict.get("per_domain_win_rate")),
            boot_p50=(None if verdict.get("boot_p50") is None else float(verdict["boot_p50"])),
            boot_p95=(None if verdict.get("boot_p95") is None else float(verdict["boot_p95"])),
            # Absent on every pre-cohort receipt; the defaults round-trip back to
            # an omitted key in _verdict_body, so the signed bytes are preserved.
            cohort_k=int(verdict.get("cohort_k", 0) or 0),
            cohort_lcbs=(
                {str(h): (None if v is None else float(v))
                 for h, v in verdict["cohort_lcbs"].items()}
                if verdict.get("cohort_lcbs") else None
            ),
            cohort_stats=_clean_cohort_stats(verdict.get("cohort_stats")),
            init_baseline_geomean=(
                None if verdict.get("init_baseline_geomean") is None
                else float(verdict["init_baseline_geomean"])
            ),
            init_floor_passed=verdict.get("init_floor_passed"),
            # DEC-CA-0054 / DEC-CA-0061: absent ⇒ None ⇒ dropped again.
            bench_gate_passed=verdict.get("bench_gate_passed"),
            bench_gate_detail=(dict(verdict["bench_gate_detail"])
                               if isinstance(verdict.get("bench_gate_detail"), dict) else None),
            per_horizon=_clean_per_horizon(verdict.get("per_horizon")),
            cohort_geomeans=(
                {str(h): (None if v is None else float(v))
                 for h, v in verdict["cohort_geomeans"].items()}
                if verdict.get("cohort_geomeans") else None
            ),
            cohort_per_horizon=(
                {str(h): _clean_per_horizon(ph)
                 for h, ph in verdict["cohort_per_horizon"].items()}
                if verdict.get("cohort_per_horizon") else None
            ),
            per_domain=_clean_per_domain_scores(verdict.get("per_domain")),
            cohort_per_domain=(
                {str(h): _clean_per_domain_scores(pd)
                 for h, pd in verdict["cohort_per_domain"].items()}
                if verdict.get("cohort_per_domain") else None
            ),
        ) if verdict else None,
        reward_uids=tuple(int(u) for u in obj.get("reward_uids", ())),
        weights=tuple(float(w) for w in obj.get("weights", ())),
        reject_reason=obj.get("reject_reason"),
        validator_hotkey=str(obj.get("validator_hotkey", "")),
        era_start_block=int(obj.get("era_start_block", 0) or 0),
        era_base_seed=int(obj.get("era_base_seed", 0) or 0),
        activation_block=int(obj.get("activation_block", 0) or 0),
        margin_v2_block=int(obj.get("margin_v2_block", 0) or 0),
        v2_block=int(obj.get("v2_block", 0) or 0),
        lane_b=(None if obj.get("lane_b") is None
                else tuple(dict(r) for r in obj["lane_b"])),
        lane_b_pay=(dict(obj["lane_b_pay"]) if isinstance(obj.get("lane_b_pay"), dict)
                    else None),
        prelaunch_burn_block=int(obj.get("prelaunch_burn_block", 0) or 0),
        emission=(dict(obj["emission"]) if isinstance(obj.get("emission"), dict) else None),
        king_check=(dict(obj["king_check"]) if isinstance(obj.get("king_check"), dict)
                    else None),
        king_check_scores=tuple(_entry_scores_from_obj(e)
                                for e in obj.get("king_check_scores", ())),
        lane_c=(None if obj.get("lane_c") is None
                else tuple(dict(r) for r in obj["lane_c"])),
        lane_c_pay=(dict(obj["lane_c_pay"]) if isinstance(obj.get("lane_c_pay"), dict)
                    else None),
        netconfig_seq=int(obj.get("netconfig_seq", 0) or 0),
        netconfig_digest=str(obj.get("netconfig_digest", "") or ""),
        manifest_sha256=str(obj.get("manifest_sha256", "") or ""),
        state_prior_sha256=str(obj.get("state_prior_sha256", "") or ""),
        state_post=(dict(obj["state_post"]) if isinstance(obj.get("state_post"), dict)
                    else None),
        bench_report=(dict(obj["bench_report"]) if isinstance(obj.get("bench_report"), dict)
                      else None),
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
    from .v2_status import v2_index_fields as _v2_index_fields

    entries = receipt.manifest.get("entries", []) if isinstance(receipt.manifest, dict) else []

    # The challenger the verdict belongs to: the LAST one scored. Under the cohort
    # duel (DEC-CA-0012) a round can carry several, and the validator scores the
    # crowned clearer last — taking the first would attribute the round to a
    # challenger that did not take the throne.
    duelled = [es.hotkey for es in receipt.entry_scores if es.role == "challenger"]
    decided_hotkey = duelled[-1] if duelled else None

    def _gen_ref(role: str) -> str | None:
        want = decided_hotkey if role == "challenger" else None
        for e in entries:
            if not isinstance(e, dict) or e.get("role") != role:
                continue
            if want is None or e.get("miner_hotkey") == want:
                return e.get("gen_ref")
        return None

    def _scorer(role: str):
        want = decided_hotkey if role == "challenger" else None
        for es in receipt.entry_scores:
            if es.role == role and (want is None or es.hotkey == want):
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
            # How decisive the screen was (shadow; never gated the pick). A
            # leader_lcb at or below 0 means first and second were statistically
            # interchangeable on the heat's evidence.
            "leader_lcb": heat.get("leader_lcb"),
            "leader_p_best": next((e.get("p_best") for e in ents if isinstance(e, dict)
                                   and e.get("rank") == 1), None),
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
        # this round's runs initialised from the previous round's promoted
        # checkpoint (manifest warm_start_ckpt) — the dashboard counts the
        # consecutive chain of these to state the champion's cumulative
        # pretraining. Absent/empty = trained from scratch.
        "warm_start": bool(receipt.manifest.get("warm_start_ckpt")
                           if isinstance(receipt.manifest, dict) else False),
        "n_participants": len(receipt.participants),
        # verdict headline
        "n_windows": (receipt.eval_context.n_windows if receipt.eval_context
                      else (v.n_windows if v else None)),
        "n_clusters": n_clusters,
        "king_geomean": v.king_geomean if v else None,
        "chal_geomean": v.chal_geomean if v else None,
        "lcb": v.lcb if v else None,
        "margin": v.margin if v else None,
        # Cohort duel (DEC-CA-0012). ``cohort_k`` is 0/absent on a single-challenger
        # round. ``cohort_alpha`` is derived, not stored: the correction tightens
        # the bootstrap QUANTILE (the LCB becomes this percentile of the paired
        # distribution) while ``margin`` above stays flat.
        "cohort_k": (v.cohort_k or None) if v else None,
        "cohort_alpha": (
            v.params.get("bootstrap_alpha", 0) / v.cohort_k
            if v and v.cohort_k else None
        ),
        "cohort_lcbs": (v.cohort_lcbs or None) if v else None,
        # shadow diagnostics that explain the LCB (see VerdictRecord) — the
        # geomean-vs-winrate pairing is what exposes a rare-but-big-wins verdict
        "win_rate": v.win_rate if v else None,
        "wilcoxon_p": v.wilcoxon_p if v else None,
        "per_domain_win_rate": v.per_domain_win_rate if v else None,
        # scored-ladder breakdown (None before the ladder) and the whole
        # judged cohort's observed geomeans — the "all miners" duel view
        "per_horizon": v.per_horizon if v else None,
        "cohort_geomeans": v.cohort_geomeans if v else None,
        "cohort_per_horizon": v.cohort_per_horizon if v else None,
        # per-domain scores — the decided pair's and every judged challenger's
        # ("which domains did I beat the king in, and by how much")
        "per_domain": v.per_domain if v else None,
        "cohort_per_domain": v.cohort_per_domain if v else None,
        "boot_p50": v.boot_p50 if v else None,
        "boot_p95": v.boot_p95 if v else None,
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
        # DEC-CA-0045: the rollover block this validator resolved from
        # validator signals (0 = none / typed into chain.toml).
        "activation_block": int(receipt.activation_block or 0),
        # DEC-CA-0049: the margin-v2 block resolved from validator signals.
        "margin_v2_block": int(receipt.margin_v2_block or 0),
        **_v2_index_fields(receipt),
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
    from .sr25519 import verify_hotkey  # raises SignatureVerifierUnavailable, never False
    # A receipt loaded from text verifies over the bytes it arrived with, so a
    # field added by newer code cannot break an older reader. A constructed
    # (or ``replace``-d) receipt carries no received body and verifies over
    # its own canonical body — any field tamper still kills the signature.
    body = getattr(receipt, "_received_body", None) or receipt.canonical_body()
    return verify_hotkey(validator_hotkey, body, receipt.signature)
