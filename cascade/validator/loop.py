"""Validator loop — manifest → eval → KOTH decision → weights.

The validator never trains. Each round it:

1. Reads the current :class:`TrainingManifest` from the owner dataset repo and
   verifies its signature + that king and challenger share the contract digest
   (the controlled-experiment guarantee).
2. Pulls the king's and challenger's trained checkpoints and scores both on the
   *same* held-out eval windows.
3. Runs the paired-bootstrap KOTH verdict and folds it into the sticky
   champion state (``dethrone_cp`` consecutive wins to take the throne;
   ``dethrone_cp = 1`` makes it single-round).
4. Sets weights: equal share across the current king plus up to
   ``[scoring] reward_prior_kings`` registered prior kings (teutonic-style),
   collapsing to winner-take-all when ``reward_prior_kings = 0``.

The pure orchestration in :meth:`ValidatorRunner.process_round` is testable by
injecting ``evaluate_fn`` and ``windows``; HF + torch + chain are isolated
behind the defaults.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ..eval.koth import RoundResult, cohort_maxt_lcb_map, evaluate_round
from ..eval.scoring import WindowScore
from ..eval.window import EvalWindow
from ..shared.activation import (
    ActivationRecord,
    ActivationStore,
    Tally,
)
from ..shared.config import (
    ChainConfig,
    cohort_maxt_active,
    cohort_maxt_increment_active,
    effective_epoch_blocks,
)
from ..shared.emission import crown_units
from ..shared.era import (
    bench_gate_mode_at,
    bench_gate_rule_at,
    legacy_king_anchor,
    settlement_contract_block,
    tenure_blocks_active,
    tenure_rounds_at,
    v2_settlement_active,
)
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    load_manifest,
)
from ..shared.receipt import (
    EntryScores,
    cohort_stats_of,
)
from . import state as state_mod
from .activation_mixin import ActivationMixin
from .bench import BENCH_REPORT_RETRY_ROUNDS, CascadeBenchMixin  # noqa: F401 — re-export
from .era_gate import EraGateMixin, _merge_error_type, _NoMergeError  # noqa: F401 — re-export
from .evaluation import EvaluationMixin
from .gating import POOL_PIN_READ_GRACE_SECONDS, ManifestGateMixin  # noqa: F401 — re-export
from .king_check import KingCheckMixin
from .lane_b import LaneBMixin, LaneBTransient, _LaneBRound
from .lane_c import LaneCMixin, LaneCTransient
from .manifest_chain import (  # noqa: F401 — re-exported for callers of cascade.validator.loop
    CHAIN_WALK_DEPTH,
    CHAIN_WALK_MAX_DEPTH,
    ManifestChain,
    chain_manifests,
    participants_from_commitments,
    ref_as_of,
)
from .marketcast import MarketCastMixin
from .promotion_gate import PromotionGateMixin
from .receipts import ReceiptMixin
from .state import ChampionState, StateTransition
from .validator_bench import ValidatorBenchHold, ValidatorBenchMixin
from .weights import WEIGHTS_RATE_LIMIT_BLOCKS, WeightsMixin  # noqa: F401 — re-export

if TYPE_CHECKING:
    from ..trainer.remote import RemoteHost
    from .cascade import CascadeController

log = logging.getLogger("cascade.validator")

# Resolve a trained entry to its per-window scores on the eval set.
EvaluateFn = Callable[[TrainedEntry, list[EvalWindow]], list[WindowScore]]
# Resolve a trained entry to its gift-eval ratio rows for the public-benchmark
# gate: ``{"status", "rows", "revision"}`` (see ``eval.benchmarks.run_gift_rows``)
# or ``None`` when the sidecar produced nothing.
GiftRowsFn = Callable[[TrainedEntry], dict | None]


@dataclass(frozen=True)
class RoundOutcome:
    result: RoundResult
    transition: StateTransition
    # Every per-window score that fed the verdict, one record per evaluated
    # (role, size) entry in evaluation order — threaded out so the live loop can
    # publish them in the round's signed public receipt (cascade.shared.receipt).
    entry_scores: tuple[EntryScores, ...] = ()
    # The king's tenure AT decision time (it set the margin); recorded in the
    # receipt so an auditor can recompute margin_for_tenure without validator state.
    king_tenure_rounds: int = 0
    # The cohort this round judged (DEC-CA-0012), best-clearer-first among those
    # that cleared. ``decided_hotkey`` is the crowned challenger — or, when none
    # cleared, the cohort's best point estimate, i.e. the challenger the verdict
    # in ``result`` belongs to. Internal only: NOT receipt fields (VerdictRecord
    # cannot grow — see cascade/shared/receipt.py), and an auditor re-derives all
    # of this from the signed manifest plus ``entry_scores``.
    decided_hotkey: str | None = None
    duelled_hotkeys: tuple[str, ...] = ()
    # Published on the receipt (DEC-CA-0012): the k that set the per-challenger
    # alpha, and every duelled challenger's LCB under it. With the verdict's
    # ``margin`` these are the complete input to the selection rule, so a third
    # party checks "who cleared, and was the crowned one the best" without
    # re-running the bootstrap or trusting this validator. ``cohort_k = 0`` on a
    # single-challenger round keeps the signed body byte-identical to a
    # pre-cohort receipt.
    cohort_k: int = 0
    cohort_lcbs: dict[str, float] = field(default_factory=dict)
    # Every judged challenger's observed geomean and per-horizon breakdown
    # (``{hotkey: {horizon: {king, chal, win_rate, n}}}``) — display fields
    # for the "all miners" duel view; the verdict itself is unchanged.
    cohort_geomeans: dict[str, float] = field(default_factory=dict)
    cohort_per_horizon: dict[str, dict] = field(default_factory=dict)
    # … and per-DOMAIN breakdown (``{hotkey: {domain: {king, chal, win_rate,
    # n}}}``): which domains each challenger beat the king in, and by how much.
    cohort_per_domain: dict[str, dict] = field(default_factory=dict)
    # Every duelled challenger's shadow diagnostics (geomean, win_rate, …), keyed
    # by hotkey — what the headline verdict records for the decided challenger
    # only. Published on the receipt as ``cohort_stats`` (same drop-when-default
    # rule as ``cohort_lcbs``); display/audit material, never a gate.
    cohort_stats: dict[str, dict] = field(default_factory=dict)
    # Challengers that LOST by entry fault ([scoring] entry_fault_from_block):
    # ``{hotkey: reason}``. Internal; the receipt carries it in the note.
    entry_faults: dict[str, str] = field(default_factory=dict)
    # The units of ``result.lcb`` (:func:`cascade.shared.emission.crown_units`):
    # a crowning's DEC-CA-0055 start share is tiered on it in "increment"
    # units only. Internal; the ladder record stamps it and the audit
    # re-derives it from the receipt.
    lcb_units: str = "level"


@dataclass
class ValidatorRunner(
    ActivationMixin,
    ManifestGateMixin,
    EraGateMixin,
    PromotionGateMixin,
    EvaluationMixin,
    CascadeBenchMixin,
    ReceiptMixin,
    WeightsMixin,
    LaneBMixin,
    LaneCMixin,
    MarketCastMixin,
    KingCheckMixin,
    ValidatorBenchMixin,
):
    cfg: ChainConfig
    state: ChampionState = field(default_factory=ChampionState)
    evaluate_fn: EvaluateFn | None = None     # injected in tests; defaults to registry+torch
    gift_rows_fn: GiftRowsFn | None = None    # injected in tests; defaults to the sidecar bridge
    cache_dir: Path | None = None
    device: str = "cpu"
    # Optional resolver for the GPU eval-offload pod, called AT EACH offloaded
    # eval (see cascade.validator.eval_offload.make_eval_host_fn): the pod is
    # elastic — rented per round by the provisioner, torn down on the receipt —
    # so it must be re-resolved lazily, not captured at startup. ``None`` (or a
    # call returning ``None``) ⇒ that eval runs on ``device`` locally. The
    # wallet and every consensus decision stay on this box either way; the pod
    # is never used for the private-pool duel.
    eval_host_fn: Callable[[], RemoteHost | None] | None = None
    # Private contexts cross ONLY to this separately authorized trusted host.
    private_eval_host_fn: Callable[[], RemoteHost | None] | None = None
    private_eval_workers: int = 1
    private_eval_timeout_s: int = 3600
    verify_signatures: bool = True            # gate manifests on the trainer-hotkey signature
    # Cascade — king-reign promotion (see cascade.validator.cascade). When wired,
    # the reign clock is reset on each dethrone and every benched duel
    # checkpoint's public-benchmark numbers (GIFT-Eval/BOOM/TIME, read from the
    # trainer's signed settlement bench report, falling back to older
    # manifests' in-entry bench_scores) are logged as promotion candidates. The
    # validator never selects: the trainer declares the promoted set in a
    # signed PromotionRecord and the warm-start gate verifies it against the
    # envelope (DEC-CA-0013), re-crowning the same king on acceptance
    # (DEC-CA-0004). None ⇒ Cascade is disabled (pure KOTH).
    cascade: CascadeController | None = None
    # Cascade bench-report source (cascade.shared.bench_report): duck-types
    # ``get_text``. ``None`` ⇒ opened lazily from [storage] through
    # open_manifest_store, so the report reads ride the same R2/HF fallbacks
    # as the manifest. Injected in tests.
    bench_report_store: object | None = None
    # Era ref binding (DEC-CA-0043): returns each hotkey's FULL reveal history
    # (``client.poll_commitments(include_history=True)``) so an era manifest's
    # entries are verified against the ref revealed AS OF their train_block.
    # None ⇒ the binding check is skipped (tests / no chain); a raising fn is
    # a transient — the round is retried next poll, never latched.
    commitment_history_fn: Callable[[], list] | None = None
    # Manifests the chain walk may collect per poll; doubles (to
    # CHAIN_WALK_MAX_DEPTH) each time the walk fails to reach the last handled
    # settlement, resets when it does.
    chain_walk_depth: int = CHAIN_WALK_DEPTH
    # Block of the last SUCCESSFUL weight-set (a rejected push is not
    # stamped); drives the between-rounds freshness push in
    # _maybe_reassert_weights. None ⇒
    # never set this process, so the first live-loop tick re-asserts
    # immediately (a restart is also a manual "refresh last_update now").
    _last_weight_block: int | None = None
    # First-failure clock (time.monotonic) per ``<round>|lane_b`` key for an
    # UNREADABLE lane-B expert manifest. Within POOL_PIN_READ_GRACE_SECONDS the
    # settlement is re-tried on the next poll; past it lane A proceeds with the
    # lane-B part unfolded. In-memory on purpose: a restart merely restarts the
    # grace clock, which is harmless.
    _pin_read_first_failure: dict[str, float] = field(default_factory=dict, repr=False)
    # Lane-A gate read holds (pool pin / warm-start provenance / era refs),
    # keyed ``<round>|<gate>``: ``(first failure monotonic, attempts)``. A read
    # failure there NEVER latches a reject — it holds until the read succeeds;
    # this is only the visibility clock (see ManifestGateMixin._hold_read_failure).
    _read_holds: dict[str, tuple[float, int]] = field(default_factory=dict, repr=False)
    # Digests of manifests skipped WITHOUT being recorded (bad signature, older
    # than the handled position), so each is logged once, not every poll.
    _skipped_digests: set = field(default_factory=set, repr=False)
    # The in-process follow-up (receipt / benchmarks / Cascade step) of the
    # round named by ``state.followup_round_id``: ``(round_id, digest, fn)``.
    # Lost on a restart — the round is then re-judged from ``state.pre_round``.
    _followup: tuple | None = field(default=None, repr=False)
    # Why process_round returned None for a gated-in settlement, when it is not
    # the stock "no king/challenger pair" (e.g. every challenger entry-faulted).
    _round_none_reason: str | None = field(default=None, repr=False)
    # Rounds whose reigning-king checkpoint is still awaiting its bench report:
    # the report is published AFTER the manifest (the bench sweep runs
    # post-publish and takes ~an hour), so it routinely lands between this
    # round's cascade step and the next round's. Each entry retries at the
    # following rounds' cascade steps and expires after
    # BENCH_REPORT_RETRY_ROUNDS misses. In-memory on purpose: losing it on a
    # restart just means those rounds contribute no bench numbers, which
    # Cascade tolerates (promotion selects over the rounds that have them).
    _pending_bench: list = field(default_factory=list, repr=False)
    # Per-round cache of verified bench reports (positive results only, small
    # FIFO): one round's cascade step joins several entries — and a promotion
    # verification several members — against the SAME report, and each miss is
    # a store GET + JSON parse + signature verification. Negative results are
    # deliberately NOT cached: "not there yet" is the normal pre-publish state
    # the pending-bench queue exists to re-probe.
    _bench_report_cache: dict = field(default_factory=dict, repr=False)
    # Stake-weighted activation (DEC-CA-0045): the persisted decision, the
    # last boundary tally (for status/chain.json), and the hotkey this
    # validator signals from. ``activation_store`` None ⇒ nothing persists
    # (tests); the loop still resolves in memory.
    activation_store: ActivationStore | None = None
    _activation: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _activation_tally: Tally | None = field(default=None, repr=False)
    _signal_hotkey: str = field(default="", repr=False)
    _signal_sent: str = field(default="", repr=False)
    # DEC-CA-0048 forfeiture decided on chain: its own record + tally, persisted
    # beside the rollover's (``activation_forfeit_state.json``).
    _forfeit: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _forfeit_tally: Tally | None = field(default=None, repr=False)
    # DEC-CA-0049 dethrone bar v2 decided on chain: its own record + tally,
    # persisted beside the others (``activation_margin_v2_state.json``).
    _margin_v2: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _margin_v2_tally: Tally | None = field(default=None, repr=False)
    # DEC-CA-0055 pre-launch burn decided on chain: its own record + tally,
    # persisted beside the others (``activation_prelaunch_burn_state.json``).
    _prelaunch: ActivationRecord = field(default_factory=ActivationRecord, repr=False)
    _prelaunch_tally: Tally | None = field(default=None, repr=False)
    # Lane B (DEC-CA-0050): the domain-winner state (``<state_db>.lane_b.json``),
    # the store its expert manifests are read from, the settlement block of the
    # last judged manifest (the weight vector's decay clock), and the merged
    # era inits already verified bit-exact.
    _lane_b_state: object = field(default=None, repr=False)
    _lane_b_store: object = field(default=None, repr=False)
    _lane_b_block: int | None = field(default=None, repr=False)
    _merged_init_ok: set = field(default_factory=set, repr=False)
    # the pool lane-B catch-up / on-demand judging re-draws windows from
    _lane_b_window_source: object = field(default=None, repr=False)
    # the payout record of the last vector _apply_weights set (None = the
    # legacy lane-A vector) — stamped on that round's receipt
    _last_lane_b_pay: dict | None = field(default=None, repr=False)
    # DEC-CA-0055: the emission record of the last vector _apply_weights set
    # (None = no DEC-CA-0055 rule shaped it), and this settlement's ladder
    # transition / king check — ``(round_id, record)`` — stamped on its receipt.
    _last_emission: dict | None = field(default=None, repr=False)
    _king_eval: tuple | None = field(default=None, repr=False)
    _settlement_ladder: tuple | None = field(default=None, repr=False)
    _settlement_king_check: tuple | None = field(default=None, repr=False)
    # Lane C (DEC-CA-0056): the per-target slot state (``<state_db>.lane_c.json``),
    # the store its manifests are read from (None = the lane-B / manifest
    # store), the settlement block of the last judged manifest (the payout
    # clock) and the payout record of the last vector set (stamped on that
    # receipt). Lane C has no pool of its own (MarketCast-only lane C:
    # MarketCast reads [marketcast] pool_prefix).
    _lane_c_state: object = field(default=None, repr=False)
    _lane_c_store: object = field(default=None, repr=False)
    _lane_c_block: int | None = field(default=None, repr=False)
    _last_lane_c_pay: dict | None = field(default=None, repr=False)
    # DEC-CA-0057 owner-signed network config: the resolver (None, or one with
    # an empty owner key = inert), the base config while a settlement runs
    # under an overlay view (``_settlement_cfg``), that settlement's block,
    # and the head block of the last activation pass (for the note).
    netconfig: object | None = None
    _cfg_base: ChainConfig | None = field(default=None, repr=False)
    _settlement_block: int | None = field(default=None, repr=False)
    # DEC-CA-0058: ``(round_id, settlement state)`` captured when the
    # settlement started — its digest is the receipt's ``state_prior_sha256``.
    _settlement_prior: tuple | None = field(default=None, repr=False)
    _netconfig_now: int | None = field(default=None, repr=False)
    # DEC-CA-0063 (lane-A benches on the main validator's GPU): the wallet
    # this validator signs its bench report with (its hotkey == [netconfig]
    # owner_hotkey makes it the MAIN validator — the only one that sweeps);
    # an injectable sweep (tests: entry -> raw sidecar report); the
    # settlement's bench state, the per-pointer sweep cache (the era king /
    # init are swept once) and verified validator reports read later.
    bench_signer: object | None = None
    bench_sweep_fn: Callable[[TrainedEntry], dict | None] | None = None
    # the paired pass ([eval] bench_paired; tests: entries -> one raw report
    # each). With only ``bench_sweep_fn`` injected, sweeps stay one at a time.
    bench_sweep_many_fn: Callable[[list], list] | None = None
    _vbench: dict | None = field(default=None, repr=False)
    _vbench_cache: dict = field(default_factory=dict, repr=False)
    _vbench_reports: dict = field(default_factory=dict, repr=False)
    # DEC-CA-0063: the raw manifests the settlement walk has pending AFTER the
    # one being judged — where the trainer's record of its validator bench
    # report (``validator_bench_reports``) is looked up.
    _vbench_later_raw: list = field(default_factory=list, repr=False)

    def process_round(
        self,
        manifest: TrainingManifest,
        windows: list[EvalWindow],
        base_seed: int | str,
    ) -> RoundOutcome | None:
        """Evaluate one manifest against the eval windows and update state.

        A round carries one king per trained size (the primary plus any
        ``[[training.sizes]]``) and a cohort of challengers — every leg the
        settlement judges (archived boundary rounds: the heat's advanced
        cohort, DEC-CA-0012). For each
        challenger, every size's pair is scored on the SAME windows and the
        per-size scores are POOLED — king's across sizes vs challenger's across
        sizes, in identical order — so a single paired bootstrap decides ONE
        throne on the combined score (scaling-aware KOTH). Pooling preserves
        pairing because each size's king and challenger share the window
        ``abs_target``.

        The WHOLE cohort is judged; the duel deliberately does NOT stop at the
        first challenger to clear the margin. Judging is cheap (the king is
        evaluated once and reused, so an extra challenger is one extra eval),
        stopping early would let the screen's noisy ranking decide the throne
        among challengers it could not separate, and an early stop makes the
        family-wise alpha below unsound (optional stopping). Clearers are then
        ranked by OBSERVED relative improvement, not by their bound — gate on the
        bound, select on the point estimate (DEC-CA-0006 measured ranking a
        selection by a bound at −20pp in the adverse case).

        Because ``k`` challengers each get a draw at the ``bootstrap_alpha`` tail,
        a merely-equal cohort would multiply the king's false-dethrone risk. The
        per-challenger alpha is therefore tightened to ``bootstrap_alpha / k``.
        ``k`` is the duelled-cohort size, derivable from the signed manifest, so
        the receipt keeps recording the UNMODIFIED config params (a mutated alpha
        there would fail ``check_koth_params`` against ``chain.toml``) and the
        audit re-derives the correction itself. At ``k = 1`` this is bit-identical
        to the pre-DEC-CA-0012 rule.

        Returns None (king holds, no state change) when the manifest carries no
        challenger paired with the king at every size, or fails the contract gate.
        Otherwise returns the round outcome with the (already-applied) transition.
        """
        self._round_none_reason = None
        self._apply_forfeiture(manifest)
        reason = self.check_manifest(manifest)
        if reason is not None:
            log.warning("rejecting manifest round=%s: %s", manifest.round_id, reason)
            return None

        king_by_size = {e.size: e for e in manifest.entries_for_role("king")}
        # The cohort and the sizes it is judged on, derived from the signed
        # manifest by a helper SHARED with cascade-audit so the two cannot derive
        # a different `k` (see TrainingManifest.duel_cohort).
        cohort, paired_sizes = manifest.duel_cohort()
        advanced = {hk for hk, _ in cohort}
        for hk, by_size in manifest.challenger_cohort():
            if hk in advanced:
                continue
            log.info("round=%s challenger %s dropped from the duel: trained at %s, "
                     "the cohort is judged on %s", manifest.round_id, hk,
                     sorted(by_size), paired_sizes)
        if not cohort:
            log.info("manifest round=%s has no king/challenger pair; king holds",
                     manifest.round_id)
            return None

        score_records: list[EntryScores] = []

        # The king is evaluated ONCE and reused across the cohort — the windows
        # are identical, which is what makes every comparison paired, and it is
        # what makes an extra challenger cost one extra eval instead of two.
        import time as _time

        king_scores: list[WindowScore] = []
        _t0 = _time.perf_counter()
        for size in paired_sizes:
            ks = self._evaluate(king_by_size[size], windows, self._training_for(manifest))
            king_scores += ks
            score_records.append(EntryScores(
                role="king", size=size,
                hotkey=king_by_size[size].miner_hotkey, uid=king_by_size[size].miner_uid,
                scores=self._window_records(ks, manifest),
            ))
        _t_king = _time.perf_counter() - _t0

        # Decision parameters for THIS round's epoch: a scheduled margin
        # change resolves from the boundary block, never from restart timing.
        epoch_block = self._epoch_start_block(manifest)
        base_params = self.cfg.koth_params(block=epoch_block)
        # Score the shared warm-start init when either consumer needs it: the
        # increment margin (DEC-CA-0027) or the init-baseline floor
        # ([scoring] init_gate_mode). The increment fallback mutates only the
        # JUDGED params — the receipt keeps recording the unmodified config
        # params (check_koth_params compares them against chain.toml) — and
        # the audit re-derives the judged margin rule as "params say
        # increment AND baseline rows exist" (rows alone do not imply
        # increment: a level round with the floor on records them too).
        baseline_scores: list[WindowScore] | None = None
        if (base_params.margin_mode == "increment"
                or base_params.init_gate_mode != "off"):
            baseline_scores = self._score_increment_baseline(
                manifest, paired_sizes, windows, score_records)
            if baseline_scores is None and base_params.margin_mode == "increment":
                # Level units ⇒ the level bar (DEC-CA-0049: never the v2
                # increment bar priced as a share of the absolute score).
                base_params = self.cfg.judged_level_params(base_params, epoch_block)
            if baseline_scores is None and base_params.init_gate_mode != "off":
                # No baseline (random-init round / size mismatch): the floor
                # cannot run this round — recorded None, judged as if off.
                log.info("round=%s init-baseline floor (%s): no baseline this "
                         "round; floor not applied",
                         manifest.round_id, base_params.init_gate_mode)
        k = len(cohort)
        # Cohort family-wise correction (DEC-CA-0038, block-gated). max-T reads
        # the critical value off the ACTUAL joint spread (one shared resample,
        # exact under the challengers' real positive correlation); Bonferroni
        # alpha/k is the pre-gate rule (loose under that correlation — it
        # over-protects the king). Under max-T the per-challenger evaluate_round
        # below runs at the UNCORRECTED alpha (its lcb is discarded and replaced
        # by the joint bound); the geomeans/floors/diagnostics it computes are
        # unaffected. k <= 1 is bit-identical either way (no multiplicity).
        use_maxt = k > 1 and cohort_maxt_active(
            self.cfg.scoring, self._epoch_start_block(manifest))
        if use_maxt:
            duel_params = base_params
            log.info("round=%s cohort duel: k=%d challengers, max-T family-wise "
                     "correction (alpha %.4f, shared resample)", manifest.round_id,
                     k, base_params.bootstrap_alpha)
        else:
            duel_params = (
                base_params if k <= 1
                else replace(base_params, bootstrap_alpha=base_params.bootstrap_alpha / k)
            )
            if k > 1:
                log.info("round=%s cohort duel: k=%d challengers, Bonferroni "
                         "alpha %.4f -> %.5f", manifest.round_id, k,
                         base_params.bootstrap_alpha, duel_params.bootstrap_alpha)

        # Tenure as the margin schedule counts it (DEC-CA-0043: in blocks
        # from tenure_blocks_from_block; the counter before it).
        self.state = self._anchor_legacy_king(self.state, self._epoch_start_block(manifest))
        tenure_at_decision = tenure_rounds_at(
            self.cfg.round, self.cfg.scoring, block=self._epoch_start_block(manifest),
            tenure_rounds=self.state.tenure_rounds,
            king_since_block=self.state.king_since_block)
        # DEC-CA-0055: the v2 king check (judged after this verdict) reuses the
        # king's and the init's rows instead of re-scoring them.
        self._king_eval = (str(manifest.round_id), king_scores, baseline_scores,
                           tenure_at_decision)
        judged: list[tuple[str, TrainedEntry, RoundResult]] = []
        cohort_scores: list[tuple[str, list[WindowScore]]] = []
        inconclusive: RoundResult | None = None
        inconclusive_at: tuple[str, TrainedEntry] | None = None
        # Entry faults ([scoring] entry_fault_from_block): a challenger whose
        # evaluation fails deterministically LOSES — no score rows, out of the
        # judged cohort, reason in the verdict note — instead of raising the
        # settlement into an endless retry. It still counts in ``k`` (the
        # family the audit derives from the signed manifest). The king above
        # is NOT covered: a king that cannot be scored must not crown anyone
        # by default, so its failure stays an infra retry (as a missing king
        # checkpoint always was).
        fault_from = int(getattr(self.cfg.scoring, "entry_fault_from_block", 0) or 0)
        faults_on = bool(fault_from) and epoch_block >= fault_from
        faults: dict[str, str] = {}
        for hk, by_size in cohort:
            chal_scores: list[WindowScore] = []
            chal_records: list[EntryScores] = []
            _t1 = _time.perf_counter()
            try:
                for size in paired_sizes:
                    cs = self._evaluate(by_size[size], windows, self._training_for(manifest))
                    chal_scores += cs
                    chal_records.append(EntryScores(
                        role="challenger", size=size,
                        hotkey=by_size[size].miner_hotkey, uid=by_size[size].miner_uid,
                        scores=self._window_records(cs, manifest),
                    ))
            except Exception as e:  # noqa: BLE001 — classified; transients re-raise
                fault = self._entry_fault_reason(e) if faults_on else None
                if fault is None:
                    raise
                log.warning("round=%s challenger %s ENTRY FAULT (loses the duel): %s",
                            manifest.round_id, hk, fault)
                faults[hk] = fault
                continue
            score_records.extend(chal_records)
            _t_chal = _time.perf_counter() - _t1
            log.info(
                "round=%s eval-timing challenger=%s device=%s n_windows=%d "
                "num_samples=%d king=%.1fs challenger=%.1fs",
                manifest.round_id, hk, self.device, len(windows),
                self.cfg.eval.num_samples, _t_king, _t_chal,
            )
            result = evaluate_round(
                king_scores, chal_scores, duel_params,
                seed=base_seed, king_tenure_rounds=tenure_at_decision,
                baseline_scores=baseline_scores,
            )
            if result.inconclusive:
                # ``min_windows``/``min_clusters`` are properties of the window
                # slice, identical for every challenger — if one duel is
                # inconclusive they all are. Hold the throne and stop rather than
                # evaluate the rest for a verdict that cannot arrive.
                log.info("round=%s duel inconclusive (n_windows=%d n_clusters=%d); "
                         "king holds, remaining cohort not evaluated",
                         manifest.round_id, result.n_windows, result.n_clusters)
                inconclusive = result
                inconclusive_at = (hk, by_size[paired_sizes[0]])
                break
            judged.append((hk, by_size[paired_sizes[0]], result))
            cohort_scores.append((hk, chal_scores))
            if result.init_floor_passed is not None:
                log.info(
                    "round=%s init-baseline floor (%s) %s: chal=%.5f vs "
                    "init=%.5f (tol=%.3f) — %s%s",
                    manifest.round_id, base_params.init_gate_mode, hk,
                    result.chal_geomean, result.baseline_geomean,
                    base_params.init_gate_tolerance,
                    "passes" if result.init_floor_passed else "WORSE THAN INIT",
                    ("" if base_params.init_gate_mode == "enforce"
                     or result.init_floor_passed else " (shadow: not gating)"),
                )

        # Cohort family-wise correction (DEC-CA-0038): replace each judged
        # challenger's solo lcb with its joint max-T bound — one shared
        # resample over king + the whole cohort, exact under the real
        # correlation — and re-decide the win under the same margin/floor rule.
        # Only when armed and the cohort completed (an inconclusive round holds
        # regardless); the per-challenger evaluate_round above ran at the
        # uncorrected alpha, so its geomeans/floors carry over unchanged.
        if not judged and inconclusive is None:
            # Every challenger faulted: nobody to judge, the king holds with
            # no state change (as for a manifest with no pair).
            self._round_none_reason = "entry_fault: every challenger faulted — " + "; ".join(
                f"{hk}: {why}" for hk, why in faults.items())
            log.warning("round=%s %s", manifest.round_id, self._round_none_reason)
            return None
        if use_maxt and faults and inconclusive is None and len(judged) == 1:
            # Faults left ONE judged challenger of a k > 1 cohort: no joint
            # bound to take, so it is judged at the cohort's Bonferroni
            # alpha/k — what the audit's solo replay derives from the signed
            # manifest (never the uncorrected alpha the max-T loop ran at).
            hk, entry, _ = judged[0]
            judged = [(hk, entry, evaluate_round(
                king_scores, cohort_scores[0][1],
                replace(base_params, bootstrap_alpha=base_params.bootstrap_alpha / k),
                seed=base_seed, king_tenure_rounds=tenure_at_decision,
                baseline_scores=baseline_scores,
            ))]
        lcb_maxt = False      # the decided lcb is the cohort's joint max-T bound
        if use_maxt and inconclusive is None and len(judged) > 1:
            from ..eval.koth import with_cohort_lcb

            lcb_maxt = True

            # Increment units for the joint bound (DEC-CA-0039 stacked on
            # DEC-CA-0038, block-gated): base_params already fell back to
            # "level" when no baseline could be scored this round, so passing
            # the rows here is exactly "increment judged AND gate reached".
            maxt_baseline = (
                baseline_scores
                if (base_params.margin_mode == "increment"
                    and cohort_maxt_increment_active(
                        self.cfg.scoring, self._epoch_start_block(manifest)))
                else None)
            if maxt_baseline is not None:
                log.info("round=%s cohort max-T bound judged in INCREMENT units "
                         "(baseline paired)", manifest.round_id)
            lcbs = cohort_maxt_lcb_map(
                king_scores, cohort_scores, base_params, seed=base_seed,
                baseline_scores=maxt_baseline)
            judged = [(hk, entry, with_cohort_lcb(res, lcbs[hk], base_params))
                      for hk, entry, res in judged]
        for hk, _, res in judged:
            log.info("round=%s duel %s lcb=%.4f margin=%.4f clears=%s",
                     manifest.round_id, hk, res.lcb, res.margin,
                     res.challenger_wins_round)

        # DEC-CA-0063: clearers the validator bench could not judge for an
        # INFRA reason (trainer-recorded absent report, infra-failed sweep,
        # exhausted bench budget) — skipped, never a loss: {hotkey: reason}.
        bench_infra: dict[str, str] = {}
        # DEC-CA-0063 items 9-10: clearers the settlement's sweep cap deferred
        # (skipped, never a loss — the trainer carries the leg into a later
        # settlement) and challengers whose sweep was a reproduced CHECKPOINT
        # fault (the gate failed: a loss): {hotkey: reason}.
        bench_deferred: dict[str, str] = {}
        bench_fault: dict[str, str] = {}
        if inconclusive is not None:
            assert inconclusive_at is not None
            decided_hotkey, chal_entry, result = (
                inconclusive_at[0], inconclusive_at[1], inconclusive)
            duelled: tuple[str, ...] = tuple(hk for hk, _, _ in judged) + (decided_hotkey,)
        else:
            # Gate on the bound, select on the point estimate: among the
            # challengers that cleared the margin, crown the best OBSERVED
            # relative improvement. The king's geomean is shared across the
            # cohort, so lowest ``chal_geomean`` IS the best improvement.
            def _pick(cands):
                return min(cands, key=lambda t: (t[2].chal_geomean,
                                                 t[1].train_block, t[1].miner_uid))

            clearers = [t for t in judged if t[2].challenger_wins_round]
            decided_hotkey, chal_entry, result = _pick(clearers or judged)
            duelled = tuple(hk for hk, _, _ in judged)
            if len(clearers) > 1:
                log.info("round=%s %d challengers cleared the margin; crowning %s "
                         "(best observed geomean %.5f)", manifest.round_id,
                         len(clearers), decided_hotkey, result.chal_geomean)
            # Public-benchmark no-regression gates: only on a private-pool win,
            # and only when enabled. Each can block a dethrone (or, uncomputable,
            # hold the round) but never grant one. The GIFT gate is a sidecar
            # sweep; the three-suite bench gate (DEC-CA-0054) reads the
            # trainer-signed rows. They run on the best clearer and fall through
            # to the next-best only if an ``enforce`` gate blocks. Gated on the
            # primary size's pair (the pooled decision spans sizes; the gates
            # screen on one).
            gift_on = self.cfg.scoring.gift_gate_mode != "off"
            bench_mode = bench_gate_mode_at(self.cfg, self._epoch_start_block(manifest))
            bench_on = bench_mode != "off"
            if (gift_on or bench_on) and clearers:
                remaining = sorted(
                    clearers, key=lambda t: (t[2].chal_geomean,
                                             t[1].train_block, t[1].miner_uid))
                gated_any = False
                infra_hold = None
                for hk, entry, res in remaining:
                    gate_size = (
                        self._training_for(manifest).arch_preset
                        if self._training_for(manifest).arch_preset in king_by_size
                        and self._training_for(manifest).arch_preset in dict(cohort)[hk]
                        else paired_sizes[0]
                    )
                    gated = res
                    if gift_on:
                        gated = self._run_gift_gate(
                            gated, king_by_size[gate_size], dict(cohort)[hk][gate_size],
                            seed=base_seed, round_id=manifest.round_id,
                        )
                    if bench_on and gated.challenger_wins_round:
                        gated = self._run_bench_gate(
                            gated, king_by_size[gate_size], dict(cohort)[hk][gate_size],
                            seed=base_seed, round_id=manifest.round_id, mode=bench_mode,
                            # DEC-CA-0061: the settlement's effective rule
                            rule=bench_gate_rule_at(self.cfg, self._epoch_start_block(manifest)),
                        )
                        infra = getattr(self, "_bench_gate_infra", None)
                        deferred = getattr(self, "_bench_gate_deferred", None)
                        fault = getattr(self, "_bench_gate_fault", None)
                        if fault is not None:
                            bench_fault[hk] = fault
                        if ((infra is not None or deferred is not None)
                                and bench_mode == "enforce"):
                            # DEC-CA-0063: infra / a budget deferral is never
                            # the clearer's loss — skip it (streak untouched)
                            # and fall through; with nobody left the king
                            # holds (inconclusive).
                            if deferred is not None:
                                bench_deferred[hk] = deferred
                            else:
                                bench_infra[hk] = infra
                            infra_hold = (hk, entry, gated)
                            log.warning("round=%s %s cleared the private pool but the "
                                        "validator bench is %s (%s); skipped, falling "
                                        "through", manifest.round_id, hk,
                                        "DEFERRED" if deferred is not None else "INFRA",
                                        deferred if deferred is not None else infra)
                            continue
                    decided_hotkey, chal_entry, result = hk, entry, gated
                    gated_any = True
                    if gated.challenger_wins_round or gated.inconclusive:
                        break
                    log.info("round=%s %s cleared the private pool but was blocked by "
                             "a public-benchmark gate; falling through to the next "
                             "clearer", manifest.round_id, hk)
                if not gated_any and infra_hold is not None:
                    decided_hotkey, chal_entry, result = infra_hold

        transition = state_mod.apply_round(
            self.state,
            challenger_hotkey=chal_entry.miner_hotkey,
            challenger_uid=chal_entry.miner_uid,
            result=result,
            dethrone_cp=self.cfg.scoring.dethrone_cp,
            keep_former_kings=self.cfg.scoring.reward_prior_kings,
            # A faulted challenger lost: its streak resets like a duelled one.
            # A bench-infra clearer (DEC-CA-0063) did not lose: its streak stays.
            defeated_hotkeys=tuple(h for h in duelled + tuple(faults)
                                   if h not in bench_infra and h not in bench_deferred),
            crowned_block=self._epoch_start_block(manifest),
            crowned_pointer=chal_entry.trained_pointer,
        )
        if faults:
            # The signed record of WHY a manifest challenger has no score rows
            # (VerdictRecord cannot grow; the note is free text).
            transition = replace(transition, note=transition.note + "; entry_fault " + "; ".join(
                f"{hk}: {why[:200]}" for hk, why in faults.items()))
        if bench_infra:
            # The signed record of which clearers the validator bench skipped
            # as infra (free-text note, like entry_fault; the audit's
            # bench-coverage row checks each against the report)
            transition = replace(transition, note=transition.note + "; bench_infra " + "; ".join(
                f"{hk}: {why[:200]}" for hk, why in sorted(bench_infra.items())))
        for seg, named in (("bench_deferred", bench_deferred), ("bench_fault", bench_fault)):
            if named:
                # DEC-CA-0063 items 9-10, same free-text convention (the
                # audit's bench-coverage row checks each against the report)
                transition = replace(transition, note=transition.note + f"; {seg} " + "; ".join(
                    f"{hk}: {why[:200]}" for hk, why in sorted(named.items())))
        self.state = transition.state
        log.info(
            "round=%s k=%d decided=%s lcb=%.4f margin=%.4f win=%s %s king=%s tenure=%d",
            manifest.round_id, k, decided_hotkey, result.lcb, result.margin,
            result.challenger_wins_round, transition.note,
            self.state.king_hotkey, self.state.tenure_rounds,
        )
        # The decided challenger's records go LAST (DEC-CA-0012 convention).
        # Every consumer of the signed receipt resolves "which challenger does
        # the verdict belong to" positionally — audit's _pooled_scores /
        # check_duel_cohort / check_transition and the public summary all read
        # the last-recorded challenger — but the eval loop above appends in
        # cohort order and the crown (best clearer, or a gift-gate
        # fallthrough) is only known now. Without this reorder any cohort
        # whose winner is not ranked last publishes a receipt those consumers
        # misattribute and the audit fails legitimately-crowned rounds.
        # k <= 1 is untouched: a single challenger is trivially last, so
        # pre-cohort receipts keep their exact signed bytes.
        if k > 1:
            decided = [r for r in score_records
                       if r.role == "challenger" and r.hotkey == decided_hotkey]
            others = [r for r in score_records
                      if not (r.role == "challenger" and r.hotkey == decided_hotkey)]
            score_records = others + decided
            duelled = tuple(hk for hk in duelled if hk != decided_hotkey) + (decided_hotkey,)
        # Shadow diagnostics: never gate the verdict. A rank-based view that
        # disagrees with the LCB, or a per-domain win-rate sign flip, means the
        # pool composition is doing the deciding — alert-worthy, not decisive.
        if result.win_rate is not None:
            log.info(
                "round=%s diag n_clusters=%d win_rate=%.3f wilcoxon_p=%s per_domain=%s",
                manifest.round_id, result.n_clusters, result.win_rate,
                f"{result.wilcoxon_p:.4g}" if result.wilcoxon_p is not None else "n/a",
                {d: f"{wr:.2f}/n{n}" for d, (wr, n) in (result.per_domain_win_rate or {}).items()},
            )
        return RoundOutcome(
            result=result, transition=transition, entry_scores=tuple(score_records),
            king_tenure_rounds=tenure_at_decision,
            decided_hotkey=decided_hotkey, duelled_hotkeys=duelled,
            # Only on a genuine cohort: k <= 1 must leave the receipt's signed
            # bytes exactly as a pre-DEC-CA-0012 round's.
            cohort_k=(k if k > 1 else 0),
            cohort_lcbs=({hk: r.lcb for hk, _, r in judged} if k > 1 else {}),
            # Display fields for every judged challenger (never gating).
            cohort_geomeans={hk: r.chal_geomean for hk, _, r in judged},
            cohort_per_horizon={hk: r.per_horizon for hk, _, r in judged
                                if r.per_horizon},
            cohort_per_domain={hk: r.per_domain for hk, _, r in judged
                               if r.per_domain},
            cohort_stats=({hk: cohort_stats_of(r) for hk, _, r in judged} if k > 1 else {}),
            entry_faults=dict(faults),
            lcb_units=crown_units(
                margin_mode=base_params.margin_mode,
                has_baseline=baseline_scores is not None, maxt=lcb_maxt,
                maxt_increment=cohort_maxt_increment_active(
                    self.cfg.scoring, self._epoch_start_block(manifest))),
        )

    def _pending_manifests(self, store: object, raw_latest: str,
                           last_round: str | None) -> list[tuple[str, str]]:
        """:func:`chain_manifests`, applied only when the latest settlement is
        past ``era_king_from_block`` (pre-rollover: ``[latest]``, exactly as
        before). A walk that does not reach the last handled settlement
        handles NOTHING (position untouched) and doubles the depth for the
        next poll. Unreadable latest ⇒ handled by the caller's load."""
        latest = load_manifest(raw_latest)
        if not self._era_active(latest):
            return [(raw_latest, hashlib.sha256(raw_latest.encode("utf-8")).hexdigest())]
        walk = chain_manifests(store, raw_latest, last_round, max_depth=self.chain_walk_depth)
        if not walk.reached:
            depth = self.chain_walk_depth
            self.chain_walk_depth = min(depth * 2, CHAIN_WALK_MAX_DEPTH)
            log.error("manifest chain: latest round %s does not reach this validator's last "
                      "handled settlement %s within %d manifests (hole or depth cap) — "
                      "NOTHING handled this poll; retrying with depth %d. A permanent "
                      "hole needs the missing round-<id>.json restored to the bucket",
                      latest.round_id, last_round, depth, self.chain_walk_depth)
            return []
        self.chain_walk_depth = CHAIN_WALK_DEPTH
        return list(walk.manifests)

    def _anchor_legacy_king(self, state: ChampionState, block: int) -> ChampionState:
        """CONSENSUS (DEC-CA-0043): at the reigning king's first settlement
        past ``tenure_blocks_from_block`` impute ``king_since_block`` from the
        ``tenure_rounds`` counter (:func:`legacy_king_anchor`) — whether or
        not a crowning block was recorded before the gate — and KEEP it
        (``tenure_anchor_gate`` marks it done). Every later ``tenure_rounds_at`` then counts blocks from a
        fixed anchor. Without this the anchor was re-imputed each settlement
        from a counter that kept advancing on the new grid, and the tenure
        grew old_grid/new_grid (4× on mainnet) per settlement."""
        if not state.king_hotkey or not tenure_blocks_active(self.cfg.scoring, block):
            return state
        gate = int(self.cfg.scoring.tenure_blocks_from_block)
        if state.tenure_anchor_gate == gate:
            return state                         # anchored once at this gate already
        # EVERY validator re-anchors from the counter at the gate — a
        # crowning block recorded before the gate (a validator that upgraded
        # before that king's crowning) is discarded, or two upgraded
        # validators would count tenure from different anchors (real block
        # vs. counter × old grid) and reach different margins at the same
        # settlement. The counter is the quantity the fleet agreed on
        # before the gate; the real block is only known to some.
        anchor = legacy_king_anchor(self.cfg.round, self.cfg.scoring, block=int(block),
                                    tenure_rounds=state.tenure_rounds)
        if state.king_since_block is not None and int(state.king_since_block) != anchor:
            log.info("tenure: king %s… recorded crowning block %d predates the gate; "
                     "re-anchored from the counter for consensus",
                     state.king_hotkey[:8], int(state.king_since_block))
        log.info("tenure: king %s… anchored at block %d (%d pre-gate rounds before "
                 "settlement %d, gate %d); persisted as king_since_block",
                 state.king_hotkey[:8], anchor, state.tenure_rounds, int(block), gate)
        return replace(state, king_since_block=anchor, tenure_anchor_gate=gate)

    def _epoch_start_block(self, manifest: TrainingManifest) -> int:
        """The round's epoch-boundary block: ``created_block`` floored to the
        epoch grid. Monotonic and identical for every validator (from the shared
        manifest), so it is the consensus key for daily eval-pool snapshot
        selection — unlike the round id, which is a block *hash* (non-monotonic).

        The length is resolved AT ``created_block`` (not at the head) so a
        scheduled cadence change is consensus-safe: every validator floors this
        round on the same grid whether it restarted before or after the switch,
        and a round that predates the switch keeps its original epoch forever.
        """
        epoch_blocks = effective_epoch_blocks(self.cfg.round, manifest.created_block)
        return (manifest.created_block // epoch_blocks) * epoch_blocks

    # ── cascade v2 (DEC-CA-0050): the contract a settlement is judged under ──

    def _training_for(self, manifest: TrainingManifest):
        """The training contract ``manifest``'s settlement was trained under:
        ``[training]`` resolved at the start of the era the settlement belongs
        to (pure block arithmetic, never the era stamp) — the v2 contract for
        a v2 era, the base contract byte-for-byte otherwise. Without a v2
        schedule this IS ``self.cfg.training`` (same object)."""
        t = self.cfg.training
        if not int(getattr(t, "v2_from_block", 0) or 0):
            return t
        return t.at_block(settlement_contract_block(self.cfg.round,
                                                    self._epoch_start_block(manifest)))

    def _v2_receipt_block(self, boundary: int) -> int:
        """The receipt's ``v2_block``: ``[v2] from_block`` when the settlement
        at ``boundary`` judges a v2 era, else 0 (dropped from the body)."""
        if v2_settlement_active(self.cfg, boundary):
            return int(self.cfg.v2.from_block)
        return 0

    # ── live loop ────────────────────────────────────────────────────────────

    def _maybe_calibrate_horizons(
        self, manifest: TrainingManifest, base_seed: int | str, window_source: object
    ) -> None:
        """Multi-horizon calibration telemetry (``[eval] calib_horizons``) —
        LOG-ONLY, runs strictly AFTER weights and the receipt, and is wrapped
        so no failure mode can reach the round path. Scores the round's duel
        checkpoints (already fetched by the verdict eval, so this re-uses the
        local cache) at each extra rung on its own seeded even-by-domain draw
        and writes ``_calib/round-<id>.json`` beside the state file."""
        horizons = tuple(getattr(self.cfg.eval, "calib_horizons", ()) or ())
        if not horizons:
            return
        try:
            from .calibration import run_horizon_calibration

            dir_fn = getattr(window_source, "snapshot_dir_for_round", None)
            pool_dir = dir_fn(block=self._epoch_start_block(manifest)) if dir_fn else None
            if pool_dir is None:
                log.info("calib: window source exposes no series dir; skipping")
                return
            primary = self._training_for(manifest).primary_size.arch_preset
            checkpoints: dict[str, Path] = {}
            for entry in manifest.entries:
                if (entry.size or primary) != primary:
                    continue
                label = ("king" if entry.role == "king"
                         else f"{entry.role}-u{entry.miner_uid}")
                checkpoints[label] = self._fetch_checkpoint_dir(entry)
            if "king" not in checkpoints:
                return
            # King first: the paired deltas in the report are vs the king.
            ordered = {"king": checkpoints.pop("king"), **checkpoints}
            doc = run_horizon_calibration(
                ordered, Path(pool_dir),
                horizons=horizons,
                n_windows=int(self.cfg.eval.calib_windows),
                num_samples=int(self.cfg.eval.calib_num_samples),
                context_length=int(self.cfg.eval.context_length),
                seed=int(str(base_seed)) % (2**63),
                device=self.device,
            )
            out = Path("_calib")
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"round-{manifest.round_id}.json"
            p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            log.info("calib: round=%s horizons=%s → %s",
                     manifest.round_id, list(horizons), p)
        except Exception as e:  # noqa: BLE001 — telemetry must never hurt the round
            log.warning("calib: skipped for round=%s (%s)", manifest.round_id, e)

    def run_forever(self, client: object, *, window_source: object) -> None:  # pragma: no cover
        """Poll the manifest bucket → evaluate → set weights, once per round.

        ``window_source`` is a :class:`cascade.validator.windows.WindowSource`
        (the loaded private pool). Each new manifest's ``round_id`` is the base
        seed; the same seed drives the rotating window slice so every validator
        scores the identical set.
        """
        import time

        from ..shared.hippius import StorageError, open_manifest_store, read_latest_manifest
        from ..shared.manifest import load_manifest

        store = open_manifest_store(self.cfg.storage)
        self._lane_b_store = store         # DEC-CA-0050 expert manifests live beside
        self._lane_b_window_source = window_source   # lane-B catch-up re-draws its windows
        if self.bench_report_store is None:
            # Bench reports live in the same bucket — reuse this store (and its
            # R2/HF fallbacks) instead of lazily building a second client.
            self.bench_report_store = store
        poll = self.cfg.manifest.poll_seconds
        # Dedup on CONTENT, not round_id: a re-published manifest for an
        # already-seen round id (same-round-id rerun, e.g. after a contract
        # fix) must be re-judged, not silently skipped.
        # The latch is PERSISTED (mirrors the trainer's): a restart resumes
        # from the last handled (round, sha) instead of re-judging the already-
        # published round — under a changed scoring rule that re-judgement can
        # flip a settled verdict (e.g. DEC-CA-0009). State files from
        # before the marker existed are seeded from this validator's own
        # signature-verified receipt trail; a same-round re-publish with
        # different content still re-judges (the sha differs).
        if self.commitment_history_fn is None:
            # DEC-CA-0043 ref binding needs the FULL reveal history; wired here
            # (not by the caller) so no live validator runs the era gate
            # without it — _check_era_refs fails closed on None.
            self.commitment_history_fn = lambda: client.poll_commitments(include_history=True)
        last_round: str | None = self.state.last_handled_round_id
        if last_round is None:
            seeded = self._probe_own_receipt_round(client, store)
            if seeded is not None:
                try:
                    raw = read_latest_manifest(store)
                    seeded_manifest = load_manifest(raw)
                    if seeded_manifest.round_id == seeded:
                        last_round = seeded
                        self.state = replace(
                            self.state,
                            last_handled_round_id=str(seeded),
                            last_handled_manifest_sha=hashlib.sha256(
                                raw.encode("utf-8")).hexdigest(),
                            last_handled_block=int(seeded_manifest.created_block),
                        )
                        self._persist_state()
                except (StorageError, ValueError) as e:
                    log.warning("re-entry probe could not read the latest manifest "
                                "(%s); proceeding without a seeded marker", e)
        if last_round is not None:
            log.info("round %s already handled (persisted/receipt marker); resuming poll",
                     last_round)
        # Stake-weighted activation (DEC-CA-0045): restore/resolve the
        # rollover and post this validator's readiness note BEFORE the first
        # manifest is judged, so a restart never judges a settlement under
        # the wrong rules.
        self._activation_startup(client)
        from ..shared.calendar_launch import check_calendar_launch

        check_calendar_launch(self._base_cfg(), client)
        from .settlement_transaction import recover

        self.registration_at_block_fn = getattr(client, "uid_for_hotkey_at_block", None)
        recover(self)
        if self.cfg.lane_b.from_block:
            # DEC-CA-0050: a missing / torn lane-B state is rebuilt from the
            # recent chain before the first weight push (best-effort; the next
            # settlement's walk retries a deferred rebuild).
            try:
                self._lane_b_catch_up_startup()
            except Exception as e:  # noqa: BLE001
                log.warning("lane B: startup catch-up failed (%s); retried at the next "
                            "settlement", e)
        if self.cfg.lane_c.from_block:
            try:
                self._lane_c_catch_up_startup()
            except Exception as e:  # noqa: BLE001
                log.warning("lane C: startup catch-up failed (%s); retried at the next "
                            "settlement", e)
        while True:
            try:
                self._poll_once(client, store, window_source)
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round processing failed; retrying after poll: %s", e)
            try:
                self._reassert_weights_at_head(client)
            except Exception as e:  # noqa: BLE001
                log.warning("weight re-assert check failed: %s", e)
            time.sleep(poll)

    def _reassert_weights_at_head(self, client: object) -> None:
        """The freshness re-assert, under the config in force at the head
        (DEC-CA-0057; exactly ``_maybe_reassert_weights`` while inert)."""
        from .settlement_transaction import recover

        recover(self)
        if self.netconfig is None or not self.netconfig.enabled:
            self._maybe_reassert_weights(client)
            return
        with self._settlement_cfg(int(client.current_block())):  # type: ignore[attr-defined]
            self._maybe_reassert_weights(client)

    def _poll_once(self, client: object, store: object, window_source: object) -> None:
        """One manifest poll. The loop's position is ``state.last_handled_*``
        (persisted): a round is HANDLED (scored, rejected or resync-held) once
        its judgement is persisted together with a ``pre_round`` snapshot and a
        ``followup_round_id`` marker; its follow-up (weights, receipt,
        benchmarks, Cascade step) then runs and clears the marker. A follow-up
        that raises is finished at the top of the next poll BEFORE anything
        newer is judged (in-process: from the kept closure; after a restart:
        by re-judging the round from its snapshot — a replacement, never a
        second application)."""
        from ..shared.hippius import read_latest_manifest
        from .settlement_transaction import recover

        recover(self)

        # The activation tally runs once per boundary (cheap otherwise) ahead
        # of the manifests, so a lock-in seen this poll governs the settlement
        # judged this poll.
        self._activation_tick(client)
        from ..shared.calendar_launch import resolve_calendar_launch

        base = self._base_cfg()
        resolved = resolve_calendar_launch(base, client)
        if resolved is not base:
            if self._cfg_base is None:
                self.cfg = resolved
            else:
                self._cfg_base = resolved
        # Live dashboard telemetry first, every poll: between receipts this is
        # the page's only fresh view of the chain (stage strip + live
        # submissions). Best-effort; never affects the round.
        self._publish_chain_status(client, store)
        self._resume_followup()
        force = self._reject_config_changed()
        raw_latest = read_latest_manifest(store)
        digest_latest = hashlib.sha256(raw_latest.encode("utf-8")).hexdigest()
        if digest_latest == self.state.last_handled_manifest_sha and not force:
            log.debug("manifest unchanged (round=%s sha=%s…); skipping",
                      self.state.last_handled_round_id, digest_latest[:12])
            caught_up = True
        else:
            caught_up = self._handle_pending(client, store, window_source, raw_latest)
        if caught_up:
            # DEC-CA-0048 on the block clock: the successor is crowned the poll
            # the chain reaches forfeit_from_block, manifest or not — but only
            # once every missed settlement has been walked: a settlement whose
            # boundary precedes the gate must be judged under the pre-gate
            # throne (each settlement applies forfeiture by ITS block in
            # process_round), not one this tick forfeited at the head block.
            self._forfeit_tick(client)

    def _handle_pending(self, client: object, store: object, window_source: object,
                        raw_latest: str) -> bool:
        """Walk and handle the pending manifests (oldest first). Returns True
        when the walk reached the latest manifest (caught up), False when it
        stopped early (a hold, a skip, or an incomplete chain walk)."""
        # DEC-CA-0043: from era_king_from_block the manifests are hash-chained
        # and walked forward from the last handled settlement (oldest first);
        # before it, latest.json alone.
        pending = self._pending_manifests(store, raw_latest, self.state.last_handled_round_id)
        if not pending:
            return False
        for i, (raw, digest) in enumerate(pending):
            # DEC-CA-0063: the later manifests carry the trainer's record of
            # this settlement's validator bench report
            self._vbench_later_raw = [r for r, _ in pending[i + 1:]]
            if not self._handle_manifest(client, window_source, load_manifest(raw), digest):
                return False
        return True

    def _skip_unrecorded(self, digest: str, msg: str, *args: object) -> None:
        """Log a skipped-without-recording manifest once per digest."""
        if digest in self._skipped_digests:
            log.debug("skipping manifest sha=%s… again (" + msg + ")", digest[:12], *args)
            return
        self._skipped_digests.add(digest)
        log.warning("skipping manifest sha=%s… WITHOUT recording it: " + msg,
                    digest[:12], *args)

    def _handle_manifest(self, client: object, window_source: object,
                         manifest: TrainingManifest, digest: str) -> bool:
        """Gate, judge and follow up ONE manifest under the config in force at
        its block (DEC-CA-0057 ``cfg_at``; the base config when no overlay is
        adopted). Returns False when it was held or skipped without being
        recorded (the walk stops; the next poll retries), True when it was
        handled."""
        # The overlay in force at created_block is the one in force at the
        # settlement's boundary: an overlay applies at an era start, a
        # boundary on both grids, so the two never straddle it.
        with self._settlement_cfg(int(manifest.created_block)):
            from .settlement_transaction import begin, recover

            recover(self)
            begin(self)
            try:
                handled = self._handle_manifest_inner(client, window_source, manifest, digest)
            except BaseException:
                # A process kill leaves the same undo journal for startup.
                # If already committed, recovery preserves the pending follow-up.
                recover(self)
                raise
            recover(self)  # held/skipped: undo preparation; committed: no-op
            return handled

    def _handle_manifest_inner(self, client: object, window_source: object,
                               manifest: TrainingManifest, digest: str) -> bool:
        from ..shared.chain import ChainError
        from ..shared.hippius import StorageError

        rid = str(manifest.round_id)
        base_seed = int(manifest.round_id)
        # Never move the handled position backwards: an older signed manifest
        # re-published as latest is skipped, never recorded.
        lhb = self.state.last_handled_block
        if lhb is not None and int(manifest.created_block) < int(lhb):
            self._skip_unrecorded(
                digest, "round=%s created_block %d is older than the handled position "
                "(round=%s, block %d)", rid, int(manifest.created_block),
                self.state.last_handled_round_id, int(lhb))
            return False
        # Re-judging the last handled round (re-publish, changed config, or an
        # unfinished follow-up): start from the state BEFORE it was applied, so
        # the new judgement replaces the old one instead of stacking on it.
        rollback = None
        if rid == self.state.last_handled_round_id:
            log.warning("manifest for already-handled round=%s re-judged (sha %s… -> %s…)",
                        rid, (self.state.last_handled_manifest_sha or "")[:12], digest[:12])
            rollback = self._rollback_point()
            if not self._restore_pre_round(rid):
                log.warning("round=%s has no pre-round snapshot (state from an older "
                            "release); the re-judgement applies on top of it", rid)
        log.info(
            "new manifest round=%s entries=%d (%s); gating + scoring …",
            manifest.round_id, len(manifest.entries),
            ",".join(f"{e.role}:uid{e.miner_uid}" for e in manifest.entries),
        )
        pre = self._snapshot(rid)
        # DEC-CA-0058: the settlement state BEFORE this settlement (its digest
        # is stamped on the receipt for satellite continuity)
        self._capture_settlement_prior(rid)
        self._settlement_ladder = None
        self._settlement_king_check = None

        def _not_recorded() -> bool:
            if rollback is not None:
                self._apply_rollback(rollback)
            return False

        # Gate first so a rejected manifest never moves weights. A gate whose
        # evidence cannot be READ holds the round (no latch, no receipt) —
        # see ManifestGateMixin._hold_read_failure. Each gate has its own key.
        try:
            reason = self.check_manifest(manifest)
        except StorageError as e:
            # Warm-start / era-init provenance (promotion record / bench report).
            self._hold_read_failure(f"{rid}|warm_start", e, gate="warm_start_unverifiable")
            return _not_recorded()
        except (ChainError, OSError) as e:
            # The era ref binding's commitment history (chain RPC) — a
            # transient, not a verdict.
            self._hold_read_failure(f"{rid}|era_refs", e, gate="era_ref_unverifiable")
            return _not_recorded()
        self._clear_read_hold(f"{rid}|warm_start")
        self._clear_read_hold(f"{rid}|era_refs")
        if reason == "signature_invalid":
            # Not the trainer's manifest: nothing about it may move the
            # handled position (it is not even evidence a round exists).
            self._skip_unrecorded(digest, "round=%s signature does not verify against the "
                                  "pinned trainer hotkey", rid)
            return _not_recorded()
        if reason is None:
            # Pool-pin gate: the signed snapshot pin must match this
            # validator's own deterministic selection for the round.
            try:
                reason = self.check_pool_pin(
                    manifest, window_source, block=self._epoch_start_block(manifest))
            except StorageError as e:
                self._hold_read_failure(f"{rid}|pool_pin", e, gate="pool_pin_unverifiable")
                return _not_recorded()
            self._clear_read_hold(f"{rid}|pool_pin")
        # Lane B (DEC-CA-0050): READ and JUDGE this settlement's bound expert
        # manifest (plus any unfolded predecessors) BEFORE lane A mutates
        # anything, so a transient read retries the whole settlement cleanly.
        lane_b_rounds = None
        if reason is None:
            self._lane_b_block = self._epoch_start_block(manifest)
            lb_key = f"{rid}|lane_b"
            try:
                lane_b_rounds = self._lane_b_prepare(manifest)
                self._pin_read_first_failure.pop(lb_key, None)
            except LaneBTransient as e:
                if self._gate_read_failed(lb_key, e, gate="lane_b_unverifiable") is None:
                    return _not_recorded()
                # Past the grace: lane A proceeds; this round's lane-B part
                # stays UNFOLDED, so the next settlement's catch-up walk
                # re-judges it.
                lane_b_rounds = [_LaneBRound(
                    rid, self._lane_b_block, [{"error": f"lane_b_unverifiable: {e}"}], [],
                    apply=False)]
        # Lane C (DEC-CA-0056): same — READ and JUDGED before lane A mutates
        # anything. A read failure HOLDS the settlement (no latch, no receipt)
        # until it reads: a missing lane-C verdict would move the vector.
        lane_c_rounds = None
        if reason is None:
            self._lane_c_block = self._epoch_start_block(manifest)
            lc_key = f"{rid}|lane_c"
            try:
                lane_c_rounds = self._lane_c_prepare(manifest)
            except LaneCTransient as e:
                self._hold_read_failure(lc_key, e, gate="lane_c_unverifiable")
                return _not_recorded()
            self._clear_read_hold(lc_key)

        if reason is not None:
            log.warning("rejecting manifest round=%s: %s", rid, reason)
            self._mark_handled(manifest, digest, pre, rejected=True)
            self._persist_state()
            from .settlement_transaction import commit

            commit(self)

            def _followup() -> None:
                # A rejected round still gets a public receipt carrying the
                # gate's reason — visible, not silently absent.
                self._publish_round_receipt(
                    client, manifest, base_seed,
                    reject_reason=reason, window_source=window_source)
        elif not self.king_synced(manifest):
            # The trainer trained the OLD king (incentive lags a dethrone).
            # Hold the KOTH state and keep voting the champion so incentive
            # migrates and the trainer re-syncs — bounded by the safety valve
            # (see _resync_step). A public receipt records why, not a silent
            # skip. The marker is folded in BEFORE _resync_step: it derives
            # the next state from this one, so the marker rides along
            # (demote_to_trained passes the loop fields through explicitly).
            self._mark_handled(manifest, digest, pre)
            self._install_era_promotion(manifest)
            self.state, reject_reason = self._resync_step(manifest)
            self._adopt_era_king(manifest)
            # Lane B folds on EVERY gated-in path, before the handled marker
            # is persisted (idempotent per round).
            lane_b_rows = self._lane_b_commit(lane_b_rounds)
            lane_c_rows = self._lane_c_commit(lane_c_rounds)
            self._persist_state()
            from .settlement_transaction import commit

            commit(self)

            def _followup() -> None:
                reward_uids = self._reward_uids(manifest, None, client)
                weights_vec = self._apply_weights(
                    client, manifest.round_id, reward_uids,
                    boundary=self._epoch_start_block(manifest))
                self._publish_round_receipt(
                    client, manifest, base_seed,
                    reject_reason=reject_reason, window_source=window_source,
                    reward_uids=tuple(reward_uids), weights=weights_vec,
                    **self._lane_b_receipt_kw(lane_b_rows),
                    **self._lane_c_receipt_kw(lane_c_rows),
                )
        else:
            # The epoch block selects the daily snapshot; base_seed rotates
            # the window slice within it.
            windows = self._verdict_windows(
                window_source, base_seed, block=self._epoch_start_block(manifest))
            # process_round mutates the sticky KOTH state atomically (it raises
            # before any mutation on a transient eval/fetch error, leaving
            # state untouched for a clean retry). The round is marked handled
            # as soon as it returns, so a later weight-set failure can NEVER
            # re-run it and double-count the streak/tenure.
            pre_king = self.state.king_hotkey
            # DEC-CA-0063: a settlement whose lane-A bench rows come from the
            # main validator — every other validator HOLDS until the main
            # validator's signed report is published (never a verdict).
            vb_key = f"{rid}|validator_bench"
            try:
                self._vbench_begin(manifest)
            except StorageError as e:
                self._hold_read_failure(vb_key, e, gate="validator_bench_unpublished")
                return _not_recorded()
            vb_point = self._rollback_point()
            try:
                outcome = self.process_settlement(manifest, windows, base_seed)
                none_reason = self._round_none_reason
                # DEC-CA-0055 (v2 + [v2] king_ladder): the king check when due,
                # the ladder step, the court and the key slash — folded in
                # BEFORE the handled marker persists, so a re-judgement
                # replaces them.
                self._v2_settlement_economics(manifest, windows, base_seed, outcome, pre_king)
            except ValidatorBenchHold as e:
                # DEC-CA-0063: a clearer reached the gate (or the king check
                # ran) and the trainer has not recorded the report yet — every
                # validator holds alike; nothing of the judgement stays.
                self._apply_rollback(vb_point)
                self._hold_read_failure(vb_key, e, gate="validator_bench_unrecorded")
                return _not_recorded()
            # DEC-CA-0063: the main validator publishes its report (every sweep
            # this settlement ran, possibly none) BEFORE the settlement is
            # handled; a failed publish undoes the judgement and holds.
            try:
                self._vbench_publish(manifest)
            except StorageError as e:
                self._apply_rollback(vb_point)
                self._hold_read_failure(vb_key, e, gate="validator_bench_publish")
                return _not_recorded()
            if self._vbench_awaiting_record(manifest):
                # The main validator's sweep pass that needed its report: it is
                # published; now hold with every other validator until the
                # trainer records it, and re-judge from those exact bytes.
                self._apply_rollback(vb_point)
                self._hold_read_failure(vb_key, ValidatorBenchHold(
                    f"validator_bench_unrecorded: round={rid} (published; awaiting the "
                    "trainer's record)"), gate="validator_bench_unrecorded")
                return _not_recorded()
            self._clear_read_hold(vb_key)
            # Back in sync — clear any accumulated resync holds so a future
            # desync starts the safety-valve count from zero.
            if self.state.resync_holds or self.state.last_resync_round_id:
                self.state = replace(self.state, resync_holds=0, last_resync_round_id=None)
            self._mark_handled(manifest, digest, pre)
            # Lane B (DEC-CA-0050): fold this settlement's judged expert
            # manifest BEFORE the handled marker persists and before weights,
            # so a new domain win is paid from the settlement that merged it.
            lane_b_rows = self._lane_b_commit(lane_b_rounds)
            lane_c_rows = self._lane_c_commit(lane_c_rounds)
            self._persist_state()
            from .settlement_transaction import commit

            commit(self)

            def _followup() -> None:
                reward_uids = self._reward_uids(manifest, outcome, client)
                weights_vec = self._apply_weights(
                    client, manifest.round_id, reward_uids,
                    boundary=self._epoch_start_block(manifest))
                # The public receipt — strictly after weights, so it records
                # what was actually set (empty vector = the weight extrinsic
                # failed this round).
                if outcome is not None:
                    self._publish_round_receipt(
                        client, manifest, base_seed,
                        outcome=outcome, windows=windows, window_source=window_source,
                        reward_uids=tuple(reward_uids), weights=weights_vec,
                        **self._lane_b_receipt_kw(lane_b_rows),
                        **self._lane_c_receipt_kw(lane_c_rows),
                    )
                else:
                    # Gated in but nothing to score (no king/challenger pair at
                    # any size, or every challenger entry-faulted): a public
                    # record still exists.
                    self._publish_round_receipt(
                        client, manifest, base_seed,
                        reject_reason=none_reason or "no_king_challenger_pair",
                        window_source=window_source,
                        reward_uids=tuple(reward_uids), weights=weights_vec,
                        **self._lane_b_receipt_kw(lane_b_rows),
                        **self._lane_c_receipt_kw(lane_c_rows),
                    )
                # Log-only public benchmarks for a freshly crowned king.
                # Strictly after weights are decided; never affects them.
                self._maybe_run_benchmarks(manifest, outcome)
                # Cascade step — strictly last, so this round's weights and
                # receipt already recorded the outgoing king. Resets the reign
                # clock on a dethrone, records the king's checkpoint, and fires
                # the promotion when the clock is ripe.
                self._cascade_round(manifest, outcome)
        self._run_followup(rid, digest, _followup)
        # Multi-horizon calibration telemetry — after everything
        # consensus-relevant; log-only, inert unless configured.
        self._maybe_calibrate_horizons(manifest, base_seed, window_source)
        return True

    # ── handled position, pre-round snapshot and follow-up ───────────────────

    def _config_identity(self) -> str:
        """Identity of the config (and release) a verdict is reached under:
        the parsed ``chain.toml``, the effective ``[scoring]`` (on-chain
        activations rewrite it in memory) and the package version. A latched
        reject is re-judged once this changes."""
        from .. import __version__

        cfg = self._base_cfg()
        parts = [
            json.dumps(getattr(cfg, "raw", {}) or {}, sort_keys=True, default=str),
            repr(getattr(cfg, "scoring", None)),
            str(__version__),
        ]
        # DEC-CA-0057: the adopted overlays are part of the rules (appended
        # only when there are any, so the identity is unchanged without).
        if self.netconfig is not None and self.netconfig.enabled:
            hist = self.netconfig.history_identity()
            if hist:
                parts.append(hist)
        body = "\n".join(parts)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    def _reject_config_changed(self) -> bool:
        """True when the last handled round is a reject latched under a
        different config identity: it is re-judged this poll (rejects only —
        an accepted judgement is never re-opened by a config change)."""
        cid = self.state.reject_config_id
        if not cid:
            return False
        now = self._config_identity()
        if cid == now:
            return False
        log.warning("round=%s was REJECTED under config %s; config is now %s — "
                    "re-judging it", self.state.last_handled_round_id, cid, now)
        return True

    def _snapshot(self, round_id: str) -> dict:
        """The ``pre_round`` record for ``round_id``: this state (without its
        own snapshot / follow-up marker) and the Cascade state, as they are
        before the round is applied."""
        st = replace(self.state, pre_round=None, followup_round_id=None)
        casc = None
        if self.cascade is not None:
            from . import cascade as cascade_mod

            casc = cascade_mod.dumps(self.cascade.state)
        snap = {"round_id": str(round_id), "state": json.loads(state_mod.dumps(st)),
                "cascade": casc}
        lane_b = self._lane_b_snapshot()
        if lane_b is not None:
            snap["lane_b"] = lane_b
        lane_c = self._lane_c_snapshot()
        if lane_c is not None:
            # DEC-CA-0056: the lane-C fold rides the snapshot (only once lane C
            # is configured, so earlier snapshots keep their exact shape)
            snap["lane_c"] = lane_c
        return snap

    def _rollback_point(self) -> tuple:
        return (self.state, self.cascade.state if self.cascade is not None else None,
                self._lane_c_snapshot(), self._lane_b_snapshot())

    def _apply_rollback(self, point: tuple) -> None:
        """Undo an in-memory pre-round restore (the re-judgement did not
        happen: skipped or held)."""
        self.state = point[0]
        if self.cascade is not None and point[1] is not None:
            self.cascade.state = point[1]
        if len(point) > 2 and point[2] is not None:
            self._lane_c_restore(point[2])
        if len(point) > 3 and point[3] is not None:
            self._lane_b_restore(point[3])

    def _restore_pre_round(self, round_id: str) -> bool:
        """Swap in the state (and Cascade state) from before ``round_id`` was
        applied, in memory only (persisted when the round is handled again).
        False when no snapshot for that round exists."""
        pre = self.state.pre_round
        if not pre or str(pre.get("round_id")) != str(round_id):
            return False
        self.state = state_mod.loads(json.dumps(pre["state"]))
        if self.cascade is not None and pre.get("cascade") is not None:
            from . import cascade as cascade_mod

            self.cascade.state = cascade_mod.loads(pre["cascade"])
        if pre.get("lane_c") is not None:
            self._lane_c_restore(pre["lane_c"])
        if pre.get("lane_b") is not None:
            self._lane_b_restore(pre["lane_b"])
        self._followup = None
        return True

    def _mark_handled(self, manifest: TrainingManifest, digest: str, pre: dict,
                      *, rejected: bool = False) -> None:
        """Record ``manifest`` as the handled position, with its pre-round
        snapshot and a pending follow-up (persisted by the caller)."""
        self.state = replace(
            self.state,
            last_handled_round_id=str(manifest.round_id),
            last_handled_manifest_sha=digest,
            last_handled_block=int(manifest.created_block),
            reject_config_id=(self._config_identity() if rejected else ""),
            pre_round=pre,
            followup_round_id=str(manifest.round_id),
        )
        if self.cascade is not None and pre.get("cascade") is not None:
            # The Cascade state may have been restored in memory to this
            # round's snapshot: persist it with the handled marker.
            self.cascade._persist()

    def _run_followup(self, round_id: str, digest: str, fn: Callable[[], None]) -> None:
        """Run a handled round's follow-up; clear the marker once it completes.
        A raise propagates (the poll stops) and the closure is kept, so the
        next poll finishes it before judging anything newer."""
        blk = self._settlement_block
        if blk is not None:
            # DEC-CA-0057: a follow-up finished on a later poll runs under the
            # same overlay view as the settlement that queued it.
            inner = fn

            def fn() -> None:
                with self._settlement_cfg(blk):
                    inner()
        self._followup = (str(round_id), digest, fn)
        fn()
        self._followup = None
        self.state = replace(self.state, followup_round_id=None)
        self._persist_state()

    def _resume_followup(self) -> None:
        """Finish the handled round's follow-up left by a failed poll. With
        the closure gone (restart), re-judge the round from its pre-round
        snapshot — the judgement is deterministic, so the replacement equals
        the original, and its follow-up then runs normally."""
        rid = self.state.followup_round_id
        if rid is None:
            return
        fu = self._followup
        if fu is not None and fu[0] == rid and fu[1] == self.state.last_handled_manifest_sha:
            log.warning("round=%s: finishing its follow-up (weights / receipt / Cascade "
                        "step) before anything newer", rid)
            self._run_followup(fu[0], fu[1], fu[2])
            return
        if self._restore_pre_round(rid):
            log.warning("round=%s: follow-up did not complete before the restart; "
                        "re-judging it from its pre-round snapshot", rid)
            return
        log.error("round=%s: follow-up pending but no snapshot to re-judge from; "
                  "dropping the marker (its receipt / Cascade step may be missing)", rid)
        self.state = replace(self.state, followup_round_id=None)
        self._persist_state()

    def _persist_state(self) -> None:
        """Atomic write (tmp + fsync + rename) keeping the previous good file
        as ``.bak`` — a crash never leaves a torn state file."""
        from ..shared.atomic import write_text_atomic

        try:
            write_text_atomic(self.cfg.validator.state_db_path,
                              state_mod.dumps(self.state), backup=True)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist validator state: %s", e)
            raise


class ValidatorStateUnreadable(RuntimeError):
    """``state_db_path`` exists but neither it nor its ``.bak`` parses.
    Raised instead of starting fresh: a fresh state resets the king's tenure,
    streaks, court and the handled position — the validator would re-judge
    already-settled rounds and vote a different throne than its peers."""


def _load_state(path: str) -> ChampionState:
    """Load persisted champion state from ``state_db_path`` (JSON), falling
    back to ``<path>.bak`` when the main file is torn. Absent (both) ⇒ a fresh
    state; present but unparseable (both) ⇒ :class:`ValidatorStateUnreadable`
    — never a silent fresh start."""
    from ..shared.atomic import read_text_with_backup

    state, src = read_text_with_backup(path, state_mod.loads)
    if src == "absent":
        return ChampionState()
    if state is None:
        raise ValidatorStateUnreadable(
            f"validator state at {path} (and {path}.bak) exists but does not parse; "
            "refusing to start fresh. Restore the file (or remove both deliberately "
            "to start from a blank state / the receipt bootstrap)")
    if src == "backup":
        log.warning("validator state at %s is torn/unreadable; resumed from %s.bak",
                    path, path)
    return state


# First-boot receipt walk: each verified receipt costs one storage GET, so the
# court derivation is bounded — a trail whose king never changed stops here
# rather than reading every indexed round. 64 rounds ≈ a month of 12h rounds,
# far past any plausible run of consecutive same-king receipts hiding a court.
_BOOTSTRAP_MAX_RECEIPT_READS = 64


def _bootstrap_state_from_receipts(
    store: object, anchor: str, *, keep_former_kings: int = 0
) -> ChampionState | None:
    """Champion (and court) inherited from the signed receipt trail, or ``None``.

    First-boot inheritance for a validator with no local state: the throne
    otherwise lives only in each validator's private state DB, so a validator
    joining mid-reign would judge the next manifest blind (``king_synced``
    treats an unset champion as synced) and crown whichever king it happened
    to see win first — a different champion than every validator that
    witnessed the real dethrone.

    ``anchor`` is the pinned receipt-signing ss58 (``[manifest]
    validator_hotkey``, falling back to ``trainer_hotkey``) — the same trust
    anchor the validator already applies to manifests, extended once, at
    first boot, to the receipt trail. Reads the anchor's
    ``receipts/<anchor>/latest.json`` (legacy shared pointer as fallback) and
    adopts the throne recorded by a *scored* receipt whose signature
    verifies. When ``latest.json`` is a hold/rejected receipt (verdict-less
    by construction), the receipt *index* is consulted — but only as an
    UNTRUSTED pointer to candidate round ids: nothing is adopted except from
    a per-round receipt whose signature verifies against the anchor.

    ``keep_former_kings`` (``[scoring] reward_prior_kings``) additionally
    seeds ``former_kings``: walking the index newest-first, each *distinct*
    king recorded by a verified scored receipt is a reign, in recency order —
    exactly the court ``apply_round`` would have accumulated had this
    validator witnessed the dethrones itself. Without it a joining validator
    votes the whole reward mass on the current king while established
    validators split it across the court, and its vtrust is capped at the
    consensus share of the top slot (the court head's consensus weight). A
    trail too short to fill the court seeds what it has;
    the rest accrues from live dethrones. The walk is bounded
    (``_BOOTSTRAP_MAX_RECEIPT_READS``) so first boot cannot stall on a long
    index.

    Only a TRULY fresh validator reaches this path (see the caller's gate) —
    deliberately. ``demote_to_trained`` clears ``former_kings`` because a
    dropped champion is not an honourably retired one; a bootstrap that
    re-seeded the court on a validator with existing state would resurrect
    exactly what the demotion semantics erased.

    Anything short of a verified scored receipt — missing objects, unreadable
    JSON, a bad signature, a genesis throne (``king_hotkey`` unset) — yields
    ``None`` for the champion and the caller proceeds with the stock
    blank-slate behaviour. Storage faults must never block validator startup.
    """
    if not anchor:
        return None
    from ..shared.hippius import (
        RECEIPT_INDEX_KEY,
        RECEIPT_LATEST_KEY,
        receipt_latest_key,
        receipt_round_key,
    )
    from ..shared.receipt import load_receipt, verify_receipt_signature

    def _verified_throne(key: str) -> tuple[str, int, str] | None:
        """(king_hotkey, king_uid, round_id) from a verified scored receipt."""
        try:
            text = store.get_text(key)
        except Exception:  # noqa: BLE001 — absent/unreachable key ⇒ next candidate
            return None
        try:
            receipt = load_receipt(text)
        except Exception as e:  # noqa: BLE001
            log.warning("receipt bootstrap: unreadable receipt at %s (%s); skipped", key, e)
            return None
        if not verify_receipt_signature(receipt, anchor):
            log.warning("receipt bootstrap: receipt at %s is not signed by the pinned "
                        "hotkey %s…; skipped", key, anchor[:8])
            return None
        v = receipt.verdict
        if receipt.status != "scored" or v is None or not v.king_hotkey or v.king_uid is None:
            return None
        return str(v.king_hotkey), int(v.king_uid), str(receipt.round_id)

    throne = (_verified_throne(receipt_latest_key(anchor))
              or _verified_throne(RECEIPT_LATEST_KEY))
    rows: list = []
    if throne is None or keep_former_kings > 0:
        try:
            rows = json.loads(store.get_text(RECEIPT_INDEX_KEY)).get("rounds", [])
        except Exception:  # noqa: BLE001 — no index ⇒ nothing beyond latest.json
            rows = []
    # Index rows are chronological (oldest first, capped at most-recent); walk
    # newest-first. The first verified scored round supplies the champion when
    # latest.json did not; every older distinct king fills the court.
    court: list[str] = []
    reads = 0
    for row in reversed(rows):
        if throne is not None and len(court) >= keep_former_kings:
            break
        if str(row.get("status")) != "scored":
            continue
        if reads >= _BOOTSTRAP_MAX_RECEIPT_READS:
            log.warning("receipt bootstrap: stopping after %d receipt reads with "
                        "%d/%d former kings seeded; the rest accrue from live "
                        "dethrones", reads, len(court), keep_former_kings)
            break
        reads += 1
        found = _verified_throne(receipt_round_key(str(row.get("round_id", "")), anchor))
        if found is None:
            continue
        if throne is None:
            throne = found
        elif found[0] != throne[0] and found[0] not in court:
            court.append(found[0])
    if throne is None:
        return None
    king_hotkey, king_uid, round_id = throne
    log.info("receipt bootstrap: adopting champion %s (uid %d) from signed scored "
             "receipt round=%s; former_kings=%s", king_hotkey, king_uid, round_id,
             court or "[]")
    return ChampionState(king_hotkey=king_hotkey, king_uid=king_uid,
                         former_kings=tuple(court))


def _build_cascade(cfg: ChainConfig) -> CascadeController:
    """Construct the Cascade controller from config, restoring the persisted reign
    clock + checkpoint log so it resumes across restarts."""
    from ..shared.era import effective_reign_threshold_rounds
    from .cascade import CascadeController, load_state

    state_path = Path(cfg.validator.cascade_state_db_path)
    return CascadeController(
        reign_days=cfg.scoring.cascade_reign_days,
        state=load_state(state_path),
        state_path=state_path,
        # The clock divides by the ROUND length in force, not a fixed day, so
        # the threshold keeps meaning "survived N challenges" under any cadence.
        round_cfg=cfg.round,
        # DEC-CA-0043: from tenure_blocks_from_block the threshold is
        # cascade_reign_blocks on the grid in force at the block.
        threshold_fn=lambda b: effective_reign_threshold_rounds(cfg.round, cfg.scoring, b),
    )


def build_runner(
    *,
    chain_toml: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
    eval_host_fn: Callable[[], RemoteHost | None] | None = None,
    private_eval_host_fn: Callable[[], RemoteHost | None] | None = None,
    private_eval_workers: int = 1,
    private_eval_timeout_s: int = 3600,
) -> ValidatorRunner:
    """Construct a runner from ``chain.toml``, restoring persisted champion
    state. Wallet/chain wiring for live weight-setting is attached by
    ``cascade-validator`` (see main.py). ``eval_host_fn`` (optional) resolves
    the GPU pod to offload heavy evals to — re-invoked per eval, so an elastic
    provisioner-rented pod is picked up lazily; the wallet stays on this box."""
    from ..shared.config import load_chain_config

    cfg = load_chain_config(chain_toml)
    state = _load_state(cfg.validator.state_db_path)
    if (cfg.validator.bootstrap_from_receipts and state.king_hotkey is None
            and state.rounds_seen == 0 and not state.former_kings):
        # Truly fresh validator (no champion, no history): inherit the throne
        # from the signed receipt trail before the first manifest is judged.
        from ..shared.hippius import open_manifest_store

        anchor = cfg.manifest.validator_hotkey or cfg.manifest.trainer_hotkey
        try:
            adopted = _bootstrap_state_from_receipts(
                open_manifest_store(cfg.storage), anchor,
                keep_former_kings=cfg.scoring.reward_prior_kings)
        except Exception as e:  # noqa: BLE001 — storage must never block startup
            log.warning("receipt bootstrap skipped (%s); starting blank", e)
            adopted = None
        if adopted is not None:
            state = adopted
    # Cascade is opt-in ([scoring] cascade_enabled); off ⇒ no controller is wired
    # and the runner is pure KOTH.
    cascade = _build_cascade(cfg) if cfg.scoring.cascade_enabled else None
    # DEC-CA-0045: the resolved rollover persists beside the champion state.
    activation_store = ActivationStore(
        Path(cfg.validator.state_db_path).with_name("activation_state.json"))
    # DEC-CA-0057: inert (no resolver) while [netconfig] owner_hotkey is empty.
    netconfig = None
    if cfg.netconfig.enabled:
        from ..shared.hippius import open_manifest_store
        from ..shared.netconfig import NetConfigResolver

        netconfig = NetConfigResolver(
            cfg, state_path=Path(cfg.validator.state_db_path).with_name("netconfig_state.json"),
            store=lambda: open_manifest_store(cfg.storage), signal=True)
    return ValidatorRunner(
        cfg=cfg, state=state,
        cache_dir=cache_dir, device=device, cascade=cascade, eval_host_fn=eval_host_fn,
        private_eval_host_fn=private_eval_host_fn, private_eval_workers=private_eval_workers,
        private_eval_timeout_s=private_eval_timeout_s,
        activation_store=activation_store, netconfig=netconfig,
    )
