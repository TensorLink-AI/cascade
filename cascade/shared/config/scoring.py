"""[eval] and [scoring] config plus the block-scheduled margin/scoring resolvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    """Held-out eval windows scored each round (same set for king and challenger).

    ``eval_dataset`` is the identifier the manifest carries and the validator
    matches on. The windows are drawn from an owner-controlled private pool
    and the *slice rotates per round*
    (seeded by the round's block hash) — TIME-style contamination resistance, so
    a generator cannot distribution-match a fixed public benchmark. The concrete
    pool loader (``window_pool``) is a boundary; the seeded rotation/selection
    lives in ``cascade.validator.windows``.
    """

    eval_dataset: str
    window_pool: str
    num_samples: int
    n_windows: int
    context_length: int
    horizon: int
    # ── Public-benchmark logging (log-only; never feeds scoring/weights) ──────
    # Off by default. When on, the validator runs the out-of-process sidecar
    # (``benchmarks/``) on a dethroned challenger and logs GIFT-Eval/BOOM/TIME
    # numbers. ``benchmark_suites = ()`` runs all three; ``benchmark_num_samples
    # = 0`` reuses ``num_samples``; ``benchmark_max_series = 0`` runs the full
    # benchmark (use a small cap for a smoke run). Defaults keep old toml loading.
    run_benchmarks: bool = False
    benchmark_project_dir: str = "benchmarks"
    benchmark_suites: tuple[str, ...] = ()
    benchmark_num_samples: int = 0
    benchmark_max_series: int = 0
    # ── Public-benchmark no-regression gate (CONSENSUS; see gift_gate) ────────
    # Consumed only when ``[scoring] gift_gate_mode`` != "off". Runs the
    # gift-eval sidecar on the primary king/challenger checkpoints on a
    # private-pool win. ``gift_gate_datasets`` pins the config subset
    # ("" = full 97 configs); ``gift_gate_num_samples = 0`` reuses
    # ``num_samples``; ``gift_gate_data_dir`` points at pinned benchmark data on
    # the validator ("" = the suite's own env vars). Defaults keep old toml.
    gift_gate_datasets: str = ""
    gift_gate_num_samples: int = 0
    gift_gate_data_dir: str = ""
    gift_gate_timeout_s: int = 3600
    # Cascade duel-eval coverage (see cascade.validator.cascade). Cap on datasets
    # per suite when the trainer scores the duel checkpoints on GIFT-Eval / BOOM /
    # TIME. ``0`` = the FULL battery (all configs) — the default, since Cascade's
    # promotion should see the whole eval. Kept separate from the log-only
    # ``benchmark_max_series`` so tightening telemetry never quietly shrinks the
    # Cascade decision. Set a positive cap only to speed up testnet iteration.
    cascade_bench_max_series: int = 0
    # ── Multi-horizon calibration telemetry (LOG-ONLY; never feeds the verdict)
    # Calibrates the scored horizon ladder (DEC-CA-0037). Empty tuple = off (inert
    # default). When set, the validator scores the round's duel checkpoints at
    # each extra horizon on its OWN seeded, even-by-domain, eligibility-filtered
    # window draw AFTER the receipt publishes, and logs per-(horizon, domain)
    # geomeans + king/challenger deltas. This is the calibration measurement
    # for the scored 16/64/256/720 ladder — candidate spread vs noise per cell
    # feeds the margin recalibration and the cell admission mask. The scored
    # flip itself is a separate, coordinated consensus release; these knobs
    # deliberately cannot cause one.
    calib_horizons: tuple[int, ...] = ()
    calib_windows: int = 256        # windows per horizon (even-by-domain draw)
    calib_num_samples: int = 32     # sample paths per window (cost control)
    # ── Scored horizon ladder (CONSENSUS, block-gated) ───────────────────────
    # ``scored_from_block = 0`` keeps the single-horizon verdict draw
    # everywhere (byte-identical history). When ``scored_horizons`` is set and
    # a round's epoch-boundary block is >= ``scored_from_block``, the round's
    # verdict — and the heat screen, which must rank on the metric the duel
    # judges — draws its windows as a horizon ladder instead: each horizon
    # gets its own seeded even-by-domain draw over the snapshot's raw series
    # (``n_windows`` split evenly across horizons, rung sizes equalised), and
    # the pooled rows flow through the unchanged round statistic. With equal
    # rung sizes the geometric mean weights horizons equally, and the cluster
    # bootstrap resamples upstream feeds jointly across horizons (rows from
    # one feed share a cluster key at every horizon). Block-gating is the
    # ``mix_from_block`` rollout shape: every validator AND the trainer must
    # run the same values before the activation block, or verdicts fork from
    # that block on; audit replay applies each round's own rule via the
    # receipt's ``epoch_start_block``.
    scored_horizons: tuple[int, ...] = ()
    scored_from_block: int = 0
    # Ceiling, in hours, on one public-benchmark sweep of a duel checkpoint
    # when the battery is uncapped (``cascade_bench_max_series = 0``): the
    # remote bench timeout trainer.main hands the BenchPlan (a capped battery
    # gets a fixed 30 min). Only bounds a crashed/wedged bench; inert while
    # ``[scoring] cascade_enabled`` is off (no bench plan is built).
    bench_hold_max_hours: float = 2.0
    # ── DEC-CA-0063 validator sweep: concurrency (OPERATOR-LOCAL) ────────────
    # Only the MAIN validator sweeps; every other validator replays its signed
    # rows, and the sidecar's report is byte-identical for every worker count
    # and with or without pairing — so these are per-box tuning, never
    # consensus. ``bench_workers`` = the sidecar's ``--workers``: ``"auto"``
    # (sized to the GPU's free memory / CPU cores, benchmarks/README.md) or an
    # int >= 1 (1 = the sequential sweep). ``bench_paired`` sweeps the
    # checkpoints a decision needs together in one pass (challenger + an
    # unswept king; king + init at the king check) so each benchmark dataset
    # is built once; see docs/OPERATOR.md for how a paired pass is charged
    # against ``[v2] bench_sweep_hours``.
    bench_workers: str = "auto"
    bench_paired: bool = True
    # ── Jittered round mix (DEC-TB-0003 port; CONSENSUS, block-gated) ────────
    # ``mix_from_block = 0`` keeps the legacy uniform permutation everywhere
    # (byte-identical history). When set to an epoch-boundary block, rounds at
    # or after that block draw a Dirichlet-jittered, class-rotated,
    # bag-filtered mix instead — every validator AND the trainer must run this
    # code (with the same value) before the activation block, or verdicts fork
    # from that block on. Audit replay applies each round's own rule via the
    # receipt's epoch_start_block. ``mix_target_windows`` (0 = n_windows) is
    # the jittered draw size — smaller than the pool is what gives the
    # Dirichlet room (calibration: 1200 of a ~2400 pool rotates the mix hard;
    # 2000 barely moves it).
    mix_from_block: int = 0
    mix_jitter_alpha: float = 4.0
    mix_block_slots: int = 8
    mix_class_keep_frac: float = 0.7
    mix_series_bag_frac: float = 0.7
    mix_target_windows: int = 0
    # DEC-CA-0032 two-tier domain split: scarce domains draw at capacity, the
    # rest split evenly under a tight Dirichlet. Consensus constants — flip
    # only via release-then-activate at mix_tier_from_block (0 = flat split).
    mix_tier_from_block: int = 0
    mix_tier_jitter_alpha: float = 75.0


@dataclass(frozen=True)
class ScoringConfig:
    win_margin_start: float
    win_margin_end: float
    margin_warmup_rounds: int
    min_windows: int
    bootstrap_B: int
    bootstrap_alpha: float
    dethrone_cp: int
    # Scheduled change of the fresh-king margin (the epoch_blocks_prev /
    # epoch_activation_block shape): rounds whose epoch boundary precedes
    # ``margin_activation_block`` are judged at ``win_margin_start_prev``,
    # rounds from it on at ``win_margin_start``. Consensus — every validator
    # resolves the value from the round's block, so restart timing never
    # forks a verdict, and cascade-audit replays each round under its own
    # value. 0.0 / 0 = no scheduled change. Keep the pair in place after the
    # flip: retiring it makes pre-flip receipts fail the params replay.
    win_margin_start_prev: float = 0.0
    margin_activation_block: int = 0
    # Second (older) step of the same schedule, so a two-transition history
    # (e.g. 2% -> 1% -> 0.5%) stays fully audit-resolvable: rounds before
    # ``margin_activation_block2`` judge at ``win_margin_start_prev2``, between
    # the two blocks at ``win_margin_start_prev``, from ``margin_activation_block``
    # on at ``win_margin_start``. Both default 0/0.0 = single transition,
    # bit-identical to the one-step form. Same "keep the pair after the flip"
    # rule (DEC-CA-0016): retiring a step makes that era's receipts fail replay.
    win_margin_start_prev2: float = 0.0
    margin_activation_block2: int = 0
    # Cohort-duel family-wise correction (DEC-CA-0038). Bonferroni's alpha/k
    # over-protects the king: the k challengers share the king's scores and one
    # window draw, so the tests are strongly positively correlated and the
    # union bound is loose (at k=11 it reads the 0.45th bootstrap percentile).
    # From a round whose epoch boundary is >= this block the cohort is judged
    # under a shared-resample max-T (Westfall-Young step-down) instead — exact
    # under the real correlation, no alpha/k. CONSENSUS, block-gated exactly
    # like margin_activation_block: every validator resolves the rule from the
    # round's block, so restart timing never forks a verdict and audit replays
    # each round under its own rule. 0 = Bonferroni forever. Bit-identical at
    # k <= 1 (no multiplicity), so single-challenger rounds never change.
    cohort_maxt_from_block: int = 0
    # DEC-CA-0048 king forfeiture — CONSENSUS: every validator applies the same
    # list from the same settlement boundary or weights fork. Hotkeys listed
    # here leave the throne (abdicating to the most recent eligible former
    # king) and the court from `forfeit_from_block`; 0 / empty = inert.
    # With a non-empty list and ``forfeit_from_block = 0`` under an enabled
    # ``[activation]``, the block is DECIDED ON CHAIN (DEC-CA-0045 machinery):
    # each validator on the release adds a ``forfeit-<hash of the list>``
    # segment to its readiness note, the first boundary where the segment
    # holds ``threshold`` of eligible stake locks in, and the NEXT boundary is
    # the forfeiture. A typed block is the owner override.
    forfeit_hotkeys: tuple[str, ...] = ()
    forfeit_from_block: int = 0
    # CONSENSUS: the hotkey crowned when the KING is forfeited (tenure reset,
    # its era king leg trained from the first era judged under the
    # forfeiture). "" = the most recent former king not itself forfeited,
    # else a vacant throne. Never itself in ``forfeit_hotkeys``.
    forfeit_successor_hotkey: str = ""
    # Increment-margin activation (DEC-CA-0039, block-gated). Under margin_mode
    # "level" the dethrone bar is a fixed fraction of the king's ABSOLUTE score
    # (win_margin_* ), so a maturing lineage whose per-round gains fall below
    # the floor becomes structurally undethroneable. From a round whose epoch
    # boundary is >= this block the margin is judged in INCREMENT units instead
    # (DEC-CA-0027: the bar prices a fraction of the typical per-round
    # improvement over the shared warm-start init), so dethrones stay clearable
    # as the lineage converges. Needs the init scored on the round's windows —
    # already true while init_gate_mode != "off". CONSENSUS, resolved per round
    # like the other gates; audit replays each round under its own rule. 0 =
    # keep `margin_mode` for every round (no scheduled flip).
    increment_from_block: int = 0
    # Increment margin ON COHORT rounds (DEC-CA-0039 amended 2026-09-19,
    # block-gated). Until this block a cohort round (k > 1) under the max-T is
    # judged in LEVEL units whatever `increment_from_block` says — the joint
    # bound only knew the level statistic, so the increment margin never
    # judged a live funded round. From a round whose epoch boundary is >= this
    # block the max-T bound is the %-of-increment statistic (the baseline
    # rides the shared resample), the same unit the single duel judges in.
    # CONSENSUS, resolved per round like the other gates; audit replays each
    # round under its own rule. 0 = cohort rounds stay level-judged.
    cohort_maxt_increment_from_block: int = 0
    # Multivariate scoring activation (DEC-CA-0041, block-gated). From a round
    # whose epoch boundary is >= this block, a multivariate window's channels
    # are averaged into ONE per-window contribution (GIFT-Eval weighting — the
    # window counts once, not once per channel), so the round point statistic
    # agrees with the source-cluster bootstrap that already treats an MV window
    # as one resampling unit. CONSENSUS, resolved per round like the other
    # gates; audit replays each round under its own rule. BIT-IDENTICAL on any
    # univariate pool (one channel per window). 0 = per-channel forever.
    mv_score_from_block: int = 0
    # Entry-fault verdicts (block-gated). From a settlement whose epoch
    # boundary is >= this block, a CHALLENGER whose evaluation fails
    # deterministically (non-finite / wrong-shape forecasts, a tampered
    # checkpoint, a wrapper that will not import / construct, a malformed
    # pointer) LOSES that duel — dropped from the judged cohort, reason in the
    # verdict note — and the settlement proceeds. Before it such a failure
    # raised out of the round and was retried forever, wedging every later
    # settlement. A KING (or baseline) failure and every transient (storage,
    # network, CUDA OOM) still retry. CONSENSUS, resolved per settlement like
    # the other gates. 0 = off (the pre-gate behaviour).
    entry_fault_from_block: int = 0
    # Breadth floor for the verdict: below this many distinct window clusters
    # (upstream feeds, from pool metadata ``source``) the round is inconclusive.
    # 0 disables; pools without ``source`` metadata are unaffected. Default keeps
    # older chain.toml loadable.
    min_clusters: int = 0
    # Reward routing: equal weight is split across the current king plus up to
    # ``reward_prior_kings`` previous distinct kings still registered; with none
    # registered, all weight burns to ``burn_uid``. ``reward_prior_kings = 0``
    # reproduces pure winner-take-all. Defaults keep older chain.toml loadable.
    reward_prior_kings: int = 0
    burn_uid: int = 0
    # Geometric decay of the reward across [current king, prior kings by recency]:
    # the king gets a share ∝ 1, the next ∝ king_decay, the next ∝ king_decay²,
    # … normalised. ``king_decay = 1.0`` is the flat equal split (back-compat);
    # ``< 1`` skews to the current king so it is unambiguously the highest-
    # incentive UID (which is how the trainer identifies the king — a flat split
    # ties it with prior kings). E.g. 0.5 gives a 4-king court shares
    # ≈ 0.53/0.27/0.13/0.07.
    king_decay: float = 1.0
    # King-resync safety valve: max consecutive rounds the validator holds the
    # throne for a champion whose trained king disagrees (incentive lag). Past
    # this many holds it abandons the stuck champion and adopts the trainer's
    # trained king (see cascade.validator.state.demote_to_trained). A champion
    # with no usable commitment can never re-sync, so an unbounded hold would
    # wedge the subnet; this bounds recovery. ``<= 0`` disables the valve (hold
    # indefinitely — the pre-safety-valve behaviour). Normal resync is 1 round,
    # so the default leaves generous slack before tripping.
    king_resync_max_rounds: int = 5
    # Public-benchmark no-regression gate (see cascade.eval.koth / .gift_gate):
    # "off" (default) | "shadow" (compute + log, verdict unchanged) | "enforce"
    # (AND into the dethrone decision). ``gift_gate_tolerance`` is the relative
    # slack the challenger may be worse by on gift-eval; ``gift_gate_min_configs``
    # is the shared-config floor below which the gate is uncomputable
    # (→ inconclusive). Reuses ``bootstrap_B``/``bootstrap_alpha``.
    gift_gate_mode: str = "off"
    gift_gate_tolerance: float = 0.03
    gift_gate_min_configs: int = 15
    # Three-suite bench gate (DEC-CA-0054; cascade.eval.bench_gate): the
    # challenger may not be significantly worse on GIFT-Eval, BOOM or TIME
    # (paired, trainer-signed rows) and its six-number geomean must beat the
    # king's. Same off/shadow/enforce rollout as the gift gate.
    bench_gate_mode: str = "off"
    bench_gate_min_items: int = 10
    # DEC-CA-0061 rule v2 ("avg_noninf_winrate", from [v2]
    # bench_gate_v2_from_block): a suite whose ratio LCB (challenger / king,
    # per task, alpha/4) exceeds 1 + bench_gate_tolerance is a significant
    # regression; the suite-equal task win rate must be >= bench_gate_min_win_rate.
    bench_gate_tolerance: float = 0.01
    bench_gate_min_win_rate: float = 0.5
    # Init-baseline floor (see cascade.eval.koth.KothParams.init_gate_mode):
    # the shared warm-start init scored on the SAME duel windows as an absolute
    # floor under the crown — a challenger that beat the king but is worse than
    # "do nothing" cannot be crowned. Same progression as the gift gate:
    # "off" (default) | "shadow" (floor computed + recorded on the receipt,
    # verdict unchanged) | "enforce" (AND-ed into the win; can only BLOCK a
    # dethrone, never grant one — king retention untouched, worst case = king
    # holds). The measured motivation (2026-08-26): the raw init outscored
    # every trained entrant post-mix-change, invisibly, because KOTH has no
    # null baseline. Rounds with no warm_start_ckpt have no baseline — the
    # floor simply cannot run (recorded None). [scoring] is fleet-consensus:
    # identical across all validators or verdicts fork — release-then-activate,
    # shadow first, never straight to enforce.
    init_gate_mode: str = "off"
    init_gate_tolerance: float = 0.0
    # Margin denomination (DEC-CA-0027; see cascade.eval.koth.MARGIN_MODES).
    # "level" (default) = LCB of (king − chal)/king — exact current behaviour.
    # "increment" = the warm-start-era rule: the shared init is scored as a
    # third paired reference and the margin prices a fraction of a typical
    # per-round increment. [scoring] is fleet-consensus (not digest-bound):
    # keep identical across validators or verdicts fork. Arm together with the
    # warm-start era flip; rounds with no warm_start_ckpt (random init) are
    # judged at "level" automatically.
    margin_mode: str = "level"
    margin_increment_floor: float = 0.01
    # Cascade — king-reign promotion / warm-start (see cascade.validator.cascade).
    # ``cascade_enabled`` is the master switch: off (default) ⇒ pure KOTH, no
    # reign clock, no public-benchmark scoring, no warm-start promotion. When on,
    # and the reigning king holds the throne ``cascade_reign_days`` worth of
    # BLOCKS (7200/day at 12 s, anchored to the manifest's epoch block so every
    # validator fires the same round) undethroned, the reign's best checkpoint
    # (lowest geomean of the six GIFT-Eval / BOOM / TIME CRPS+MASE numbers the
    # trainer stamps onto the signed manifest) is installed as the warm-start
    # init; the king persists on the throne with a fresh reign clock
    # (DEC-CA-0004). The clock is persisted and survives restarts.
    cascade_enabled: bool = False
    # Threshold in ROUNDS ("survived N challenges"). Field name kept for
    # call-site compatibility; the config key is ``cascade_reign_rounds``.
    cascade_reign_days: int = 7
    # Promotion ENVELOPE knobs (DEC-CA-0013, propose-and-verify). The trainer
    # selects the promoted warm-start set; validators verify the signed
    # declaration against these bounds — so, like cascade_reign_rounds, they
    # are fleet-consensus values ([scoring] is not in contract_digest; keep
    # them identical across validators or their accept/reject verdicts fork).
    # cascade_top_k caps how many member checkpoints one promotion may declare
    # (the trainer may declare fewer); cascade_quality_epsilon is the quality
    # floor — every member's bench geomean must sit within (1 + epsilon) of
    # the best verifiable score in the reign, so diversity can never be bought
    # with a materially worse init.
    cascade_top_k: int = 3
    cascade_quality_epsilon: float = 0.05
    # Block-gated contract transition (the DEC-CA-0016/0019 release-then-
    # activate pattern applied to contract_digest itself). When BOTH are set,
    # a manifest whose epoch-boundary block precedes ``contract_from_block``
    # is accepted against ``prior_contract_digest`` instead of the digest
    # recomputed from the local [training] — so validators can restart onto a
    # release that changes the training contract WITHOUT sacrificing the
    # round currently in flight under the old contract, and the fleet needs
    # no synchronized restart window. Rounds at/after the block accept only
    # the new digest. [scoring] lives outside contract_digest, so these keys
    # never perturb the digest they gate. "" / 0 = gate off (legacy exact
    # match). Fleet-consensus values like every [scoring] key: keep identical
    # across validators. Retire the pin (back to ""/0) in the release AFTER
    # the transition round is beyond every validator's scoring horizon.
    prior_contract_digest: str = ""
    contract_from_block: int = 0
    # Declared-contract gating (DEC-CA-0036). From this block on, a manifest is
    # gated on the LOCKED terms only — the trainer publishes its full training
    # contract in the signed manifest (``contract_body``), the validator checks
    # that body re-hashes to the declared ``contract_digest`` and that its
    # locked projection (:func:`~cascade.shared.manifest.locked_contract_terms`
    # — arch_preset, base_arch_digest, expected_gpu, train_seed_salt, and each
    # extra size's preset/GPU pin) equals the local one. Every other [training]
    # term is RECORDED, not gated, so a trainer-side fix (runtime image,
    # recipe, budget) ships the same day instead of waiting on a fleet restart.
    #
    # 0 = off: the legacy strict gate (full digest equality against the local
    # [training]) applies to every round. Flipping this on is the LAST
    # contract change that needs a coordinated deploy — after it, [training]
    # edits never again reject a round on an un-upgraded validator. Rounds
    # before the block still take the strict path, so the flip is replayable
    # and cascade-audit judges each round under its own rule.
    #
    # A manifest with no published body always takes the strict path, whatever
    # this says — the gate can only relax once the trainer actually declares.
    declared_contract_from_block: int = 0
    # ── Era king (DEC-CA-0043, CONSENSUS, block-gated) ───────────────────────
    # From a settlement whose epoch boundary is >= this block the validator
    # verifies the ERA ENVELOPE (cascade.shared.era): the manifest's era
    # stamp is on the era grid with seeds from the previous era's start
    # block; the init is members_gen(era)[era % k] of the latest promotion
    # whose effective_era <= era; the king entry is the champion and its
    # pointer is the one this validator judged in the crowning settlement (or
    # the era's first king leg); every entry's ref is the hotkey's revealed
    # commitment AS OF its train_block; manifests are hash-chained
    # (prev_round_id) and walked in order, never jumped; the same-GPU
    # fallback of the GPU gate is lifted when no GPU is pinned. Release-then-
    # activate with the external validators; audit replays each settlement
    # under its own block. 0 = off (bit-identical to the ungated validator).
    # Must equal ``[round] rolling_from_block`` and be >=
    # ``cohort_maxt_increment_from_block`` >= ``cohort_maxt_from_block``
    # (load-checked).
    era_king_from_block: int = 0
    # Tenure / ripeness re-denominated to BLOCKS from this block (CONSENSUS):
    # ``margin_warmup_blocks`` replaces ``margin_warmup_rounds`` and
    # ``cascade_reign_blocks`` replaces ``cascade_reign_rounds``, each
    # expressed per settlement on the grid in force at the round's block, so a
    # 4× faster grid does not 4× the decay and promotion clocks in wall-time.
    # The king's tenure is then blocks reigned ÷ grid (a king crowned on the
    # old grid keeps its wall-time tenure across the switch). 0 = off (the
    # two ``*_blocks`` knobs are inert). Must equal the ROLLOVER block.
    tenure_blocks_from_block: int = 0
    margin_warmup_blocks: int = 0
    cascade_reign_blocks: int = 0
    # King-resync safety valve in BLOCKS from ``tenure_blocks_from_block``
    # (``king_resync_max_rounds`` counted settlements: 5 × 900 blocks trips in
    # 15 h, not the 60 h it meant). 0 = keep counting rounds.
    king_resync_max_blocks: int = 0
    # Dethrone bar v2 (DEC-CA-0049, CONSENSUS). The increment-unit LCB
    # (DEC-CA-0039) made the 1% -> 0.5% bar trivially easy (recent increment
    # LCBs span about -0.84..+0.47), so from ``margin_v2_from_block`` a fresh
    # king's bar is ``win_margin_start_v2`` decaying linearly to
    # ``win_margin_end_v2`` over ``margin_warmup_blocks_v2`` blocks of tenure
    # (per settlement on the grid in force, like ``margin_warmup_blocks``).
    # All three set together or none (load-checked; end > 0 is the HARD
    # guardrail, start >= end). ``margin_v2_from_block`` = 0 means DECIDED ON
    # CHAIN: the validators signal readiness (DEC-CA-0045) and the lock-in
    # writes the first ERA START after the lock-in boundary here (never
    # mid-era); a typed block is the owner override. Level-judged rounds keep
    # the level bar (``ChainConfig.judged_level_params``).
    # Rounds before the block, and the block-less steady state, are judged
    # exactly as before.
    win_margin_start_v2: float = 0.0
    win_margin_end_v2: float = 0.0
    margin_warmup_blocks_v2: int = 0
    margin_v2_from_block: int = 0



def effective_win_margin_start(scoring: ScoringConfig, block: int | None) -> float:
    """The fresh-king margin in force for the round at epoch boundary ``block``,
    resolving an up-to-two-step schedule so a full margin history stays
    audit-verifiable: ``win_margin_start_prev2`` for blocks before
    ``margin_activation_block2``, ``win_margin_start_prev`` before
    ``margin_activation_block``, ``win_margin_start`` from it on. With no
    scheduled change (activation 0 / prev 0.0) or no block, the steady value.
    Bit-identical to the single-step form when the ``*2`` fields are unset."""
    if block is None:
        return float(scoring.win_margin_start)
    b = int(block)
    if scoring.margin_activation_block > 0 and b < scoring.margin_activation_block:
        if (scoring.margin_activation_block2 > 0 and scoring.win_margin_start_prev2 > 0.0
                and b < scoring.margin_activation_block2):
            return float(scoring.win_margin_start_prev2)
        if scoring.win_margin_start_prev > 0.0:
            return float(scoring.win_margin_start_prev)
    return float(scoring.win_margin_start)


def margin_v2_configured(scoring: ScoringConfig) -> bool:
    """Whether the DEC-CA-0049 v2 bar is defined (all three values set)."""
    return (float(getattr(scoring, "win_margin_start_v2", 0.0) or 0.0) > 0.0
            and float(getattr(scoring, "win_margin_end_v2", 0.0) or 0.0) > 0.0
            and int(getattr(scoring, "margin_warmup_blocks_v2", 0) or 0) > 0)


def margin_v2_active(scoring: ScoringConfig, block: int | None) -> bool:
    """CONSENSUS gate (DEC-CA-0049): the round at epoch boundary ``block`` is
    judged under the v2 bar. False with no block (the steady state stays the
    pre-v2 values), with the bar undefined, or before ``margin_v2_from_block``
    (0 = not yet decided)."""
    gate = int(getattr(scoring, "margin_v2_from_block", 0) or 0)
    return (gate > 0 and block is not None and int(block) >= gate
            and margin_v2_configured(scoring))


def check_margin_v2(start: float, end: float, warmup_blocks: int, from_block: int) -> None:
    """Load-time guardrails for the DEC-CA-0049 bar: all three values set
    together or none; the floor strictly positive (a 0 floor turns the decay
    into a term-limit lottery); start >= end; a typed block needs the bar."""
    vals = (float(start) > 0.0, float(end) > 0.0, int(warmup_blocks) > 0)
    if any(vals) and not all(vals):
        raise ValueError("[scoring] win_margin_start_v2, win_margin_end_v2 and "
                         "margin_warmup_blocks_v2 must all be set (> 0) or all unset")
    if float(start) < 0.0 or float(end) < 0.0 or int(warmup_blocks) < 0:
        raise ValueError("[scoring] margin v2 values must be non-negative")
    if all(vals) and float(start) < float(end):
        raise ValueError(f"[scoring] win_margin_start_v2={start} must be >= "
                         f"win_margin_end_v2={end}")
    if int(from_block) < 0:
        raise ValueError(f"[scoring] margin_v2_from_block={from_block} must be >= 0")
    if int(from_block) > 0 and not all(vals):
        raise ValueError("[scoring] margin_v2_from_block is set but the v2 bar is not "
                         "(win_margin_start_v2 / win_margin_end_v2 / margin_warmup_blocks_v2)")


def cohort_maxt_active(scoring: ScoringConfig, block: int | None) -> bool:
    """Whether a cohort round at epoch boundary ``block`` is judged under the
    shared-resample max-T (DEC-CA-0038) rather than Bonferroni ``alpha/k``:
    ``cohort_maxt_from_block`` set and reached. ``0`` / unknown block ⇒ False
    (Bonferroni). Block-gated exactly like :func:`effective_win_margin_start`,
    so every validator and the audit resolve the same rule from the round's
    block."""
    return (scoring.cohort_maxt_from_block > 0 and block is not None
            and int(block) >= scoring.cohort_maxt_from_block)


def cohort_maxt_increment_active(scoring: ScoringConfig, block: int | None) -> bool:
    """Whether a cohort round at epoch boundary ``block`` judges its max-T
    bound in INCREMENT units (DEC-CA-0039 stacked on DEC-CA-0038):
    ``cohort_maxt_increment_from_block`` set and reached. Only meaningful when
    the max-T and the increment margin are both in force for that block (and a
    baseline exists — a random-init round still falls back to level); ``0`` /
    unknown block ⇒ False (cohort rounds level-judged, the pre-gate rule)."""
    return (scoring.cohort_maxt_increment_from_block > 0 and block is not None
            and int(block) >= scoring.cohort_maxt_increment_from_block)


def mv_score_active(scoring: ScoringConfig, block: int | None) -> bool:
    """Whether a round at epoch boundary ``block`` is scored with the GIFT-Eval
    multivariate weighting (DEC-CA-0041) — a window's channels averaged into one
    per-window contribution — rather than one geomean row per channel:
    ``mv_score_from_block`` set and reached. ``0`` / unknown block ⇒ False.
    Block-gated exactly like :func:`cohort_maxt_active`, so every validator and
    the audit resolve the same rule from the round's block. Bit-identical on any
    univariate pool regardless."""
    return (scoring.mv_score_from_block > 0 and block is not None
            and int(block) >= scoring.mv_score_from_block)


def effective_margin_mode(scoring: ScoringConfig, block: int | None) -> str:
    """The margin denomination in force for the round at epoch boundary
    ``block``: ``"increment"`` from ``increment_from_block`` on (DEC-CA-0039),
    else the base ``[scoring] margin_mode``. Block-gated exactly like
    :func:`effective_win_margin_start`, so validator and audit resolve the same
    rule per round and a random-init round still falls back to level in
    evaluate_round (no baseline)."""
    if (scoring.increment_from_block > 0 and block is not None
            and int(block) >= scoring.increment_from_block):
        return "increment"
    return scoring.margin_mode



_MARGIN_MODES = ("level", "increment")


def _margin_mode(value: object) -> str:
    """Validate ``[scoring] margin_mode`` at load — a typo here would silently
    judge every round under the wrong denomination."""
    mode = str(value)
    if mode not in _MARGIN_MODES:
        raise ValueError(f"[scoring] margin_mode={mode!r} invalid; one of {_MARGIN_MODES}")
    return mode


_GIFT_GATE_MODES = ("off", "shadow", "enforce")


def _bench_gate_tolerance(value: object) -> float:
    """``[scoring] bench_gate_tolerance`` (DEC-CA-0061 rule v2): a finite
    float in [0, 0.5) — the slack above ratio 1 a suite's LCB may reach."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"[scoring] bench_gate_tolerance must be a number; got {value!r}")
    v = float(value)
    if not (v == v and 0.0 <= v < 0.5):
        raise ValueError(f"[scoring] bench_gate_tolerance must be in [0, 0.5); got {v}")
    return v


def _bench_gate_win_rate(value: object) -> float:
    """``[scoring] bench_gate_min_win_rate`` (DEC-CA-0061 rule v2): in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"[scoring] bench_gate_min_win_rate must be a number; got {value!r}")
    v = float(value)
    if not (v == v and 0.0 <= v <= 1.0):
        raise ValueError(f"[scoring] bench_gate_min_win_rate must be in [0, 1]; got {v}")
    return v


def _gift_gate_mode(value: object) -> str:
    """Validate ``[scoring] gift_gate_mode`` at load time so a typo fails fast
    rather than silently disabling the consensus gate."""
    mode = str(value)
    if mode not in _GIFT_GATE_MODES:
        raise ValueError(
            f"[scoring] gift_gate_mode={mode!r} invalid; one of {_GIFT_GATE_MODES}"
        )
    return mode


_INIT_GATE_MODES = ("off", "shadow", "enforce")


def _init_gate_mode(value: object) -> str:
    """Validate ``[scoring] init_gate_mode`` at load time (gift-gate rule: a
    typo must fail fast, not silently un-enforce the floor). The heat-side
    ``[round] init_gate_mode`` is deliberately NOT run through this — its
    unknown-value semantics are warn-and-off at the use site (DEC-CA-0034)."""
    mode = str(value)
    if mode not in _INIT_GATE_MODES:
        raise ValueError(
            f"[scoring] init_gate_mode={mode!r} invalid; one of {_INIT_GATE_MODES}"
        )
    return mode


def parse_bench_workers(value) -> str:
    """``[eval] bench_workers``: ``"auto"`` or an int >= 1, normalised to the
    sidecar's ``--workers`` string. Raises ``ValueError`` otherwise."""
    if isinstance(value, bool):
        raise ValueError(f"[eval] bench_workers must be \"auto\" or an int >= 1; got {value!r}")
    text = str(value).strip().lower()
    if text == "auto":
        return text
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1 or (isinstance(value, float) and value != n):
        raise ValueError(f"[eval] bench_workers must be \"auto\" or an int >= 1; got {value!r}")
    return str(n)
