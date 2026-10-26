"""[v2] go-live block and [lane_b] domain-expert config (DEC-CA-0050)."""

from __future__ import annotations

from dataclasses import dataclass

# The DEC-CA-0055 margin-tiered start share's defaults (owner 2026-10-20;
# placeholders to calibrate against seed-replica noise on the private pool):
# ``(lcb_threshold, start_bp)``, thresholds descending.
LADDER_START_TIERS: tuple[tuple[float, int], ...] = (
    (0.50, 8000), (0.25, 6500), (0.10, 5000), (0.01, 2500))

# Gain-tiered full-pay period (owner 2026-10-20) — the lane-B / lane-C (and
# MarketCast) analogue of the tiered start share: ``(lcb_threshold,
# full_eras)``, thresholds descending. A win pays in full for the
# ``full_eras`` of the highest threshold its gain LCB meets, then the usual
# decay. UNCALIBRATED placeholders: set them from real win LCBs before arming
# (lane-B domain LCBs are noisy — few windows / sources per domain).
FULL_PAY_TIERS: tuple[tuple[float, int], ...] = ((0.05, 10), (0.02, 5), (0.01, 2))


@dataclass(frozen=True)
class V2Config:
    """``[v2]`` — the ONE typed activation block of cascade v2 (DEC-CA-0050).

    ``from_block = 0``: v2 is not active (the mainnet placeholder until the
    owner types the go-live block). ``N > 0``: every era whose start block is
    ``>= N`` runs v2 — lane A trains Cascade-MOE-28M under ``[v2.training]``
    (a new lineage from random init). B and C have independent launch blocks. ``N``
    must be an era boundary on the grid in force at ``N`` (checked at load).
    No on-chain readiness vote: the owner types the block, every validator
    must run the v2 release before it. CONSENSUS.
    """

    from_block: int = 0
    launch_at_utc: str = ""
    commit_open_at_utc: str = ""
    commit_from_block: int = 0
    prelaunch_burn_readiness: str = ""
    # DEC-CA-0054 rides the v2 go-live: the three-suite bench gate mode that
    # applies to every v2 settlement ("off" keeps [scoring] bench_gate_mode).
    bench_gate_mode: str = "off"
    # DEC-CA-0061 (CONSENSUS): the bench gate's rule switches from DEC-CA-0054
    # (three suites, six-number geomean better) to rule v2 over four suites
    # ("avg_noninf_winrate": average ratio < 1, no suite's ratio LCB above
    # 1 + [scoring] bench_gate_tolerance, win rate >= bench_gate_min_win_rate)
    # for every settlement whose era starts at or after this block (:func:`cascade.shared.era.bench_gate_rule_at`).
    # 0 = never (the DEC-CA-0054 rule). chain.toml may type the string "v2":
    # it resolves at load to ``from_block`` (0 while v2 is unset), so the
    # rule goes live with the v2 block and never touches a judged settlement.
    bench_gate_v2_from_block: int = 0
    # DEC-CA-0063 (overlay-settable runtime policy): who produces the
    # lane-A public-benchmark rows the bench gate and the king check read.
    # "trainer" (default) = the trainer-signed bench report, every checkpoint
    # benched at completion (today's behaviour, bit-identical). "validator" =
    # the MAIN validator ([netconfig] owner_hotkey) sweeps on its own GPU only
    # a challenger that cleared the private pool (plus the era king, and the
    # king + init when the king check is due), signs its own report
    # (``bench_v/round-<id>.json``); every other validator and satellite reads
    # it. Applies to every settlement whose era starts at or after
    # ``bench_source_from_block`` (an era boundary; "v2" = ``from_block``;
    # 0 = never).
    bench_source: str = "trainer"
    bench_source_from_block: int = 0
    # DEC-CA-0063 liveness bounds (overlay-settable, like the source itself):
    # ``bench_report_timeout_settlements`` — the trainer records a
    # validator-sourced settlement's report as "absent" in the signed
    # manifest of the Nth later settlement built without it (N >= 1; its
    # clearers are then infra, never a loss). ``bench_budget_hours`` — the
    # main validator's sweep CAP per settlement (wall clock, its own GPU,
    # every sweep of the settlement counted): a clearer is started only while
    # the cap still holds its whole allowance (plus the king's when the king
    # must be swept too); one it reaches without that is published
    # ``budget_deferred`` (never a loss, re-judged at a later settlement —
    # DEC-CA-0063 item 9). ``bench_sweep_hours`` — the allowance of ONE
    # checkpoint's sweep (its retries included): each clearer has its own,
    # and the king's sweep has its own, never a clearer's (<= the cap).
    bench_report_timeout_settlements: int = 8
    bench_budget_hours: float = 4.0
    bench_sweep_hours: float = 2.0
    # Random-init window: every v2 era before ``v2_first_era + max(2, N)``
    # trains from random init — no promotion initialises one (CONSENSUS,
    # :func:`cascade.shared.era.v2_first_warm_era`). 0 = the two-era notice only.
    random_init_eras: int = 0
    # King check timing (DEC-CA-0055): the check is due at the settlement
    # whose warm start would take effect this many blocks into the reign
    # (:func:`cascade.shared.era.v2_check_due`). 21600 = 3 days. 0 = off.
    reign_warm_start_blocks: int = 0
    # Emission burns (CONSENSUS on the weight vector, see
    # cascade.shared.emission). ``prelaunch_burn``: the fraction of every
    # pre-v2 vector sent to ``burn_uid`` by a validator on this release.
    # ``carryover_king_share``: from the v2 block until the first v2 crowning,
    # lane A (the carried king's court) is paid at most this fraction of the
    # total; the rest burns. 0 = off.
    prelaunch_burn: float = 0.0
    carryover_king_share: float = 0.0
    # The pre-launch burn's start (DEC-CA-0055 item 1): decided ON CHAIN as a
    # DEC-CA-0045 secondary feature (``prelaunch-<pct>``) — lock-in
    # writes the boundary after it here (``activation.resolved_prelaunch_
    # burn_block`` marks it). A typed block is the owner override; 0 = vote.
    prelaunch_burn_from_block: int = 0
    # DEC-CA-0055 items 4–6 (CONSENSUS, v2 eras only): the king check at each
    # ripe reign point, king-only (k = 1) promotions backed by a passing check,
    # the lane-A share ladder with the 5-day former-king court, and the
    # operator-funded key slash. False = the DEC-CA-0050 lane-A rules (court
    # vector, top-k promotions) — the testnet state, whose v2 is already live.
    king_ladder: bool = False
    # Lane-A dethrone bar on v2 settlements (CONSENSUS; owner 2026-10-19:
    # "increment margin to clear go from 10% to 1%"): a fresh king's
    # increment-unit bar is ``win_margin_start``, decaying linearly to
    # ``win_margin_end`` over ``margin_warmup_blocks`` of tenure. Replaces
    # DEC-CA-0049's bar on v2 settlements only (toto2 eras keep it); a round
    # that falls back to level units keeps the level bar
    # (``ChainConfig.judged_level_params``). 0.0 = inherit ``[scoring]``.
    win_margin_start: float = 0.0
    win_margin_end: float = 0.0
    margin_warmup_blocks: int = 0
    # The DEC-CA-0055 king check's pool bar (owner 2026-10-19: kept separate
    # from the dethrone bar above): a FIXED increment-unit LCB margin for the
    # king vs the init, no tenure decay. With the init as both reference and
    # baseline the statistic is the king's gain over the init in units of
    # ``[scoring] margin_increment_floor`` × the init's level (capped at 2), so
    # 0.2 ≈ an LCB of 0.2 % relative improvement. 0.0 = the dethrone bar.
    king_check_margin: float = 0.0
    # The DEC-CA-0055 lane-A split, in BASIS POINTS of lane A's pool
    # (CONSENSUS; integers so every replay agrees exactly). Overlay-settable
    # (DEC-CA-0057); the defaults are the owner's 50 % -> +10 %/pass -> 95 %,
    # fail -> 10 %, operator-key slash 1 %, former king 5 % for 36000 blocks.
    # Margin-tiered start share (owner 2026-10-20): a king crowned at an
    # INCREMENT-unit settlement starts at the start_bp of the highest
    # ``(lcb_threshold, start_bp)`` tier its crowning LCB (the family-wise
    # corrected increment-unit LCB the dethrone bar was compared against)
    # meets; sorted by threshold, descending. A crowning judged in LEVEL units
    # (random-init eras, a level fallback) or with no duel LCB (a DEC-CA-0048
    # successor) starts at the fixed ``ladder_start_level_bp`` instead.
    ladder_start_tiers: tuple[tuple[float, int], ...] = LADDER_START_TIERS
    ladder_start_level_bp: int = 5000
    ladder_step_bp: int = 1000
    ladder_cap_bp: int = 9500
    ladder_fail_bp: int = 1000
    key_slash_bp: int = 100
    court_bp: int = 500
    court_blocks: int = 36000
    # Escalating key slash (owner 2026-10-20, CONSENSUS, overlay-settable like
    # the split above): the FIRST operator-paid lapse of a reign keeps the king
    # at key_slash_bp for at least this many eras from the settlement that
    # recorded it (then until a king-funded leg); a SECOND lapse in the same
    # reign keeps it there until the reign ends. 0 = the first lapse lifts
    # with the first king-funded leg. A crowning resets the count.
    key_slash_min_eras: int = 6

    def margin_set(self) -> bool:
        return self.win_margin_start > 0.0 and self.win_margin_end > 0.0

    def active(self, block: int | None) -> bool:
        """Whether the era STARTING at (or containing) ``block`` is a v2 era.
        For a settlement boundary use :func:`cascade.shared.era.v2_settlement_active`
        (a settlement belongs to the era containing ``boundary − 1``)."""
        return self.from_block > 0 and block is not None and int(block) >= self.from_block


def check_v2_activation(*, from_block: int, has_contract: bool, epoch_blocks: int,
                        epoch_blocks_prev: int, epoch_activation_block: int,
                        era_settlements: int, rolling_from_block: int,
                        activation_feature: str) -> None:
    """Refuse a ``[v2] from_block`` that is not an era boundary on the grid in
    force at it, that precedes the DEC-CA-0043 era machinery, or that names no
    post-v2 contract. Same shape as :func:`check_rollover_alignment`."""
    n = int(from_block)
    if n <= 0:
        return
    if not has_contract:
        raise ValueError(f"[v2] from_block={n} is set but [v2.training] (the post-v2 "
                         "Cascade-MOE-28M contract) is missing")
    if int(era_settlements) < 1:
        raise ValueError("[round] era_settlements must be >= 1 when [v2] from_block is set "
                         "(v2 switches at an era boundary)")
    eab, ebp = int(epoch_activation_block), int(epoch_blocks_prev)
    grid = ebp if (eab and ebp and n < eab) else int(epoch_blocks)
    era = max(1, grid) * int(era_settlements)
    if n % era:
        raise ValueError(
            f"[v2] from_block={n} must be an era boundary: a multiple of epoch_blocks "
            f"({grid}, the grid in force at it) × era_settlements ({era_settlements}) "
            f"= {era}")
    rolling = int(rolling_from_block)
    if rolling and n < rolling:
        raise ValueError(f"[v2] from_block={n} precedes the DEC-CA-0043 rollover "
                         f"({rolling}): v2 runs on eras")
    if not rolling and not str(activation_feature or ""):
        raise ValueError(f"[v2] from_block={n} needs the DEC-CA-0043 era machinery: type "
                         "the rollover ([round] rolling_from_block …) or enable "
                         "[activation] feature")


LADDER_DEFAULTS = {"ladder_start_level_bp": 5000, "ladder_step_bp": 1000,
                   "ladder_cap_bp": 9500, "ladder_fail_bp": 1000, "key_slash_bp": 100,
                   "court_bp": 500, "court_blocks": 36000, "key_slash_min_eras": 6}


def parse_ladder_start_tiers(raw: object, *, fail_bp: int,
                             cap_bp: int) -> tuple[tuple[float, int], ...]:
    """``[v2] ladder_start_tiers``: a non-empty list of ``[lcb_threshold,
    start_bp]`` pairs. Thresholds are numbers in (0, 1), STRICTLY descending;
    each start_bp an integer in ``[ladder_fail_bp, ladder_cap_bp]`` (a fail
    never raises the share), non-increasing as the threshold falls (a bigger
    win never starts lower)."""
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"[v2] ladder_start_tiers must be a non-empty list of "
                         f"[lcb_threshold, start_bp] pairs; got {raw!r}")
    out: list[tuple[float, int]] = []
    for pair in raw:
        if isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"[v2] ladder_start_tiers: each tier is [lcb_threshold, "
                             f"start_bp]; got {pair!r}")
        t, bp = pair
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 < float(t) < 1.0:
            raise ValueError(f"[v2] ladder_start_tiers: threshold must be a number in "
                             f"(0, 1); got {t!r}")
        if isinstance(bp, bool) or not isinstance(bp, int) or not 0 < bp <= cap_bp:
            raise ValueError(f"[v2] ladder_start_tiers: start_bp must be an integer in "
                             f"(0, ladder_cap_bp={cap_bp}]; got {bp!r}")
        if bp < fail_bp:
            raise ValueError(f"[v2] ladder_start_tiers: start_bp {bp} is below "
                             f"ladder_fail_bp={fail_bp} (a failed check would raise it)")
        if out and not float(t) < out[-1][0]:
            raise ValueError("[v2] ladder_start_tiers: thresholds must be strictly "
                             f"descending; got {raw!r}")
        if out and bp > out[-1][1]:
            raise ValueError("[v2] ladder_start_tiers: start_bp must not increase as the "
                             f"threshold falls; got {raw!r}")
        out.append((float(t), int(bp)))
    return tuple(out)


def parse_full_pay_tiers(raw: object, *, section: str,
                         win_ttl_eras: int) -> tuple[tuple[float, int], ...]:
    """``[<section>] full_pay_tiers``: a non-empty list of ``[lcb_threshold,
    full_eras]`` pairs. Thresholds are numbers in (0, 1), STRICTLY
    descending; each ``full_eras`` an integer ``>= 0``, non-increasing as the
    threshold falls (a bigger win never pays in full for less) and ``<=
    win_ttl_eras`` when a TTL is set (the full period never outlives the
    win). Same shape as :func:`parse_ladder_start_tiers`."""
    name = f"[{section}] full_pay_tiers"
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{name} must be a non-empty list of [lcb_threshold, full_eras] "
                         f"pairs; got {raw!r}")
    ttl = int(win_ttl_eras)
    out: list[tuple[float, int]] = []
    for pair in raw:
        if isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}: each tier is [lcb_threshold, full_eras]; got {pair!r}")
        t, n = pair
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 < float(t) < 1.0:
            raise ValueError(f"{name}: threshold must be a number in (0, 1); got {t!r}")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"{name}: full_eras must be an integer >= 0; got {n!r}")
        if ttl > 0 and n > ttl:
            raise ValueError(f"{name}: full_eras {n} exceeds win_ttl_eras={ttl} (the win "
                             "stops paying at its TTL)")
        if out and not float(t) < out[-1][0]:
            raise ValueError(f"{name}: thresholds must be strictly descending; got {raw!r}")
        if out and n > out[-1][1]:
            raise ValueError(f"{name}: full_eras must not increase as the threshold falls; "
                             f"got {raw!r}")
        out.append((float(t), int(n)))
    return tuple(out)


def _refuse_full_rounds(raw: dict, section: str) -> None:
    """``full_rounds`` was replaced in place by ``full_pay_tiers`` (owner
    2026-10-20): a config still typing it is refused, never silently read."""
    if "full_rounds" in raw:
        raise ValueError(f"[{section}] full_rounds was replaced by full_pay_tiers (the "
                         "gain-tiered full-pay period); type the tiers instead")


def parse_v2_economics(v2raw: dict) -> dict:
    """The DEC-CA-0055 ``[v2]`` keys, validated: the eras counts are ints
    ``>= 0``; ``prelaunch_burn`` / ``carryover_king_share`` fractions in
    ``[0, 1)`` (1 would burn — or cap — everything, which is a force-burn,
    not an economics rule); ``prelaunch_burn_from_block >= 0``; ``king_ladder``
    a bool; ``bench_gate_v2_from_block`` per
    :func:`parse_bench_gate_v2_from_block`. Absent keys keep the inert
    defaults."""
    out: dict = {}
    from ..calendar_launch import launch_timestamp

    launch = v2raw.get("launch_at_utc", "")
    opening = v2raw.get("commit_open_at_utc", "")
    readiness = v2raw.get("prelaunch_burn_readiness", "")
    if not isinstance(launch, str) or not isinstance(readiness, str):
        raise ValueError("v2 launch_at_utc and prelaunch_burn_readiness must be strings")
    launch_timestamp(launch)
    if not isinstance(opening, str):
        raise ValueError("v2 commit_open_at_utc must be a UTC timestamp string")
    opened = launch_timestamp(opening)
    launched = launch_timestamp(launch)
    if opened is not None and launched is not None and opened > launched:
        raise ValueError("v2 commit_open_at_utc must not follow launch_at_utc")
    if readiness and (not readiness.isascii() or not readiness.isalnum() or len(readiness) > 12):
        raise ValueError("v2 prelaunch_burn_readiness must be up to 12 ASCII letters/digits")
    out.update(launch_at_utc=launch, commit_open_at_utc=opening, prelaunch_burn_readiness=readiness)
    for k in ("random_init_eras", "reign_warm_start_blocks", "prelaunch_burn_from_block", "commit_from_block"):
        raw = v2raw.get(k, 0)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"[v2] {k} must be an integer; got {raw!r}")
        if raw < 0:
            raise ValueError(f"[v2] {k} must be >= 0; got {raw}")
        out[k] = int(raw)
    for k in ("prelaunch_burn", "carryover_king_share"):
        raw = v2raw.get(k, 0.0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"[v2] {k} must be a number; got {raw!r}")
        if not 0.0 <= float(raw) < 1.0:
            raise ValueError(f"[v2] {k} must be in [0, 1); got {raw}")
        out[k] = float(raw)
    out["bench_gate_v2_from_block"] = parse_bench_gate_v2_from_block(v2raw)
    out["bench_source"], out["bench_source_from_block"] = parse_bench_source(v2raw)
    out.update(parse_bench_source_bounds(v2raw))
    ladder = v2raw.get("king_ladder", False)
    if not isinstance(ladder, bool):
        raise ValueError(f"[v2] king_ladder must be true/false; got {ladder!r}")
    out["king_ladder"] = ladder
    start, end = v2raw.get("win_margin_start", 0.0), v2raw.get("win_margin_end", 0.0)
    warm = v2raw.get("margin_warmup_blocks", 0)
    for k, raw in (("win_margin_start", start), ("win_margin_end", end)):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"[v2] {k} must be a number; got {raw!r}")
    if isinstance(warm, bool) or not isinstance(warm, int) or warm < 0:
        raise ValueError(f"[v2] margin_warmup_blocks must be an integer >= 0; got {warm!r}")
    start, end = float(start), float(end)
    # Same guardrail as DEC-CA-0049: the floor must stay > 0.
    if (start or end) and (not (0.0 < end <= start < 1.0) or warm < 1):
        raise ValueError("[v2] win_margin_start/win_margin_end need 0 < end <= start < 1 "
                             f"and margin_warmup_blocks >= 1; got {start}/{end}/{warm}")
    out.update(win_margin_start=start, win_margin_end=end, margin_warmup_blocks=int(warm))
    kcm = v2raw.get("king_check_margin", 0.0)
    if isinstance(kcm, bool) or not isinstance(kcm, (int, float)) or not 0.0 <= float(kcm) < 2.0:
        # The increment statistic is capped at 2: a margin at or past it never passes.
        raise ValueError(f"[v2] king_check_margin must be a number in [0, 2); got {kcm!r}")
    out["king_check_margin"] = float(kcm)
    lad = {k: v2raw.get(k, d) for k, d in LADDER_DEFAULTS.items()}
    for k, v in lad.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"[v2] {k} must be an integer >= 0; got {v!r}")
    if not (0 < lad["ladder_fail_bp"] <= lad["ladder_start_level_bp"] <= lad["ladder_cap_bp"]
            <= 10000):
        raise ValueError("[v2] need 0 < ladder_fail_bp <= ladder_start_level_bp <= "
                         f"ladder_cap_bp <= 10000; got {lad}")
    if lad["ladder_cap_bp"] + lad["court_bp"] > 10000 or lad["key_slash_bp"] > 10000:
        raise ValueError("[v2] ladder_cap_bp + court_bp must be <= 10000 and key_slash_bp "
                         f"<= 10000; got {lad}")
    out.update(lad)
    out["ladder_start_tiers"] = parse_ladder_start_tiers(
        v2raw.get("ladder_start_tiers", LADDER_START_TIERS),
        fail_bp=lad["ladder_fail_bp"], cap_bp=lad["ladder_cap_bp"])
    return out


LANE_B_REMAINDERS = ("burn", "lane_a")
# DEC-CA-0059: a horizon slot's target name (``horizon:64``) and its short
# commitment code (``h64`` — ``:`` separates the commitment's fields)
HORIZON_TARGET_PREFIX = "horizon:"


@dataclass(frozen=True)
class LaneBConfig:
    """``[lane_b]`` — the domain-expert competition (DEC-CA-0050).

    CONSENSUS terms (every validator must agree; replayed by cascade-audit):
    ``from_block`` (0 = lane B off — the deployed state), ``share``, the decay
    (``full_pay_tiers`` / ``half_life_rounds`` / ``floor`` — counted in ERAS of
    ``[round] era_settlements`` settlements), the payout expiry
    ``win_ttl_eras``, ``remainder``, ``domains`` (one emission slot each) and
    the verdict parameters.
    TRAINER POLICY (no consensus impact): the opening gate, expert selection,
    leg limits and budgets.

    ``[lane_b] from_block`` is an independent era-start launch block (0 =
    off), at or after ``[v2] from_block``. ``load_chain_config`` stores it
    here as ``launch + 1``: lane B covers every block strictly after its
    launch, so the settlement AT the launch boundary — which belongs to
    the preceding era (DEC-CA-0043: a settlement at ``B`` belongs to the era
    containing ``B − 1``) — keeps the prior allocation. A direct
    ``parse_lane_b({"from_block": …})`` (tests) still sets it verbatim.
    """

    from_block: int = 0
    share: float = 0.5
    domains: tuple[str, ...] = ()
    # payout schedule in ERAS (owner-tunable): 100 % for the first
    # ``full_eras`` eras after the merging settlement — GAIN-TIERED (owner
    # 2026-10-20): the ``full_eras`` of the highest ``full_pay_tiers``
    # threshold the win's target gain LCB meets, fixed at the win and
    # recorded on it — then halving every ``half_life_rounds`` eras down to
    # ``floor``; at ``win_ttl_eras`` the win stops paying (its slot burns) —
    # the merged weights stay in the king forever. 0 = no expiry (paid at the
    # floor until replaced).
    full_pay_tiers: tuple[tuple[float, int], ...] = FULL_PAY_TIERS
    half_life_rounds: float = 1.0
    floor: float = 0.1
    win_ttl_eras: int = 28
    remainder: str = "burn"
    # verdict (cascade.eval.expert_lane.ExpertLaneParams)
    alpha: float = 0.05
    bootstrap_B: int = 10_000
    # > 0 (refused at load otherwise): an unchanged expert scores LCB = 0 and
    # a zero bar would merge it; the verdict also needs a strictly positive
    # point gain and ``min_target_clusters`` distinct sources on the target
    gain_margin: float = 0.01
    tau_domain: float = 0.01
    tau_public: float = 0.01
    min_target_windows: int = 100
    min_target_clusters: int = 5
    min_guard_windows: int = 50
    min_public_items: int = 10
    require_public: tuple[str, ...] = ()
    # trainer policy
    open_tokens: int = 200_000_000_000
    # routing-stability gate (OLMoE router saturation), measured per domain on
    # the affinity slice between the PREVIOUS era's king and the current one:
    # EVERY MoE layer must clear both the ESFT-selection Jaccard
    # (``routing_jaccard``) and the token-level top-k overlap
    # (``routing_overlap``); once open, lane B stays open (one-way latch)
    routing_jaccard: float = 0.7
    routing_overlap: float = 0.8
    # rebase staleness: a finished leg whose domain the CURRENT king routes to
    # different experts than its base king (any layer below either bar) is
    # dropped as ``routing_stale`` instead of rebased
    rebase_jaccard: float = 0.5
    rebase_overlap: float = 0.7
    esft_p: float = 0.2
    min_experts: int = 8
    max_experts: int = 38
    legs_per_domain: int = 1
    max_base_age_eras: int = 2
    leg_hours: float = 3.0
    affinity_windows: int = 256
    # DEC-CA-0059: a commitment may declare 1 .. ``max_domains_per_leg``
    # targets (CONSENSUS: validators refuse an entry declaring more). One leg
    # trains the union of their expert selections under ONE leg budget; each
    # target is judged at ``alpha / k`` and paid as its own slot.
    max_domains_per_leg: int = 3
    # DEC-CA-0059 horizon targets: one slot per ``[eval] scored_horizons``
    # rung (``horizon:<h>``), filled by ``load_chain_config`` — empty
    # ``scored_horizons`` (testnet) ⇒ no horizon targets, the slot list is
    # exactly ``domains``. ``horizon_targets = false`` disarms them (CONSENSUS)
    horizons: tuple[int, ...] = ()
    horizon_targets: bool = True
    # validator catch-up (not a consensus term: the replay is deterministic):
    # a validator missing lane-B history (new, long downtime, lost state)
    # re-judges at most this many past settlements' expert manifests, in
    # chain order, at their own blocks / windows / king pointers. Unset in
    # chain.toml ⇒ derived to cover the payout TTL, ``(win_ttl_eras + 1) ×
    # era_settlements`` (see :func:`lane_b_replay_cover`), so a rebuilt
    # validator holds every still-paying win by construction; an explicit
    # value below that is refused at load.
    # Once acceptance is active this is only a legacy replay bound: recovery
    # walks the complete ledger because accepted deltas outlive payout TTLs.
    replay_settlements: int = 64
    # Merge guards (CONSENSUS, typed era-start block; chain.toml ``"v2"`` =
    # ``[v2] from_block``, the default; 0 = never): from it validators refuse
    # an era init whose declared lane-B merges overlap on any (layer, row),
    # and refuse non-finite expert deltas both when judging a leg and when
    # rebuilding a merged init. Typed separately from the v2 block so a
    # network whose lane B is already live (testnet) arms it at a future era
    # instead of re-judging settled history.
    merge_guard_from_block: int = 0
    # CONSENSUS: qualifications are unpaid until an incremental merge passes
    # acceptance at a scheduled warm start. 0 preserves historical payouts.
    acceptance_from_block: int = 0
    # Who pays for a lane-B leg (TRAINER POLICY): "miner" (the default,
    # DEC-CA-0059 item 8) rents on the miner's deposited key through
    # ``/v1/lane-b/fund`` with no operator fallback; "operator" keeps the
    # retired operator lane-pool dispatch (dev / emergency only).
    leg_funding: str = "miner"

    def active(self, block: int | None) -> bool:
        return self.from_block > 0 and block is not None and int(block) >= self.from_block

    def merge_guard_at_era(self, era_start: int | None) -> bool:
        """Merge guards apply to the init of an era starting at ``era_start``."""
        n = int(self.merge_guard_from_block)
        return n > 0 and era_start is not None and int(era_start) >= n

    def acceptance_at_era(self, era_start: int | None) -> bool:
        n = int(self.acceptance_from_block)
        return n > 0 and era_start is not None and int(era_start) >= n

    def acceptance_at_settlement(self, block: int | None) -> bool:
        n = int(self.acceptance_from_block)
        # The boundary settlement supplies candidates to the new era's init.
        return n > 0 and block is not None and int(block) >= n

    def merge_guard_at_settlement(self, block: int | None) -> bool:
        """Merge guards apply to the settlement at boundary ``block`` (it
        belongs to the era containing ``block − 1``, so ``block > N`` ⇔ that
        era starts at or after the era boundary ``N``)."""
        n = int(self.merge_guard_from_block)
        return n > 0 and block is not None and int(block) > n

    @property
    def horizon_target_names(self) -> tuple[str, ...]:
        """``horizon:<h>`` per scored horizon (DEC-CA-0059); ``()`` when
        ``[eval] scored_horizons`` is empty or ``horizon_targets`` is off."""
        if not self.horizon_targets:
            return ()
        return tuple(f"{HORIZON_TARGET_PREFIX}{int(h)}" for h in self.horizons)

    @property
    def targets(self) -> tuple[str, ...]:
        """The lane-B slot list (DEC-CA-0059): ``domains`` then the horizon
        targets. One emission slot each (``share / len(targets)``). Equal to
        ``domains`` whenever there are no horizon targets."""
        return tuple(self.domains) + self.horizon_target_names

    def verdict_params(self):
        from ...eval.expert_lane import ExpertLaneParams

        return ExpertLaneParams(
            alpha=self.alpha, B=self.bootstrap_B, gain_margin=self.gain_margin,
            tau_domain=self.tau_domain, tau_public=self.tau_public,
            min_target_windows=self.min_target_windows,
            min_target_clusters=self.min_target_clusters,
            min_guard_windows=self.min_guard_windows,
            min_public_items=self.min_public_items,
        )


def lane_b_replay_cover(win_ttl_eras: int, era_settlements: int) -> int:
    """Settlements a catch-up must replay to hold every still-paying win: a
    win merged up to ``win_ttl_eras`` eras back, plus one era of boundary
    slack. 0 when there is no TTL (no finite cover)."""
    if int(win_ttl_eras) <= 0:
        return 0
    return (int(win_ttl_eras) + 1) * max(1, int(era_settlements))


LANE_B_LEG_FUNDING = ("miner", "operator")


def _merge_guard_block(raw, v2_from_block: int, key="merge_guard_from_block") -> int:
    """``[lane_b] merge_guard_from_block``: ``"v2"`` (the default) = ``[v2]
    from_block`` (0 while v2 is unset), else an integer ``>= 0`` as typed."""
    if raw == "v2":
        return max(0, int(v2_from_block))
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f'[lane_b] {key} must be an integer >= 0 or "v2"; '
                         f"got {raw!r}")
    return int(raw)


def parse_lane_b(lb: dict, *, from_block: int | None = None,
                 era_settlements: int = 0,
                 horizons: tuple[int, ...] | None = None,
                 v2_from_block: int | None = None) -> LaneBConfig:
    """Parse + guard ``[lane_b]`` (absent section ⇒ inert defaults).

    ``from_block`` (``load_chain_config``) is the first settlement block,
    derived from ``[lane_b] from_block + 1``; ``None`` reads the dict's own
    key verbatim (direct construction).
    ``era_settlements`` sizes the default ``replay_settlements`` to the TTL.
    ``horizons`` (``load_chain_config``: ``[eval] scored_horizons``) are the
    DEC-CA-0059 horizon targets; ``None`` reads the dict's own ``horizons``
    key (direct construction in tests), absent ⇒ none. ``v2_from_block``
    resolves ``merge_guard_from_block = "v2"``; ``None`` = the launch block the
    derived ``from_block`` implies (``from_block − 1``; 0 when lane B is off)."""
    _refuse_full_rounds(lb, "lane_b")
    ttl = int(lb.get("win_ttl_eras", 28))
    cover = lane_b_replay_cover(ttl, era_settlements)
    if "replay_settlements" in lb:
        replay = int(lb["replay_settlements"])
        if cover and era_settlements and replay < cover:
            raise ValueError(
                f"[lane_b] replay_settlements={replay} does not cover the payout TTL "
                f"(win_ttl_eras={ttl} × era_settlements={era_settlements} + one era "
                f"= {cover}): a rebuilt validator would miss still-paying wins")
    else:
        replay = cover if (cover and era_settlements) else 64
    fb = (max(0, int(lb.get("from_block", 0) or 0)) if from_block is None
          else max(0, int(from_block)))
    cfg = LaneBConfig(
        from_block=fb,
        share=float(lb.get("share", 0.5)),
        domains=tuple(str(d) for d in lb.get("domains", ())),
        full_pay_tiers=parse_full_pay_tiers(lb.get("full_pay_tiers", FULL_PAY_TIERS),
                                            section="lane_b", win_ttl_eras=ttl),
        half_life_rounds=float(lb.get("half_life_rounds", 1.0)),
        floor=float(lb.get("floor", 0.1)),
        win_ttl_eras=ttl,
        remainder=str(lb.get("remainder", "burn")),
        alpha=float(lb.get("alpha", 0.05)),
        bootstrap_B=int(lb.get("bootstrap_B", 10_000)),
        gain_margin=float(lb.get("gain_margin", 0.01)),
        tau_domain=float(lb.get("tau_domain", 0.01)),
        tau_public=float(lb.get("tau_public", 0.01)),
        min_target_windows=int(lb.get("min_target_windows", 100)),
        min_target_clusters=int(lb.get("min_target_clusters", 5)),
        min_guard_windows=int(lb.get("min_guard_windows", 50)),
        min_public_items=int(lb.get("min_public_items", 10)),
        require_public=tuple(str(x) for x in lb.get("require_public", ())),
        open_tokens=int(lb.get("open_tokens", 200_000_000_000)),
        routing_jaccard=float(lb.get("routing_jaccard", 0.7)),
        routing_overlap=float(lb.get("routing_overlap", 0.8)),
        rebase_jaccard=float(lb.get("rebase_jaccard", 0.5)),
        rebase_overlap=float(lb.get("rebase_overlap", 0.7)),
        esft_p=float(lb.get("esft_p", 0.2)),
        min_experts=int(lb.get("min_experts", 8)),
        max_experts=int(lb.get("max_experts", 38)),
        legs_per_domain=int(lb.get("legs_per_domain", 1)),
        max_base_age_eras=int(lb.get("max_base_age_eras", 2)),
        leg_hours=float(lb.get("leg_hours", 3.0)),
        affinity_windows=int(lb.get("affinity_windows", 256)),
        max_domains_per_leg=int(lb.get("max_domains_per_leg", 3)),
        horizons=tuple(int(h) for h in (lb.get("horizons", ()) if horizons is None
                                        else horizons) or ()),
        horizon_targets=bool(lb.get("horizon_targets", True)),
        replay_settlements=replay,
        merge_guard_from_block=_merge_guard_block(
            lb.get("merge_guard_from_block", "v2"),
            int(v2_from_block) if v2_from_block is not None else max(0, fb - 1)),
        acceptance_from_block=_merge_guard_block(
            lb.get("acceptance_from_block", 0),
            int(v2_from_block) if v2_from_block is not None else max(0, fb - 1),
            "acceptance_from_block"),
        leg_funding=str(lb.get("leg_funding", "miner")),
    )
    if cfg.leg_funding not in LANE_B_LEG_FUNDING:
        raise ValueError(f"[lane_b] leg_funding must be one of {LANE_B_LEG_FUNDING}; "
                         f"got {cfg.leg_funding!r}")
    if not 0.0 <= cfg.share <= 1.0:
        raise ValueError(f"[lane_b] share must be in [0, 1]; got {cfg.share}")
    if not 0.0 < cfg.floor <= 1.0:
        raise ValueError(f"[lane_b] floor must be in (0, 1]; got {cfg.floor}")
    import math

    if not math.isfinite(cfg.half_life_rounds) or cfg.half_life_rounds <= 0:
        raise ValueError("[lane_b] half_life_rounds must be > 0")
    if not 0.0 < cfg.alpha < 1.0:
        raise ValueError("[lane_b] alpha must be in (0, 1)")
    if cfg.bootstrap_B < 1:
        raise ValueError("[lane_b] bootstrap_B must be >= 1")
    for key in ("tau_domain", "tau_public"):
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
            raise ValueError(f"[lane_b] {key} must be finite and >= 0")
    if not 0.0 < cfg.esft_p <= 1.0:
        raise ValueError("[lane_b] esft_p must be in (0, 1]")
    if not math.isfinite(cfg.leg_hours) or cfg.leg_hours <= 0:
        raise ValueError("[lane_b] leg_hours must be finite and > 0")
    if min(cfg.min_guard_windows, cfg.min_public_items, cfg.affinity_windows) < 1:
        raise ValueError("[lane_b] min_guard_windows, min_public_items and affinity_windows "
                         "must be >= 1")
    if cfg.open_tokens < 0:
        raise ValueError("[lane_b] open_tokens must be >= 0")
    if (len(set(cfg.require_public)) != len(cfg.require_public)
            or set(cfg.require_public) - {"gift", "boom", "time", "fev"}):
        raise ValueError("[lane_b] require_public must be unique supported suites "
                         "(gift, boom, time, fev)")
    if cfg.win_ttl_eras < 0:
        raise ValueError(f"[lane_b] win_ttl_eras={cfg.win_ttl_eras} must be 0 (no expiry) "
                         "or > 0")
    if cfg.remainder not in LANE_B_REMAINDERS:
        raise ValueError(f"[lane_b] remainder must be one of {LANE_B_REMAINDERS}")
    if len(set(cfg.domains)) != len(cfg.domains):
        raise ValueError("[lane_b] domains must be unique")
    if cfg.from_block and not cfg.domains:
        raise ValueError("[lane_b] from_block set but no domains")
    import re

    for d in cfg.domains:
        # a domain must be a commitment token (DEC-CA-0059): never a horizon
        # code, never a separator
        if not re.fullmatch(r"[a-z0-9_\-]{1,32}", d) or re.fullmatch(r"h[0-9]+", d):
            raise ValueError(f"[lane_b] domain {d!r} must match [a-z0-9_-]{{1,32}} and not "
                             f"look like a horizon code (h<digits>)")
    if len(set(cfg.horizons)) != len(cfg.horizons) or any(h <= 0 for h in cfg.horizons):
        raise ValueError(f"[lane_b] horizons must be unique positive ints; got {cfg.horizons}")
    if cfg.max_domains_per_leg < 1:
        raise ValueError("[lane_b] max_domains_per_leg must be >= 1")
    if not 0.0 < cfg.gain_margin < 1.0:
        raise ValueError(f"[lane_b] gain_margin must be in (0, 1) — a zero bar merges an "
                         f"unchanged expert (identical scores give LCB = 0); got "
                         f"{cfg.gain_margin}")
    if cfg.min_target_windows < 1 or cfg.min_target_clusters < 1:
        raise ValueError("[lane_b] min_target_windows and min_target_clusters must be >= 1")
    if not 1 <= cfg.min_experts <= cfg.max_experts:
        raise ValueError("[lane_b] need 1 <= min_experts <= max_experts")
    if cfg.replay_settlements < 0:
        raise ValueError("[lane_b] replay_settlements must be >= 0")
    for k in ("routing_jaccard", "routing_overlap", "rebase_jaccard", "rebase_overlap"):
        if not 0.0 <= getattr(cfg, k) <= 1.0:
            raise ValueError(f"[lane_b] {k} must be in [0, 1]; got {getattr(cfg, k)}")
    if cfg.legs_per_domain < 1 or cfg.max_base_age_eras < 0:
        raise ValueError("[lane_b] legs_per_domain >= 1 and max_base_age_eras >= 0")
    return cfg


def parse_bench_gate_v2_from_block(v2raw: dict) -> int:
    """``[v2] bench_gate_v2_from_block`` (DEC-CA-0061) → a block: an integer
    ``>= 0`` as typed, or the string ``"v2"`` = ``[v2] from_block`` (0 while v2
    is unset — the rule then never applies). Anything else is refused."""
    raw = v2raw.get("bench_gate_v2_from_block", 0)
    if raw == "v2":
        # the loader's own reading of [v2] from_block
        return max(0, int(v2raw.get("from_block", 0) or 0))
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f'[v2] bench_gate_v2_from_block must be an integer or "v2"; got {raw!r}')
    if raw < 0:
        raise ValueError(f"[v2] bench_gate_v2_from_block must be >= 0; got {raw}")
    return int(raw)


BENCH_SOURCES = ("trainer", "validator")


def parse_bench_source(v2raw: dict) -> tuple[str, int]:
    """``[v2] bench_source`` / ``bench_source_from_block`` (DEC-CA-0063) →
    ``(source, block)``. ``source`` is "trainer" (default) or "validator";
    the block is an integer ``>= 0`` or the string ``"v2"`` (= ``[v2]
    from_block``, 0 while v2 is unset). A "trainer" source resolves to block
    0 whatever is typed (nothing switches)."""
    src = v2raw.get("bench_source", "trainer")
    if src not in BENCH_SOURCES:
        raise ValueError(f"[v2] bench_source must be one of {BENCH_SOURCES}; got {src!r}")
    raw = v2raw.get("bench_source_from_block", 0)
    if raw == "v2":
        block = max(0, int(v2raw.get("from_block", 0) or 0))
    elif isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f'[v2] bench_source_from_block must be an integer or "v2"; got {raw!r}')
    elif raw < 0:
        raise ValueError(f"[v2] bench_source_from_block must be >= 0; got {raw}")
    else:
        block = int(raw)
    return str(src), (block if src == "validator" else 0)


def parse_bench_source_bounds(v2raw: dict) -> dict:
    """``[v2] bench_report_timeout_settlements`` (int >= 1, default 8),
    ``[v2] bench_budget_hours`` (number > 0, default 4.0) and ``[v2]
    bench_sweep_hours`` (number > 0 and <= ``bench_budget_hours``, default
    2.0) — DEC-CA-0063."""
    out: dict = {}
    t = v2raw.get("bench_report_timeout_settlements", 8)
    if isinstance(t, bool) or not isinstance(t, int) or t < 1:
        raise ValueError(f"[v2] bench_report_timeout_settlements must be an integer >= 1; "
                         f"got {t!r}")
    out["bench_report_timeout_settlements"] = int(t)
    b = v2raw.get("bench_budget_hours", 4.0)
    if isinstance(b, bool) or not isinstance(b, (int, float)) or not float(b) > 0.0:
        raise ValueError(f"[v2] bench_budget_hours must be a number > 0; got {b!r}")
    out["bench_budget_hours"] = float(b)
    w = v2raw.get("bench_sweep_hours", 2.0)
    if isinstance(w, bool) or not isinstance(w, (int, float)) or not float(w) > 0.0:
        raise ValueError(f"[v2] bench_sweep_hours must be a number > 0; got {w!r}")
    if float(w) > float(b):
        raise ValueError(f"[v2] bench_sweep_hours={w} must be <= bench_budget_hours={b}: "
                         "a clearer whose allowance exceeds the settlement cap could "
                         "never be swept (every one would be budget-deferred)")
    out["bench_sweep_hours"] = float(w)
    return out


def check_bench_source(*, v2: V2Config, owner_hotkey: str, era_blocks: int,
                       manifest_validator_hotkey: str = "") -> None:
    """Load checks for a DEC-CA-0063 validator bench source with a block:
    the block is an era boundary (``era_blocks`` = the era length on the grid
    in force at it) at/after ``[v2] from_block``; ``[v2] king_ladder`` is on
    (k = 1 king-only promotions — top-k promotion needs every challenger's
    numbers, which the validator never sweeps); ``[netconfig] owner_hotkey``
    names the main validator (the only one that sweeps, and the key its
    report is verified against)."""
    n = int(v2.bench_source_from_block)
    if v2.bench_source != "validator" or n <= 0:
        return
    if int(v2.from_block) <= 0 or n < int(v2.from_block):
        raise ValueError(f"[v2] bench_source_from_block={n} must be at or after [v2] "
                         f"from_block ({v2.from_block}): the validator bench source is a "
                         "v2 lane-A rule")
    if n % max(1, int(era_blocks)):
        raise ValueError(f"[v2] bench_source_from_block={n} must be an era boundary "
                         f"(a multiple of {era_blocks})")
    if not v2.king_ladder:
        raise ValueError('[v2] bench_source = "validator" needs [v2] king_ladder = true: '
                         "top-k promotion selects over every challenger's bench numbers, "
                         "and the validator benches only private-pool winners")
    if not str(owner_hotkey or "").strip():
        raise ValueError('[v2] bench_source = "validator" needs [netconfig] owner_hotkey '
                         "(the main validator: the only node that sweeps, and the key its "
                         "signed bench report is verified against)")
    # Both keys name the main validator: [manifest] validator_hotkey is the
    # trainer's / audit's receipt anchor, [netconfig] owner_hotkey the bench
    # report signer. Neither is derived from the other (a silent derivation
    # would move a trust root); when both are set they must agree.
    mv = str(manifest_validator_hotkey or "").strip()
    if mv and mv != str(owner_hotkey).strip():
        raise ValueError(f'[v2] bench_source = "validator": [manifest] validator_hotkey '
                         f"({mv}) and [netconfig] owner_hotkey ({owner_hotkey}) must name "
                         "the same main validator")
