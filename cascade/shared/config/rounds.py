"""[round] config: RoundConfig, the epoch grid in force at a block, and the rollover check."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoundConfig:
    """The epoch grid, rolling intake (DEC-CA-0043) and the trainer's admission
    and pod policy.

    ``epoch_blocks`` is the settlement grid: every boundary settles what
    finished into one manifest, and every validator derives pool-snapshot
    selection from the same grid (CONSENSUS-RELEVANT, not in contract_digest).
    Everything else here is trainer policy, consensus-inert unless noted.

    ``throne_sizes`` (and the informational ``screen_size``) are
    ``arch_preset`` names selecting from the size registry (``[training]``
    primary + each ``[[training.sizes]]``); empty ⇒ the primary size.

    Keys of the retired boundary-round trainer (heat screen, tie run-off,
    duel seating, commit witness, field-wide dedup probe, …) are accepted and
    ignored by the loader — see :data:`RETIRED_KEYS`.
    """

    epoch_blocks: int = 7200          # ≈24h at 12s blocks
    round_hours: float = 24.0         # informational: wall-clock span of an epoch
    # ── Scheduled cadence change (see :func:`effective_epoch_blocks`) ────────
    # A round-length change is CONSENSUS-RELEVANT: every validator derives the
    # epoch from the block grid, so a fleet that switches at different moments
    # selects different eval-pool snapshots for the same round. Restart timing
    # cannot coordinate that across independent operators — so the switch is
    # pinned to a BLOCK instead. Operators pull and restart whenever they like;
    # every node flips at the same block on its own.
    #
    # ``epoch_activation_block`` is the first block on which ``epoch_blocks``
    # applies; before it, ``epoch_blocks_prev`` does. Both 0 ⇒ no scheduled
    # change, ``epoch_blocks`` always applies (the steady state — clear these
    # two once the activation block is safely in the past).
    #
    # The activation block MUST be a boundary of BOTH grids, or a round is cut
    # short or double-counted at the seam; ``load_chain_config`` enforces it.
    epoch_blocks_prev: int = 0
    epoch_activation_block: int = 0
    # ── Rolling intake + era king (DEC-CA-0043, block-gated) ─────────────────
    # From ``rolling_from_block`` challengers train the moment they are funded;
    # a leg may cross boundaries; every boundary publishes ONE settlement
    # manifest (the era king + every challenger finished since the last one);
    # the king's leg is trained once per ERA and cached. 0 = decided on chain
    # (DEC-CA-0045 stake-weighted activation; the trainer idles until then).
    # Must equal the fleet's ROLLOVER block — the same value as ``[scoring]
    # era_king_from_block`` — and be a boundary of the grid (load-checked).
    rolling_from_block: int = 0
    # Settlements per era: consecutive boundaries sharing one set of training
    # seeds, one init and one cached king checkpoint. 4 on a 3h grid = 12h
    # eras, i.e. today's king-leg cadence with 3h verdicts inside it. Era
    # arithmetic (cascade.shared.era) is pure block math — trainer,
    # validators and audit derive the same era with zero discretion. 0 = one
    # settlement per era (only meaningful once a rolling gate is set).
    era_settlements: int = 0
    # The local-scoring budget miners emulate (``cascade score``) and the
    # wall-clock guard a reduced-budget leg (the worker's ``--train-hours``)
    # runs under: max(heat_guard_factor × hours, heat_guard_floor_seconds),
    # never above [training] max_train_seconds.
    heat_train_hours: float = 0.5
    heat_n_windows: int = 256         # windows `cascade score` evaluates on
    heat_guard_factor: float = 1.0
    heat_guard_floor_seconds: int = 900
    # Default rolling admission cap when ``funded_field_cap`` is 0:
    # ``finalist_cap`` = max(finalists, max_finalists) once max_finalists > 1.
    finalists: int = 1
    max_finalists: int = 1
    screen_size: str = ""             # informational ("" ⇒ primary)
    throne_sizes: tuple[str, ...] = ()  # arch_presets the throne trains/judges at (() ⇒ [primary])
    # Extra seconds a funded leg must clear before its target boundary, for
    # the completion bench that runs before an entry counts as finished
    # (0 = none).
    funded_bench_margin_seconds: int = 0
    # DEC-CA-0048 admission denylist (trainer-side, not consensus): entries from
    # these hotkeys are refused at rolling admission from `blocked_from_block`
    # (0 = immediately once the list is non-empty). Their legs never rent a pod.
    blocked_hotkeys: tuple[str, ...] = ()
    blocked_from_block: int = 0
    # Anti-spam: 1 hotkey = 1 submission (lifetime). A hotkey whose leg
    # settled (or whose submission was refused for its own fault) must
    # re-register — a new UID, paying the registration cost — to resubmit, so
    # a miner cannot cheaply re-roll a generator against the throne. The burn
    # set persists at ``submissions_db_path`` (relative ⇒ under the trainer's
    # ``work_root``). Off ⇒ a hotkey re-competes freely — testnet only.
    one_submission_per_hotkey: bool = True
    # Only commitments revealed AT/AFTER this block ever compete — the
    # official go-live gate (squatters and rehearsal commits never enter and
    # never burn their one submission). 0 = no floor (testnet).
    commit_floor_block: int = 0
    submissions_db_path: str = "trainer_submissions.json"
    # ── Miner-funded compute (DEC-CA-0036) ───────────────────────────────────
    # "off"      — the funded queue is never read: rolling intake admits
    #              nothing.
    # "shadow"   — funding status is logged only.
    # "required" — the funded queue IS the intake: every funded, revealed
    #              entry trains (seniority = reveal block), up to
    #              ``funded_field_cap`` legs in flight. Unfunded reveals wait,
    #              unburned, until their owner funds them.
    funded_mode: str = "off"
    # The funded queue file shared with cascade-intake. Relative paths resolve
    # under work_root (like submissions_db_path).
    funded_queue_path: str = "funded_queue.json"
    # Queue-entry idle TTL, hours. MUST track cascade-intake --ttl-hours (the
    # payer-key vault TTL): expiring earlier kills paid entries whose key still
    # works; later leaves keyless entries holding the skip-floor open. Idle is
    # measured from the entry's last activity (fund/requeue/promotion), so a
    # sold-out entry actively cycling never expires mid-drought.
    funded_entry_ttl_hours: float = 36.0
    # ── Per-payer funded pods (DEC-CA-0036, second arming gate) ──────────────
    # "off"  — funded challenger legs run on the operator fleet (hosts.toml),
    #          exactly the pre-wiring behaviour.
    # "rent" — each funded challenger leg rents ONE pod on ITS payer's Lium key
    #          (cascade.provision.funded), dispatches there, and tears it down
    #          when the leg ends. The king's leg and every eval pod stay on the
    #          operator fleet, always. Requires funded_mode = "required" plus
    #          the three keys below; consensus-inert ([round]).
    funded_pods: str = "off"
    # The payer-key vault directory shared with cascade-intake --vault-dir
    # (cascade.funding.vault; relative resolves under work_root). Required for
    # "rent": without the miner's key no pod can be rented on their account.
    payer_vault_dir: str = ""
    # Marketplace SKU for funded pods (e.g. "RTX4090") when
    # ``funded_sku_per_leg`` is off: one locked type for every leg.
    funded_pod_sku: str = ""
    # Digest-pinned worker image funded pods launch from (full ref,
    # ``…@sha256:…`` — the same image the operator's final fleet runs).
    funded_pod_image: str = ""
    # sha256 fingerprint of the pinned image's ``cascade/**/*.py``
    # (cascade.provision.codeprint; scripts/worker_code_fingerprint.py derives
    # it from the registry). Checked from the pod's FILES before any final /
    # funded dispatch — a host that starts a stale image under the pinned tag
    # still echoes the injected digest env; this it cannot fake. Empty ⇒ off.
    worker_code_fingerprint: str = ""
    # Ceiling on one funded pod's boot (launch → SSH-ready), seconds.
    funded_ready_timeout_seconds: float = 900.0
    # Harbor expiry (days) on the per-pod Hub robot a funded leg gets
    # (cascade.funding.robots). Revocation at teardown is the real bound;
    # this is the backstop if a revoke ever fails.
    funded_robot_duration_days: int = 1
    # How a funded leg's checkpoint leaves its payer pod. "harvest" (default,
    # DEC-CA-0036 credential-free pods): the worker runs --local-only, the pod
    # holds NO Hub credential at all, and the orchestrator pulls the checkpoint
    # over the pinned SSH, ingest-verifies it, and uploads it under its own
    # identity. "robot": the legacy per-pod push-only Harbor robot (needs a
    # worker image that predates --local-only, or a Hub whose anonymous pulls
    # are off). Requires a funded_pod_image built from a --local-only-aware
    # release when set to "harvest".
    funded_pod_checkpoint: str = "harvest"
    # Shared admission cap for miner-funded A/B/C preparation and training;
    # queued tasks wait FIFO. King work has reserved admission. Also bounds
    # lane-A in-flight entries; 0 ⇒ ``finalist_cap``. Overlay-settable.
    funded_field_cap: int = 0
    # Allowed GPU types, PREFERENCE-ordered; with ``funded_sku_per_leg`` every
    # leg rents the cheapest fitting executor across them. Needs [training]
    # expected_gpu = "" (a hard pin freezes the type).
    funded_pod_skus: tuple[str, ...] = ()
    # Executors whose CPU model contains any of these substrings (case-
    # insensitive; Lium specs.cpu.model) are never rented — the king and every
    # leg alike (generators stalled on whole host families).
    funded_cpu_blocklist: tuple[str, ...] = ()
    # Per-SKU floor for the fixed calibration bench (host_probe.host_bench's
    # composite leg, tokens/s) every freshly rented pod — payer legs and the
    # JIT king — runs over ssh BEFORE dispatch (~5 s): a slow host eats the
    # leg's equal wall. Below the floor the pod is released, its executor
    # stays excluded and the leg rents again; the miner is never blamed.
    # () / unknown SKU = off; an unreachable probe logs and lets the pod
    # through (a slow-host check must never itself sink a leg).
    funded_host_bench_floor: tuple[tuple[str, float], ...] = ()
    # Choose the GPU type PER LEG: every leg (the JIT king too) rents the
    # cheapest executor that fits across funded_pod_skus. Fair because
    # budgets are compute-denominated and every leg completes its full token
    # budget; safe because validators lift the gpu_mismatch gate at the
    # rollover. Off = one locked type (``funded_pod_sku``).
    funded_sku_per_leg: bool = True
    # Price guards on marketplace executors (payer-billed legs and the JIT
    # king alike; USD; 0 = off). max_price_per_hour is a plain $/h ceiling;
    # max_leg_cost_usd caps price_per_hour × the SKU's measured wall
    # (funded_sku_wall_seconds; the contract's max_train_seconds when the
    # SKU has no entry) — the figure that actually matters (a fast expensive
    # GPU can be the cheaper leg).
    funded_max_price_per_hour: float = 1.5
    funded_max_leg_cost_usd: float = 1.6
    # Measured wall (seconds) of one full-budget leg per SKU. Drives the
    # per-SKU latest safe start — a fast SKU may still start late — and the
    # per-leg cost cap. Unknown SKU ⇒ max_train_seconds.
    funded_sku_wall_seconds: tuple[tuple[str, int], ...] = (
        ("H100", 6000), ("L40", 11400), ("L40S", 10200), ("RTX4090", 13500))
    # Rent the era KING's pod just-in-time on the OPERATOR's account (the
    # trainer ledgers and sweeps it; the provisioner's reaper never matches
    # its name).
    funded_king_rent: bool = False
    # Detached dispatch: every remote leg and every pod-side bench runs in its
    # OWN session on the pod (setsid + nohup, stdout/stderr/exit code in a
    # per-leg run dir) and the orchestrator polls with short ssh calls. An
    # ssh transport drop is retried for `dispatch_reattach_grace_seconds` of
    # consecutive unreachability before the leg is declared lost (the
    # attached form turned one dropped session into a failed leg).
    # Credentials still travel on stdin only. False = the attached form.
    detached_dispatch: bool = True
    dispatch_poll_seconds: int = 30
    dispatch_reattach_grace_seconds: int = 900
    # Hybrid fallback: a funded leg (or the JIT king) still WAITING for
    # marketplace capacity takes an OPERATOR final lane from hosts.toml as
    # soon as one is on file — operator-billed for that leg only. Off ⇒
    # funded legs never run on operator lanes (the bill never moves silently).
    funded_operator_fallback: bool = False
    # Marketplace first, operator lanes for the spillover: a REQUEUED leg
    # takes an operator lane the moment one is on file; a FRESH leg keeps
    # polling the marketplace (payer-billed) and falls back only inside this
    # many seconds before its latest safe start. 0 ⇒ every waiting leg falls
    # back at once. The JIT king is never held back.
    funded_operator_fallback_fresh_window_seconds: int = 3600
    # Trainer recovery controls; scientific contracts and reward rules are unchanged.
    ops_max_active_per_coldkey: int = 0
    ops_gpu_failure_threshold: int = 0
    ops_gpu_failure_window_seconds: int = 3600
    ops_gpu_cooldown_seconds: int = 10800
    ops_alert_interval_seconds: int = 60
    ops_low_balance_usd: float = 10.0
    ops_oldest_wait_seconds: int = 7200
    ops_settlement_hold_seconds: int = 900
    ops_acceptance_timeout_seconds: int = 1200
    ops_clock_drift_seconds: int = 120
    ops_operator_providers: tuple[str, ...] = ("lium",)
    ops_operator_daily_cap_usd: float = 0.0
    ops_operator_leg_cap_usd: float = 0.0
    ops_paid_fallback_after_seconds: int = 0
    ops_king_backup_after_seconds: int = 0
    ops_king_backup_cap_usd: float = 0.0
    ops_alert_webhook_env: str = ""
    ops_continuation_interval_seconds: int = 0
    ops_continuation_max_bytes: int = 17179869184
    ops_continuation_image_digest: str = ""
    # A harvested funded checkpoint whose worker receipt carries no in-memory
    # tensor digests is refused as tamper (true) or accepted with a warning
    # (false: a worker image that predates the digests still harvests).
    funded_require_tensor_digests: bool = False
    # ── Direct submissions + champion-only publication (DEC-CA-0036) ─────────
    # Where the intake's private submission store lives (cascade.funding.store;
    # relative resolves under work_root). "" = direct submissions off: vault
    # refs cannot resolve and are dropped from the field with a log line. The
    # dir is operator-local — off git, off any rsync'd pod tree; a dispatch
    # stages exactly ONE entry's ZIP onto its pod, never the store.
    submission_vault_dir: str = ""
    # When a vault-ref king's code goes public (cascade.funding.champion):
    # "off" (never — dev only), "crown" (immediately), "delay" (after
    # champion_publish_delay_rounds of reign), "dethrone" (only when the reign
    # ends). Under every non-off policy a dethroned vault king publishes at
    # the hand-off if it never did; losers never publish. Trainer-side
    # policy, consensus-inert.
    champion_publish: str = "off"
    champion_publish_delay_rounds: int = 2
    # Content-level dedup at admission (DEC-CA-0008, the rolling registry
    # cascade.trainer.dedup_registry): each newcomer's repo is fingerprinted
    # and compared against the queue, in-flight legs, the era king and every
    # published champion; an EXACTLY identical repo — same tree, same
    # normalized tokens, or same tokens modulo renames — is refused (the
    # earlier reveal keeps the slot) and burns its submission. There is
    # deliberately NO similarity threshold: a ratio bar is gameable by spacing
    # edits just under it. "off" = no registry; "shadow" = log verdicts, drop
    # nothing; "enforce" = drop. The shipped mainnet chain.toml enforces.
    dedup_mode: str = "off"
    # private_copy tier (rolling registry only): an entrant whose Python module
    # (a .py file or a string-packed source) contains >= private_copy_min_containment
    # of an EARLIER-committed, never-published vault submission's module from
    # another hotkey — after masking identifiers, so renames do not help — is
    # a copy of code that was never public. Copying PUBLIC code (a crowned
    # king's published tree, a public Hub generator) is the design and is never
    # flagged: a private module that itself matches public material is skipped.
    # "shadow" logs the verdict and admits; "enforce" drops like the exact
    # tiers (earliest commit keeps the entry). Only consulted when dedup_mode
    # is shadow/enforce (the registry must exist).
    private_copy_mode: str = "shadow"
    private_copy_min_containment: float = 0.8
    private_copy_min_tokens: int = 2_000
    # embedded tiers (exact identity, DEC-CA-0008): a Python source packed into
    # a string constant of an entrant's .py file (plain / base64 / zlib / hex)
    # that equals — byte-for-byte, or after identifier masking — another
    # entry's whole .py stream or one of ITS packed sources, in either
    # direction (a wrapper around someone's generator, or a plain re-submission
    # of someone's blob). "shadow" logs and admits; "enforce" drops like the
    # whole-repo tiers. Only consulted when dedup_mode is shadow/enforce.
    dedup_embedded_mode: str = "shadow"
    # Cost caps on fingerprinting. Tokenizing is linear but not cheap (~2s/MB)
    # and input size is attacker-chosen up to [generator] max_repo_mb:
    # dedup_max_text_mb bounds the tokenizer's input per repo.
    dedup_max_text_mb: int = 4
    # Timed-reveal safety margin: `cascade deploy` targets its timelock reveal at
    # `epoch boundary − reveal_margin_blocks`, so a submission stays hidden for
    # its whole window and is public only for the last few minutes before the
    # field locks. The margin must absorb commit-inclusion + drand reveal jitter
    # (a reveal landing AT/after the boundary misses the round) while staying
    # too short for a copier to fetch + re-commit + land their own reveal.
    # ~25 blocks ≈ 5 min at 12s blocks; tighten after measuring live jitter.
    reveal_margin_blocks: int = 25

    @property
    def finalist_cap(self) -> int:
        """``finalists`` while ``max_finalists <= 1``, else
        ``max(finalists, max_finalists)`` — the default rolling admission cap
        when ``funded_field_cap`` is 0."""
        if self.max_finalists > 1:
            return max(self.finalists, self.max_finalists)
        return self.finalists

    def blocked_at(self, block: int | None) -> frozenset[str]:
        """Hotkeys refused at admission at ``block`` (DEC-CA-0048): the list once
        ``blocked_from_block`` is reached (0 = as soon as the list is set)."""
        if not self.blocked_hotkeys:
            return frozenset()
        gate = int(self.blocked_from_block or 0)
        if gate > 0 and (block is None or int(block) < gate):
            return frozenset()
        return frozenset(self.blocked_hotkeys)



def effective_epoch_blocks(round_cfg: RoundConfig, block: int) -> int:
    """Round length in force AT ``block`` — the only correct way to turn a block
    into an epoch.

    A cadence change is consensus-relevant: the epoch is derived from the block
    grid (``block // epoch_blocks``), so two validators using different lengths
    for the same round select different eval-pool snapshots and reach different
    verdicts. Restart timing cannot coordinate that across independent
    operators, so the switch is pinned to ``epoch_activation_block``: every node
    flips at the same block regardless of when it restarted, and an operator who
    pulls early is not punished for it.

    Returns ``epoch_blocks_prev`` for blocks strictly before the activation
    block, ``epoch_blocks`` from it onward. With no scheduled change (both keys
    0, the steady state) this is just ``epoch_blocks`` and the call is free.

    ALWAYS pass the block the epoch is being derived FOR — a manifest's
    ``created_block``, a receipt's ``epoch_start_block``, a reveal target —
    never the current head. Deriving a historical round's epoch from the head
    is what makes an audit disagree with the validator that wrote the receipt.
    ``load_chain_config`` guarantees the activation block is a boundary of both
    grids, so the seam is unambiguous.
    """
    eb = max(1, int(round_cfg.epoch_blocks))
    act = int(round_cfg.epoch_activation_block)
    prev = int(round_cfg.epoch_blocks_prev)
    if act and prev > 0 and int(block) < act:
        return max(1, prev)
    return eb


def check_rollover_alignment(
    *,
    rolling_from_block: int,
    era_king_from_block: int,
    tenure_blocks_from_block: int,
    epoch_blocks: int,
    epoch_blocks_prev: int,
    epoch_activation_block: int,
    era_settlements: int,
    cohort_maxt_from_block: int,
    cohort_maxt_increment_from_block: int,
    funded_pods: str,
    funded_king_rent: bool,
) -> int:
    """The DEC-CA-0043 rollover invariants on plain values.

    Returns the rollover block (0 when no rollover key is set). Raises
    ``ValueError`` on a split, misaligned or under-configured rollover — the
    same refusal for a value typed into chain.toml (``load_chain_config``)
    and for one resolved from validator signals at runtime
    (:func:`cascade.shared.activation.apply_activation`), so a runtime
    rollover can never reach a state the loader would have refused.
    """
    _eb, _ebp, _eab = int(epoch_blocks), int(epoch_blocks_prev), int(epoch_activation_block)
    _rollover_keys = {
        "[round] rolling_from_block": int(rolling_from_block),
        "[scoring] era_king_from_block": int(era_king_from_block),
        "[scoring] tenure_blocks_from_block": int(tenure_blocks_from_block),
    }
    _rollover_set = {k: v for k, v in _rollover_keys.items() if v}
    if not _rollover_set:
        return 0
    if len(set(_rollover_set.values())) != 1:
        raise ValueError(
            "DEC-CA-0043 rollover keys must all name the same block: "
            + ", ".join(f"{k}={v}" for k, v in _rollover_keys.items()))
    _rollover = next(iter(_rollover_set.values()))
    if len(_rollover_set) != len(_rollover_keys):
        missing = [k for k, v in _rollover_keys.items() if not v]
        raise ValueError(
            f"DEC-CA-0043 rollover block {_rollover} is set on "
            f"{sorted(_rollover_set)} but not on {missing} — every rollover "
            "key flips at ONE block")
    if _eab and _eab != _rollover:
        raise ValueError(
            f"[round] epoch_activation_block={_eab} must equal the DEC-CA-0043 "
            f"rollover block {_rollover} (the grid switch and the era switch "
            "land on one boundary)")
    _grid_before = _ebp if (_eab and _ebp) else _eb
    if _rollover % _grid_before or _rollover % _eb:
        raise ValueError(
            f"DEC-CA-0043 rollover block {_rollover} must be a boundary of the "
            f"grid before it ({_grid_before}) and of epoch_blocks={_eb}")
    _era_settlements = int(era_settlements)
    if _era_settlements < 1:
        raise ValueError(
            "[round] era_settlements must be >= 1 when a DEC-CA-0043 rollover "
            "block is set")
    _era_len = _eb * _era_settlements
    if _rollover % _era_len:
        raise ValueError(
            f"DEC-CA-0043 rollover block {_rollover} must start an era: a "
            f"multiple of epoch_blocks × era_settlements = {_era_len} (the era "
            "grid is absolute — block // era length — so the rollover boundary "
            "is only the first era's start when it lies on that grid)")
    _cmi = int(cohort_maxt_increment_from_block)
    _cm = int(cohort_maxt_from_block)
    _era_king = int(era_king_from_block)
    if not (_era_king >= _cmi >= _cm):
        raise ValueError(
            f"[scoring] era_king_from_block={_era_king} must be >= "
            f"cohort_maxt_increment_from_block={_cmi} >= "
            f"cohort_maxt_from_block={_cm} (the era king stacks on the "
            "increment-unit max-T)")
    if _cmi == 0 or _cm == 0:
        raise ValueError(
            "[scoring] cohort_maxt_from_block and cohort_maxt_increment_from_block "
            "must be set when era_king_from_block is (the era king is judged "
            "under the increment-unit max-T)")
    if not _eab:
        raise ValueError(
            f"DEC-CA-0043 rollover block {_rollover} must also switch the grid: set "
            "[round] epoch_blocks_prev (the grid before it) and "
            f"epoch_activation_block = {_rollover} — without the switch rolling "
            f"intake would run on the old grid with {_era_settlements}-round eras "
            "of the old length, silently")
    _funded_pods = str(funded_pods or "off")
    if _funded_pods != "rent" or not bool(funded_king_rent):
        raise ValueError(
            "DEC-CA-0043 rolling intake needs [round] funded_pods = \"rent\" and "
            f"funded_king_rent = true (got funded_pods={_funded_pods!r}, "
            f"funded_king_rent={bool(funded_king_rent)}): legs and "
            "the era king are rented just-in-time, there is no round-wide pool")
    return int(_rollover)
