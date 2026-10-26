"""The trainer service: :class:`TrainerRunner` and its pod/bucket/chain plumbing.

The live loop (:meth:`TrainerRunner.run_forever`) drives the rolling scheduler
(:mod:`cascade.trainer.rolling`, DEC-CA-0043): eligible challengers wait for
compatible admission/capacity before training, the era king trains once per era, and every epoch boundary
settles what finished into one signed, hash-chained manifest. Everything the
scheduler needs from the outside world lives here — renting and attesting pods
(payer pods on the miner's own Lium key, the operator's JIT king pod, operator
fallback lanes), dispatching and harvesting legs, running trainer-sourced
benches where configured, the promotion engine's boundary step, and publishing the
manifest and its bench report.

The pod/bucket plumbing is split into mixins :class:`TrainerRunner` inherits
(``runner_capacity``, ``runner_funded_pods``, ``runner_operator_pods``,
``runner_funded_legs``, ``runner_promotion``, ``runner_rolling``); the shared
helpers live in ``faults``, ``lane_pool`` and ``runner_common``.

:meth:`TrainerRunner.train_one` / :meth:`TrainerRunner.train_one_local` are the
GPU boundary the pod worker (:mod:`cascade.trainer.worker`) runs: fetch the
generator, build the corpus in the sandbox, train under the fixed contract.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..shared.config import (
    ChainConfig,
    TrainingContractConfig,
    effective_epoch_blocks,
)
from ..shared.era import rolling_active
from ..shared.hippius import (
    HubConfig,
    LogSink,
    S3Config,
    S3Store,
    StorageError,
    fetch_from_hub,
    generator_archive_key,
    private_archive_key,
    publish_manifest,
    upload_dir_to_hub_or_hf,
)
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    dump_manifest,
    format_trained_pointer,
    parse_trained_pointer,
    sign_manifest,
)
from .contract import BaseTrainer, RoundSeeds, TrainResult
from .corpus import CorpusError

# The helpers below moved out of this module and are re-exported (noqa F401):
# callers and tests import them from ``cascade.trainer.loop``.
from .faults import (  # noqa: F401
    _STALL_MARKER,
    _STORAGE_FAILURE_MARKERS,
    KEY_LOST_REASON,
    STALL_CLASS,
    _FundedDiverged,
    _FundedKeyLost,
    _FundedLegSkip,
    _FundedOperatorFallback,
    _FundedOperatorRent,
    _FundedTamper,
    _http_status_in_chain,
    _LaneDeadlinePassed,
    _quarantine_lane_host,
    _storage_failure,
    _transport_failure,
    classify_funded_worker_failure,
)
from .host_probe import host_snapshot, host_summary_line
from .lane_pool import (  # noqa: F401
    _FinalLanePool,
    _lium_provider,
    host_cannot_bench,
    king_pod_name_prefix,
)
from .remote import error_tail
from .runner_capacity import _FundedCapacityMixin
from .runner_common import (  # noqa: F401
    ResolvedGenerator,
    _load_seen_hotkeys,
    _save_seen_hotkeys,
    _v2_scheduled,
    clear_harvest,
    era_contracts,
    expert_kw,
    harvest_dir,
    router_kw,
    settlement_contracts,
    write_pending_upload,
)
from .runner_funded_legs import _FundedLegMixin
from .runner_funded_pods import _FundedPodMixin
from .runner_generation import _GenerationMixin
from .runner_operator_pods import _OperatorPodMixin
from .runner_promotion import BENCH_ANNEAL_SALT, _PromotionMixin  # noqa: F401
from .runner_rolling import KING_RETIRE_WAIT_S, _RollingMixin  # noqa: F401
from .stream import open_round_stream
from .wandb_sink import open_wandb_run

log = logging.getLogger("cascade.trainer")

# Lane-retry backoff after a storage-layer failure (see
# faults._STORAGE_FAILURE_MARKERS). Stays here: _dispatch_on_free_lane reads it
# from this module, where tests patch it.
STORAGE_RETRY_BACKOFF_SECONDS = 45.0


# A verified harvest's upload: retried with backoff before the leg is given up
# as an operator fault (the miner's checkpoint is fine and on our disk).
HARVEST_UPLOAD_ATTEMPTS = 4
HARVEST_UPLOAD_BACKOFF_SECONDS = 60.0


def upload_harvested_checkpoint(runner, record: dict, *, base_seed) -> TrainedEntry:
    """Upload the verified local copy ``record["dest"]`` under the operator's
    identity → the leg's :class:`TrainedEntry`; delete the copy once the
    upload returned a pinned ref. Bounded retries with backoff; when every
    attempt fails the copy stays, a pending-upload record lets the requeued
    leg re-upload it instead of retraining, and :class:`_OperatorFault` is
    raised (infra, never an attempt burn)."""
    from .faults import _OperatorFault

    dest = Path(record["dest"])
    ckpt_repo = record["ckpt_repo"]
    hotkey = record["miner_hotkey"]
    sleep = getattr(runner, "_upload_retry_sleep", None) or time.sleep
    last: Exception | None = None
    for i in range(HARVEST_UPLOAD_ATTEMPTS):
        try:
            up = upload_dir_to_hub_or_hf(
                dest, ckpt_repo, runner.hub(), hf_repo=runner._hf_ckpt_repo(ckpt_repo))
            break
        except Exception as e:  # noqa: BLE001 — storage blip: ours, retried
            last = e
            log.warning("checkpoint upload %s for %s: attempt %d/%d failed: %s", ckpt_repo,
                        hotkey[:12], i + 1, HARVEST_UPLOAD_ATTEMPTS, error_tail(str(e), 200))
            if i + 1 < HARVEST_UPLOAD_ATTEMPTS:
                sleep(HARVEST_UPLOAD_BACKOFF_SECONDS * (2 ** i))
    else:
        write_pending_upload(runner.work_root, base_seed, hotkey, record)
        raise _OperatorFault(
            f"checkpoint upload failed after {HARVEST_UPLOAD_ATTEMPTS} attempts (operator "
            f"storage; the harvested checkpoint is kept for the requeued leg): {last}")
    entry = TrainedEntry(
        miner_hotkey=hotkey,
        miner_uid=int(record["miner_uid"]),
        role=record["role"],
        gen_ref=record["gen_ref"],
        trained_pointer=format_trained_pointer(up.ref.immutable_ref),
        corpus_digest=record["corpus_digest"],
        train_block=int(record["train_block"]),
        gpu_name=record.get("gpu_name", ""),
        size=record.get("receipt_size") or record["size"],
        init_pointer=record.get("warm_start_ref"),
        training_contract_digest=record.get("contract_digest", ""),
    )
    # The upload is verified (a pinned ref came back): nothing reads the
    # local copy after this — benches and validators fetch the ref.
    clear_harvest(runner.work_root, base_seed, hotkey)
    return entry


@dataclass
class TrainerRunner(_FundedCapacityMixin, _FundedPodMixin, _OperatorPodMixin, _FundedLegMixin,
                    _PromotionMixin, _RollingMixin, _GenerationMixin):
    """Owner-operated trainer. ``base_trainer`` is the GPU backend (Protocol).

    Storage is Hippius: generators + checkpoints on the Hub registry (by
    ``repo@digest``), training logs + the manifest on S3. The live service
    drives the rolling scheduler (:mod:`cascade.trainer.rolling`); this class
    owns everything that touches a pod, a bucket or the chain on its behalf.
    """

    cfg: ChainConfig
    base_trainer: BaseTrainer
    work_root: Path
    wallet: object | None = None       # bittensor wallet for signing (live)
    use_sandbox: bool = True           # run generators in the isolated subprocess
    # Eval-pool pin: ``(base_seed, block) -> (key, sha256)`` of the pool
    # snapshot a settlement is judged on, stamped (signed) into the manifest so
    # validators verify their own snapshot selection against it. None ⇒
    # manifests go out unpinned. Wired in trainer.main.
    pool_provenance_fn: object | None = None
    # Operator hosts (``[[host]]`` tables): the pod profile rented pods mirror
    # and the operator fallback lanes. ``remote_hosts_path`` is re-read on use
    # so a lane added after startup is seen without a restart.
    remote_hosts: list | None = None
    remote_hosts_path: Path | None = None
    trainer_spec: str | None = None    # the BaseTrainer 'module:Class' pods run
    remote_timeout_seconds: int = 6 * 3600
    # Detached dispatch (remote.run_detached): mirrors [round] detached_dispatch /
    # dispatch_poll_seconds / dispatch_reattach_grace_seconds.
    detached_dispatch: bool = False
    dispatch_poll_seconds: int = 30
    dispatch_reattach_grace_seconds: int = 900
    # Chain-read guards (see _block_with_freeze_guard): a height frozen this
    # long rebuilds the connection; every read runs under the hard deadline.
    stale_block_after_s: float = 300.0
    chain_clock: Callable[[], float] = time.monotonic
    chain_read_timeout_s: float = 60.0
    # The orchestrator's DEPLOYED chain.toml, pushed onto every pod the trainer
    # rents (``<workdir>/chain.deployed.toml``): the worker image bakes a
    # build-time snapshot whose ``train_image_digest`` pin is stale. None ⇒ no
    # push (offline tools, tests).
    deployed_chain_toml: Path | None = None
    # GIFT-Eval/BOOM/TIME sweep on the pod that holds a checkpoint (bench at
    # completion, DEC-CA-0043); None ⇒ no bench numbers.
    cascade_bench_plan: object | None = None
    # Cascade promotion engine (DEC-CA-0013): tracks the reign, logs benched
    # candidates, fires signed promotions. None ⇒ never promotes.
    promotion: object | None = None
    # Stake-weighted activation (DEC-CA-0045): where the trainer persists the
    # rollover the validators decided (it reads the fleet's decision, never
    # signals). None ⇒ resolved in memory only.
    activation_store: object | None = None
    _activation: object | None = field(default=None, repr=False)
    _forfeit: object | None = field(default=None, repr=False)
    # DEC-CA-0057 owner-signed network config, resolved READ-ONLY (the trainer
    # never signals): the resolver (None = inert) and, while an overlay is in
    # force, the base config ``self.cfg`` is a view of (activation lock-ins
    # are applied to the base; the view is re-derived every poll).
    netconfig: object | None = None
    _cfg_base: object | None = field(default=None, repr=False)
    _hub: HubConfig | None = field(default=None, repr=False)
    _manifest_store: S3Store | None = field(default=None, repr=False)
    _logs_store: S3Store | None = field(default=None, repr=False)
    # Summary row of every leg trained IN THIS PROCESS (round_id → log role)
    # for the public training summary; pod legs are read back from their log.
    _leg_summaries: dict = field(default_factory=dict, repr=False)
    # The pod each leg's checkpoint lives on, keyed (role, size, hotkey): the
    # bench runs where the checkpoint already is.
    _final_role_hosts: dict = field(default_factory=dict, repr=False)
    # Frozen-block tracker for _block_with_freeze_guard.
    _last_block: int | None = field(default=None, repr=False)
    _block_changed_at: float = field(default=0.0, repr=False)
    # Last king a signed receipt named (sticky across transient read failures
    # so the reign clock never flaps back to the lagging incentive king), and
    # the last receipt-read problem logged (logged once per change).
    _last_receipt_king: str | None = field(default=None, repr=False)
    _receipt_problem: str = field(default="", repr=False)
    # Serialises the funded-pod ledger's load-modify-save (legs rent
    # concurrently; an unlocked RMW drops a live pod from the crash ledger).
    _funded_ledger_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    # Serialises the RENT phase (marketplace pick + `lium up` + readiness): a
    # later rent must see earlier claims, or concurrent rents all pick the
    # market's same top executor. Training itself stays fully parallel.
    _funded_rent_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    # Funded pods kept alive past their leg for the completion bench
    # (payer hotkey → PodInstance); each is torn down when its bench ends.
    _funded_bench_pods: dict = field(default_factory=dict, repr=False)
    _funded_bench_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    # Per-leg verdicts for the queue (hotkey → (msg, miner_fault, class, burn)).
    _funded_leg_failures: dict = field(default_factory=dict, repr=False)
    # DEC-CA-0064: hotkey -> the harvested MarketCast leg's receipt record
    _mkt_harvested: dict = field(default_factory=dict, repr=False)
    # Executors claimed by a live rent (never re-picked by a sibling), and
    # each claim's (claimed_at, pod instance id): released when that pod is
    # confirmed torn down, aged out as a backstop (runner_common).
    _funded_claimed_execs: set = field(default_factory=set, repr=False)
    _funded_claim_meta: dict = field(default_factory=dict, repr=False)
    _funded_exec_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    # The era king's JIT pod (rented once per era) and its rent state: the
    # king rents first and challengers yield until it is served or gives up.
    _funded_king_host: object | None = field(default=None, repr=False)
    _funded_king_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    _king_rent_done: bool = field(default=True, repr=False)
    _funded_wait_abort: threading.Event = field(
        default_factory=threading.Event, repr=False)
    _rolling_king_host_era: int | None = field(default=None, repr=False)

    # ── storage handles (lazy so offline/tests need no Hippius) ──────────────

    def hub(self) -> HubConfig:
        if self._hub is None:
            self._hub = HubConfig.from_storage(self.cfg.storage)
        return self._hub

    def _hf_ckpt_repo(self, ckpt_repo: str) -> str | None:
        """The HuggingFace **model** repo to mirror a checkpoint to when the Hub is
        down, or ``None`` when no HF fallback is configured (Hub-only).

        Reuses the HF account from ``[storage] hf_backup_repo`` — the same mirror
        the manifest/receipt store already falls back to; its namespace owns the
        model repos too — and keeps the Hub repo's basename, so a fallback ref
        reads ``<hf_ns>/ckpt-r…``. Empty/namespaceless ``hf_backup_repo`` ⇒ no
        checkpoint fallback, matching the manifest store's own gating."""
        backup = self.cfg.storage.hf_backup_repo
        if not backup or "/" not in backup:
            return None
        ns = backup.split("/", 1)[0]
        return f"{ns}/{ckpt_repo.rsplit('/', 1)[-1]}"

    def manifest_store(self):
        # HF-backed when [storage] hf_backup_repo is set, else plain S3 — so the
        # trainer's manifest write survives a Hippius S3 outage (writes to HF).
        if self._manifest_store is None:
            from ..shared.hippius import open_manifest_store

            self._manifest_store = open_manifest_store(self.cfg.storage)
        return self._manifest_store

    def logs_store(self) -> S3Store:
        if self._logs_store is None:
            self._logs_store = S3Store(
                S3Config.from_storage(self.cfg.storage, bucket=self.cfg.storage.logs_bucket)
            )
        return self._logs_store

    # ── anti-spam: 1 hotkey = 1 submission (lifetime) ────────────────────────

    def _submissions_path(self) -> Path:
        """Where the burn set is persisted. A relative ``submissions_db_path`` is
        resolved under ``work_root`` (a stable per-deployment dir; per-test tmp)."""
        p = Path(self.cfg.round.submissions_db_path)
        return p if p.is_absolute() else (self.work_root / p)

    # ── miner-funded compute (DEC-CA-0036) ───────────────────────────────────

    def _funded_harvest(self) -> bool:
        """Credential-free funded pods (``[round] funded_pod_checkpoint =
        "harvest"``, the default): the worker runs ``--local-only``, the pod
        holds NO Hub credential, and the orchestrator harvests + verifies +
        uploads the checkpoint itself. ``"robot"`` keeps the per-pod
        push-only Harbor robot (the pre-harvest image path)."""
        mode = str(getattr(self.cfg.round, "funded_pod_checkpoint", "harvest"))
        if mode not in ("harvest", "robot"):
            raise RuntimeError(
                f"[round] funded_pod_checkpoint must be 'harvest' or 'robot'; got {mode!r}")
        return mode == "harvest"

    def _funded_queue(self):
        """The funded queue, or ``None`` while ``[round] funded_mode = "off"``.

        A fresh file-backed instance per call (single-writer process; the
        intake service appends via its own instance and atomic replace, so the
        newest state is always one load away). Relative paths resolve under
        ``work_root``, like ``submissions_db_path``.
        """
        local = self.__dict__.get("_leg_local_obj")
        if getattr(local, "funded_is_king", False):
            return None
        queue = getattr(local, "funded_queue", None)
        if queue is not None:
            return queue
        from .runner_common import funded_queue_of

        return funded_queue_of(self)

    def _submission_store(self):
        """The private direct-submission store, or None while unset.

        Also exports the resolved dir as ``$CASCADE_VAULT_DIR`` (unless the
        operator already set one): the FETCH path — the dedup screen, a
        vault-ref king fetch, local training — resolves vault refs through
        that env, and without this bridge a fully configured store would be
        invisible to every fetch on the orchestrator.
        """
        d = self.cfg.round.submission_vault_dir
        if not d:
            return None
        import os

        from ..funding.store import VAULT_DIR_ENV, SubmissionStore

        p = Path(d)
        resolved = p if p.is_absolute() else (self.work_root / p)
        os.environ.setdefault(VAULT_DIR_ENV, str(resolved))
        return SubmissionStore(resolved)

    def _verify_vault_ownership(
        self, challengers: list[ResolvedGenerator]
    ) -> list[ResolvedGenerator]:
        """Drop vault-ref challengers whose digest belongs to someone else.

        Content-addressing cuts both ways: a miner who learns a digest (the
        published champion's, most obviously) could chain-commit
        ``vault/direct@sha256:<that digest>`` and claim bytes they never had.
        The store records who UPLOADED each digest; a mismatched claim is
        dropped here — before any burn, screen, or pod — and an unresolvable
        vault ref (no store configured, digest never uploaded) drops the same
        way, since nothing could ever train it. Hub refs pass untouched.
        """
        from ..funding.store import parse_vault_ref

        store = self._submission_store()
        kept: list[ResolvedGenerator] = []
        for c in challengers:
            digest = parse_vault_ref(c.ref)
            if digest is None:
                kept.append(c)
                continue
            owner = store.owner(digest) if store is not None else None
            if owner is None:
                log.warning("dropping %s: vault ref %s has no stored submission "
                            "(store %sconfigured)", c.hotkey, c.ref,
                            "" if store is not None else "NOT ")
                continue
            if owner != c.hotkey:
                log.warning("dropping %s: vault digest %s was uploaded by a "
                            "different hotkey — a copied digest is not a submission",
                            c.hotkey, digest)
                continue
            kept.append(c)
        return kept

    def _maybe_publish_champion(self, king: ResolvedGenerator, round_id: str) -> None:
        """Run the champion-publication policy for this round's resolved king.

        Best-effort by contract (a bucket must never sink a round) and inert
        unless BOTH ``[round] champion_publish`` and ``submission_vault_dir``
        are set. State (reign counter, published flag) lives in
        ``champion_publisher.json`` under work_root.
        """
        policy = self.cfg.round.champion_publish
        store = self._submission_store()
        if policy == "off" or store is None:
            return
        try:
            from ..funding.champion import ChampionPublisher

            publisher = ChampionPublisher(
                store, self.manifest_store(), policy=policy,
                delay_rounds=self.cfg.round.champion_publish_delay_rounds,
                state_path=self.work_root / "champion_publisher.json",
            )
            published = publisher.note_king(king.hotkey, king.ref, round_id)
            for digest in published:
                log.info("round %s: champion code published (%s)", round_id, digest)
        except Exception as e:  # noqa: BLE001 — publication must never sink the round
            log.warning("champion publication step failed (retries next round): %s", e)

    # ── stake-weighted activation (DEC-CA-0045) ─────────────────────────────

    def apply_activation_block(self, block: int) -> bool:
        """Arm the DEC-CA-0043 rollover at ``block`` on the live config and
        refresh the promotion engine's captured grid and tenure gate (its
        reign clock and ripeness threshold both follow the armed config).
        Returns True when the
        config changed. The rolling scheduler is built lazily from
        ``self.cfg`` only once ``rolling_active`` is true, so it always sees
        the armed config."""
        from ..shared.activation import apply_activation

        base = self._base_cfg()
        new_cfg = apply_activation(base, block)
        if new_cfg is base:
            return False
        self._set_base_cfg(new_cfg)
        promo = getattr(self, "promotion", None)
        if promo is not None and hasattr(promo, "round_cfg"):
            promo.round_cfg = new_cfg.round
        if promo is not None and hasattr(promo, "scoring_cfg"):
            promo.scoring_cfg = new_cfg.scoring
        log.warning("activation: DEC-CA-0043 rollover ARMED at block %d (grid %d → %d): "
                    "rolling intake + era king start there", block,
                    new_cfg.round.epoch_blocks_prev, new_cfg.round.epoch_blocks)
        return True

    # ── owner-signed network config (DEC-CA-0057), read-only ────────────────

    def _base_cfg(self):
        """The live config without an overlay view (``self.cfg`` while inert)."""
        return self._cfg_base if self._cfg_base is not None else self.cfg

    def _set_base_cfg(self, cfg) -> None:
        """Replace the base config (an activation lock-in); a view in place is
        re-derived from it at the next :meth:`_netconfig_tick`."""
        if self._cfg_base is not None:
            self._cfg_base = cfg
        else:
            self.cfg = cfg

    def cfg_at(self, block: int | None):
        """The config the validators judge a settlement at ``block`` under —
        the same resolver, read-only (the base config itself while inert)."""
        base = self._base_cfg()
        nc = self.netconfig
        if nc is None or not getattr(nc, "enabled", False):
            return base
        return nc.config_at(base, block)

    def _netconfig_tick(self, client, block: int) -> None:
        """Resolve the owner's overlay exactly as the validators do (fetch,
        verify, read the ``cfg-`` lock-in from their notes — never write one)
        and point ``self.cfg``, the rolling scheduler and the promotion engine
        at the config in force at the current boundary. Never raises; a no-op
        while inert."""
        nc = self.netconfig
        if nc is None or not getattr(nc, "enabled", False):
            return
        try:
            base = self._base_cfg()
            nc.tick(base, client, now_block=int(block))
            # an overlay applies at an era start (a boundary on both grids),
            # so the view at the head is the view at its boundary
            view = nc.config_at(base, int(block))
            if view is base:
                if self._cfg_base is not None:
                    self.cfg, self._cfg_base = base, None
                    self._point_cfg_consumers(base)
                return
            if view is not self.cfg:
                self._cfg_base, self.cfg = base, view
                self._point_cfg_consumers(view)
                log.info("netconfig: overlay seq %d (%s…) in force from block %d",
                         view.netconfig.seq, view.netconfig.digest[:12],
                         view.netconfig.applied_block)
        except Exception as e:  # noqa: BLE001 — config adoption must never sink a tick
            log.warning("netconfig step failed (%s); retrying next poll", e)

    def _point_cfg_consumers(self, cfg) -> None:
        sched = self.__dict__.get("_rolling_sched")
        if sched is not None:
            sched.update_config(cfg)
        promo = getattr(self, "promotion", None)
        if promo is not None and hasattr(promo, "round_cfg"):
            promo.round_cfg = cfg.round
        if promo is not None and hasattr(promo, "scoring_cfg"):
            promo.scoring_cfg = cfg.scoring

    def _activation_tick(self, client, block: int) -> None:
        """One resolver pass per poll (a chain read only at a new boundary or
        while notes are pending). Never raises."""
        from ..shared.activation import ActivationRecord, record_for, resolve_activation

        base = self._base_cfg()
        if not getattr(base, "activation", None) or not base.activation.enabled:
            return
        try:
            rec = self._activation
            if rec is None:
                store = self.activation_store
                rec = store.load() if store is not None else ActivationRecord()
                rec = record_for(self._base_cfg(), rec)      # a renamed feature's record arms nothing
                self._activation = rec
                if rec.locked and rec.source != "config":
                    log.info("activation: restored lock-in (block %d, rollover %d, via %s)",
                             rec.lock_block, rec.activation_block, rec.source)
            res = resolve_activation(self._base_cfg(), client, now_block=int(block), record=rec)
            if res.changed:
                self._activation = res.record
                if self.activation_store is not None:
                    self.activation_store.save(res.record)
            rec = self._activation
            if rec.locked and rec.source != "config":
                self.apply_activation_block(rec.activation_block)
            self._forfeit_tick(client, block)
        except Exception as e:  # noqa: BLE001 — activation must never sink a tick
            log.warning("activation step failed (%s); retrying next poll", e)

    def _forfeit_tick(self, client, block: int) -> None:
        """DEC-CA-0048 forfeiture decided on chain: the trainer resolves the
        same tally the validators do (it never signals) so the rolling
        scheduler hands the king leg to the successor at the block they
        will apply."""
        from ..shared.activation import (
            ActivationRecord,
            ActivationStore,
            apply_forfeit_activation,
            forfeit_feature,
            record_for,
            resolve_activation,
        )

        spec = forfeit_feature(self._base_cfg())
        if spec is None or not spec.decided_on_chain:
            return
        store = None
        base = getattr(self.activation_store, "path", None)
        if base is not None:
            store = ActivationStore(Path(base).with_name("activation_forfeit_state.json"))
        rec = self._forfeit
        if rec is None:
            rec = record_for(self._base_cfg(), store.load() if store is not None else ActivationRecord(),
                             spec)
            self._forfeit = rec
            if rec.locked and rec.source != "config":
                log.info("activation: restored forfeiture lock-in (block %d, forfeit at %d, via %s)",
                         rec.lock_block, rec.activation_block, rec.source)
        res = resolve_activation(self._base_cfg(), client, now_block=int(block), record=rec,
                                 feature=spec)
        if res.changed:
            self._forfeit = res.record
            if store is not None:
                store.save(res.record)
        rec = self._forfeit
        if rec.locked and rec.source != "config":
            base = self._base_cfg()
            new_cfg = apply_forfeit_activation(base, rec.activation_block)
            if new_cfg is not base:
                self._set_base_cfg(new_cfg)
                promo = getattr(self, "promotion", None)
                if promo is not None and hasattr(promo, "scoring_cfg"):
                    promo.scoring_cfg = new_cfg.scoring
                log.warning("activation: DEC-CA-0048 forfeiture ARMED at block %d — the era whose "
                            "first settlement reaches it trains the successor's king leg",
                            new_cfg.scoring.forfeit_from_block)

    def _harvest_leg(self, host, receipt, *, base_seed: int, repo_suffix: str = ""):
        """Dispatcher hook for ISOLATED hosts that were not dispatched
        ``local_checkpoint`` explicitly (operator lanes declared
        ``isolated = true``, the JIT king pod): the worker trained
        ``--local-only``; pull, verify and upload its checkpoint exactly like
        a payer leg, resolving the contract from the receipt's size."""
        contract = self.cfg.training.primary_size
        size = str(getattr(receipt, "size", "") or "")
        if size and size != contract.arch_preset:
            spec = next((sp for sp in self.cfg.training.extra_sizes
                         if sp.arch_preset == size), None)
            if spec is None:
                raise RuntimeError(f"harvest: receipt names unknown size {size!r}")
            contract = self.cfg.training.for_size(spec)
        seeds = RoundSeeds.derive(int(base_seed), self.cfg.training)
        return self._harvest_funded_checkpoint(host, receipt, contract, seeds, repo_suffix)

    def _harvest_funded_checkpoint(self, host, receipt, contract, seeds,
                                   suffix: str):
        """Pull a ``--local-only`` funded leg's checkpoint off its payer pod,
        ingest-verify it, upload it under the operator's identity, and return
        the real :class:`TrainedEntry` (DEC-CA-0036 credential-free pods).

        Any guard deviation raises :class:`_FundedTamper` (miner fault,
        terminal); a transport/upload failure raises plain (infra — the
        miner's training was fine, the operator's pull was not)."""
        from ..eval.checkpoint_guard import (
            CheckpointDiverged,
            CheckpointTampered,
            verify_checkpoint,
            verify_weights_finite,
        )
        from .progress import report_progress
        from .remote import harvest_remote_dir

        report_progress(self, "harvesting")

        dest = harvest_dir(self.work_root, seeds.base_seed, receipt.miner_hotkey)
        harvest_remote_dir(host, receipt.checkpoint_dir, dest)
        if getattr(receipt, "marketcast", None):
            # DEC-CA-0064: a Timestral fine-tune, not a Cascade-MoE checkpoint
            from .marketcast_runner import verify_harvested_mkt_checkpoint

            why = verify_harvested_mkt_checkpoint(self, dest, receipt)
            if why:
                raise _FundedTamper(f"checkpoint: {why}")
            ckpt_repo = (f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-"
                         f"{receipt.role}-mkt{suffix}")
            self._mkt_harvested[receipt.miner_hotkey] = dict(receipt.marketcast)
            return upload_harvested_checkpoint(self, {
                "dest": str(dest), "ckpt_repo": ckpt_repo, "suffix": suffix, "size": "mkt",
                "miner_hotkey": receipt.miner_hotkey, "miner_uid": int(receipt.miner_uid),
                "role": receipt.role, "gen_ref": receipt.gen_ref,
                "corpus_digest": receipt.corpus_digest, "train_block": int(receipt.train_block),
                "gpu_name": receipt.gpu_name, "receipt_size": "mkt"},
                base_seed=seeds.base_seed)
        # The harvested tensor files must be the ones the worker hashed in
        # memory at save time; anything else on disk is not the run's product.
        from .ckpt_digest import tensor_mismatches

        expected = dict(getattr(receipt, "tensor_digests", None) or {})
        if expected:
            bad = tensor_mismatches(dest, expected)
            if bad:
                raise _FundedTamper("checkpoint: " + "; ".join(bad))
        elif self.cfg.round.funded_require_tensor_digests:
            raise _FundedTamper("checkpoint: the worker receipt carries no tensor digests")
        else:
            log.warning("funded leg %s: worker receipt carries no tensor digests (worker "
                        "image predates them) — checkpoint digest check skipped",
                        receipt.miner_hotkey)
        try:
            verify_checkpoint(dest, contract)
            # Only after the header bounded the file: a NaN/inf checkpoint is
            # the miner's diverged run, refused BEFORE the upload.
            verify_weights_finite(dest)
        except CheckpointTampered as e:
            raise _FundedTamper(f"checkpoint: {e}") from e
        except CheckpointDiverged as e:
            raise _FundedDiverged(str(e)) from e
        size = contract.arch_preset
        ckpt_repo = (f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-"
                     f"{receipt.role}-{size}{suffix}")
        record = {
            "dest": str(dest), "ckpt_repo": ckpt_repo, "suffix": suffix, "size": size,
            "miner_hotkey": receipt.miner_hotkey, "miner_uid": int(receipt.miner_uid),
            "role": receipt.role, "gen_ref": receipt.gen_ref,
            "corpus_digest": receipt.corpus_digest, "train_block": int(receipt.train_block),
            "gpu_name": receipt.gpu_name, "receipt_size": receipt.size or "",
            "warm_start_ref": receipt.init_pointer,
            "contract_digest": receipt.training_contract_digest,
        }
        return upload_harvested_checkpoint(self, record, base_seed=seeds.base_seed)

    def _archive_generator_tree(self, ref: str, d: Path) -> None:
        """Best-effort audit archive of a fetched generator tree.

        Miners routinely delete their repos after a leg (breaking later
        anneals and audit replays); the fetch is the one guaranteed moment the
        code exists. Tar the tree to the manifest bucket under its ref key —
        the mirror store dual-writes to R2 like every other put.
        Skip-if-present keeps it idempotent across retries; ANY failure is
        swallowed — the archive must never disturb the leg it records.

        PRIVATE (``vault/direct``) submissions never touch the manifest bucket:
        it is a shared surface (every reader of the bucket sees every object
        under ``generators/``), and an unpublished submission's code must stay
        with the operator until the crown policy publishes it. They archive to
        the operator-only king-archive bucket instead (``vault/<digest>.tar``,
        :func:`_private_archive_store`); when that store is unavailable the
        tree is simply not archived — the vault ZIP on-box already retains it.
        """
        import io
        import tarfile

        from ..funding.store import parse_vault_ref
        digest = parse_vault_ref(ref)
        try:
            if digest is not None:
                store = self._private_archive_store()
                if store is None:
                    log.debug("generator archive: private submission %s stays on-box "
                              "(no private archive store)", ref[:60])
                    return
                key = private_archive_key(digest)
            else:
                store = self.manifest_store()
                key = generator_archive_key(ref)
            try:
                store.get_bytes(key)
                return  # already archived (tars are KBs; a GET probe is fine)
            except Exception:  # noqa: BLE001 — missing/unreadable ⇒ (re)write
                pass
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                tar.add(str(d), arcname=".")
            store.put_bytes(key, buf.getvalue())
            log.info("archived generator %s (%d bytes)", ref[:60], buf.tell())
        except Exception as e:  # noqa: BLE001 — never sink a round for the archive
            log.warning("generator archive failed for %s (ignored): %s", ref[:60], e)

    def _private_archive_store(self):
        """The operator-only store private submissions archive to (the R2
        king-archive bucket, :func:`cascade.shared.king_archive.king_archive_config`),
        or ``None`` when it is not configured / not reachable. Resolved once
        per process; a failure is remembered so a misconfigured archive costs
        one warning, not one per entrant. Never falls back to the manifest
        store — a private tree must not land on a shared surface."""
        cached = getattr(self, "_private_archive", None)
        if cached is not None:
            return cached or None
        try:
            from ..shared.king_archive import king_archive_config
            s3cfg, _endpoint, _bucket = king_archive_config(self.cfg.storage)
            store = S3Store(cfg=s3cfg)
            store.client()  # credentials + endpoint resolve here, not mid-archive
        except Exception as e:  # noqa: BLE001 — archive is best-effort
            log.warning("private generator archive unavailable (private submissions "
                        "stay on-box only): %s", e)
            self._private_archive = False
            return None
        self._private_archive = store
        return store

    # ── per-generator train (GPU + registry + S3 boundary) ───────────────────

    def _train_checkpoint(
        self,
        gen: ResolvedGenerator,
        seeds: RoundSeeds,
        contract: TrainingContractConfig,
        token_budget: int,
        out_dir: Path,
        *,
        log_role: str,
        warm_start_dir: Path | None = None,
        expert_selection: dict | None = None,
        lane_info: dict | None = None,
    ) -> tuple[TrainResult, str, int, int]:
        """Fetch generator (registry) → build corpus → train into ``out_dir``,
        streaming per-step metrics to S3. No upload — the caller decides where
        the checkpoint goes (uploaded by the worker, or harvested by the
        orchestrator).

        ``contract`` is the per-size training contract (the base recipe with this
        size's width/depth/digest/throughput); ``token_budget`` is its compute
        budget for this leg. ``warm_start_dir`` is a fetched promoted-init
        checkpoint dir: the run initialises from its weights instead of random
        (Cascade consumption — forwarded to the backend only when set, so
        custom BaseTrainers without the kwarg keep working random-init).
        Returns ``(result, corpus_digest, n_series, total_points)``. Raises on
        any failure.
        """
        gen_dir = out_dir.parent / "generator"
        try:
            if contract.corpus_mode == "shards":
                # DEC-CA-0053: the CPU tier already ran the generator; this pod
                # reads shards and never receives miner code.
                gen_dir.mkdir(parents=True, exist_ok=True)
            else:
                fetch_from_hub(gen.ref, gen_dir, self.hub())
        except StorageError as e:
            status = _http_status_in_chain(e)
            if status in (401, 403, 404):
                # The MINER's repo, not our infra: a private or missing artifact
                # is the submitter's fault (Hippius Harbor projects must be
                # public for the trainer to pull them — see docs/MINER.md).
                raise CorpusError(
                    f"generator_artifact_unreachable: HTTP {status} for {gen.ref}"
                ) from e
            raise
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "round=%s run=%s: fetched generator %s — building corpus + training "
            "(mode=%s, budget=%s point-passes) …",
            seeds.base_seed, log_role, gen.ref[:48],
            contract.corpus_mode, f"{token_budget:,}",
        )

        # Stream per-step metrics to S3 (best-effort: logging must never abort a
        # training run). ``log_role`` carries the size/heat tag so each run's log
        # lands at a distinct key (king-toto2-4m, challenger-toto2-22m, heat-<hk>).
        sink: LogSink | None = None
        try:
            sink = LogSink(self.logs_store(), round_id=str(seeds.base_seed), role=log_role)
        except Exception as e:  # noqa: BLE001
            log.warning("log sink unavailable (continuing without S3 logs): %s", e)
        # Optional live wandb mirror (observability only — the same per-step
        # records, so miners can watch this run train as it occurs). Best-effort:
        # disabled/unavailable ⇒ None, and every wandb call swallows its errors.
        # ``lane_info`` ({lane, target, recipe}) labels the leg's lane on W&B
        # only: a lane-B / lane-C leg trains under the ``challenger`` role
        # (its receipt role), so without it every lane logs alike.
        li = dict(lane_info or {})
        wandb_sink = open_wandb_run(
            self.cfg.wandb,
            round_id=str(seeds.base_seed), role=log_role,
            hotkey=gen.hotkey, uid=gen.uid, size=contract.arch_preset,
            config={"corpus_mode": contract.corpus_mode, "token_budget": token_budget,
                    "contract_digest": contract_digest(contract)},
            lane=str(li.get("lane") or "A"), target=str(li.get("target") or ""),
            recipe=li.get("recipe"),
        )
        emitters = [s for s in (sink, wandb_sink) if s is not None]
        logger = (lambda record: [s.emit(record) for s in emitters]) if emitters else None

        # Host telemetry, taken HERE — after the generator is fetched but before
        # the corpus stream (and therefore the sandbox child) exists, so the
        # calibration bench measures the pod and not the submission competing
        # with it. It cannot eat the compute budget: max_train_seconds anchors at
        # the first training batch, and token_budget is a token count.
        host_facts = self._host_snapshot()
        if host_facts:
            for s in emitters:
                s.emit({"event": "host", "role": log_role, **host_facts})
            log.info(
                "round=%s run=%s host: %s",
                seeds.base_seed, log_role, host_summary_line(host_facts),
            )

        continuation = None
        continuation_args = self.__dict__.get("_continuation_args")
        if continuation_args is not None and getattr(continuation_args, "continuation_root", None):
            from .continuation import PROTOCOL, digest_file, worker_context
            if getattr(contract, "continuation_protocol", "") != PROTOCOL:
                raise ValueError("continuation requires an explicitly activated training contract")
            base_files = {p.name: digest_file(p) for p in sorted(Path(warm_start_dir).glob('*.safetensors'))} if warm_start_dir else {}
            continuation = worker_context(continuation_args, self.cfg, {
                "backend": contract.base_arch, "contract": contract_digest(contract),
                "base_files": base_files, "gen_ref": gen.ref, "hotkey": gen.hotkey,
                "generation_seed": seeds.generation_seed, "training_seed": seeds.training_seed,
                "token_budget": token_budget, "experts": expert_selection})

        with open_round_stream(
            contract.corpus_mode,
            gen_dir, seeds.generation_seed, self.cfg.generator_for_contract(contract),
            token_budget=token_budget,
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
            max_wall_seconds=contract.max_train_seconds,
            seed_mix=int(getattr(contract, "gen_seed_mix", 1) or 1),
            budget_denomination=getattr(contract, "budget_denomination", "points"),
            corpus_epochs=int(getattr(contract, "corpus_epochs", 0) or 0),
            **({"continuation_protocol": contract.continuation_protocol} if getattr(contract, "continuation_protocol", "") else {}),
        ) as rs:
            result = self.base_trainer.train(
                rs.series(),
                contract,
                training_seed=seeds.training_seed,
                token_budget=token_budget,
                out_dir=out_dir,
                logger=logger,
                **({"warm_start_dir": warm_start_dir} if warm_start_dir is not None else {}),
                # DEC-CA-0050 lane-B leg: only the selected expert rows train
                **expert_kw(expert_selection),
                **({"continuation": continuation} if continuation is not None else {}),
            )
            corpus_digest, n_series, total_points = rs.digest, rs.n_series, rs.total_points

        # The host facts are repeated on the summary row, not just left on their
        # own "host" record: the whole point is regressing realized throughput on
        # host capability, and that is a one-liner only when both live on the same
        # row. The standalone record is what a run that DIES before the summary
        # leaves behind — on the live wandb mirror and the stderr line, since the
        # S3 blob is only written at flush.
        summary = {"event": "summary", "role": log_role, "corpus_digest": corpus_digest,
                   "n_series": n_series, "total_points": total_points,
                   "train_seconds": result.train_seconds, **host_facts, **result.metrics}
        for s in emitters:
            s.emit(summary)
        self._leg_summaries.setdefault(str(seeds.base_seed), {})[log_role] = summary
        while len(self._leg_summaries) > 8:          # bounded: a round that never publishes
            del self._leg_summaries[next(iter(self._leg_summaries))]
        if sink is not None:
            try:
                sink.flush()
            except Exception as e:  # noqa: BLE001
                log.warning("failed to flush S3 training logs: %s", e)
        if wandb_sink is not None:
            wandb_sink.finish()

        log.info(
            "round=%s run=%s hotkey=%s mode=%s n=%d points=%d digest=%s",
            seeds.base_seed, log_role, gen.hotkey, contract.corpus_mode,
            n_series, total_points, corpus_digest[:12],
        )
        # One parseable key=value telemetry line per run. TrainResult.metrics
        # never crosses the remote boundary (the receipt is a TrainedEntry),
        # so this stderr line IS how a pod run's starvation/deadline telemetry
        # reaches the orchestrator's captured dispatch output.
        m = result.metrics or {}
        if "deadline_hit" in m:
            log.info(
                "round=%s run=%s telemetry: deadline_hit=%s tokens_frac=%s "
                "data_wait_s=%s data_wait_frac=%s",
                seeds.base_seed, log_role, m.get("deadline_hit"),
                m.get("tokens_frac"), m.get("data_wait_s"), m.get("data_wait_frac"),
            )
        return result, corpus_digest, n_series, total_points

    def _host_snapshot(self) -> dict:
        """Host facts for the run about to start, per ``[telemetry]`` — ``{}`` when
        disabled or wholly unavailable.

        The device comes from the backend when it exposes one: ``BaseTrainer`` is a
        Protocol and only the reference implementation carries a ``.device``, so a
        custom backend benches on whatever torch finds rather than not at all.
        """
        tcfg = getattr(self.cfg, "telemetry", None)
        if tcfg is not None and not getattr(tcfg, "host_probe", True):
            return {}
        run_bench = getattr(tcfg, "host_bench", True) if tcfg is not None else True
        try:
            return host_snapshot(
                device=getattr(self.base_trainer, "device", None),
                run_bench=bool(run_bench),
            )
        except Exception as e:  # noqa: BLE001 — telemetry must never abort a round
            log.warning("host telemetry unavailable (continuing): %s", e)
            return {}

    def train_one(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        block: int,
        *,
        contract: TrainingContractConfig | None = None,
        token_budget: int | None = None,
        repo_suffix: str = "",
        heat: bool = False,
        warm_start_ref: str | None = None,
        expert_selection: dict | None = None,
        finetune: bool = False,
        lane_info: dict | None = None,
    ) -> TrainedEntry:
        """Train one generator at one size, upload its checkpoint, return the receipt.

        ``contract`` defaults to the primary (smallest) size; ``token_budget`` to
        that size's full ``train_tokens``. The checkpoint is uploaded to a
        size-tagged registry repo (``ckpt-r<seed>-<role>-<size><repo_suffix>``)
        and the entry carries the ``size`` tag so the validator can pair king
        and challenger per size. ``repo_suffix`` (``-u<uid>``) keeps the
        challengers of one era from overwriting each other's checkpoint.

        ``heat`` tags a REDUCED-budget run (the worker's ``--train-hours``:
        the bench-anneal leg) so its S3 and wandb telemetry lands at a distinct
        ``heat-<hotkey>`` key instead of colliding with the full leg's
        ``<role>-<size>`` key. The receipt's ``role`` is unaffected.

        Raises on any failure.
        """
        contract = contract if contract is not None else self.cfg.training.primary_size
        token_budget = token_budget if token_budget is not None else contract.train_tokens
        size = contract.arch_preset
        out_dir, corpus_digest, gpu_name = self._train_for_entry(
            gen, role, seeds, contract=contract, token_budget=token_budget,
            repo_suffix=repo_suffix, heat=heat, warm_start_ref=warm_start_ref,
            **expert_kw(expert_selection), **({"finetune": True} if finetune else {}),
            **({"lane_info": lane_info} if lane_info else {}))
        ckpt_repo = f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-{role}-{size}{repo_suffix}"
        # Hub is priority-one; mirror to HF only if the Hub is down (keeps a round
        # alive through a Hub outage instead of failing the checkpoint upload).
        up = upload_dir_to_hub_or_hf(
            out_dir, ckpt_repo, self.hub(),
            hf_repo=self._hf_ckpt_repo(ckpt_repo),
        )
        return TrainedEntry(
            miner_hotkey=gen.hotkey,
            miner_uid=gen.uid,
            role=role,
            gen_ref=gen.ref,
            trained_pointer=format_trained_pointer(up.ref.immutable_ref),
            corpus_digest=corpus_digest,
            train_block=block,
            gpu_name=gpu_name,
            size=size,
            init_pointer=warm_start_ref or "",
            training_contract_digest=contract_digest(contract),
        )

    def train_one_local(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        block: int,
        *,
        contract: TrainingContractConfig | None = None,
        token_budget: int | None = None,
        repo_suffix: str = "",
        heat: bool = False,
        warm_start_ref: str | None = None,
        expert_selection: dict | None = None,
        finetune: bool = False,
        lane_info: dict | None = None,
    ) -> dict:
        """:meth:`train_one` WITHOUT the registry upload: train, stamp the
        marker, and return a plain receipt dict — every ``TrainedEntry`` field
        except ``trained_pointer`` plus ``local_checkpoint_dir``.

        This is the credential-free pod mode (DEC-CA-0036): a payer-controlled
        box holds NO Hub credential of any kind — the orchestrator harvests the
        checkpoint over the pinned SSH, verifies it, and uploads it under its
        own identity. Deliberately a dict, not a ``TrainedEntry``: the entry
        class refuses anything but a real hub pointer, which is exactly the
        property that keeps a local receipt out of a manifest."""
        contract = contract if contract is not None else self.cfg.training.primary_size
        token_budget = token_budget if token_budget is not None else contract.train_tokens
        out_dir, corpus_digest, gpu_name = self._train_for_entry(
            gen, role, seeds, contract=contract, token_budget=token_budget,
            repo_suffix=repo_suffix, heat=heat, warm_start_ref=warm_start_ref,
            **expert_kw(expert_selection), **({"finetune": True} if finetune else {}),
            **({"lane_info": lane_info} if lane_info else {}))
        return {
            "miner_hotkey": gen.hotkey,
            "miner_uid": gen.uid,
            "role": role,
            "gen_ref": gen.ref,
            "corpus_digest": corpus_digest,
            "train_block": int(block),
            "gpu_name": gpu_name,
            "size": contract.arch_preset,
            "local_checkpoint_dir": str(out_dir),
            "init_pointer": warm_start_ref or "",
            "training_contract_digest": contract_digest(contract),
            "tensor_digests": self._tensor_digests_of(out_dir),
        }

    def _tensor_digests_of(self, out_dir: Path) -> dict[str, str]:
        """``{tensor file: sha256}`` as hashed in memory at save time: this
        process's own writes, else the ones the ``.train_complete`` marker
        carries (a complete checkpoint reused from an earlier process)."""
        from .ckpt_digest import digests_for

        got = digests_for(out_dir)
        if got:
            return got
        try:
            payload = json.loads((out_dir / self.TRAIN_COMPLETE_MARKER).read_text())
            return {str(k): str(v) for k, v in dict(payload.get("tensor_digests") or {}).items()}
        except Exception:  # noqa: BLE001 — no marker ⇒ no digests to report
            return {}

    def _train_for_entry(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        *,
        contract: TrainingContractConfig,
        token_budget: int,
        repo_suffix: str,
        heat: bool,
        warm_start_ref: str | None,
        expert_selection: dict | None = None,
        finetune: bool = False,
        lane_info: dict | None = None,
    ) -> tuple[Path, str, str]:
        """The shared training body of :meth:`train_one` / :meth:`train_one_local`:
        train (or reuse a complete prior run), stamp the marker, and return
        ``(checkpoint dir, corpus_digest, gpu_name)`` — everything but the upload."""
        size = contract.arch_preset
        out_dir = self.work_root / f"{seeds.base_seed}" / size / f"{role}{repo_suffix}" / "checkpoint"
        log_role = f"heat-{gen.hotkey}" if heat else f"{role}-{size}"
        # W&B lane label (observability only): the caller's, else derived from
        # the leg's shape — a fine-tune leg is lane C, an expert leg lane B
        from .wandb_sink import lane_of

        lane_info = {"lane": lane_of(finetune=finetune, expert=expert_selection is not None),
                     **(lane_info or {})}
        # Retry-without-retrain: if a PRIOR run of this exact job already trained
        # a complete checkpoint here (marker written post-train, pre-upload) and
        # only the upload failed, reuse it instead of burning the budget again.
        # Sound because training is deterministic — same generator/seeds/contract
        # rederives byte-identical weights.
        job_identity = {"warm_start_ref": warm_start_ref or "", "training_seed": seeds.training_seed,
                        "generation_seed": seeds.generation_seed, "token_budget": token_budget,
                        "expert_selection": expert_selection}
        reused = self._reusable_checkpoint(out_dir, contract, gen, job_identity=job_identity)
        if reused is not None:
            log.warning(
                "round=%s %s %s: COMPLETE checkpoint already at %s (prior run "
                "trained it; only its upload failed) — skipping straight to "
                "upload", seeds.base_seed, role, gen.hotkey, out_dir,
            )
            corpus_digest = str(reused["corpus_digest"])
            gpu_name = str(reused.get("gpu_name", ""))
        else:
            # Cascade warm-start: fetch the pinned promoted init (content-addressed;
            # the OCI digest verifies the bytes). A fetch failure RAISES — the run
            # must never silently fall back to random init (DEC-CA-0005).
            ws_dir = self._fetch_checkpoint_dir(warm_start_ref) if warm_start_ref else None
            if finetune:
                # DEC-CA-0056 lane-C leg: the base king's SCORED weights only —
                # no optimizer / EMA / stable-branch file, so the run starts a
                # fresh optimizer from exactly the bytes validators score
                from .finetune import stage_finetune_base

                if ws_dir is None:
                    raise ValueError("a lane-C fine-tune leg needs the base checkpoint "
                                     "(warm_start_ref)")
                ws_dir = stage_finetune_base(ws_dir, out_dir.parent / "finetune_base")
            result, corpus_digest, _, _ = self._train_checkpoint(
                gen, seeds, contract, token_budget, out_dir, log_role=log_role,
                warm_start_dir=ws_dir, **expert_kw(expert_selection),
                # forwarded only off lane A, so a lane-A call (and test doubles
                # without the kwarg) stays unchanged
                **({"lane_info": lane_info}
                   if lane_info["lane"] != "A" or lane_info.get("target") else {}),
            )
            gpu_name = str(result.metrics.get("gpu_name", ""))
            from .lineage import write_measurement

            write_measurement(
                out_dir, metrics=result.metrics, gen_ref=gen.ref,
                corpus_digest=corpus_digest,
                contract_digest=contract_digest(contract),
                warm_start_ref=warm_start_ref or "", token_budget=token_budget)
            self._write_train_complete_marker(
                out_dir, contract, gen, corpus_digest=corpus_digest, gpu_name=gpu_name,
                job_identity=job_identity,
            )
        return out_dir, corpus_digest, gpu_name

    # ── retry-without-retrain (the .train_complete marker) ────────────────────

    TRAIN_COMPLETE_MARKER = ".train_complete"

    def _write_train_complete_marker(
        self, out_dir: Path, contract: TrainingContractConfig,
        gen: ResolvedGenerator, *, corpus_digest: str, gpu_name: str,
        job_identity: dict | None = None,
    ) -> None:
        """Stamp ``out_dir`` as a COMPLETE training product (written after
        training returns, before the upload is attempted) so an upload-failure
        retry can skip the retrain. Atomic (tmp + rename); best-effort — a
        marker miss just costs the old retrain-on-retry behaviour."""
        from .ckpt_digest import digests_for

        try:
            payload = json.dumps({
                "contract_digest": contract_digest(contract),
                "gen_ref": gen.ref,
                "corpus_digest": corpus_digest,
                "gpu_name": gpu_name,
                "job_identity": job_identity,
                # The in-memory digests of the tensor files this run wrote,
                # so a reuse of the checkpoint still reports them.
                "tensor_digests": digests_for(out_dir),
            }, sort_keys=True)
            tmp = out_dir / (self.TRAIN_COMPLETE_MARKER + ".tmp")
            tmp.write_text(payload)
            tmp.rename(out_dir / self.TRAIN_COMPLETE_MARKER)
        except Exception as e:  # noqa: BLE001 — never sink a finished run
            log.warning("train-complete marker write failed for %s: %s", out_dir, e)

    def _reusable_checkpoint(
        self, out_dir: Path, contract: TrainingContractConfig, gen: ResolvedGenerator,
        *, job_identity: dict | None = None,
    ) -> dict | None:
        """The marker payload iff ``out_dir`` holds a complete checkpoint for
        EXACTLY this job (same contract digest + same generator ref, weights
        present) — else None. ``out_dir`` is already round/size/role-scoped, so
        the digest+ref match is a belt-and-braces identity check, not the only
        one. A torn/stale/mismatched marker reads as None (retrain, the safe
        direction)."""
        marker = out_dir / self.TRAIN_COMPLETE_MARKER
        try:
            if not marker.exists() or not (out_dir / "weights.safetensors").exists():
                return None
            payload = json.loads(marker.read_text())
            if job_identity is not None and payload.get("job_identity") != json.loads(json.dumps(job_identity)):
                return None
            if (payload.get("contract_digest") == contract_digest(contract)
                    and payload.get("gen_ref") == gen.ref
                    and payload.get("corpus_digest")):
                return payload
        except Exception as e:  # noqa: BLE001 — unreadable marker ⇒ retrain
            log.warning("train-complete marker unreadable at %s (%s); retraining",
                        out_dir, e)
        return None

    def _fetch_checkpoint_dir(self, trained_pointer: str) -> Path:
        """Fetch a just-trained checkpoint from the registry to a local dir (the
        OCI digest self-verifies the bytes). Uniform for local and remote training,
        since every final checkpoint is uploaded to the registry."""
        ref = parse_trained_pointer(trained_pointer)
        if ref is None:
            raise ValueError(f"malformed trained_pointer: {trained_pointer!r}")
        from ..shared.hippius import HubRef

        dest = self.work_root / "_bench_ckpts" / HubRef.parse(ref).digest.replace(":", "-")
        fetch_from_hub(ref, dest, self.hub())
        return dest

    def _dispatch_mode(self) -> dict:
        """RemoteDispatcher kwargs for the detached/attached choice (see
        remote.run_detached; [round] detached_dispatch)."""
        if not self.detached_dispatch:
            return {}          # attached: the dispatcher's defaults (and test doubles) unchanged
        return {"detached": True,
                "poll_seconds": int(self.dispatch_poll_seconds),
                "reattach_grace_seconds": int(self.dispatch_reattach_grace_seconds)}

    def _pod_extra_forward_env(self) -> tuple[str, ...]:
        """Env vars every pod dispatch forwards on top of each host's own list.

        When [wandb] is enabled the training runs on the pod, so the POD is where
        ``open_wandb_run`` needs ``WANDB_API_KEY`` — forwarding it here means
        pod-side wandb logs land without the operator having to name the key in
        every host's ``forward_env`` (the silent-no-op that leaves wandb runs with
        no training logs). Only forwarded if actually present in the orchestrator
        env; absent ⇒ the pod's wandb no-ops exactly as before."""
        return ("WANDB_API_KEY",) if getattr(self.cfg.wandb, "enabled", False) else ()

    def _pod_isolated_forward_env(self) -> tuple[tuple[str, str], ...]:
        """Credentials an ISOLATED (payer-paid) pod may receive: a project-scoped
        wandb key named by ``[wandb] funded_key_env``, delivered as WANDB_API_KEY
        so funded legs stream live like operator legs. The payer can read it, so
        the operator's own key name is refused — silently keeping the old
        no-wandb behaviour would hide a misconfiguration, so it warns."""
        wb = getattr(self.cfg, "wandb", None)
        name = str(getattr(wb, "funded_key_env", "") or "").strip()
        if not name or not getattr(wb, "enabled", False):
            return ()
        if name == "WANDB_API_KEY":
            log.warning("[wandb] funded_key_env names the operator's own key — refusing "
                        "to forward it to payer pods (use a project-scoped key)")
            return ()
        return (("WANDB_API_KEY", name),)

    @staticmethod
    def _dispatch_on_free_lane(disp, free_lanes, hosts: list, *, describe: str,
                               used_host: list | None = None, prepare=None,
                               deadline: float | None = None, **kw):
        """Dispatch on the next IDLE lane, retrying once on whichever lane is
        free after a failure (a different one whenever one is available).
        ``deadline`` (wall-clock) bounds BOTH lane waits — see
        :meth:`_FinalLanePool.get`; ``_LaneDeadlinePassed`` propagates.

        Lane occupancy is tracked through ``free_lanes`` (a ``queue.Queue`` of
        hosts), never a static pin, so a leg never lands on a GPU another leg
        still holds. A checked-out lane always returns to the pool, success
        or failure: a lane that failed for a challenger-specific reason
        (import error, OOM) is still good silicon — only a transport failure
        drains the pod's lanes."""
        from .remote import pod_lane_count

        # ``prepare(host) -> host`` runs on the CHOSEN lane right before each
        # dispatch (first try and the retry): the vault-ZIP staging hook — a
        # private submission's bytes must be on the lane before the worker
        # starts, or the leg burns an attempt on a pod that can't resolve it.
        _ready = prepare or (lambda h: h)
        _take = ((lambda: free_lanes.get(deadline=deadline)) if deadline is not None
                 else free_lanes.get)
        host = _take()
        try:
            entry = disp.dispatch(_ready(host), lane_count=pod_lane_count(host, hosts), **kw)
        except Exception as e:  # noqa: BLE001 — any dispatch failure is retryable once
            transport = _transport_failure(e)
            if transport and hasattr(free_lanes, "mark_dead"):
                # ssh refused / worker unreachable: the POD is gone, not the
                # leg. Its lanes leave the rotation and the host is
                # quarantined, so the retry (and every later leg) lands
                # elsewhere.
                free_lanes.mark_dead(host, error_tail(e, 160))
                _quarantine_lane_host(host, f"lane {getattr(host, 'name', host)}: {error_tail(e, 200)}")
            else:
                free_lanes.put(host)             # failed lane rejoins the rotation
            if _storage_failure(e):
                # Registry blips are global, so wait before re-dispatching.
                # Sleep BEFORE taking the retry lane so no idle GPU is held
                # through the backoff.
                log.warning("%s failed on %s at the storage layer; backing off "
                            "%.0fs before the retry", describe,
                            getattr(host, "name", host), STORAGE_RETRY_BACKOFF_SECONDS)
                time.sleep(STORAGE_RETRY_BACKOFF_SECONDS)
            retry_host = _take()                 # next idle lane; different when one exists
            if transport:
                # Never the dead pod again: a sibling lane handed back by a
                # concurrent failure may still be in the rotation.
                skipped = []
                same = getattr(free_lanes, "same_pod", None)
                while same is not None and same(retry_host, host) and len(skipped) <= len(hosts):
                    skipped.append(retry_host)
                    retry_host = _take()
                for sk in skipped:
                    if not (same is not None and same(sk, host)):
                        free_lanes.put(sk)
            log.warning("%s failed on %s (%s); retrying on %s", describe,
                        getattr(host, "name", host), e,
                        getattr(retry_host, "name", retry_host))
            try:
                entry = disp.dispatch(_ready(retry_host),
                                      lane_count=pod_lane_count(retry_host, hosts), **kw)
                if used_host is not None:
                    used_host.append(retry_host)
                return entry
            finally:
                free_lanes.put(retry_host)
        else:
            free_lanes.put(host)
            if used_host is not None:
                used_host.append(host)
            return entry

    def publish(self, manifest: TrainingManifest) -> None:
        """Sign the manifest with the trainer hotkey and write it to the Hippius
        S3 manifest bucket (``round-<id>.json`` + ``latest.json``)."""
        if self.wallet is not None:
            manifest = sign_manifest(manifest, self.wallet)
        elif manifest.signature is None:
            log.warning("publishing an UNSIGNED manifest (no wallet); validators will reject it")
        key = publish_manifest(self.manifest_store(), dump_manifest(manifest), manifest.round_id)
        log.info(
            "published manifest round=%s entries=%d signed=%s → s3://%s/%s",
            manifest.round_id, len(manifest.entries), manifest.signature is not None,
            self.cfg.storage.manifest_bucket, key,
        )
        # It reads each pod leg's log back from the logs bucket; nothing
        # consensus-relevant waits on it.
        self._publish_training_summary(manifest)

    def _publish_training_summary(self, manifest: TrainingManifest) -> str | None:
        """Unsigned per-round training telemetry — what each duel leg actually
        trained (steps, budget count, channel tokens, wall stop) — published
        public-read as ``training/round-<id>.json`` for the dashboard's
        Training tab (:mod:`cascade.shared.training_summary`). Local legs fold
        in from their in-process summary row; pod legs are read back from the
        log the worker flushed. Best-effort: never raises, never delays the
        manifest (which is already out)."""
        try:
            from ..shared.hippius import log_key
            from ..shared.training_summary import (
                build_training_summary,
                collect_round_legs,
                contract_block,
                dump_training_summary,
                publish_training_summary,
            )

            contracts = list(settlement_contracts(self.cfg, int(manifest.created_block)))
            primary = contracts[0].arch_preset
            rid = str(manifest.round_id)
            cached = self._leg_summaries.pop(rid, {})
            logs = self.logs_store()

            def read_log(log_role: str) -> str | None:
                try:
                    return logs.get_text(log_key(rid, log_role))
                except Exception:  # noqa: BLE001 — absent/unreadable ⇒ unmeasured leg
                    return None

            legs = collect_round_legs(manifest.entries, primary, cached=cached, read_log=read_log)
            by_size = {c.arch_preset: c for c in contracts}
            for leg in legs:
                c = by_size.get(leg.get("size"))
                if c is not None and c is not contracts[0]:
                    leg["contract"] = contract_block(c)
            doc = build_training_summary(
                rid, manifest.created_block, contract_block(contracts[0]), legs,
                warm_start_ckpt=str(getattr(manifest, "warm_start_ckpt", "") or ""),
                warm_start_size=str(getattr(manifest, "warm_start_size", "") or ""))
            key = publish_training_summary(self.manifest_store(), dump_training_summary(doc), rid)
            log.info("published training summary round=%s legs=%d measured=%d → s3://%s/%s",
                     rid, len(legs), sum(1 for x in legs if x.get("measured")),
                     self.cfg.storage.manifest_bucket, key)
            return key
        except Exception as e:  # noqa: BLE001 — telemetry must never fail a round
            log.warning("round=%s: training summary publish failed (ignored): %s",
                        manifest.round_id, e)
            return None

    # ── live loop ────────────────────────────────────────────────────────────

    @staticmethod
    def _with_deadline(fn, seconds: float):
        """Run ``fn()`` under a HARD wall-clock deadline in a helper thread.

        bittensor's websocket calls have no client-side timeout and can hang
        indefinitely; a loop that must publish once per round cannot block on
        one. A timeout bounds the caller's wait; it does not stop the underlying call.
        Shared with the provisioner through ``shared.concurrency``."""
        from ..shared.concurrency import call_with_deadline

        return call_with_deadline(fn, seconds)

    def _block_with_freeze_guard(self, client: object) -> int:
        """The chain height, rebuilding the connection when it hangs or freezes.

        A bittensor websocket can die quietly in three ways: it keeps answering
        ``current_block()`` with a stale value (~20-min-old block), it raises,
        or it hangs without erroring — and on a stale read the trainer
        re-derives an already-published round and re-enters it. Guard all three,
        exactly like the provisioner's ``_current_block``:

        * the read runs under ``chain_read_timeout_s`` so a hang becomes a
          ``TimeoutError`` instead of a wedged loop;
        * a raised/hung read rebuilds the connection (``client.reconnect()``)
          and retries once — the fresh client's answer is trusted;
        * a height that has not advanced for ``stale_block_after_s`` (blocks are
          ~12s, so a multi-minute freeze is anomalous) rebuilds and re-reads.

        A client without ``reconnect`` (offline fakes) cannot be rebuilt: a
        raise propagates and a freeze just returns the last read, unchanged."""
        now = self.chain_clock()
        reconnect = getattr(client, "reconnect", None)
        try:
            block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
        except Exception as e:  # noqa: BLE001 — a dead/hung client is rebuildable
            if reconnect is None:
                raise
            log.warning("chain read failed/hung (%s); rebuilding substrate connection",
                        type(e).__name__)
            reconnect()
            block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
            self._block_changed_at = now
        if self._last_block is None or block != self._last_block:
            self._last_block = block
            self._block_changed_at = now
        elif reconnect is not None and now - self._block_changed_at > self.stale_block_after_s:
            log.warning("chain block frozen at %d for %.0fs — rebuilding substrate "
                        "connection (quietly dead websocket?)",
                        block, now - self._block_changed_at)
            reconnect()
            block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
            self._last_block = block
            self._block_changed_at = now
        timestamp = getattr(client, 'block_timestamp', None)
        drift_limit = float(getattr(self.cfg.round, 'ops_clock_drift_seconds', 0))
        checked = self.__dict__.get('_chain_timestamp_checked', float('-inf'))
        if callable(timestamp) and drift_limit > 0 and now - checked >= 60:
            import math
            def fresh(height):
                stamp = float(self._with_deadline(lambda: timestamp(height), self.chain_read_timeout_s))
                return math.isfinite(stamp) and abs(time.time() - stamp) <= drift_limit
            if not fresh(block):
                if reconnect is not None:
                    reconnect()
                    block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
                if not fresh(block):
                    raise RuntimeError('chain timestamp is stale; preserving work until a fresh head is available')
            self.__dict__['_chain_timestamp_checked'] = now
        return block

    UPKEEP_INTERVAL_SECONDS = 1800.0
    UPKEEP_DEADLINE_SECONDS = 600.0

    def _upkeep(self) -> None:
        """Housekeeping off the chain path, each step best-effort and under a
        hard deadline (a hung provider API must not wedge the loop): reap
        generation VMs left billing (DEC-CA-0053), and age out harvested
        checkpoint copies nothing will read again."""
        from .runner_common import prune_harvest_dirs

        for what, fn in (("generation VM reap", self._reap_generation_vms),
                         ("harvest prune", lambda: prune_harvest_dirs(self.work_root))):
            try:
                self._with_deadline(fn, self.UPKEEP_DEADLINE_SECONDS)
            except Exception as e:  # noqa: BLE001 — housekeeping never sinks the loop
                log.warning("upkeep: %s failed (%s: %s)", what, type(e).__name__, e)

    def run_forever(self, client: object) -> None:  # pragma: no cover — glue over tick
        """Poll the chain and drive the rolling scheduler (DEC-CA-0043).

        Every poll: read the height (freeze-guarded), resolve the fleet's
        stake-weighted activation (DEC-CA-0045, which arms the rollover on the
        live config), then — once rolling intake is active at the current
        epoch boundary — tick the scheduler. Before the rollover the trainer
        has nothing to train: it idles and says so once per boundary. A
        failed tick is logged and retried next poll; the scheduler owns every
        pod's lifetime, so nothing is torn down on a failure."""
        from ..shared.calendar_launch import check_calendar_launch
        from .launch_preflight import check_worker_release
        from .operator_health import OperatorMonitor

        check_calendar_launch(self._base_cfg(), client)
        try:
            check_worker_release(self._base_cfg())
        except ValueError as exc:
            # Legacy work can continue; v2 rental still requires qualification.
            self.__dict__["_v2_launch_preflight_status"] = {"state": "blocked", "reason": str(exc)}
            log.error("v2 launch preflight blocked: %s", exc)
        monitor = OperatorMonitor(self, client)
        monitor.start()
        self.__dict__["_operator_monitor"] = monitor
        poll = self.cfg.manifest.poll_seconds
        waiting_logged: int | None = None
        last_upkeep = float("-inf")
        while True:
            if time.monotonic() - last_upkeep >= self.UPKEEP_INTERVAL_SECONDS:
                # Startup, then every interval: before the first tick, so a
                # crash's leftovers are reaped before anything new rents.
                last_upkeep = time.monotonic()
                self._upkeep()
            try:
                block = self._block_with_freeze_guard(client)
                # BEFORE the grid is derived: a lock-in arms rolling intake at
                # the very boundary it names.
                self._activation_tick(client, block)
                from ..shared.calendar_launch import resolve_calendar_launch

                base = self._base_cfg()
                resolved = resolve_calendar_launch(base, client)
                if resolved is not base:
                    self._set_base_cfg(resolved)
                    self._point_cfg_consumers(resolved)
                # DEC-CA-0057: the overlay in force at this boundary (read-only).
                self._netconfig_tick(client, block)
                # Resolved per tick: a scheduled cadence change must not pin
                # the pre-switch grid until a restart.
                epoch_blocks = effective_epoch_blocks(self.cfg.round, block)
                epoch_start = (block // epoch_blocks) * epoch_blocks
                if rolling_active(self.cfg.round, epoch_start):
                    self._rolling().tick(client, block)
                elif waiting_logged != epoch_start:
                    waiting_logged = epoch_start
                    log.info("block %d: rolling intake not active yet (rolling_from_block=%d; "
                             "awaiting the DEC-CA-0045 rollover) — idle", block,
                             self.cfg.round.rolling_from_block)
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one tick
                log.exception("trainer tick failed; retrying after the poll interval: %s", e)
            time.sleep(poll)
