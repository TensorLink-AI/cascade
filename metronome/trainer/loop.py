"""Trainer service loop — the owner-operated training round.

Each round the trainer:

1. Resolves on-chain generator commitments to ``(hotkey, uid, ref)``.
2. Identifies the reigning king (the highest-incentive UID on the metagraph in
   live mode; a caller-supplied hotkey offline) and selects challengers.
3. For the king and each challenger, under one shared :class:`RoundSeeds`:
   fetches the generator from the Hippius Hub registry by ref, builds the corpus,
   trains a fresh base model via the owner's :class:`BaseTrainer` (streaming
   per-step metrics to Hippius S3), and uploads the checkpoint to the registry.
4. Assembles a :class:`TrainingManifest`, signs it with the trainer hotkey, and
   (live) publishes it to the Hippius S3 manifest bucket for validators.

The pure planning + assembly logic is testable without GPUs, a chain, or
Hippius; the GPU / registry / S3 / chain calls are isolated in
:meth:`TrainerRunner.train_one`, :meth:`TrainerRunner.publish`, and the live
:meth:`TrainerRunner.run_forever`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..interface.validation import parse_commit
from ..shared.chain import Commitment
from ..shared.config import ChainConfig
from ..shared.hippius import (
    HubConfig,
    LogSink,
    S3Config,
    S3Store,
    fetch_from_hub,
    publish_manifest,
    upload_dir_to_hub,
)
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    dump_manifest,
    format_trained_pointer,
    sign_manifest,
)
from .contract import BaseTrainer, RoundSeeds
from .queue import QueuedSubmission, SubmissionQueue
from .queue import dumps as dump_queue
from .queue import loads as load_queue
from .stream import open_round_stream

log = logging.getLogger("metronome.trainer")


@dataclass(frozen=True)
class ResolvedGenerator:
    hotkey: str
    uid: int
    ref: str           # generator's Hippius Hub reference (repo@digest)


@dataclass(frozen=True)
class RoundPlan:
    king: ResolvedGenerator | None
    challengers: list[ResolvedGenerator]


def resolve_commitments(commitments: list[Commitment]) -> list[ResolvedGenerator]:
    """Parse each commitment's generator pointer, dropping malformed ones.

    A later commit from the same hotkey wins (miners re-deploy by committing a
    new ref), so we keep the highest ``commit_block`` per hotkey.
    """
    best: dict[str, tuple[int, ResolvedGenerator]] = {}
    for c in commitments:
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        rg = ResolvedGenerator(hotkey=c.hotkey, uid=c.uid, ref=parsed.ref)
        prev = best.get(c.hotkey)
        if prev is None or c.commit_block >= prev[0]:
            best[c.hotkey] = (c.commit_block, rg)
    return [rg for _, rg in best.values()]


def plan_round(
    resolved: list[ResolvedGenerator],
    king_hotkey: str | None,
) -> RoundPlan:
    """Split the field into the king and the challengers.

    ``king_hotkey`` is the reigning champion. When it is None or not present in
    the field (genesis, or the king deregistered), the lowest-UID resolved
    generator is promoted to interim king so there is always something to compare
    against. Challengers are returned in a stable order (by UID).

    Two cheap anti-duplicate filters run here, before any generator is fetched or
    trained (a round is ~3h of GPU per generator):

    * **duplicate-of-king** — a challenger whose generator ref equals the king's
      (same ``repo@digest``) is byte-identical to the king (the OCI digest is the
      content hash). It can only tie the king, never clear the win margin, so it
      is dropped rather than handed a wasted round. This is the metronome
      analogue of teutonic's ``check_model_copy`` "same repo + same digest →
      instant reject".
    * **same-ref dedup** — if two hotkeys committed the *same* generator ref,
      only the first (lowest UID) is kept; the others would be identical runs.
    """
    by_hotkey = {rg.hotkey: rg for rg in resolved}
    king = by_hotkey.get(king_hotkey) if king_hotkey else None
    field_ = sorted(resolved, key=lambda r: r.uid)
    if king is None:
        king = field_[0] if field_ else None
    king_ref = king.ref if king is not None else None

    challengers: list[ResolvedGenerator] = []
    seen_refs: set[str] = set()
    for rg in field_:
        if king is not None and rg.hotkey == king.hotkey:
            continue
        if king_ref is not None and rg.ref == king_ref:
            log.info("dropping challenger %s: generator ref is identical to the king", rg.hotkey)
            continue
        if rg.ref in seen_refs:
            log.info("dropping challenger %s: duplicate of an already-planned ref", rg.hotkey)
            continue
        seen_refs.add(rg.ref)
        challengers.append(rg)
    return RoundPlan(king=king, challengers=challengers)


@dataclass
class TrainerRunner:
    """Owner-operated trainer. ``base_trainer`` is the GPU backend (Protocol).

    Storage is Hippius: generators + checkpoints on the Hub registry (by
    ``repo@digest``), training logs + the manifest on S3.
    """

    cfg: ChainConfig
    base_trainer: BaseTrainer
    work_root: Path
    wallet: object | None = None       # bittensor wallet for signing (live)
    use_sandbox: bool = True           # run generators in the isolated subprocess
    # Remote (two-device) training: when ``remote_hosts`` is set, each round's
    # king and challenger train on separate SSH GPU pods in parallel (see
    # metronome.trainer.remote). ``trainer_spec`` is the BaseTrainer 'module:Class'
    # the pods run. None ⇒ local sequential training on this box.
    remote_hosts: list | None = None
    trainer_spec: str | None = None
    remote_timeout_seconds: int = 6 * 3600
    _hub: HubConfig | None = field(default=None, repr=False)
    _manifest_store: S3Store | None = field(default=None, repr=False)
    _logs_store: S3Store | None = field(default=None, repr=False)

    # ── storage handles (lazy so offline/tests need no Hippius) ──────────────

    def hub(self) -> HubConfig:
        if self._hub is None:
            self._hub = HubConfig.from_storage(self.cfg.storage)
        return self._hub

    def manifest_store(self) -> S3Store:
        if self._manifest_store is None:
            self._manifest_store = S3Store(
                S3Config.from_storage(self.cfg.storage, bucket=self.cfg.storage.manifest_bucket)
            )
        return self._manifest_store

    def logs_store(self) -> S3Store:
        if self._logs_store is None:
            self._logs_store = S3Store(
                S3Config.from_storage(self.cfg.storage, bucket=self.cfg.storage.logs_bucket)
            )
        return self._logs_store

    # ── per-generator train (GPU + registry + S3 boundary) ───────────────────

    def train_one(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        block: int,
    ) -> TrainedEntry:
        """Fetch generator (registry) → build corpus → train (logging to S3) →
        upload checkpoint (registry) → receipt for one generator.

        Raises on any failure; the caller decides whether a failed challenger
        simply doesn't qualify (it does) or a failed king aborts the round (it
        does — there's nothing to defend against).
        """
        gen_dir = self.work_root / f"{seeds.base_seed}" / role / "generator"
        fetch_from_hub(gen.ref, gen_dir, self.hub())

        out_dir = self.work_root / f"{seeds.base_seed}" / role / "checkpoint"
        out_dir.mkdir(parents=True, exist_ok=True)
        token_budget = self.cfg.training.train_tokens

        # Stream per-step metrics to S3 (best-effort: logging must never abort a
        # training run).
        sink: LogSink | None = None
        try:
            sink = LogSink(self.logs_store(), round_id=str(seeds.base_seed), role=role)
        except Exception as e:  # noqa: BLE001
            log.warning("log sink unavailable (continuing without S3 logs): %s", e)
        logger = sink.emit if sink is not None else None

        with open_round_stream(
            self.cfg.training.corpus_mode,
            gen_dir, seeds.generation_seed, self.cfg.generator,
            token_budget=token_budget,
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
        ) as rs:
            result = self.base_trainer.train(
                rs.series(),
                self.cfg.training,
                training_seed=seeds.training_seed,
                token_budget=token_budget,
                out_dir=out_dir,
                logger=logger,
            )
            corpus_digest, n_series, total_points = rs.digest, rs.n_series, rs.total_points

        if sink is not None:
            sink.emit({"event": "summary", "role": role, "corpus_digest": corpus_digest,
                       "n_series": n_series, "total_points": total_points,
                       "train_seconds": result.train_seconds, **result.metrics})
            try:
                sink.flush()
            except Exception as e:  # noqa: BLE001
                log.warning("failed to flush S3 training logs: %s", e)

        log.info(
            "round=%s role=%s hotkey=%s mode=%s n=%d points=%d digest=%s",
            seeds.base_seed, role, gen.hotkey, self.cfg.training.corpus_mode,
            n_series, total_points, corpus_digest[:12],
        )

        ckpt_repo = f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-{role}"
        up = upload_dir_to_hub(result.local_dir, ckpt_repo, self.hub())
        return TrainedEntry(
            miner_hotkey=gen.hotkey,
            miner_uid=gen.uid,
            role=role,
            gen_ref=gen.ref,
            trained_pointer=format_trained_pointer(up.ref.immutable_ref),
            corpus_digest=corpus_digest,
            train_block=block,
            gpu_name=str(result.metrics.get("gpu_name", "")),
        )

    def run_round(
        self,
        commitments: list[Commitment],
        king_hotkey: str | None,
        base_seed: int,
        block: int,
        *,
        max_challengers: int = 1,
        queue: SubmissionQueue | None = None,
    ) -> TrainingManifest:
        """Train the king and up to ``max_challengers`` challengers, returning
        the assembled (unsigned) manifest. Does not publish; see :meth:`publish`.

        Trains locally (sequential) by default, or across ``remote_hosts`` (king
        and challenger in parallel on separate GPU pods) when configured.

        When a ``queue`` is supplied, challengers are drawn from the persistent
        FIFO backlog (oldest-first, deduplicated against the king and against
        refs already trained this reign) instead of straight from this round's
        field; the queue is mutated in place (entries enqueued, selected ones
        marked trained), so the caller persists it after the round.
        """
        resolved = resolve_commitments(commitments)
        plan = plan_round(resolved, king_hotkey)
        if plan.king is None:
            raise RuntimeError("no resolvable generators on the netuid; nothing to train")

        challengers = self._select_challengers(plan, block, max_challengers, queue)

        seeds = RoundSeeds.derive(base_seed, self.cfg.training)
        jobs: list[tuple[ResolvedGenerator, str]] = [(plan.king, "king")]
        jobs += [(c, "challenger") for c in challengers]

        entries = (
            self._train_remote(jobs, seeds, block)
            if self.remote_hosts
            else self._train_local(jobs, seeds, block)
        )
        if queue is not None:
            # An attempt (win, loss, or a generator that failed to train) consumes
            # the challenger's shot for this reign so it is not re-run every round.
            for c in challengers:
                queue.mark_trained(c.ref)
        if not entries or entries[0].role != "king":
            raise RuntimeError("king training produced no entry; aborting round")

        return TrainingManifest(
            round_id=str(base_seed),
            created_block=block,
            contract_digest=contract_digest(self.cfg.training),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset,
            entries=entries,
        )

    def _select_challengers(
        self,
        plan: RoundPlan,
        block: int,
        max_challengers: int,
        queue: SubmissionQueue | None,
    ) -> list[ResolvedGenerator]:
        """Pick this round's challengers — straight off the planned field when
        there is no queue, otherwise from the FIFO backlog.

        ``plan.challengers`` has already had the duplicate-of-king and same-ref
        filters applied (:func:`plan_round`). With a queue, those survivors are
        the current on-chain field: the backlog is pruned to it (dropping
        re-deployed/deregistered refs), the field is enqueued (cheap dedup), and
        the front ``max_challengers`` still-eligible entries are returned.
        """
        if queue is None:
            return plan.challengers[:max_challengers]

        king_ref = plan.king.ref if plan.king is not None else None
        if queue.note_king(king_ref):
            log.info("new reign (king ref %s…); cleared per-reign trained cache", str(king_ref)[:24])

        by_ref = {c.ref: c for c in plan.challengers}
        for dropped in queue.prune_to_field(set(by_ref) | ({king_ref} if king_ref else set())):
            log.info("pruning queued %s: ref no longer in the on-chain field", dropped.hotkey)
        for c in plan.challengers:
            reason = queue.enqueue(QueuedSubmission(c.hotkey, c.uid, c.ref, block))
            if reason is not None:
                log.info("skip enqueue %s (ref %s…): %s", c.hotkey, c.ref[:24], reason)

        selected = queue.select(max_challengers)
        # Map the queued picks back to the resolved generators for this round.
        return [by_ref[s.ref] for s in selected if s.ref in by_ref]

    def _load_queue(self) -> SubmissionQueue:
        """Load the persistent submission backlog from ``[queue] state_db_path``
        (a fresh, empty queue when the file is absent or unreadable)."""
        path = Path(self.cfg.queue.state_db_path)
        try:
            return load_queue(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return SubmissionQueue(max_trained_cache=self.cfg.queue.trained_cache_size)
        except Exception as e:  # noqa: BLE001 — a corrupt backlog must not wedge the trainer
            log.warning("submission queue unreadable (%s); starting from empty", e)
            return SubmissionQueue(max_trained_cache=self.cfg.queue.trained_cache_size)

    def _save_queue(self, queue: SubmissionQueue) -> None:
        path = Path(self.cfg.queue.state_db_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(dump_queue(queue), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — persistence is best-effort
            log.warning("failed to persist submission queue to %s: %s", path, e)

    def _train_local(
        self, jobs: list[tuple[ResolvedGenerator, str]], seeds: RoundSeeds, block: int
    ) -> list[TrainedEntry]:
        """Sequential training on this box: king first (its failure aborts the
        round), then each challenger (a failure just drops that challenger)."""
        entries: list[TrainedEntry] = []
        for gen, role in jobs:
            try:
                entries.append(self.train_one(gen, role, seeds, block))
            except Exception as e:  # noqa: BLE001
                if role == "king":
                    raise
                log.warning("challenger %s failed to train: %s", gen.hotkey, e)
        return entries

    def _train_remote(
        self, jobs: list[tuple[ResolvedGenerator, str]], seeds: RoundSeeds, block: int
    ) -> list[TrainedEntry]:
        """Parallel training across ``remote_hosts`` (e.g. king→pod A, challenger→
        pod B over SSH). Equal compute is preserved (fixed token budget); audit is
        tolerance-based on rented hardware. King failure aborts the round; a
        challenger failure drops only that challenger."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .remote import RemoteDispatcher

        if not self.trainer_spec:
            raise RuntimeError("remote training requires trainer_spec (BaseTrainer 'module:Class')")
        hosts = self.remote_hosts
        disp = RemoteDispatcher(
            trainer_spec=self.trainer_spec, timeout_seconds=self.remote_timeout_seconds
        )

        def _run(i: int, gen: ResolvedGenerator, role: str) -> TrainedEntry:
            host = hosts[i % len(hosts)]
            return disp.dispatch(
                host, gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
                role=role, base_seed=seeds.base_seed, block=block,
            )

        results: list[TrainedEntry | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as ex:
            futs = {ex.submit(_run, i, gen, role): (i, gen, role)
                    for i, (gen, role) in enumerate(jobs)}
            for fut in as_completed(futs):
                i, gen, role = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # noqa: BLE001
                    if role == "king":
                        raise RuntimeError(f"king training failed on remote: {e}") from e
                    log.warning("challenger %s failed on remote: %s", gen.hotkey, e)
        return [r for r in results if r is not None]

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

    # ── live loop ────────────────────────────────────────────────────────────

    def run_forever(self, client: object, *, max_challengers: int = 1) -> None:  # pragma: no cover
        """Poll → train → publish, once per new round.

        A *round* is keyed by the chain block hash at the time the trainer wakes
        and finds a fresh king/field; the block hash is the shared base seed (so
        every honest party re-derives the same seeds). The reigning king is the
        highest-incentive UID on the metagraph (validators own the dethrone
        decision; the trainer just reads their weights).
        """
        poll = self.cfg.manifest.poll_seconds
        last_round: str | None = None
        queue = self._load_queue()
        log.info("submission queue loaded: %d pending, %d trained this reign",
                 len(queue.pending), len(queue.trained_refs))
        while True:
            try:
                block = client.current_block()
                base_seed = client.block_seed(block)
                round_id = str(base_seed)
                if round_id == last_round:
                    time.sleep(poll)
                    continue
                commitments = client.poll_commitments()
                king_hotkey = client.highest_incentive_hotkey()
                log.info("starting round=%s block=%d king=%s field=%d queued=%d",
                         round_id, block, king_hotkey, len(commitments), len(queue.pending))
                manifest = self.run_round(
                    commitments, king_hotkey, base_seed, block,
                    max_challengers=max_challengers, queue=queue,
                )
                self._save_queue(queue)
                self.publish(manifest)
                last_round = round_id
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round failed; retrying after poll interval: %s", e)
            time.sleep(poll)
