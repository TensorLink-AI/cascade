"""Trainer service loop — the owner-operated training round.

Each round the trainer:

1. Resolves on-chain generator commitments to ``(hotkey, uid, cid)``.
2. Identifies the reigning king (the highest-incentive UID on the metagraph in
   live mode; a caller-supplied hotkey offline) and selects challengers.
3. For the king and each challenger, under one shared :class:`RoundSeeds`:
   fetches the generator from the Hippius registry by CID, builds the corpus,
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
    LogSink,
    RegistryConfig,
    S3Config,
    S3Store,
    fetch_from_registry,
    publish_manifest,
    upload_dir_to_registry,
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
from .stream import open_round_stream

log = logging.getLogger("metronome.trainer")


@dataclass(frozen=True)
class ResolvedGenerator:
    hotkey: str
    uid: int
    cid: str           # generator's Hippius registry CID


@dataclass(frozen=True)
class RoundPlan:
    king: ResolvedGenerator | None
    challengers: list[ResolvedGenerator]


def resolve_commitments(commitments: list[Commitment]) -> list[ResolvedGenerator]:
    """Parse each commitment's generator pointer, dropping malformed ones.

    A later commit from the same hotkey wins (miners re-deploy by committing a
    new CID), so we keep the highest ``commit_block`` per hotkey.
    """
    best: dict[str, tuple[int, ResolvedGenerator]] = {}
    for c in commitments:
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        rg = ResolvedGenerator(hotkey=c.hotkey, uid=c.uid, cid=parsed.cid)
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
    """
    by_hotkey = {rg.hotkey: rg for rg in resolved}
    king = by_hotkey.get(king_hotkey) if king_hotkey else None
    field_ = sorted(resolved, key=lambda r: r.uid)
    if king is None:
        king = field_[0] if field_ else None
    challengers = [rg for rg in field_ if king is None or rg.hotkey != king.hotkey]
    return RoundPlan(king=king, challengers=challengers)


@dataclass
class TrainerRunner:
    """Owner-operated trainer. ``base_trainer`` is the GPU backend (Protocol).

    Storage is Hippius: generators + checkpoints on the registry (IPFS, by CID),
    training logs + the manifest on S3.
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
    _registry: RegistryConfig | None = field(default=None, repr=False)
    _manifest_store: S3Store | None = field(default=None, repr=False)
    _logs_store: S3Store | None = field(default=None, repr=False)

    # ── storage handles (lazy so offline/tests need no Hippius) ──────────────

    def registry(self) -> RegistryConfig:
        if self._registry is None:
            self._registry = RegistryConfig.from_storage(self.cfg.storage)
        return self._registry

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
        fetch_from_registry(gen.cid, gen_dir, self.registry())

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

        up = upload_dir_to_registry(result.local_dir, self.registry())
        return TrainedEntry(
            miner_hotkey=gen.hotkey,
            miner_uid=gen.uid,
            role=role,
            gen_cid=gen.cid,
            trained_pointer=format_trained_pointer(up.cid),
            corpus_digest=corpus_digest,
            train_block=block,
            tar_digest=up.tar_digest,
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
    ) -> TrainingManifest:
        """Train the king and up to ``max_challengers`` challengers, returning
        the assembled (unsigned) manifest. Does not publish; see :meth:`publish`.

        Trains locally (sequential) by default, or across ``remote_hosts`` (king
        and challenger in parallel on separate GPU pods) when configured.
        """
        resolved = resolve_commitments(commitments)
        plan = plan_round(resolved, king_hotkey)
        if plan.king is None:
            raise RuntimeError("no resolvable generators on the netuid; nothing to train")

        seeds = RoundSeeds.derive(base_seed, self.cfg.training)
        jobs: list[tuple[ResolvedGenerator, str]] = [(plan.king, "king")]
        jobs += [(c, "challenger") for c in plan.challengers[:max_challengers]]

        entries = (
            self._train_remote(jobs, seeds, block)
            if self.remote_hosts
            else self._train_local(jobs, seeds, block)
        )
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
                host, gen_cid=gen.cid, uid=gen.uid, hotkey=gen.hotkey,
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
                log.info("starting round=%s block=%d king=%s field=%d",
                         round_id, block, king_hotkey, len(commitments))
                manifest = self.run_round(
                    commitments, king_hotkey, base_seed, block, max_challengers=max_challengers
                )
                self.publish(manifest)
                last_round = round_id
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round failed; retrying after poll interval: %s", e)
            time.sleep(poll)
