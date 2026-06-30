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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..interface.validation import parse_commit
from ..shared.chain import Commitment
from ..shared.config import ChainConfig, TrainingContractConfig
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
from .contract import BaseTrainer, RoundSeeds, TrainResult
from .stream import open_round_stream

# Screens one heat checkpoint: given the trained heat-model directory, the
# generator that produced its corpus, and the round's base seed (so the screening
# window slice can rotate per round), return a heat score (LOWER is better, e.g.
# geomean(CRPS, MASE) on the held-out windows). Injected so the trainer's
# screening stays a testable boundary — the default wiring (torch evaluator +
# eval pool) is attached in cascade.trainer.main.
ScreenFn = Callable[[Path, "ResolvedGenerator", int], float]

log = logging.getLogger("cascade.trainer")


@dataclass(frozen=True)
class ResolvedGenerator:
    hotkey: str
    uid: int
    ref: str           # generator's Hippius Hub reference (repo@digest)


@dataclass(frozen=True)
class RoundPlan:
    king: ResolvedGenerator | None
    challengers: list[ResolvedGenerator]


def resolve_commitments(
    commitments: list[Commitment], cutoff_block: int | None = None
) -> list[ResolvedGenerator]:
    """Parse each commitment's generator pointer, dropping malformed ones.

    A later commit from the same hotkey wins (miners re-deploy by committing a
    new ref), so we keep the highest ``commit_block`` per hotkey.

    When ``cutoff_block`` is given (the round's epoch boundary), only commits
    revealed STRICTLY BEFORE it are eligible — this is the daily submission
    deadline. A miner who commits at or after the boundary competes in the next
    round, not this one, and because the boundary is deterministic every honest
    party re-derives the identical field. The latest-commit-wins rule applies only
    among a hotkey's eligible (pre-cutoff) commits.
    """
    best: dict[str, tuple[int, ResolvedGenerator]] = {}
    for c in commitments:
        if cutoff_block is not None and c.commit_block >= cutoff_block:
            continue
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
      is dropped rather than handed a wasted round. This is the cascade
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
    # Heat screener: scores a trained heat checkpoint (lower better) to rank the
    # field down to [round] finalists before the expensive final. None ⇒ no
    # internal screen (the field's natural order is taken). Wired in trainer.main.
    screen_fn: ScreenFn | None = None
    # Remote (two-device) training: when ``remote_hosts`` is set, each round's
    # king and challenger train on separate SSH GPU pods in parallel (see
    # cascade.trainer.remote). ``trainer_spec`` is the BaseTrainer 'module:Class'
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

    def _train_checkpoint(
        self,
        gen: ResolvedGenerator,
        seeds: RoundSeeds,
        contract: TrainingContractConfig,
        token_budget: int,
        out_dir: Path,
        *,
        log_role: str,
    ) -> tuple[TrainResult, str, int, int]:
        """Fetch generator (registry) → build corpus → train into ``out_dir``,
        streaming per-step metrics to S3. No upload — the caller decides whether
        the checkpoint is uploaded (final) or thrown away after screening (heat).

        ``contract`` is the per-size training contract (the base recipe with this
        size's width/depth/digest/throughput); ``token_budget`` is its compute
        budget for this stage. Returns ``(result, corpus_digest, n_series,
        total_points)``. Raises on any failure.
        """
        gen_dir = out_dir.parent / "generator"
        fetch_from_hub(gen.ref, gen_dir, self.hub())
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
        logger = sink.emit if sink is not None else None

        with open_round_stream(
            contract.corpus_mode,
            gen_dir, seeds.generation_seed, self.cfg.generator,
            token_budget=token_budget,
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
        ) as rs:
            result = self.base_trainer.train(
                rs.series(),
                contract,
                training_seed=seeds.training_seed,
                token_budget=token_budget,
                out_dir=out_dir,
                logger=logger,
            )
            corpus_digest, n_series, total_points = rs.digest, rs.n_series, rs.total_points

        if sink is not None:
            sink.emit({"event": "summary", "role": log_role, "corpus_digest": corpus_digest,
                       "n_series": n_series, "total_points": total_points,
                       "train_seconds": result.train_seconds, **result.metrics})
            try:
                sink.flush()
            except Exception as e:  # noqa: BLE001
                log.warning("failed to flush S3 training logs: %s", e)

        log.info(
            "round=%s run=%s hotkey=%s mode=%s n=%d points=%d digest=%s",
            seeds.base_seed, log_role, gen.hotkey, contract.corpus_mode,
            n_series, total_points, corpus_digest[:12],
        )
        return result, corpus_digest, n_series, total_points

    def train_one(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        block: int,
        *,
        contract: TrainingContractConfig | None = None,
        token_budget: int | None = None,
    ) -> TrainedEntry:
        """Train one generator at one size for the FINAL stage, upload its
        checkpoint, and return the receipt.

        ``contract`` defaults to the primary (smallest) size; ``token_budget`` to
        that size's full ``train_tokens``. The checkpoint is uploaded to a
        size-tagged registry repo (``ckpt-r<seed>-<role>-<size>``) and the entry
        carries the ``size`` tag so the validator can pair king and challenger
        per size before combining their scores.

        Raises on any failure; the caller decides whether a failed challenger
        simply doesn't qualify (it does) or a failed king aborts the round (it
        does — there's nothing to defend against).
        """
        contract = contract if contract is not None else self.cfg.training.primary_size
        token_budget = token_budget if token_budget is not None else contract.train_tokens
        size = contract.arch_preset
        out_dir = self.work_root / f"{seeds.base_seed}" / size / role / "checkpoint"
        result, corpus_digest, _, _ = self._train_checkpoint(
            gen, seeds, contract, token_budget, out_dir, log_role=f"{role}-{size}",
        )

        ckpt_repo = f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-{role}-{size}"
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
            size=size,
        )

    def run_round(
        self,
        commitments: list[Commitment],
        king_hotkey: str | None,
        base_seed: int,
        block: int,
        *,
        cutoff_block: int | None = None,
    ) -> TrainingManifest:
        """Run one daily round and return the assembled (unsigned) manifest.

        Two stages, both under one shared :class:`RoundSeeds` (identical random
        init for the whole round):

        1. **Heat** — every eligible challenger is trained cheaply
           (``[round] heat_train_hours`` on the primary size) and screened; the
           top ``[round] finalists`` advance.
        2. **Final** — the king and the surviving finalists are trained to the
           full ``[training] target_train_hours`` at EVERY configured size
           (primary + ``[[training.sizes]]``). Each (king, challenger) pair is
           tagged with its size so the validator can combine scores across sizes
           into one throne.

        ``cutoff_block`` (the epoch boundary) is the submission deadline: only
        commitments revealed before it are eligible (see
        :func:`resolve_commitments`). Does not publish; see :meth:`publish`.
        Trains locally (sequential) by default, or across ``remote_hosts`` when
        configured. A king failure at any size aborts the round.
        """
        resolved = resolve_commitments(commitments, cutoff_block=cutoff_block)
        plan = plan_round(resolved, king_hotkey)
        if plan.king is None:
            raise RuntimeError("no resolvable generators on the netuid; nothing to train")

        seeds = RoundSeeds.derive(base_seed, self.cfg.training)

        finalists = self._run_heat(plan.challengers, seeds, block)
        jobs: list[tuple[ResolvedGenerator, str]] = [(plan.king, "king")]
        jobs += [(c, "challenger") for c in finalists]

        entries = self._train_final(jobs, seeds, block)
        if not any(e.role == "king" for e in entries):
            raise RuntimeError("king training produced no entry; aborting round")

        return TrainingManifest(
            round_id=str(base_seed),
            created_block=block,
            contract_digest=contract_digest(self.cfg.training),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset,
            entries=entries,
        )

    def _run_heat(
        self, challengers: list[ResolvedGenerator], seeds: RoundSeeds, block: int
    ) -> list[ResolvedGenerator]:
        """Screen the field down to ``[round] finalists`` for the final stage.

        Each challenger is trained for ``[round] heat_train_hours`` on the primary
        (smallest) size and scored by the injected ``screen_fn`` (lower is
        better); the cheapest ``finalists`` advance, UID breaking ties for
        determinism. When the field already fits within ``finalists``, or no
        ``screen_fn`` is wired, the field's natural order (lowest UID first) is
        taken without spending heat compute. A challenger that fails to train or
        screen is dropped (it simply doesn't qualify).
        """
        n = max(0, self.cfg.round.finalists)
        if not challengers or n == 0:
            return []
        if self.screen_fn is None or len(challengers) <= n:
            if self.screen_fn is None and len(challengers) > n:
                log.warning("no screen_fn wired; taking %d of %d challengers by UID order",
                            n, len(challengers))
            return list(challengers[:n])

        heat_contract = self.cfg.screen_contract()
        heat_tokens = heat_contract.tokens_for_hours(self.cfg.round.heat_train_hours)
        scored: list[tuple[float, int, ResolvedGenerator]] = []
        for c in challengers:
            out_dir = self.work_root / f"{seeds.base_seed}" / "heat" / c.hotkey / "checkpoint"
            try:
                result, *_ = self._train_checkpoint(
                    c, seeds, heat_contract, heat_tokens, out_dir, log_role=f"heat-{c.hotkey}",
                )
                score = float(self.screen_fn(result.local_dir, c, seeds.base_seed))
            except Exception as e:  # noqa: BLE001 — a broken heat entry just doesn't qualify
                log.warning("heat: challenger %s failed to screen: %s", c.hotkey, e)
                continue
            log.info("heat: challenger %s score=%.5f", c.hotkey, score)
            scored.append((score, c.uid, c))

        scored.sort(key=lambda t: (t[0], t[1]))  # lower score better; UID tiebreak
        winners = [c for _, _, c in scored[:n]]
        log.info("heat: %d/%d advance to the final: %s",
                 len(winners), len(challengers), [c.hotkey for c in winners])
        return winners

    def _train_final(
        self, jobs: list[tuple[ResolvedGenerator, str]], seeds: RoundSeeds, block: int
    ) -> list[TrainedEntry]:
        """Train the final jobs at each throne size, returning all receipts.

        One (king + finalists) pass per size in ``cfg.throne_contracts()`` (the
        ``[round] throne_sizes``); a king failure at any size aborts the round, a
        challenger failure drops only that challenger from that size."""
        entries: list[TrainedEntry] = []
        for contract in self.cfg.throne_contracts():
            token_budget = contract.train_tokens
            if self.remote_hosts:
                entries += self._train_remote(jobs, seeds, block, contract, token_budget)
            else:
                entries += self._train_local(jobs, seeds, block, contract, token_budget)
        return entries

    def _train_local(
        self,
        jobs: list[tuple[ResolvedGenerator, str]],
        seeds: RoundSeeds,
        block: int,
        contract: TrainingContractConfig,
        token_budget: int,
    ) -> list[TrainedEntry]:
        """Sequential training on this box for one size: king first (its failure
        aborts the round), then each challenger (a failure just drops it)."""
        entries: list[TrainedEntry] = []
        for gen, role in jobs:
            try:
                entries.append(
                    self.train_one(gen, role, seeds, block,
                                   contract=contract, token_budget=token_budget)
                )
            except Exception as e:  # noqa: BLE001
                if role == "king":
                    raise
                log.warning("challenger %s failed to train (%s): %s",
                            gen.hotkey, contract.arch_preset, e)
        return entries

    def _train_remote(
        self,
        jobs: list[tuple[ResolvedGenerator, str]],
        seeds: RoundSeeds,
        block: int,
        contract: TrainingContractConfig,
        token_budget: int,  # noqa: ARG002 — budget travels via chain.toml on the pod
    ) -> list[TrainedEntry]:
        """Parallel training across ``remote_hosts`` for one size (king→pod A,
        challenger→pod B over SSH). Equal compute is preserved (fixed token
        budget); audit is tolerance-based on rented hardware. King failure aborts
        the round; a challenger failure drops only that challenger."""
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
                arch_preset=contract.arch_preset,
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
                    log.warning("challenger %s failed on remote (%s): %s",
                                gen.hotkey, contract.arch_preset, e)
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

    def run_forever(self, client: object) -> None:  # pragma: no cover
        """Poll → train → publish, once per daily round (epoch).

        A *round* is one ``[round] epoch_blocks`` window (~24h). It is keyed by
        the chain block hash at the EPOCH BOUNDARY (``epoch_start = block //
        epoch_blocks × epoch_blocks``), which is the shared base seed — so the
        whole day's heat and final trainings share one :class:`RoundSeeds`, and
        every honest party re-derives the same seeds and the same eligible field.
        The reigning king is the highest-incentive UID on the metagraph
        (validators own the dethrone decision; the trainer just reads weights).
        """
        poll = self.cfg.manifest.poll_seconds
        epoch_blocks = max(1, self.cfg.round.epoch_blocks)
        last_round: str | None = None
        while True:
            try:
                block = client.current_block()
                epoch = block // epoch_blocks
                epoch_start = epoch * epoch_blocks
                base_seed = client.block_seed(epoch_start)
                round_id = str(base_seed)
                if round_id == last_round:
                    time.sleep(poll)
                    continue
                commitments = client.poll_commitments()
                king_hotkey = client.highest_incentive_hotkey()
                log.info("starting round=%s epoch=%d epoch_start=%d king=%s field=%d",
                         round_id, epoch, epoch_start, king_hotkey, len(commitments))
                manifest = self.run_round(
                    commitments, king_hotkey, base_seed, block, cutoff_block=epoch_start,
                )
                self.publish(manifest)
                last_round = round_id
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round failed; retrying after poll interval: %s", e)
            time.sleep(poll)
