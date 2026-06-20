"""Trainer service loop — the owner-operated training round.

Each round the trainer:

1. Resolves on-chain generator commitments to ``(hotkey, uid, repo, revision)``.
2. Identifies the reigning king (the highest-incentive UID on the metagraph in
   live mode; a caller-supplied hotkey offline) and selects challengers.
3. For the king and each challenger, under one shared :class:`RoundSeeds`:
   builds the corpus from the generator, trains a fresh base model via the
   owner's :class:`BaseTrainer`, and uploads the checkpoint to HF.
4. Assembles a :class:`TrainingManifest` and (live) publishes it to the
   owner-controlled dataset repo for validators to read.

The pure planning + assembly logic is testable without GPUs or a chain; the
GPU/HF/chain calls are isolated in :meth:`TrainerRunner.train_one` and
:meth:`TrainerRunner.publish`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..interface.validation import parse_commit
from ..shared.chain import Commitment
from ..shared.config import ChainConfig
from ..shared.hf import upload_folder
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
)
from .contract import BaseTrainer, RoundSeeds
from .corpus import build_round_corpus

log = logging.getLogger("metronome.trainer")


@dataclass(frozen=True)
class ResolvedGenerator:
    hotkey: str
    uid: int
    repo: str
    revision: str


@dataclass(frozen=True)
class RoundPlan:
    king: ResolvedGenerator | None
    challengers: list[ResolvedGenerator]


def resolve_commitments(commitments: list[Commitment]) -> list[ResolvedGenerator]:
    """Parse each commitment's generator pointer, dropping malformed ones.

    A later commit from the same hotkey wins (miners re-deploy by committing a
    new revision), so we keep the highest ``commit_block`` per hotkey.
    """
    best: dict[str, tuple[int, ResolvedGenerator]] = {}
    for c in commitments:
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        rg = ResolvedGenerator(
            hotkey=c.hotkey, uid=c.uid, repo=parsed.repo, revision=parsed.revision
        )
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
    the field (genesis, or the king deregistered), the first resolved generator
    is promoted to interim king so there is always something to compare against.
    Challengers are returned in a stable order (by UID).
    """
    by_hotkey = {rg.hotkey: rg for rg in resolved}
    king = by_hotkey.get(king_hotkey) if king_hotkey else None
    field = sorted(resolved, key=lambda r: r.uid)
    if king is None:
        king = field[0] if field else None
    challengers = [rg for rg in field if king is None or rg.hotkey != king.hotkey]
    return RoundPlan(king=king, challengers=challengers)


@dataclass
class TrainerRunner:
    """Owner-operated trainer. ``base_trainer`` is the GPU backend (Protocol)."""

    cfg: ChainConfig
    base_trainer: BaseTrainer
    work_root: Path
    trained_repo_prefix: str           # e.g. "tensorlink-ai/metro-trained"
    hf_token: str | None = None
    hf_cache_dir: Path | None = None
    use_sandbox: bool = True           # run generators in the isolated subprocess

    def train_one(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        block: int,
    ) -> TrainedEntry:
        """Build corpus → train → upload → receipt for one generator.

        GPU + HF boundary. Raises on any failure; the caller decides whether a
        failed challenger simply doesn't qualify (it does) or a failed king
        aborts the round (it does — there's nothing to defend against).
        """
        from ..shared.hf import fetch_revision

        fetched = fetch_revision(
            gen.repo, gen.revision, cache_dir=self.hf_cache_dir, token=self.hf_token
        )
        corpus = build_round_corpus(
            fetched.local_dir, seeds.generation_seed, self.cfg.generator,
            self.cfg.training.corpus_mode,
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
        )
        log.info(
            "round=%s role=%s hotkey=%s corpus n=%d points=%d digest=%s",
            seeds.base_seed, role, gen.hotkey, corpus.n_series, corpus.total_points,
            corpus.digest[:12],
        )

        out_dir = self.work_root / f"{seeds.base_seed}" / role
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self.base_trainer.train(
            corpus.series,
            self.cfg.training,
            training_seed=seeds.training_seed,
            out_dir=out_dir,
        )
        repo = f"{self.trained_repo_prefix}-{role}-{seeds.base_seed}"
        commit_sha = upload_folder(
            result.local_dir, repo, token=self.hf_token,
            commit_message=f"metronome round {seeds.base_seed} {role} ({gen.hotkey})",
        )
        return TrainedEntry(
            miner_hotkey=gen.hotkey,
            miner_uid=gen.uid,
            role=role,
            gen_repo=gen.repo,
            gen_revision=gen.revision,
            trained_pointer=format_trained_pointer(repo, commit_sha),
            corpus_digest=corpus.digest,
            train_block=block,
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
        the assembled (unsigned) manifest. Does not publish; see
        :meth:`publish`."""
        resolved = resolve_commitments(commitments)
        plan = plan_round(resolved, king_hotkey)
        if plan.king is None:
            raise RuntimeError("no resolvable generators on the netuid; nothing to train")

        seeds = RoundSeeds.derive(base_seed, self.cfg.training)
        entries: list[TrainedEntry] = [self.train_one(plan.king, "king", seeds, block)]
        for chal in plan.challengers[:max_challengers]:
            try:
                entries.append(self.train_one(chal, "challenger", seeds, block))
            except Exception as e:  # noqa: BLE001
                log.warning("challenger %s failed to train: %s", chal.hotkey, e)

        return TrainingManifest(
            round_id=str(base_seed),
            created_block=block,
            contract_digest=contract_digest(self.cfg.training),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset,
            entries=entries,
        )

    def publish(self, manifest: TrainingManifest) -> None:
        """TODO: write ``manifest.json`` to the owner HF dataset repo and sign
        it with the trainer hotkey. Offline this is a no-op log line."""
        log.info(
            "publish manifest round=%s entries=%d (TODO: push to %s, sign with %s)",
            manifest.round_id, len(manifest.entries),
            self.cfg.manifest.hf_dataset_repo, self.cfg.manifest.trainer_hotkey,
        )
