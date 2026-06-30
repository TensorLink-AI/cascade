"""Owner-operated trainer: the GPU boundary.

Builds each generator's corpus, trains a fresh base model under the fixed
contract, uploads checkpoints, and publishes the training manifest.
"""

from __future__ import annotations

from .contract import BaseTrainer, RoundSeeds, TrainResult
from .corpus import CorpusError, CorpusResult, assert_corpus_reproducible, build_corpus
from .loop import ResolvedGenerator, RoundPlan, TrainerRunner, plan_round, resolve_commitments

__all__ = [
    "BaseTrainer",
    "RoundSeeds",
    "TrainResult",
    "CorpusError",
    "CorpusResult",
    "assert_corpus_reproducible",
    "build_corpus",
    "ResolvedGenerator",
    "RoundPlan",
    "TrainerRunner",
    "plan_round",
    "resolve_commitments",
]
