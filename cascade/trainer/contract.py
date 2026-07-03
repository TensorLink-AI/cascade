"""The fixed training contract and the pluggable base-trainer protocol.

cascade's central invariant: in a given round, the king's generator and the
challenger's generator are trained into models under *byte-identical* terms —
same base architecture, same epochs/batch/lr, same generation seed, same
training seed. The ONLY difference is the generator code that produced the
corpus. That is what turns the downstream eval into a controlled measurement of
data quality rather than a confound of data + luck + hyperparameters.

Seeds are shared across king and challenger on purpose:

* ``generation_seed`` — passed to each generator's ``__init__``. Same value for
  both so neither draws a "luckier" data seed; each generator is still
  deterministic in it.
* ``training_seed`` — model init + data-order RNG inside the base trainer. Same
  value for both so weight initialisation and shuffling are identical.

Both derive deterministically from the round's base seed (the chain block hash
at round start), so a second honest trainer reproduces the exact run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from ..shared.config import TrainingContractConfig

# A training logger: the trainer loop passes one in so a :class:`BaseTrainer`
# can stream per-step metrics (loss, lr, throughput, tokens) out to Hippius S3
# for observability. ``None`` means "don't log" (offline / tests).
TrainLogger = Callable[[dict], None]

# The architecture fields folded into base_arch_digest. These pin the *shape* of
# the frozen base model; together with the model source they fully determine the
# from-scratch architecture + init that king and challenger must share.
_ARCH_FIELDS = (
    "base_arch", "arch_preset", "d_model", "num_layers", "num_heads", "head_dim",
    "patch_size", "mlp_expansion", "d_ff", "num_quantiles", "masking", "cpm_c_max",
    "cpm_p_max", "input_transform", "context_length", "horizon",
)


def compute_base_arch_digest(contract: TrainingContractConfig) -> str:
    """Deterministic sha256 of the frozen base architecture + init code.

    Hashes the architecture fields from the contract *and* the bytes of the
    reference model source (``toto2_model.py``), so the digest changes if either
    the integers or the model definition change. The operator computes this once
    (``cascade-trainer --offline`` prints it) and pins the result in
    ``chain.toml [training] base_arch_digest``; the validator's controlled-
    experiment gate then asserts every manifest was trained under that exact arch.

    Reads the model source as bytes (no torch import), so it runs in any
    environment.
    """
    arch = {k: getattr(contract, k) for k in _ARCH_FIELDS}
    model_src = (Path(__file__).with_name("toto2_model.py")).read_bytes()
    h = hashlib.sha256()
    h.update(json.dumps(arch, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(b"\x00arch_src\x00")
    h.update(model_src)
    return h.hexdigest()


def _mix(base_seed: int, tag: str, salt: int = 0) -> int:
    """Deterministically mix a base seed, a string tag, and an int salt into a
    64-bit seed. Stable across platforms (blake2b)."""
    h = hashlib.blake2b(
        f"{base_seed}:{tag}:{salt}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big", signed=False)


@dataclass(frozen=True)
class RoundSeeds:
    """The two seeds for a round, shared identically across king and challenger."""

    base_seed: int
    generation_seed: int
    training_seed: int

    @classmethod
    def derive(cls, base_seed: int, contract: TrainingContractConfig) -> RoundSeeds:
        return cls(
            base_seed=base_seed,
            generation_seed=_mix(base_seed, "generation"),
            training_seed=_mix(base_seed, "training", contract.train_seed_salt),
        )


@dataclass(frozen=True)
class TrainResult:
    """What a :class:`BaseTrainer` returns after training one model.

    ``metrics`` is a small summary of the run (e.g. ``final_loss``, ``steps``,
    ``tokens_seen``, ``throughput_tokens_per_s``) — recorded into the training
    log alongside the per-step records the trainer streamed via the
    :data:`TrainLogger`. Keep it JSON-serialisable.
    """

    local_dir: Path        # directory holding the trained checkpoint to upload
    param_count: int
    train_seconds: float
    metrics: dict = field(default_factory=dict)


class BaseTrainer(Protocol):
    """Owner-supplied training backend.

    Implementations train a fresh copy of the base architecture by pulling
    series from ``stream`` and write a complete, uploadable checkpoint into
    ``out_dir``. ``stream`` yields canonical ``(C, L)`` float64 series and is
    already budget-capped by the caller — iterate it to exhaustion. Use
    ``token_budget`` (the contract's ``train_tokens``) to shape the LR schedule
    (warmup/decay) so it lands as the stream ends.

    The stream's nature follows ``contract.corpus_mode`` but the trainer need not
    care: ``cache_reuse`` cycles a fixed corpus (data repeats), ``stream_cpu``
    delivers fresh series with no reuse. Either way the trainer just trains on
    what it pulls.

    The same instance is reused for king and challenger within a round, so it
    MUST be stateless across calls beyond the (immutable) base architecture — any
    leakage from king's run into challenger's would break the controlled
    comparison.

    This Protocol is the GPU boundary: everything above it in cascade is
    numpy/CPU and unit-testable; the concrete trainer is the operator's to
    provide. See ``docs/ARCHITECTURE.md`` for the reference shape.
    """

    def train(
        self,
        stream: Iterator[np.ndarray],
        contract: TrainingContractConfig,
        *,
        training_seed: int,
        token_budget: int,
        out_dir: Path,
        logger: TrainLogger | None = None,
    ) -> TrainResult:
        ...
