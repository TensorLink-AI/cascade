"""Validator: read the manifest, evaluate king vs challenger, decide the throne,
set weights. Never trains."""

from __future__ import annotations

from .loop import RoundOutcome, ValidatorRunner, build_runner
from .state import ChampionState, StateTransition, apply_round, genesis

__all__ = [
    "RoundOutcome",
    "ValidatorRunner",
    "build_runner",
    "ChampionState",
    "StateTransition",
    "apply_round",
    "genesis",
]
