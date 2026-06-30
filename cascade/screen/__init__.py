"""LLM-as-a-judge submission screen.

Runs after the deterministic checks (static guard + determinism) to catch what a
parser can't: distilled weights, benchmark-targeting, and restructured copies of
the king. The deterministic mechanics (literal-density evidence, source
similarity) are pure and offline; only the adjudication touches the network, via
an injectable :class:`~cascade.screen.judge.JudgeClient`.
"""

from __future__ import annotations

from .canonicalize import canonical_source, source_similarity
from .judge import (
    JudgeClient,
    JudgeError,
    JudgeVerdict,
    OpenRouterJudge,
    build_judge,
    judge_copy_of_king,
    judge_static,
)
from .screen import INTERFACE_RULE, CheckResult, ScreenReport, screen_repo
from .signals import NumericEvidence, numeric_evidence

__all__ = [
    "INTERFACE_RULE",
    "CheckResult",
    "JudgeClient",
    "JudgeError",
    "JudgeVerdict",
    "NumericEvidence",
    "OpenRouterJudge",
    "ScreenReport",
    "build_judge",
    "canonical_source",
    "judge_copy_of_king",
    "judge_static",
    "numeric_evidence",
    "screen_repo",
    "source_similarity",
]
