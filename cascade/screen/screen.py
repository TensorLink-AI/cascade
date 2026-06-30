"""Orchestrates the LLM-judge submission screen — the step that runs after the
deterministic checks (static guard + determinism) in ``cascade verify`` and,
trainer-side, before the heat.

Three checks, in cost order:

1. **distillation** (hard-fail) — mechanical literal-density evidence +
   one LLM call asking prior-vs-fitted-weights.
2. **benchmark-targeting** (hard-fail if blatant, warn otherwise) — folded into
   the *same* LLM call.
3. **copy-of-king** (mechanical reject above a high similarity; LLM only for the
   middle band) — skipped entirely unless the king's source is published.

Returns a :class:`ScreenReport`; the caller (``verify_repo`` / the trainer) maps
``fail`` to a hard rejection and ``warn`` to a recorded warning. A judge transport
error is resolved by ``[judge] fail_closed`` — reject (default, safe for a
security screen) or pass with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .canonicalize import source_similarity
from .judge import JudgeClient, JudgeError, JudgeVerdict, judge_copy_of_king, judge_static
from .signals import numeric_evidence

# The code-only rule handed to the distillation judge verbatim. Mirrors
# docs/INTERFACE.md and cascade.interface.validation.
INTERFACE_RULE = (
    "A submission is a PURELY ALGORITHMIC data generator: it samples synthetic "
    "time-series from a prior (GP/kernel families, causal DAGs, parametric "
    "trend/seasonality/noise). It is code-only — NO shipped or embedded learned "
    "weights of any kind. torch/gpytorch are allowed only as compute libraries "
    "for GP/kernel priors, never to load or replay a pretrained forecaster."
)

# Files fed to the static judge (the whole submission contract is tiny).
_JUDGE_FILES = ("generator.py", "config.json", "requirements.txt")


@dataclass(frozen=True)
class CheckResult:
    name: str          # "distillation" | "benchmark_targeting" | "copy_of_king"
    verdict: str       # "pass" | "warn" | "fail"
    reason: str
    evidence: dict | None = None

    @property
    def failed(self) -> bool:
        return self.verdict == "fail"

    @property
    def warned(self) -> bool:
        return self.verdict == "warn"


@dataclass(frozen=True)
class ScreenReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.failed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.failed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.warned]


def _read_files(repo_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in _JUDGE_FILES:
        p = repo_dir / name
        if p.is_file():
            try:
                files[name] = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                files[name] = "<unreadable>"
    return files


def _on_error(names: list[str], err: Exception, *, fail_closed: bool) -> list[CheckResult]:
    verdict = "fail" if fail_closed else "warn"
    reason = f"judge unavailable ({err}); {'rejecting (fail-closed)' if fail_closed else 'passing with warning (fail-open)'}"
    return [CheckResult(n, verdict, reason) for n in names]


def screen_repo(
    repo_dir: Path | str,
    cfg,
    judge: JudgeClient,
    *,
    king_source: str | None = None,
) -> ScreenReport:
    """Run the LLM-judge screen on a materialised generator repo.

    ``judge`` is the (required) LLM client; build it with
    :func:`cascade.screen.judge.build_judge` and skip calling this when it is
    ``None``. ``king_source`` enables the copy-of-king check (only when the king
    is published — gated by ``[judge] publish_king``).
    """
    d = Path(repo_dir)
    jc = cfg.judge
    fail_closed = jc.fail_closed
    checks: list[CheckResult] = []

    # 1+2. distillation + benchmark-targeting (one LLM call) with mechanical evidence.
    evidence = numeric_evidence(d)
    files = _read_files(d)
    try:
        distill, bench = judge_static(
            files, evidence.as_dict(), judge, interface_rule=INTERFACE_RULE
        )
        checks.append(CheckResult("distillation", distill.verdict, distill.reason, evidence.as_dict()))
        checks.append(CheckResult("benchmark_targeting", bench.verdict, bench.reason))
    except JudgeError as e:
        checks += _on_error(["distillation", "benchmark_targeting"], e, fail_closed=fail_closed)

    # 3. copy-of-king — mechanical first, LLM only for the middle band.
    if jc.publish_king and king_source is not None:
        checks.append(_copy_check(files.get("generator.py", ""), king_source, cfg, judge))

    return ScreenReport(checks=checks)


def _copy_check(challenger_src: str, king_src: str, cfg, judge: JudgeClient) -> CheckResult:
    jc = cfg.judge
    sim = source_similarity(challenger_src, king_src)
    ev = {"canonical_similarity": round(sim, 4),
          "reject_at": jc.copy_reject_similarity, "review_at": jc.copy_review_similarity}
    if sim >= jc.copy_reject_similarity:
        return CheckResult(
            "copy_of_king", "fail",
            f"canonical-source similarity {sim:.3f} >= {jc.copy_reject_similarity} "
            "(mechanical copy-of-king reject, no LLM)", ev,
        )
    if sim >= jc.copy_review_similarity:
        try:
            v: JudgeVerdict = judge_copy_of_king(challenger_src, king_src, sim, judge)
        except JudgeError as e:
            return _on_error(["copy_of_king"], e, fail_closed=jc.fail_closed)[0]
        return CheckResult("copy_of_king", v.verdict, v.reason or f"similarity {sim:.3f}", ev)
    return CheckResult("copy_of_king", "pass", f"similarity {sim:.3f} below review band", ev)
