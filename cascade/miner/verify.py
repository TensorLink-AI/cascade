"""Library form of ``cascade verify`` — runs the checks the trainer runs at
fetch time so a miner sees the same failure messages locally.

Sequence:

1. Repo layout + size — required files present, no pickle weight files
   (safetensors OK), and total bytes <= max_repo_mb.
2. Config — ``config.json`` parses as an object.
3. Static guard on ``generator.py`` — AST scan for blocked imports.
4. Requirements — hash-locked, allowlisted, <= max_packages.
5. (optional) Determinism — import ``Generator``, draw the corpus twice at a
   fixed seed, and assert identical digests. This is the load-bearing extra
   check cascade has that horizon doesn't: a non-deterministic generator
   breaks auditability and is rejected.

Step 5 requires numpy and the generator's runtime deps importable in the
current interpreter; ``--skip-runtime`` runs the static path only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..interface.static_guard import scan_file
from ..interface.validation import (
    ValidationResult,
    check_config,
    check_repo_layout,
    check_repo_size,
    check_requirements_hash_locked,
)
from ..shared.config import ChainConfig


@dataclass(frozen=True)
class VerifyReport:
    """Empty ``failures`` means the generator would be accepted by the trainer."""

    ok: bool
    failures: list[tuple[str, ValidationResult]] = field(default_factory=list)
    runtime_skipped: bool = False
    corpus_digest: str | None = None

    def render(self) -> str:
        lines = []
        if self.ok:
            lines.append("OK: generator would be accepted by the trainer.")
        else:
            lines.append("FAIL: generator would be rejected by the trainer:")
            for step, r in self.failures:
                lines.append(f"  [{step}] {r.reason}  {r.details or ''}")
        if self.runtime_skipped:
            lines.append("  (determinism check skipped: --skip-runtime)")
        elif self.corpus_digest is not None:
            lines.append(f"  corpus_digest (seed=0): {self.corpus_digest[:16]}…  [deterministic]")
        return "\n".join(lines)


def verify_repo(
    repo_dir: Path | str,
    cfg: ChainConfig,
    *,
    skip_runtime: bool = False,
) -> VerifyReport:
    """Run every check the trainer runs before training on a generator."""
    d = Path(repo_dir)
    failures: list[tuple[str, ValidationResult]] = []

    layout = check_repo_layout(d)
    if not layout.ok:
        failures.append(("repo_layout", layout))
        return VerifyReport(ok=False, failures=failures, runtime_skipped=skip_runtime)

    size = check_repo_size(d, cfg.generator.max_repo_mb)
    if not size.ok:
        failures.append(("repo_size", size))

    config = check_config(d)
    if not config.ok:
        failures.append(("config", config))

    guard = scan_file(d / "generator.py", cfg.static_guard.blocked)
    if not guard.ok:
        failures.append((
            "static_guard",
            ValidationResult.fail(
                "blocked_import",
                blocked_module=guard.blocked_module,
                reason=guard.reason,
            ),
        ))

    reqs = check_requirements_hash_locked(
        d / "requirements.txt",
        allowed=cfg.dependencies.allowed,
        max_packages=cfg.dependencies.max_packages,
    )
    if not reqs.ok:
        failures.append(("requirements", reqs))

    if failures or skip_runtime:
        return VerifyReport(ok=not failures, failures=failures, runtime_skipped=skip_runtime)

    digest, runtime_err = _determinism_check(d, cfg)
    if runtime_err is not None:
        failures.append(("determinism", runtime_err))
        return VerifyReport(ok=False, failures=failures)

    return VerifyReport(ok=True, failures=[], runtime_skipped=False, corpus_digest=digest)


def _determinism_check(
    repo_dir: Path,
    cfg: ChainConfig,
) -> tuple[str | None, ValidationResult | None]:
    """Build the corpus twice at a fixed seed and assert identical digests."""
    try:
        from ..trainer.corpus import CorpusError, assert_corpus_reproducible
    except ImportError as e:
        return None, ValidationResult.fail("missing_runtime_dep", error=str(e))
    try:
        digest = assert_corpus_reproducible(repo_dir, 0, cfg.generator)
    except CorpusError as e:
        return None, ValidationResult.fail("corpus_check_failed", error=str(e))
    return digest, None
