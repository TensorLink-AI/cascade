"""Library form of ``cascade verify`` — runs the checks the trainer runs at
fetch time so a miner sees the same failure messages locally.

Sequence:

1. Repo layout + size — required files present, no shipped weight files of any
   kind (pickle checkpoints or code-free containers like safetensors), and total
   bytes <= max_repo_mb.
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

from ..interface.static_guard import scan_tree
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
    """A local report; only scope=full covers the full materialised corpus."""

    ok: bool
    failures: list[tuple[str, ValidationResult]] = field(default_factory=list)
    runtime_skipped: bool = False
    corpus_digest: str | None = None
    scope: str = "full"
    # The effective output/training geometry checked (see :func:`geometry`).
    geometry: dict | None = None

    def to_dict(self) -> dict:
        out = {"ok": self.ok, "runtime_skipped": self.runtime_skipped, "scope": self.scope,
               "corpus_digest": self.corpus_digest,
               "failures": [{"step": step, "reason": r.reason, "details": r.details}
                            for step, r in self.failures]}
        if self.geometry is not None:
            out["geometry"] = self.geometry
        return out

    def render(self) -> str:
        lines = []
        if self.ok:
            lines.append("OK: static checks passed; output bounds and determinism remain unchecked."
                         if self.runtime_skipped else
                         "OK: quick checks passed; run full verify before submission."
                         if self.scope == "quick" else "OK: full local verification passed under the selected generator policy.")
        else:
            lines.append("FAIL: generator would be rejected by the trainer:")
            for step, r in self.failures:
                lines.append(f"  [{step}] {r.reason}  {r.details or ''}")
        if self.runtime_skipped:
            lines.append("  (determinism check skipped: --skip-runtime)")
        elif self.corpus_digest is not None:
            lines.append(f"  corpus_digest (seed=0): {self.corpus_digest[:16]}…  [deterministic]")
        if self.geometry is not None:
            lines.extend(render_geometry(self.geometry))
        return "\n".join(lines)


def _contract_geometry(label: str, contract, channels: int | None) -> dict:
    """How the trainer batches a ``(C, L)`` yield under ``contract``
    (:func:`cascade.trainer.toto2_trainer.iter_training_batches`): a
    ``"sequences"`` batch packs ``max(1, batch_size // C)`` series per step,
    a ``"series"`` batch ``batch_size`` series of C channels each."""
    size = int(contract.batch_size)
    denom = str(contract.batch_denomination)
    row = {"contract": label, "batch_size": size, "batch_denomination": denom,
           "micro_batches": int(getattr(contract, "micro_batches", 1) or 1),
           "context_length": int(contract.context_length),
           "long_series": str(getattr(contract, "long_series", "last") or "last"),
           "corpus_mode": str(contract.corpus_mode)}
    if channels:
        series = max(1, size // channels) if denom == "sequences" else size
        row.update(series_per_step=series, channels=channels,
                   channel_sequences_per_step=series * channels)
    return row


def geometry(cfg: ChainConfig, policy, across_launch: bool) -> dict:
    """The output and batch geometry a pass is checked against, from the loaded
    config — so a miner sees e.g. "exactly 32 channels; 16 series x 32 channels
    per step" instead of inferring it."""
    v2 = cfg.training.v2_contract
    both = across_launch and v2 is not None
    contracts = [("pre-v2" if both else "effective", cfg.screen_contract())]
    if both:
        contracts.append(("v2", v2))
    fixed = int(policy.max_channels) if int(policy.min_channels) == int(policy.max_channels) else None
    return {"min_channels": int(policy.min_channels), "max_channels": int(policy.max_channels),
            "min_length": int(policy.min_length), "max_length": int(policy.max_length),
            "groups_accepted": "groups" in tuple(policy.accepted_fields),
            "packed_sources": str(getattr(cfg.static_guard, "packed_sources", "scan")),
            "training": [_contract_geometry(label, c, fixed) for label, c in contracts]}


def render_geometry(g: dict) -> list[str]:
    lo, hi = g["min_channels"], g["max_channels"]
    width = (f"exactly {hi} channels per yield (pack narrower processes into the {hi} rows "
             "yourself; nothing pads or merges yields)" if lo == hi
             else f"{lo}-{hi} channels per yield")
    lines = [f"  checked geometry: {width}, length {g['min_length']}-{g['max_length']}; "
             + ("optional `groups` packing accepted" if g["groups_accepted"]
                else "values-only (`groups` not accepted by every checked contract)")
             + f"; packed sources: {g['packed_sources']}"]
    for t in g["training"]:
        if "series_per_step" in t and t["batch_denomination"] == "sequences":
            batch = (f"{t['batch_size']} sequences = {t['series_per_step']} series"
                     f" x {t['channels']} channels per step")
        elif "series_per_step" in t:
            batch = f"{t['series_per_step']} series x {t['channels']} channels per step"
        else:
            batch = (f"{t['batch_size']} {t['batch_denomination']} per step"
                     + (" (series = batch_size // C)" if t["batch_denomination"] == "sequences"
                        else " (x C channels each)"))
        lines.append(f"  training [{t['contract']}]: {batch}, {t['micro_batches']} micro-batch(es); "
                     f"context {t['context_length']}, long series: {t['long_series']}; "
                     f"corpus: {t['corpus_mode']}")
    return lines


def _policy(cfg: ChainConfig, across_launch: bool):
    from dataclasses import replace

    base = cfg.generator_for_contract(cfg.screen_contract())
    v2 = cfg.training.v2_contract
    if not across_launch or v2 is None:
        return base
    after = cfg.generator_for_contract(v2)
    return replace(base,
                   accepted_fields=tuple(sorted(set(base.accepted_fields) & set(after.accepted_fields))),
                   min_channels=max(base.min_channels, after.min_channels),
                   max_channels=min(base.max_channels, after.max_channels),
                   min_length=max(base.min_length, after.min_length),
                   max_length=min(base.max_length, after.max_length),
                   max_memory_mb=min(base.max_memory_mb, after.max_memory_mb),
                   max_generate_seconds=min(base.max_generate_seconds, after.max_generate_seconds))


def _verification_cores(cfg: ChainConfig, across_launch: bool = False):
    """Explicit child cpuset for shard verification; never mutate the caller's env.

    Match the tier's configured cores per child, within the local cpuset. Legacy
    verification keeps dispatch geometry (or its historical uncapped default).
    """
    import os

    contracts = [cfg.screen_contract()]
    if across_launch and cfg.training.v2_contract is not None:
        contracts.append(cfg.training.v2_contract)
    if not any(c.corpus_mode == "shards" for c in contracts):
        return None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 1))
    count = max(1, cfg.generation.vcpus // cfg.generation.n_children)
    return tuple(allowed[:count])


def verify_repo(
    repo_dir: Path | str,
    cfg: ChainConfig,
    *,
    skip_runtime: bool = False,
    quick: bool = False,
    across_launch: bool = False,
) -> VerifyReport:
    """Run every check the trainer runs before training on a generator.

    ``across_launch`` (offline, no contract bound) checks the intersection of
    the base and ``[v2.training]`` policies: a pass is admissible whichever
    side of the v2 launch trains the entry.
    """
    from dataclasses import replace

    cpu_cores = _verification_cores(cfg, across_launch)
    policy = _policy(cfg, across_launch)
    try:
        geo = geometry(cfg, policy, across_launch)
    except (AttributeError, TypeError, ValueError):  # a partial test config: report without it
        geo = None
    cfg = replace(cfg, generator=policy)
    d = Path(repo_dir)
    scope = "quick" if quick else "static" if skip_runtime else "full"
    failures: list[tuple[str, ValidationResult]] = []

    layout = check_repo_layout(d)
    if not layout.ok:
        failures.append(("repo_layout", layout))
        return VerifyReport(ok=False, failures=failures, runtime_skipped=skip_runtime, scope=scope, geometry=geo)

    size = check_repo_size(d, cfg.generator.max_repo_mb)
    if not size.ok:
        failures.append(("repo_size", size))

    config = check_config(d)
    if not config.ok:
        failures.append(("config", config))

    guard = scan_tree(d, cfg.static_guard.blocked,
                      packed_sources=getattr(cfg.static_guard, "packed_sources", "scan"))
    if not guard.ok:
        failures.append((
            "static_guard",
            ValidationResult.fail(
                "packed_source" if (guard.reason or "").startswith("packed_source")
                else "blocked_import",
                blocked_module=guard.blocked_module,
                guard_reason=guard.reason,
                file=guard.file,
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
        return VerifyReport(ok=not failures, failures=failures, runtime_skipped=skip_runtime, scope=scope, geometry=geo)

    digest, runtime_err = (_quick_determinism_check(d, cfg, cpu_cores=cpu_cores) if quick
                           else _determinism_check(d, cfg, cpu_cores=cpu_cores))
    if runtime_err is not None:
        failures.append(("determinism", runtime_err))
        return VerifyReport(ok=False, failures=failures, scope=scope, geometry=geo)

    return VerifyReport(ok=True, failures=[], runtime_skipped=False, corpus_digest=digest,
                        scope=scope, geometry=geo)


def _quick_determinism_check(repo_dir: Path, cfg: ChainConfig, *, cpu_cores=None):
    """Bounded child process: two small corpora, with the full output validators."""
    import json
    import subprocess
    import sys
    from dataclasses import asdict, replace

    generator = replace(cfg.generator, corpus_target_points=65_536, corpus_n_series=128)
    try:
        run = subprocess.run(
            [sys.executable, "-m", "cascade.miner.verify_worker"],
            input=json.dumps({"repo": str(repo_dir.resolve()), "generator": asdict(generator),
                              "cpu_cores": cpu_cores}),
            capture_output=True, text=True, timeout=30, check=False,
        )
        doc = json.loads(run.stdout)
        if run.returncode or not doc.get("ok"):
            return None, ValidationResult.fail("corpus_check_failed", error=doc.get("error", "worker failed"))
        return str(doc["digest"]), None
    except subprocess.TimeoutExpired:
        return None, ValidationResult.fail("runtime_timeout", seconds=30)
    except (OSError, ValueError, KeyError) as exc:
        return None, ValidationResult.fail("corpus_check_failed", error=str(exc))


def _determinism_check(
    repo_dir: Path,
    cfg: ChainConfig,
    *, cpu_cores=None,
) -> tuple[str | None, ValidationResult | None]:
    """Build the corpus twice at a fixed seed and assert identical digests."""
    try:
        from ..trainer.corpus import CorpusError
        from ..trainer.sandbox import run_in_sandbox
    except ImportError as e:
        return None, ValidationResult.fail("missing_runtime_dep", error=str(e))
    try:
        first = run_in_sandbox(repo_dir, 0, cfg.generator, blocked=cfg.static_guard.blocked,
                               cpu_cores=cpu_cores)
        second = run_in_sandbox(repo_dir, 0, cfg.generator, blocked=cfg.static_guard.blocked,
                               cpu_cores=cpu_cores)
        if not first.n_series or not second.n_series:
            raise CorpusError("generator produced no usable series")
        if first.digest != second.digest:
            raise CorpusError("generator is not deterministic")
        digest = first.digest
    except CorpusError as e:
        return None, ValidationResult.fail("corpus_check_failed", error=str(e))
    return digest, None
