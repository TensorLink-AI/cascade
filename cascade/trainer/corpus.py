"""Build a training corpus by running a miner's generator.

Given a materialised generator repo, import ``generator.Generator``, construct
it with the round's generation seed, and drain exactly ``corpus_n_series``
validated series. The result is a list of float64 arrays plus its digest — the
auditable record of what the model was trained on.

Isolation boundary: the generator is miner-controlled code. :func:`build_corpus`
runs it IN-PROCESS — fine for tests, ``cascade verify``, and trusted offline
smoke. In production the trainer runs this same path inside the network-isolated,
rlimited subprocess in :mod:`cascade.trainer.sandbox` (:func:`build_round_corpus`
with ``use_sandbox=True``, the default).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..interface.generator import CAST_SAFE_MAX_FLOAT32, DataGenerator, drain_generator
from ..interface.validation import check_repo_size
from ..shared.config import GeneratorConfig
from ..shared.manifest import corpus_digest


@dataclass(frozen=True)
class CorpusResult:
    series: list[np.ndarray]
    digest: str
    n_series: int
    total_points: int


class CorpusError(RuntimeError):
    """Importing or running the generator failed, or its output was rejected."""


def _load_generator(repo_dir: Path, generation_seed: int) -> DataGenerator:
    wrapper_py = repo_dir / "generator.py"
    if not wrapper_py.is_file():
        raise CorpusError("missing generator.py")
    spec = importlib.util.spec_from_file_location("cascade_submitted_generator", wrapper_py)
    if spec is None or spec.loader is None:
        raise CorpusError("generator_spec_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cascade_submitted_generator"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        raise CorpusError(f"generator_import_failed: {type(e).__name__}: {e}") from e

    Generator = getattr(module, "Generator", None)
    if Generator is None:
        raise CorpusError("generator_class_missing (expected `Generator` in generator.py)")
    try:
        gen = Generator(str(repo_dir), seed=generation_seed)
    except Exception as e:  # noqa: BLE001
        raise CorpusError(f"generator_construct_failed: {type(e).__name__}: {e}") from e
    if not isinstance(gen, DataGenerator):
        raise CorpusError("Generator must subclass cascade.interface.DataGenerator")
    return gen


def build_corpus(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
) -> CorpusResult:
    """Import the generator, draw a validated corpus, and digest it.

    Raises :class:`CorpusError` on any failure; the trainer catches it and the
    offending generator simply fails to qualify this round (a bad generator can
    never affect the king's run).
    """
    size = check_repo_size(repo_dir, cfg.max_repo_mb)
    if not size.ok:
        raise CorpusError(f"submission_too_large: {size.details}")
    gen = _load_generator(Path(repo_dir), generation_seed)
    try:
        series = drain_generator(
            gen,
            cfg.corpus_n_series,
            min_length=cfg.min_length,
            max_length=cfg.max_length,
            max_total_points=cfg.max_total_points,
            max_channels=cfg.max_channels,
            max_abs=cfg.max_abs_value or CAST_SAFE_MAX_FLOAT32,
            reject_constant=cfg.reject_constant,
            max_dup_fraction=cfg.max_dup_fraction,
        )
    except ValueError as e:
        raise CorpusError(f"generator_output_rejected: {e}") from e
    total = int(sum(int(s.size) for s in series))
    return CorpusResult(
        series=series,
        digest=corpus_digest(series),
        n_series=len(series),
        total_points=total,
    )


# Feed modes that stream fresh data with no reuse (vs. cache_reuse, which draws a
# fixed corpus once and lets the trainer pass over it multiple times).
STREAMING_MODES = ("stream_cpu", "stream_gpu")


def build_round_corpus(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    mode: str,
    *,
    use_sandbox: bool = True,
    blocked: tuple[str, ...] = (),
    allow_netns: bool = True,
) -> CorpusResult:
    """Build a round's corpus according to the selected feed ``mode``.

    * ``cache_reuse`` — draw a fixed corpus once (materialised) and let the base
      trainer make multiple passes over it under the token budget. Byte-exact
      auditable; reuses data. This is the path :func:`build_corpus` implements.
    * ``stream_cpu`` / ``stream_gpu`` — streaming feed modes, handled by
      :func:`cascade.trainer.stream.open_round_stream`, not here.
      ``build_round_corpus`` is the *materialised* helper (cache_reuse only) and
      rejects stream modes so a miswired caller fails loudly rather than silently
      falling back to reuse.

    ``use_sandbox`` (default True) runs the generator in the network-isolated,
    rlimited subprocess from :mod:`cascade.trainer.sandbox`; ``blocked`` is the
    static-guard import blocklist enforced before the generator is imported. Pass
    ``use_sandbox=False`` only for trusted offline / in-process test runs.

    Raises :class:`CorpusError` for an unwired or unknown mode.
    """
    if mode == "cache_reuse":
        if use_sandbox:
            from .sandbox import run_in_sandbox

            return run_in_sandbox(
                repo_dir, generation_seed, cfg, blocked=tuple(blocked), allow_netns=allow_netns
            )
        return build_corpus(repo_dir, generation_seed, cfg)
    if mode in STREAMING_MODES:
        raise CorpusError(
            f"corpus_mode={mode!r} streams via stream.open_round_stream, not "
            "build_round_corpus (the materialised cache_reuse-only helper)."
        )
    raise CorpusError(f"unknown corpus_mode={mode!r}")


def assert_corpus_reproducible(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
) -> str:
    """Run :func:`build_corpus` twice and assert identical digests.

    The determinism check used by ``cascade verify`` and (optionally) by the
    trainer before committing a run. Raises :class:`CorpusError` if the
    generator is non-deterministic in its seed. Returns the shared digest.
    """
    first = build_corpus(repo_dir, generation_seed, cfg)
    second = build_corpus(repo_dir, generation_seed, cfg)
    if first.digest != second.digest:
        raise CorpusError(
            "generator is non-deterministic: two runs at the same seed produced "
            "different corpora (digests differ)"
        )
    return first.digest
