"""Build a training corpus by running a miner's generator.

Given a materialised generator repo, import ``generator.Generator``, construct
it with the round's generation seed, and drain exactly ``corpus_n_series``
validated series. The result is a list of float64 arrays plus its digest — the
auditable record of what the model was trained on.

Isolation boundary: the generator is miner-controlled code. The static guard
(:mod:`metronome.interface.static_guard`) is the cheap pre-check; running the
generator MUST happen inside a network-isolated, rlimited sandbox subprocess in
production. :func:`build_corpus` here runs in-process for tests and offline
smoke; ``run_in_sandbox`` is the TODO boundary that wraps it for live use (see
OPEN_QUESTIONS.md #2).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..interface.generator import DataGenerator, drain_generator
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
    spec = importlib.util.spec_from_file_location("metronome_submitted_generator", wrapper_py)
    if spec is None or spec.loader is None:
        raise CorpusError("generator_spec_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules["metronome_submitted_generator"] = module
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
        raise CorpusError("Generator must subclass metronome.interface.DataGenerator")
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
    gen = _load_generator(Path(repo_dir), generation_seed)
    try:
        series = drain_generator(
            gen,
            cfg.corpus_n_series,
            min_length=cfg.min_length,
            max_length=cfg.max_length,
            max_total_points=cfg.max_total_points,
            max_channels=cfg.max_channels,
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


def assert_corpus_reproducible(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
) -> str:
    """Run :func:`build_corpus` twice and assert identical digests.

    The determinism check used by ``metronome verify`` and (optionally) by the
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


def run_in_sandbox(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
) -> CorpusResult:
    """TODO: spawn a network-isolated, rlimited subprocess that calls
    :func:`build_corpus` and returns the corpus over a pipe.

    Until the sandbox lands this delegates to the in-process path, which is
    acceptable for trusted offline runs but NOT for adversarial mainnet use.
    Mirrors horizon's ``validator/scorer/sandbox.py`` design.
    """
    return build_corpus(repo_dir, generation_seed, cfg)
