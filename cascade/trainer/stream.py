"""Per-round corpus stream handed to the :class:`BaseTrainer`.

Both feed modes share one shape — a budget-capped iterator of canonical
``(C, L)`` float64 series the trainer pulls — so the GPU code never branches on
the mode:

* ``cache_reuse`` — draw a fixed corpus once (sandboxed), then ``cycle`` it; the
  model sees data again. Digest is the unique-corpus digest (``corpus_digest``).
* ``stream_cpu`` — stream *fresh* series with no reuse, each hashed into a
  rolling digest as it passes. Digest covers exactly the consumed prefix.
* ``stream_gpu`` — same fresh-series streaming, but from a CUDA/torch-resident
  generator under the sandbox's GPU profile (relaxed address-space rlimit + CUDA
  env passthrough). High throughput; audit is tolerance/same-hardware, so its
  rolling digest reproduces only on equivalent hardware.

Both stop at ``token_budget`` points. :func:`open_round_stream` is a context
manager; after the trainer drains ``series()``, read ``digest`` / ``n_series`` /
``total_points`` for the manifest. The two modes use different but
internally-reproducible digest schemes — ``corpus_mode`` is in
``contract_digest``, so an auditor re-derives in the same mode and matches.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..shared.config import GeneratorConfig
from . import sandbox
from .corpus import CorpusError, build_round_corpus


class _StreamDigest:
    """Rolling sha256 over canonical ``(C, L)`` float64 series; count finalised."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self._n = 0

    def update(self, arr: np.ndarray) -> None:
        self._h.update(int(arr.shape[0]).to_bytes(8, "big"))
        self._h.update(int(arr.shape[1]).to_bytes(8, "big"))
        self._h.update(arr.tobytes())
        self._n += 1

    def hexdigest(self) -> str:
        h = self._h.copy()
        h.update(b"\x00count")
        h.update(self._n.to_bytes(8, "big"))
        return h.hexdigest()


def _inprocess_stream(
    repo: Path, seed: int, cfg: GeneratorConfig, token_budget: int
) -> Iterator[np.ndarray]:
    """In-process fresh-series stream (no sandbox) for offline / test runs."""
    from ..interface.generator import CAST_SAFE_MAX_FLOAT32, check_series
    from .corpus import _load_generator

    n_upper = int(token_budget) // max(int(cfg.min_length), 1) + 2
    gen = _load_generator(repo, int(seed))
    for i, arr in enumerate(gen.generate(n_upper)):
        check_series(
            arr, min_length=cfg.min_length, max_length=cfg.max_length,
            max_channels=cfg.max_channels,
            max_abs=cfg.max_abs_value or CAST_SAFE_MAX_FLOAT32,
            reject_constant=cfg.reject_constant, index=i,
        )
        yield np.ascontiguousarray(np.atleast_2d(np.asarray(arr, dtype=np.float64)))


class RoundStream:
    """Context manager: ``series()`` plus digest / counts read after consumption."""

    def __enter__(self) -> RoundStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def series(self) -> Iterator[np.ndarray]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def digest(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def n_series(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def total_points(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError


class _CacheReuseStream(RoundStream):
    """Materialise once (sandboxed), then cycle under the token budget."""

    def __init__(
        self, repo_dir: Path | str, generation_seed: int, cfg: GeneratorConfig,
        token_budget: int, *, use_sandbox: bool, blocked: tuple[str, ...],
        allow_netns: bool = True,
    ) -> None:
        self._budget = int(token_budget)
        self._corpus = build_round_corpus(
            repo_dir, generation_seed, cfg, "cache_reuse",
            use_sandbox=use_sandbox, blocked=blocked, allow_netns=allow_netns,
        )
        self._consumed = 0

    def series(self) -> Iterator[np.ndarray]:
        total = 0
        for arr in itertools.cycle(self._corpus.series):
            yield arr
            total += int(arr.size)
            self._consumed = total
            if total >= self._budget:
                break

    @property
    def digest(self) -> str:
        return self._corpus.digest

    @property
    def n_series(self) -> int:
        return self._corpus.n_series

    @property
    def total_points(self) -> int:
        return self._consumed or self._corpus.total_points


class _FreshSeriesStream(RoundStream):
    """Stream fresh series (no reuse) from the sandbox, rolling-digesting each.

    Backs both ``stream_cpu`` and ``stream_gpu``; ``gpu=True`` selects the
    sandbox's GPU profile (relaxed address-space rlimit + CUDA env passthrough)
    for a torch-resident generator. The rolling digest is byte-exact for
    ``stream_cpu`` and tolerance/same-hardware for ``stream_gpu``.
    """

    def __init__(
        self, repo_dir: Path | str, generation_seed: int, cfg: GeneratorConfig,
        token_budget: int, *, use_sandbox: bool, blocked: tuple[str, ...],
        allow_netns: bool = True, gpu: bool = False,
    ) -> None:
        self._repo = Path(repo_dir)
        self._seed = int(generation_seed)
        self._cfg = cfg
        self._budget = int(token_budget)
        self._use_sandbox = use_sandbox
        self._allow_netns = allow_netns
        self._blocked = tuple(blocked)
        self._gpu = gpu
        self._dig = _StreamDigest()
        self._n = 0
        self._points = 0
        self._cm: object | None = None

    def _raw_source(self) -> Iterator[np.ndarray]:
        if self._use_sandbox:
            self._cm = sandbox.stream_series(
                self._repo, self._seed, self._cfg, self._budget,
                blocked=self._blocked, allow_netns=self._allow_netns, gpu=self._gpu,
            )
            return self._cm.__enter__()
        return _inprocess_stream(self._repo, self._seed, self._cfg, self._budget)

    def series(self) -> Iterator[np.ndarray]:
        total = 0
        for arr in self._raw_source():
            yield arr
            self._dig.update(arr)
            self._n += 1
            total += int(arr.size)
            self._points = total
            if total >= self._budget:
                break

    def close(self) -> None:
        if self._cm is not None:
            with contextlib.suppress(Exception):
                self._cm.__exit__(None, None, None)
            self._cm = None

    @property
    def digest(self) -> str:
        return self._dig.hexdigest()

    @property
    def n_series(self) -> int:
        return self._n

    @property
    def total_points(self) -> int:
        return self._points


def open_round_stream(
    mode: str,
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    *,
    token_budget: int,
    use_sandbox: bool = True,
    blocked: tuple[str, ...] = (),
    allow_netns: bool = True,
) -> RoundStream:
    """Open the round's corpus stream for ``mode`` (see module docstring)."""
    if mode == "cache_reuse":
        return _CacheReuseStream(
            repo_dir, generation_seed, cfg, token_budget,
            use_sandbox=use_sandbox, blocked=tuple(blocked), allow_netns=allow_netns,
        )
    if mode in ("stream_cpu", "stream_gpu"):
        return _FreshSeriesStream(
            repo_dir, generation_seed, cfg, token_budget,
            use_sandbox=use_sandbox, blocked=tuple(blocked), allow_netns=allow_netns,
            gpu=(mode == "stream_gpu"),
        )
    raise CorpusError(f"unknown corpus_mode={mode!r}")
