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

import hashlib
import itertools
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..shared.config import GeneratorConfig
from . import sandbox
from .corpus import CorpusError, build_round_corpus


class _StreamDigest:
    """Rolling sha256 over canonical ``(C, L)`` float64 series; count finalised.

    Extended record elements (``{"values", "mask"/"roles"}`` dicts, present
    only when ``[training] accepted_fields`` is armed) hash via their
    0xFF-sentinel frame — see :func:`cascade.shared.manifest.corpus_digest`
    for the collision argument. Values-only streams keep their frozen bytes.
    """

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self._n = 0

    def update(self, arr: np.ndarray | dict) -> None:
        if isinstance(arr, dict):
            from ..interface.generator import canonicalize_record, record_frame_bytes

            self._h.update(record_frame_bytes(canonicalize_record(arr)))
            self._n += 1
            return
        self._h.update(int(arr.shape[0]).to_bytes(8, "big"))
        self._h.update(int(arr.shape[1]).to_bytes(8, "big"))
        self._h.update(arr.tobytes())
        self._n += 1

    def hexdigest(self) -> str:
        h = self._h.copy()
        h.update(b"\x00count")
        h.update(self._n.to_bytes(8, "big"))
        return h.hexdigest()


def _element_points(arr: np.ndarray | dict) -> int:
    """Token-budget points of one stream element: values entries only (a mask
    marks points missing; it does not add points)."""
    if isinstance(arr, dict):
        return int(arr["values"].size)
    return int(arr.size)


def _inprocess_stream(
    repo: Path, seed: int, cfg: GeneratorConfig, token_budget: int
) -> Iterator[np.ndarray]:
    """In-process fresh-series stream (no sandbox) for offline / test runs.

    The full carrier pipeline (record fields, validation, budgets, the corr
    gate) is :class:`~cascade.interface.generator.SeriesValidator` — one
    implementation shared with the drain and the sandbox child, so the feed
    modes cannot drift.
    """
    from ..interface.generator import SeriesValidator
    from .channel_stats import corr_enforce_gate
    from .corpus import _load_generator, resolve_real_corpus

    n_upper = int(token_budget) // max(int(cfg.min_length), 1) + 2
    cfg = resolve_real_corpus(cfg)
    validator = SeriesValidator.from_config(
        cfg, extra_series_check=corr_enforce_gate(cfg)
    )
    gen = _load_generator(repo, int(seed), interface_version=cfg.interface_version,
                          real_corpus_dir=cfg.real_corpus_dir)
    for i, item in enumerate(gen.generate(n_upper)):
        yield validator.process(item, i)


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
            total += _element_points(arr)
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
        max_wall_seconds: int | None = None,
    ) -> None:
        self._repo = Path(repo_dir)
        self._seed = int(generation_seed)
        self._cfg = cfg
        self._budget = int(token_budget)
        self._use_sandbox = use_sandbox
        self._allow_netns = allow_netns
        self._blocked = tuple(blocked)
        self._gpu = gpu
        self._max_wall_seconds = max_wall_seconds
        self._dig = _StreamDigest()
        self._n = 0
        self._points = 0
        self._cm: object | None = None

    def _raw_source(self) -> Iterator[np.ndarray]:
        if self._use_sandbox:
            self._cm = sandbox.stream_series(
                self._repo, self._seed, self._cfg, self._budget,
                blocked=self._blocked, allow_netns=self._allow_netns, gpu=self._gpu,
                max_wall_seconds=self._max_wall_seconds,
            )
            return self._cm.__enter__()
        return _inprocess_stream(self._repo, self._seed, self._cfg, self._budget)

    def series(self) -> Iterator[np.ndarray]:
        total = 0
        for arr in self._raw_source():
            yield arr
            self._dig.update(arr)
            self._n += 1
            total += _element_points(arr)
            self._points = total
            if total >= self._budget:
                break

    def close(self) -> None:
        if self._cm is not None:
            cm, self._cm = self._cm, None
            try:
                cm.__exit__(None, None, None)
            except CorpusError:
                # A sandbox VERDICT at stream close (e.g. the CPU-mode
                # GPU-use rejection) must propagate and fail the entry —
                # only ordinary teardown noise is safe to swallow.
                raise
            except Exception:  # noqa: BLE001, S110 - teardown must not mask training
                pass

    @property
    def digest(self) -> str:
        return self._dig.hexdigest()

    @property
    def n_series(self) -> int:
        return self._n

    @property
    def total_points(self) -> int:
        return self._points


class _SeedMixStream(RoundStream):
    """Interleave N single-seed streams (``gen_seed_mix``, DEC-CA-0033).

    Each child runs the generator under its own derived generation seed with
    ~1/N of the token budget; series are yielded round-robin, one per child
    per turn, skipping exhausted children. The digest is a rolling digest of
    the INTERLEAVED sequence actually consumed — the same rule for training
    and for an audit re-derivation, whatever the child mode's own digest
    semantics.
    """

    def __init__(self, children: list[RoundStream]) -> None:
        self._children = children
        self._dig = _StreamDigest()
        self._n = 0
        self._points = 0

    def series(self) -> Iterator[np.ndarray]:
        iters = [c.series() for c in self._children]
        while iters:
            live = []
            for it in iters:
                try:
                    arr = next(it)
                except StopIteration:
                    continue
                yield arr
                self._dig.update(arr)
                self._n += 1
                self._points += _element_points(arr)
                live.append(it)
            iters = live

    def close(self) -> None:
        first_err: Exception | None = None
        for c in self._children:
            try:
                c.close()
            except Exception as e:  # noqa: BLE001 - close every child first
                first_err = first_err or e
        if first_err is not None:
            raise first_err

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
    max_wall_seconds: int | None = None,
    seed_mix: int = 1,
) -> RoundStream:
    """Open the round's corpus stream for ``mode`` (see module docstring).

    ``max_wall_seconds`` (streaming modes only) is the upper bound on how long
    the stream will be consumed — pass the contract's ``max_train_seconds`` so
    the sandbox child's cumulative CPU rlimit scales with the training budget
    instead of the per-frame stall window (see ``sandbox.stream_cpu_rlimit``).

    ``seed_mix`` > 1 ([training] gen_seed_mix, DEC-CA-0033) opens N child
    streams — one per derived generation seed (``_mix(generation_seed,
    "seed-mix", i)``), each with ~1/N of the token budget (the first child
    absorbs the remainder) — and interleaves them round-robin. Callers pass
    the contract's value so the trainer, the audit re-derivation, and the
    miner's local scorer all replay the same rule.
    """
    n_mix = 1 if seed_mix is None else int(seed_mix)
    if n_mix < 1:
        raise CorpusError(f"seed_mix must be >= 1; got {seed_mix}")
    if n_mix > 1:
        from .contract import _mix

        share = int(token_budget) // n_mix
        budgets = [int(token_budget) - share * (n_mix - 1)] + [share] * (n_mix - 1)
        return _SeedMixStream([
            open_round_stream(
                mode, repo_dir, _mix(generation_seed, "seed-mix", i), cfg,
                token_budget=b, use_sandbox=use_sandbox, blocked=tuple(blocked),
                allow_netns=allow_netns, max_wall_seconds=max_wall_seconds,
            )
            for i, b in enumerate(budgets)
        ])
    if mode == "cache_reuse":
        return _CacheReuseStream(
            repo_dir, generation_seed, cfg, token_budget,
            use_sandbox=use_sandbox, blocked=tuple(blocked), allow_netns=allow_netns,
        )
    if mode in ("stream_cpu", "stream_gpu"):
        return _FreshSeriesStream(
            repo_dir, generation_seed, cfg, token_budget,
            use_sandbox=use_sandbox, blocked=tuple(blocked), allow_netns=allow_netns,
            gpu=(mode == "stream_gpu"), max_wall_seconds=max_wall_seconds,
        )
    raise CorpusError(f"unknown corpus_mode={mode!r}")
