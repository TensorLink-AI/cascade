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
* ``shards`` — read a corpus the CPU generation tier (DEC-CA-0053) already
  materialised into digest-pinned parquet shards
  (:mod:`cascade.generation.shards`). No sandbox and no miner code run here;
  each shard is sha256-verified before its first series is read, and the
  rolling digest uses the same framing as the fresh streams, over the stored
  (float32-widened) values. ``corpus_epochs`` > 0 cycles the corpus at most
  that many times; 0 stops at the end of one pass.

All modes stop at ``token_budget`` points. :func:`open_round_stream` is a context
manager; after the trainer drains ``series()``, read ``digest`` / ``n_series`` /
``total_points`` for the manifest. The two modes use different but
internally-reproducible digest schemes — ``corpus_mode`` is in
``contract_digest``, so an auditor re-derives in the same mode and matches.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import math
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..shared.config import GeneratorConfig, budget_denomination_parts
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
        if isinstance(arr, dict) and "groups" in arr and arr["groups"] is None:
            arr = {k: v for k, v in arr.items() if k != "groups"}
            if set(arr) == {"values"}:
                arr = arr["values"]
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


def element_points(arr: np.ndarray | dict, denomination: str = "points") -> int:
    """Token-budget points of one stream element under ``denomination``
    (``[training] budget_denomination``, DEC-CA-0042).

    ``"points"``: every values entry is a point — a ``(C, L)`` series costs
    ``C×L`` (a mask marks points missing; it does not add points).
    ``"series_points"``: a ``(C, L)`` series costs ``L`` — per time-step
    positions, so a wide series draws ``C×`` the channel tokens per budget
    point. ``"points+mvPCT"`` (DEC-CA-0047): ``C×L`` billed at
    ``100/(100+PCT)`` when ``C > 1`` — token billing with a width bonus.
    Identical for a 1-D / single-channel series. The trainer's token
    counter (:func:`~cascade.trainer.toto2_trainer.batch_points`) applies the
    same rule, so the stream's stop and the loop's stop agree.
    """
    vals = arr["values"] if isinstance(arr, dict) else arr
    if denomination == "series_points":
        return int(np.shape(vals)[-1]) if np.ndim(vals) else 0
    size = int(np.size(vals))
    _base, pct = budget_denomination_parts(denomination)
    if pct and np.ndim(vals) == 2 and int(np.shape(vals)[0]) > 1:
        # DEC-CA-0047: a multichannel series is billed 100/(100+PCT) of its
        # points (ceil, exact integers) — the width bonus under token billing.
        return -(-size * 100 // (100 + pct))
    return size


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
        allow_netns: bool = True, budget_denomination: str = "points",
    ) -> None:
        self._budget = int(token_budget)
        self._denom = budget_denomination
        self._corpus = build_round_corpus(
            repo_dir, generation_seed, cfg, "cache_reuse",
            use_sandbox=use_sandbox, blocked=blocked, allow_netns=allow_netns,
        )
        self._consumed = 0

    def series(self) -> Iterator[np.ndarray]:
        total = 0
        for arr in itertools.cycle(self._corpus.series):
            yield arr
            total += element_points(arr, self._denom)
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
        max_wall_seconds: int | None = None, budget_denomination: str = "points",
    ) -> None:
        self._repo = Path(repo_dir)
        self._seed = int(generation_seed)
        self._cfg = cfg
        self._budget = int(token_budget)
        self._denom = budget_denomination
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
        from ..interface.generator import SeriesValidator

        validator = SeriesValidator.from_config(self._cfg)
        total = 0
        for index, arr in enumerate(self._raw_source()):
            try:
                arr = validator.process(arr, index)
            except ValueError as e:
                raise CorpusError(f"generator_output_rejected: {e}") from e
            yield arr
            self._dig.update(arr)
            self._n += 1
            total += element_points(arr, self._denom)
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


# Where a ``shards`` leg's materialised corpus sits on the pod when the caller
# passes no ``corpus_dir`` (the worker sets it from ``--corpus-dir``).
CORPUS_DIR_ENV = "CASCADE_CORPUS_DIR"


class _ShardStream(RoundStream):
    """Read a materialised corpus (DEC-CA-0053) under the token budget.

    Series come from :func:`cascade.generation.shards.iter_corpus` in its
    training order; the stream stops at ``token_budget`` points under the
    contract's denomination, exactly like the fresh streams, and digests the
    consumed prefix with the same :class:`_StreamDigest` framing. With
    ``epochs`` > 0 the corpus is re-read from the start at most ``epochs``
    times in total; with 0 it is read once.
    """

    def __init__(self, corpus_dir: Path | str, token_budget: int, *,
                 budget_denomination: str = "points", epochs: int = 0, eager_accounting=False,
                 channel_bounds: tuple[int, int] | None = None, accepted_fields: tuple[str, ...] = ()) -> None:
        from ..generation.shards import read_index, verify_shards

        # A missing or corrupt shard is an infrastructure fault (the tier or
        # the transfer), never the miner's: ShardError is deliberately NOT a
        # CorpusError, so the worker exits as infra and the leg requeues.
        self._dir = Path(corpus_dir)
        verify_shards(self._dir, read_index(self._dir))
        self._budget = int(token_budget)
        self._denom = budget_denomination
        self._passes = max(1, int(epochs or 0))
        self._eager_accounting = eager_accounting
        self._channel_bounds = channel_bounds
        self._accepted_fields = accepted_fields
        self._dig = _StreamDigest()
        self._n = 0
        self._points = 0

    def series(self) -> Iterator[np.ndarray]:
        from ..generation.shards import iter_corpus

        total = 0
        for _ in range(self._passes):
            # shards were verified at open; a re-read pass skips the re-hash
            for arr in iter_corpus(self._dir, verify=False):
                if isinstance(arr, dict) and set(arr) - {"values", *self._accepted_fields}:
                    from ..generation.shards import ShardError

                    raise ShardError("shard record contains fields not accepted by the contract")
                vals = arr["values"] if isinstance(arr, dict) else arr
                if self._channel_bounds is not None:
                    from ..generation.shards import ShardError

                    lower, upper = self._channel_bounds
                    if not lower <= vals.shape[0] <= upper:
                        raise ShardError(f"shard series has {vals.shape[0]} channels outside [{lower}, {upper}]")
                if self._eager_accounting:
                    self._dig.update(arr)
                    self._n += 1
                    total += element_points(arr, self._denom)
                    self._points = total
                yield arr
                if not self._eager_accounting:
                    self._dig.update(arr)
                    self._n += 1
                    total += element_points(arr, self._denom)
                    self._points = total
                if total >= self._budget:
                    return

    @property
    def digest(self) -> str:
        return self._dig.hexdigest()

    @property
    def n_series(self) -> int:
        return self._n

    @property
    def total_points(self) -> int:
        return self._points


# Read-ahead per seed-mix child (DEC-CA-0050 generation surface). Each child
# is its own sandboxed generator process, but the round-robin reader only
# drains one series per child per turn, so a child otherwise idles once its
# OS pipe (~2 series) fills. A bounded queue filled by a thread keeps every
# child generating concurrently. CONSUMPTION ORDER IS UNCHANGED — the
# interleave still takes one series per child per turn — so the rolling
# corpus digest is byte-identical with or without read-ahead.
PREFETCH_SERIES = 64
_END = object()


class _Prefetch:
    """Iterate ``it`` on a daemon thread into a bounded queue; re-raises the
    producer's exception at the consumer, in order.

    Every producer ``put`` (items, the end marker AND a raised exception) is
    stop-aware: once the consumer is gone (the leg hit its wall and
    ``close()`` ran) the thread gives up instead of blocking forever on a full
    queue. :meth:`stop` also drains the queue so a blocked put frees at once,
    and :meth:`join` lets the owner wait for the thread to leave the child
    iterator before the child is closed (never two threads in one child)."""

    _PUT_TIMEOUT = 0.1

    def __init__(self, it: Iterator, maxsize: int = PREFETCH_SERIES) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max(1, maxsize))
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(it,), daemon=True)
        self._t.start()

    def _put(self, item) -> bool:
        """Put ``item`` unless stopped first; False ⇒ the consumer is gone."""
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=self._PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, it: Iterator) -> None:
        try:
            for item in it:
                if not self._put(item) or self._stop.is_set():
                    return
            self._put(_END)
        except BaseException as e:  # noqa: BLE001 - surfaced to the consumer
            if not self._stop.is_set():
                self._put(e)

    def __iter__(self) -> _Prefetch:
        return self

    def __next__(self):
        item = self._q.get()
        if item is _END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self) -> None:
        """Signal the producer and drain the queue (unblocks a pending put)."""
        self._stop.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the producer thread; True once it has exited."""
        self._t.join(timeout)
        return not self._t.is_alive()

    @property
    def alive(self) -> bool:
        return self._t.is_alive()


# How long close() waits for a prefetch thread to leave its child iterator
# before closing the child under it (a producer blocked INSIDE a slow
# generator read only returns once the child process is killed).
PREFETCH_JOIN_SECONDS = 2.0


class _SeedMixStream(RoundStream):
    """Interleave N single-seed streams (``gen_seed_mix``, DEC-CA-0033).

    Each child runs the generator under its own derived generation seed with
    ~1/N of the token budget; series are yielded round-robin, one per child
    per turn, skipping exhausted children. The digest is a rolling digest of
    the INTERLEAVED sequence actually consumed — the same rule for training
    and for an audit re-derivation, whatever the child mode's own digest
    semantics.
    """

    def __init__(self, children: list[RoundStream],
                 budget_denomination: str = "points") -> None:
        self._children = children
        self._denom = budget_denomination
        self._dig = _StreamDigest()
        self._n = 0
        self._points = 0

    def series(self) -> Iterator[np.ndarray]:
        self._prefetch = [_Prefetch(c.series()) for c in self._children]
        iters = list(self._prefetch)
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
                self._points += element_points(arr, self._denom)
                live.append(it)
            iters = live

    def close(self) -> None:
        prefetch = list(getattr(self, "_prefetch", ()))
        for pf in prefetch:
            pf.stop()
        # Let each producer leave its child iterator BEFORE the child is
        # closed, so a child is never driven by two threads at once. A
        # producer still blocked inside a slow generator read after the grace
        # only returns once its child process is killed below.
        for pf in prefetch:
            pf.join(PREFETCH_JOIN_SECONDS)
        first_err: Exception | None = None
        for c in self._children:
            try:
                c.close()
            except Exception as e:  # noqa: BLE001 - close every child first
                first_err = first_err or e
        for pf in prefetch:
            if pf.alive:
                pf.stop()                 # drain anything put after the first drain
                pf.join(PREFETCH_JOIN_SECONDS)
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
    budget_denomination: str = "points",
    corpus_epochs: int = 0,
    corpus_dir: Path | str | None = None,
    continuation_protocol: str = "",
) -> RoundStream:
    """Open the round's corpus stream for ``mode`` (see module docstring).

    ``corpus_dir`` (``shards`` only) is the materialised corpus directory;
    omitted ⇒ ``$CASCADE_CORPUS_DIR``. A ``shards`` stream never runs the
    generator, so ``repo_dir``/``seed_mix`` are ignored there. ``cfg`` still
    checks channel bounds; the tier already fanned generation over its children.

    ``budget_denomination`` ([training] budget_denomination, DEC-CA-0042) is
    what one point of ``token_budget`` is at C > 1 — see
    :func:`element_points`. It decides WHERE the stream stops, and the corpus
    digest is the rolling digest of the consumed prefix, so the trainer, the
    audit re-derivation, and the miner's local scorer must all pass the
    contract's value.

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
    epochs = int(corpus_epochs or 0)
    if epochs < 0:
        raise CorpusError(f"corpus_epochs must be >= 0; got {corpus_epochs}")
    if mode == "shards":
        import os

        where = corpus_dir if corpus_dir is not None else os.environ.get(CORPUS_DIR_ENV, "")
        if not where:
            # operator misconfiguration, not a miner fault (not CorpusError)
            raise ValueError(
                f"corpus_mode='shards' needs a corpus_dir (or ${CORPUS_DIR_ENV})")
        return _ShardStream(where, token_budget,
                            budget_denomination=budget_denomination, epochs=epochs,
                            eager_accounting=continuation_protocol == "cascade-continuation/1",
                            channel_bounds=(cfg.min_channels, cfg.max_channels) if cfg is not None else None,
                            accepted_fields=tuple(cfg.accepted_fields) if cfg is not None else ())
    if epochs and mode != "cache_reuse":
        raise CorpusError("corpus_epochs requires corpus_mode='cache_reuse' or 'shards'")
    if n_mix > 1:
        from concurrent.futures import ThreadPoolExecutor

        from .contract import _mix

        share = int(token_budget) // n_mix
        budgets = [int(token_budget) - share * (n_mix - 1)] + [share] * (n_mix - 1)

        def _child(i: int, b: int) -> RoundStream:
            return open_round_stream(
                mode, repo_dir, _mix(generation_seed, "seed-mix", i), cfg,
                token_budget=b, use_sandbox=use_sandbox, blocked=tuple(blocked),
                allow_netns=allow_netns, max_wall_seconds=max_wall_seconds,
                budget_denomination=budget_denomination, corpus_epochs=epochs,
            )

        # cache_reuse children materialise their corpus in the constructor:
        # build them CONCURRENTLY (each is its own sandboxed process), in a
        # fixed list order, so a prefill costs one child's wall, not N.
        with ThreadPoolExecutor(max_workers=n_mix) as ex:
            futs = [ex.submit(_child, i, b) for i, b in enumerate(budgets)]
        children, first_err = [], None
        for f in futs:
            try:
                children.append(f.result())
            except Exception as e:  # noqa: BLE001 - re-raised after cleanup
                first_err = first_err or e
        if first_err is not None:
            # one child failed to build: never leak its siblings' processes
            for c in children:
                with contextlib.suppress(Exception):   # the build error wins
                    c.close()
            raise first_err
        return _SeedMixStream(children, budget_denomination=budget_denomination)
    if epochs:
        # DEC-CA-0050 generation surface: the unique corpus is exactly
        # budget / epochs points — materialised before step 0 (the clock
        # starts at the first batch), then cycled at most `epochs` times.
        import dataclasses

        cfg = dataclasses.replace(
            cfg, corpus_target_points=int(math.ceil(int(token_budget) / epochs)))
    if mode == "cache_reuse":
        return _CacheReuseStream(
            repo_dir, generation_seed, cfg, token_budget,
            use_sandbox=use_sandbox, blocked=tuple(blocked), allow_netns=allow_netns,
            budget_denomination=budget_denomination,
        )
    if mode in ("stream_cpu", "stream_gpu"):
        return _FreshSeriesStream(
            repo_dir, generation_seed, cfg, token_budget,
            use_sandbox=use_sandbox, blocked=tuple(blocked), allow_netns=allow_netns,
            gpu=(mode == "stream_gpu"), max_wall_seconds=max_wall_seconds,
            budget_denomination=budget_denomination,
        )
    raise CorpusError(f"unknown corpus_mode={mode!r}")
