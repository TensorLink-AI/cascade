"""Build a training corpus by running a miner's generator.

Given a materialised generator repo, import ``generator.Generator``, construct
it with the round's generation seed, and drain the corpus: exactly
``corpus_n_series`` validated series, or — with ``corpus_target_points`` armed
(DEC-CA-0031) — a free number of series up to that many points. The result is a
list of float64 arrays plus its digest — the auditable record of what the model
was trained on.

Isolation boundary: the generator is miner-controlled code. :func:`build_corpus`
runs it IN-PROCESS — fine for tests, ``cascade verify``, and trusted offline
smoke. In production the trainer runs this same path inside the network-isolated,
rlimited subprocess in :mod:`cascade.trainer.sandbox` (:func:`build_round_corpus`
with ``use_sandbox=True``, the default).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..interface.generator import (
    CAST_SAFE_MAX_FLOAT32,
    SUPPORTED_INTERFACE_VERSION,
    DataGenerator,
    drain_generator,
)
from ..interface.validation import check_repo_size
from ..shared.config import GeneratorConfig
from ..shared.manifest import corpus_digest


@dataclass(frozen=True)
class CorpusResult:
    # Elements are canonical (C, L) float64 arrays, or — only when [training]
    # accepted_fields is armed and a series carries an extra — extended-record
    # dicts {"values": (C, L) f64, "mask": (C, L) u8, "roles": (C,) u8}.
    series: list[np.ndarray | dict[str, np.ndarray]]
    digest: str
    n_series: int
    total_points: int


def _values_of(element: np.ndarray | dict) -> np.ndarray:
    return element["values"] if isinstance(element, dict) else element


def _corr_gate(cfg: GeneratorConfig):
    from .channel_stats import corr_enforce_gate

    return corr_enforce_gate(cfg)


class CorpusError(RuntimeError):
    """Importing or running the generator failed, or its output was rejected."""


# ─────────────── shared real corpus (DEC-CA-0028; inert until armed) ─────────

# Override for the machine-local materialisation cache of the owner-published
# shared real corpus. Default keeps one digest-keyed copy per machine, shared
# by trainer, audit, and `cascade verify` alike.
REAL_CORPUS_CACHE_ENV = "CASCADE_REAL_CORPUS_CACHE"

# Completion marker inside a materialised corpus directory: fetch_from_hub
# clears the destination before writing, so a directory that carries the
# marker was fully fetched (partial fetches never gain one).
_REAL_CORPUS_MARKER = ".cascade_real_corpus_ok"


def real_corpus_cache_root() -> Path:
    env = os.environ.get(REAL_CORPUS_CACHE_ENV, "")
    return Path(env) if env else Path.home() / ".cache" / "cascade" / "real-corpus"


def resolve_real_corpus(cfg: GeneratorConfig, *, fetch=None) -> GeneratorConfig:
    """Return ``cfg`` with ``real_corpus_dir`` resolved, or unchanged when unarmed.

    The PARENT side of every sandbox calls this before spawning (children are
    network-isolated and cannot fetch); the in-process paths call it too, so
    a run can never silently proceed without the pinned corpus while
    ``real_corpus_ref`` is armed. Resolution is: an already-set dir is only
    verified to exist; otherwise the ref is materialised once per digest into
    the machine-local cache (fetched into a private temp dir, atomically
    renamed into place, so concurrent lanes race safely). ``fetch`` is the
    injectable fetcher for tests; the default is the digest-pinned
    :func:`cascade.shared.hippius.fetch_from_hub`.
    """
    if not cfg.real_corpus_ref:
        return cfg
    if cfg.real_corpus_dir:
        if not Path(cfg.real_corpus_dir).is_dir():
            raise CorpusError(
                f"real_corpus_missing: real_corpus_dir={cfg.real_corpus_dir!r} does "
                "not exist (the parent must materialise the pinned corpus before "
                "the sandbox runs)"
            )
        return cfg
    from dataclasses import replace

    digest_key = cfg.real_corpus_ref.rsplit("@", 1)[-1].replace(":", "-")
    dest = real_corpus_cache_root() / digest_key
    if not (dest / _REAL_CORPUS_MARKER).is_file():
        if fetch is None:
            from ..shared.hippius import fetch_from_hub as fetch
        tmp = dest.parent / f".fetch-{os.getpid()}-{digest_key}"
        try:
            fetch(cfg.real_corpus_ref, tmp)
        except Exception as e:  # noqa: BLE001 — a pinned-but-unfetchable corpus is loud
            raise CorpusError(
                f"real_corpus_unfetchable: {cfg.real_corpus_ref} "
                f"({type(e).__name__}: {e})"
            ) from e
        (tmp / _REAL_CORPUS_MARKER).touch()
        try:
            tmp.rename(dest)
        except OSError:
            # Another process won the race; its rename only happens after a
            # complete fetch, so the marker must be there.
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            if not (dest / _REAL_CORPUS_MARKER).is_file():
                raise CorpusError(
                    f"real_corpus_cache_corrupt: {dest} exists without its "
                    "completion marker — remove it and retry"
                ) from None
    return replace(cfg, real_corpus_dir=str(dest))


def _check_declared_interface(
    repo_dir: Path, supported: int = SUPPORTED_INTERFACE_VERSION
) -> None:
    """Reject a repo declaring an interface_version newer than ``supported``.

    DEC-CA-0020 layering: a generator MAY declare ``"interface_version"`` in
    its config.json (absent ⇒ 1, the deployed fleet). A declared version above
    the bar fails HERE, before any miner code runs, with an error naming the
    mismatch — instead of a confusing mid-drain rejection of a record field
    the miner believed was accepted. An unparseable config.json is ignored
    (the generator itself owns that file's schema).

    ``supported`` is the ``[generator] interface_version`` config value
    (callers pass ``cfg.interface_version``), capped at what this CODE
    implements — config can hold the bar BELOW the code's ceiling during a
    staged rollout, never raise it above (accepting a declaration the code
    cannot honour would be a lie to the miner).
    """
    supported = min(int(supported), SUPPORTED_INTERFACE_VERSION)
    cfg_p = repo_dir / "config.json"
    if not cfg_p.is_file():
        return
    try:
        declared = json.loads(cfg_p.read_text(encoding="utf-8")).get("interface_version")
    except (OSError, ValueError):
        return
    if declared is None:
        return
    try:
        version = int(declared)
    except (TypeError, ValueError):
        raise CorpusError(
            f"generator_interface_invalid: config.json interface_version={declared!r} "
            "is not an integer"
        ) from None
    if version > supported:
        raise CorpusError(
            f"generator_interface_too_new: config.json declares interface_version="
            f"{version}, but this trainer supports <= {supported}"
        )


def _ctor_accepts_real_corpus(generator_cls: type) -> bool:
    """Whether the submitted constructor declares the opt-in ``real_corpus_dir``
    keyword (named parameter or ``**kwargs``). Signature inspection, not a
    TypeError retry — a constructor that raises TypeError for its own reasons
    must stay a plain construct failure, never a silent no-corpus rerun."""
    import inspect

    try:
        params = inspect.signature(generator_cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "real_corpus_dir" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _load_generator(
    repo_dir: Path, generation_seed: int, *,
    interface_version: int = SUPPORTED_INTERFACE_VERSION,
    real_corpus_dir: str = "",
) -> DataGenerator:
    wrapper_py = repo_dir / "generator.py"
    if not wrapper_py.is_file():
        raise CorpusError("missing generator.py")
    _check_declared_interface(repo_dir, interface_version)
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
    # DEC-CA-0028 opt-in: the shared-corpus path is passed ONLY when armed AND
    # the constructor declares it — every deployed two-argument generator stays
    # valid under an armed config, and while unarmed the call is byte-identical
    # to the legacy form even for generators that declare the kwarg.
    kwargs: dict[str, str] = {}
    if real_corpus_dir and _ctor_accepts_real_corpus(Generator):
        kwargs["real_corpus_dir"] = str(real_corpus_dir)
    try:
        gen = Generator(str(repo_dir), seed=generation_seed, **kwargs)
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

    With ``cfg.corpus_target_points`` armed (DEC-CA-0031) the drain is
    points-denominated: the generator is asked for the same prefix upper bound
    the streaming feed would use (``target // min_length + 2``) and the drain
    stops once the corpus reaches the target points — count free, budget
    fixed. At the default 0 the legacy exactly-``corpus_n_series`` drain runs
    byte-identically.
    """
    size = check_repo_size(repo_dir, cfg.max_repo_mb)
    if not size.ok:
        raise CorpusError(f"submission_too_large: {size.details}")
    cfg = resolve_real_corpus(cfg)
    gen = _load_generator(
        Path(repo_dir), generation_seed, interface_version=cfg.interface_version,
        real_corpus_dir=cfg.real_corpus_dir,
    )
    target = int(cfg.corpus_target_points)
    n_request = (
        target // max(int(cfg.min_length), 1) + 2 if target > 0
        else cfg.corpus_n_series
    )
    try:
        series = drain_generator(
            gen,
            n_request,
            min_length=cfg.min_length,
            max_length=cfg.max_length,
            max_total_points=cfg.max_total_points,
            max_channels=cfg.max_channels,
            max_abs=cfg.max_abs_value or CAST_SAFE_MAX_FLOAT32,
            reject_constant=cfg.reject_constant,
            max_dup_fraction=cfg.max_dup_fraction,
            max_payload_bytes=cfg.max_payload_bytes,
            accepted_fields=tuple(cfg.accepted_fields),
            max_missing_frac=cfg.max_missing_frac,
            allow_future_known=cfg.allow_future_known,
            target_points=target,
            extra_series_check=_corr_gate(cfg),
        )
    except ValueError as e:
        raise CorpusError(f"generator_output_rejected: {e}") from e
    total = int(sum(int(_values_of(s).size) for s in series))
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
