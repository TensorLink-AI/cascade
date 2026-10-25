"""The MarketCast base-model interface (DEC-CA-0064).

MarketCast judges fine-tunes of a pinned third-party base,
``mistralai/Timestral-28M`` — a different architecture from the Cascade-MoE
king (never touches ``cascade/trainer/moe_model.py``). Its modelling code
comes from the separate ``timestral`` library (``github.com/mistralai/
timestral`` @ ``a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff``, version 1.0.0,
Apache-2.0 code; NOT on PyPI), an OPTIONAL extra (``pip install
'.[marketcast]'``) the main validator and the MarketCast training pods
need. Nothing here imports it at module load. This module ships:

* :class:`QuantileForecaster` — the Protocol every MarketCast scorer talks
  to (the base, a fine-tune, or a test stub);
* :func:`load_timestral` — the optional import, raising
  :class:`TimestralUnavailable` with a clear message when it is missing;
* :func:`sha256_file` / :func:`check_base_sha256` — verify a downloaded
  checkpoint against ``[marketcast] base_sha256`` before it is ever loaded;
* :func:`forecast_windows` — runs a forecaster over windows, checks the
  output shape and maps its knots onto the requested levels.

* :class:`TimestralQuantileForecaster` — the concrete :class:`QuantileForecaster`
  over the library's ``TimestralForecaster.forecast`` (context ``(B, V, T)``
  with NaN = missing, ``past_covariates`` ``(B, C, T)``, ``known_covariates``
  ``(B, C, T + H)``; horizons past one 1024-step block are decoded block by
  block by the library, feeding each block's median back; the library's
  defaults ``symmetric_averaging`` and ``clip_nonnegative`` stay ON for the
  base and every fine-tune alike);
* :func:`load_pinned_base` — resolve the pinned base (``<repo>@<commit>``),
  verify ``model.safetensors`` against ``base_sha256`` BEFORE loading, load;
* the fine-tune hook — :func:`select_trainable` (``full`` |
  ``last_n_blocks`` | ``frozen_embeddings`` | ``lora``; the library has no
  LoRA, so :class:`LoRALinear` wraps the transformer's ``nn.Linear`` layers
  here and :func:`merge_lora` folds it back), :func:`forward_block` (one
  differentiable decode block through the library's ``predict_block``) and
  :func:`export_checkpoint` (``config.json`` + ``model.safetensors`` that the
  library's strict loader reads back, + ``marketcast_meta.json``);
* :func:`safetensors_header` / :func:`safetensors_all_finite` /
  :func:`same_structure` — pure checks the trainer's harvest runs on a
  fine-tuned checkpoint (no torch).

The training path calls two PRIVATE methods of the library's forecaster
(``_as_context``, ``_assemble_rows`` — the covariate row assembly the public
``forecast`` uses); they exist at the pinned commit, which is why the pin is
a commit and not a range.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..shared.config.marketcast import (
    TIMESTRAL_KNOTS,
    TIMESTRAL_MAX_CONTEXT,
    TIMESTRAL_MAX_HORIZON,
)
from .marketcast import map_quantiles

TIMESTRAL_MODULE = "timestral"
# The library pin (github.com/mistralai/timestral; not on PyPI).
TIMESTRAL_LIB_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
TIMESTRAL_LIB_VERSION = "1.0.0"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"
META_FILE = "marketcast_meta.json"


class TimestralUnavailable(ImportError):
    """The ``timestral`` library is not installed (MarketCast is inert)."""


@runtime_checkable
class QuantileForecaster(Protocol):
    """Forecast marginal quantiles given a context (+ covariates).

    ``context`` is ``(T,)`` (NaN = missing); ``past_covariates`` ``(C_p, T)``
    aligned with it; ``future_covariates`` ``(C_f, T + horizon)`` known-future
    inputs (calendar / session flags, scheduled events). Returns
    ``(horizon, K)`` quantiles on :attr:`quantile_levels` (ascending)."""

    quantile_levels: tuple[float, ...]

    def forecast_quantiles(self, context: np.ndarray, horizon: int, *,
                           past_covariates: np.ndarray | None = None,
                           future_covariates: np.ndarray | None = None) -> np.ndarray: ...


def load_timestral():
    """Import the ``timestral`` library or raise :class:`TimestralUnavailable`."""
    try:
        return importlib.import_module(TIMESTRAL_MODULE)
    except ImportError as e:
        raise TimestralUnavailable(
            "MarketCast (DEC-CA-0064) needs the 'timestral' library to load the "
            f"Timestral-28M base; it is not installed — install the pinned extra "
            f"(pip install '.[marketcast]', timestral {TIMESTRAL_LIB_VERSION} @ "
            f"{TIMESTRAL_LIB_COMMIT[:12]})") from e


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_base_sha256(actual: str, expected: str) -> None:
    """Refuse a checkpoint whose sha256 is not ``[marketcast] base_sha256``.
    An empty ``expected`` (the inert placeholder) is refused too — an
    unpinned base is never loaded."""
    exp = str(expected or "").strip().lower()
    if not exp:
        raise ValueError("[marketcast] base_sha256 is unset: the base checkpoint is not pinned")
    if str(actual or "").strip().lower() != exp:
        raise ValueError(f"base checkpoint sha256 {actual} != pinned base_sha256 {exp}")


def verify_checkpoint(path: str | Path, expected: str) -> str:
    """:func:`sha256_file` + :func:`check_base_sha256`; returns the digest."""
    d = sha256_file(path)
    check_base_sha256(d, expected)
    return d


def forecast_windows(forecaster: QuantileForecaster, contexts: Sequence[np.ndarray],
                     horizon: int, *, levels: Sequence[float] = TIMESTRAL_KNOTS,
                     past_covariates: Sequence[np.ndarray | None] | None = None,
                     future_covariates: Sequence[np.ndarray | None] | None = None,
                     max_context: int = TIMESTRAL_MAX_CONTEXT) -> np.ndarray:
    """Forecast every window: ``(N, horizon, len(levels))``.

    Contexts longer than ``max_context`` keep their most recent points (the
    base's limit). ``horizon`` above :data:`TIMESTRAL_MAX_HORIZON` is refused:
    the base would autoregress on its median, so a long-horizon quantity is
    scored as its own coarser series instead (DEC-CA-0064)."""
    h = int(horizon)
    if not 1 <= h <= TIMESTRAL_MAX_HORIZON:
        raise ValueError(f"horizon {h} must be in [1, {TIMESTRAL_MAX_HORIZON}] (one pass)")
    src = tuple(float(x) for x in forecaster.quantile_levels)
    n = len(contexts)
    pc = list(past_covariates) if past_covariates is not None else [None] * n
    fc = list(future_covariates) if future_covariates is not None else [None] * n
    if len(pc) != n or len(fc) != n:
        raise ValueError("covariate lists must match the number of contexts")
    out = np.empty((n, h, len(levels)))
    for i, ctx in enumerate(contexts):
        c = np.asarray(ctx, dtype=np.float64)
        if c.ndim != 1 or c.size == 0:
            raise ValueError(f"context {i} must be a non-empty 1-D array")
        p, fu = pc[i], fc[i]
        if c.size > max_context:
            c = c[-max_context:]
            if p is not None:
                p = np.asarray(p)[..., -max_context:]
            if fu is not None:
                fu = np.asarray(fu)[..., -(max_context + h):]
        q = np.asarray(forecaster.forecast_quantiles(
            c, h, past_covariates=p, future_covariates=fu), dtype=np.float64)
        if q.shape != (h, len(src)):
            raise ValueError(f"forecaster returned {q.shape}; expected {(h, len(src))}")
        out[i] = map_quantiles(q, src, levels)
    return out


# ── the concrete forecaster ─────────────────────────────────────────────────


class TimestralQuantileForecaster:
    """:class:`QuantileForecaster` over a ``timestral.TimestralForecaster``
    (``inner``). ``forecast_quantiles`` takes one series; ``forecast_batch``
    a uniform-length batch (the eval path). Returns numpy float64."""

    def __init__(self, inner, *, symmetric_averaging: bool = True,
                 clip_nonnegative: bool = True) -> None:
        self.inner = inner
        self.quantile_levels = tuple(float(q) for q in inner.quantile_levels)
        self.symmetric_averaging = bool(symmetric_averaging)
        self.clip_nonnegative = bool(clip_nonnegative)

    @classmethod
    def from_dir(cls, path: str | Path, *, device: str = "cpu",
                 max_context: int | None = None) -> TimestralQuantileForecaster:
        """Load an exported checkpoint directory (``config.json`` +
        ``model.safetensors``) — the base after :func:`load_pinned_base`'s
        digest check, or a harvested fine-tune."""
        lib = load_timestral()
        inner = lib.TimestralForecaster.from_pretrained(
            str(path), device=device,
            **({"max_context": int(max_context)} if max_context else {}))
        return cls(inner)

    def forecast_batch(self, contexts: np.ndarray, horizon: int, *,
                       past_covariates: np.ndarray | None = None,
                       future_covariates: np.ndarray | None = None) -> np.ndarray:
        """``contexts`` ``(B, T)`` (NaN = missing) → ``(B, horizon, K)``."""
        c = np.asarray(contexts, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError(f"contexts must be (B, T); got {c.shape}")
        fc = self.inner.forecast(
            c, int(horizon),
            past_covariates=None if past_covariates is None else np.asarray(past_covariates),
            known_covariates=None if future_covariates is None else np.asarray(future_covariates),
            symmetric_averaging=self.symmetric_averaging, clip_nonnegative=self.clip_nonnegative)
        q = fc.quantiles
        q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
        return np.asarray(q, dtype=np.float64)[:, 0]          # (B, V=1, H, Q) → (B, H, Q)

    def forecast_quantiles(self, context: np.ndarray, horizon: int, *,
                           past_covariates: np.ndarray | None = None,
                           future_covariates: np.ndarray | None = None) -> np.ndarray:
        pc = None if past_covariates is None else np.asarray(past_covariates)[None]
        fc = None if future_covariates is None else np.asarray(future_covariates)[None]
        return self.forecast_batch(np.asarray(context, dtype=np.float64)[None], horizon,
                                   past_covariates=pc, future_covariates=fc)[0]


def load_pinned_base(base_ref: str, base_sha256: str, *, cache_dir: str | Path | None = None,
                     device: str = "cpu") -> tuple[TimestralQuantileForecaster, Path]:
    """Resolve the pinned base (``<repo>@<40-hex commit>``) to a local
    directory, verify its ``model.safetensors`` against ``base_sha256``
    BEFORE any load, then load it. Returns ``(forecaster, directory)``."""
    from ..shared.config.marketcast import split_base_ref

    repo, rev = split_base_ref(base_ref)
    if not rev:
        raise ValueError(f"[marketcast] base_ref {base_ref!r} is not pinned to a commit")
    check_base_sha256(base_sha256, base_sha256)          # refuses an unset pin
    load_timestral()
    from timestral.loading import resolve_checkpoint  # type: ignore[import-not-found]

    d = Path(resolve_checkpoint(repo, revision=rev, cache_dir=cache_dir))
    verify_checkpoint(d / WEIGHTS_FILE, base_sha256)
    return TimestralQuantileForecaster.from_dir(d, device=device), d


def batched_forecasts(forecaster, contexts: list[np.ndarray], horizon: int, *,
                      past: list | None = None, known: list | None = None,
                      batch: int = 64, levels=None) -> np.ndarray:
    """Forecast many contexts of possibly different lengths: each batch is
    left-padded with NaN (missing) to its longest context. Uses
    ``forecast_batch`` when the forecaster has one, else one call per
    window. ``(N, horizon, len(levels))`` mapped onto ``levels`` (default:
    the forecaster's own)."""
    src = tuple(float(x) for x in forecaster.quantile_levels)
    lv = src if levels is None else tuple(levels)
    n = len(contexts)
    out = np.empty((n, int(horizon), len(lv)))
    fb = getattr(forecaster, "forecast_batch", None)
    for i0 in range(0, n, max(1, int(batch))):
        idx = list(range(i0, min(n, i0 + int(batch))))
        if fb is None:
            for i in idx:
                q = forecaster.forecast_quantiles(
                    contexts[i], horizon,
                    past_covariates=None if past is None else past[i],
                    future_covariates=None if known is None else known[i])
                out[i] = map_quantiles(q, src, lv)
            continue
        T = max(int(np.asarray(contexts[i]).size) for i in idx)
        ctx = np.full((len(idx), T), np.nan)
        pc = kc = None
        if past is not None and past[idx[0]] is not None:
            pc = np.full((len(idx), np.asarray(past[idx[0]]).shape[0], T), np.nan)
        if known is not None and known[idx[0]] is not None:
            kc = np.full((len(idx), np.asarray(known[idx[0]]).shape[0], T + int(horizon)), np.nan)
        for j, i in enumerate(idx):
            c = np.asarray(contexts[i], dtype=np.float64)
            ctx[j, T - c.size:] = c
            if pc is not None:
                pc[j, :, T - c.size:] = np.asarray(past[i])
            if kc is not None:
                k = np.asarray(known[i])
                kc[j, :, kc.shape[-1] - k.shape[-1]:] = k
        q = fb(ctx, horizon, past_covariates=pc, future_covariates=kc)
        out[idx[0]:idx[-1] + 1] = map_quantiles(q, src, lv)
    return out


# ── safetensors checks (pure, no torch) ─────────────────────────────────────

_ST_DTYPES = {"F32": np.float32, "F64": np.float64, "F16": np.float16}


def safetensors_header(path: str | Path) -> dict:
    """The JSON header of a safetensors file (``{name: {dtype, shape,
    data_offsets}}``, ``__metadata__`` dropped). Raises ``ValueError``."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        n = int.from_bytes(raw, "little")
        if not 0 < n < 100 * 1024 * 1024:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        import json

        hdr = json.loads(fh.read(n).decode("utf-8"))
    if not isinstance(hdr, dict):
        raise ValueError(f"{path}: malformed safetensors header")
    hdr.pop("__metadata__", None)
    return hdr


def same_structure(a: str | Path, b: str | Path) -> list[str]:
    """Differences between two checkpoints' tensor names / shapes / dtypes
    (empty = the same architecture; a fine-tune must match its base)."""
    ha, hb = safetensors_header(a), safetensors_header(b)
    out = [f"missing {k}" for k in sorted(set(ha) - set(hb))]
    out += [f"unexpected {k}" for k in sorted(set(hb) - set(ha))]
    for k in sorted(set(ha) & set(hb)):
        if ha[k].get("shape") != hb[k].get("shape") or ha[k].get("dtype") != hb[k].get("dtype"):
            out.append(f"{k}: {hb[k].get('dtype')}{hb[k].get('shape')} != "
                       f"{ha[k].get('dtype')}{ha[k].get('shape')}")
    return out


def safetensors_all_finite(path: str | Path) -> bool:
    """Whether every tensor in the file is finite (F32 / F64 / F16 only;
    any other dtype raises ``ValueError``)."""
    hdr = safetensors_header(path)
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name, info in hdr.items():
        dt = _ST_DTYPES.get(str(info.get("dtype")))
        if dt is None:
            raise ValueError(f"{name}: dtype {info.get('dtype')} not checked")
        lo, hi = (int(x) for x in info["data_offsets"])
        arr = np.frombuffer(mm[base + lo:base + hi].tobytes(), dtype=dt)
        if not np.isfinite(arr).all():
            return False
    return True


# ── fine-tune hook (torch; the training leg + its tests) ────────────────────


def _lora_class():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """``base(x) + (alpha / r) · x Aᵀ Bᵀ`` with ``base`` frozen, ``A``
        Kaiming-uniform, ``B`` zero (so the wrapped layer starts exactly at
        the base). ``weight`` / ``bias`` proxy the base layer (the library
        reads ``.weight.dtype`` of some linears)."""

        def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
            super().__init__()
            if rank < 1:
                raise ValueError(f"lora rank must be >= 1; got {rank}")
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.rank = int(rank)
            self.scale = float(alpha if alpha is not None else rank) / float(rank)
            dev, dt = base.weight.device, base.weight.dtype
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale

        def merged(self) -> nn.Linear:
            with torch.no_grad():
                self.base.weight += self.scale * (self.lora_B @ self.lora_A)
            for p in self.base.parameters():
                p.requires_grad_(True)
            return self.base

    return LoRALinear


def LoRALinear(base, rank: int, alpha: float | None = None):  # noqa: N802 — a factory
    """Wrap ``base`` (an ``nn.Linear``) in a LoRA adapter (torch imported
    lazily)."""
    return _lora_class()(base, rank, alpha)


def _is_lora(m) -> bool:
    return type(m).__name__ == "LoRALinear" and hasattr(m, "lora_A")


def select_trainable(model, scope: str, *, n_blocks: int = 0, lora_rank: int = 0) -> list:
    """Set ``requires_grad`` for a recipe's ``trainable`` scope on a
    Timestral-shaped model (``representation`` / ``transformer.layers`` /
    ``transformer.out_norm`` / ``head``) and return the trainable
    parameters:

    * ``full``              everything;
    * ``last_n_blocks``     the last ``n_blocks`` transformer layers + the
                            output norm + the quantile head;
    * ``frozen_embeddings`` everything but the patch embedding and the
                            forecast-query token (``representation``);
    * ``lora``              rank-``lora_rank`` adapters on every ``nn.Linear``
                            of the transformer layers (the library has no
                            LoRA — :class:`LoRALinear`); base weights frozen.
    """
    from torch import nn

    layers = model.transformer.layers
    for p in model.parameters():
        p.requires_grad_(scope in ("full", "frozen_embeddings"))
    if scope == "full":
        pass
    elif scope == "frozen_embeddings":
        for p in model.representation.parameters():
            p.requires_grad_(False)
    elif scope == "last_n_blocks":
        n = int(n_blocks)
        if not 1 <= n <= len(layers):
            raise ValueError(f"n_blocks {n} must be in [1, {len(layers)}]")
        for mod in (*list(layers)[-n:], model.transformer.out_norm, model.head):
            for p in mod.parameters():
                p.requires_grad_(True)
    elif scope == "lora":
        cls = _lora_class()
        for layer in layers:
            for _name, child in list(layer.named_modules()):
                for attr, sub in list(child.named_children()):
                    if isinstance(sub, nn.Linear) and not _is_lora(child):
                        setattr(child, attr, cls(sub, int(lora_rank)))
        if not any(_is_lora(m) for m in model.modules()):
            raise ValueError("lora: no nn.Linear found in the transformer layers")
    else:
        raise ValueError(f"unknown trainable scope {scope!r}")
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora(model) -> int:
    """Fold every :class:`LoRALinear` back into a plain ``nn.Linear`` (in
    place); returns how many were merged. After this the state dict has the
    base's exact key set."""
    n = 0
    for mod in list(model.modules()):
        for attr, sub in list(mod.named_children()):
            if _is_lora(sub):
                setattr(mod, attr, sub.merged())
                n += 1
    return n


def forward_block(forecaster_inner, contexts, horizon: int, *, past_covariates=None,
                  known_covariates=None):
    """One DIFFERENTIABLE decode block through the library: contexts
    ``(B, T)`` (NaN = missing; torch or numpy) → raw-space quantiles
    ``(B, horizon, K)`` (``horizon`` ≤ ``block_size · patch_size``). Uses the
    forecaster's ``_as_context`` / ``_assemble_rows`` (pinned private API)
    and the model's ``predict_block`` — the public ``forecast`` runs under
    ``torch.inference_mode`` and cannot train."""
    import math

    import torch
    from timestral.model import ROLE_TARGET

    fc = forecaster_inner
    ps = int(fc.config.patch_size)
    hp = math.ceil(int(horizon) / ps)
    if hp > int(fc.config.block_size):
        raise ValueError(f"horizon {horizon} exceeds one decode block "
                         f"({fc.config.block_size * ps} steps)")
    x, mask = fc._as_context(contexts, None)
    t = x.shape[-1]
    past = None if past_covariates is None else torch.as_tensor(
        past_covariates, dtype=torch.float32, device=x.device)
    known = None if known_covariates is None else torch.as_tensor(
        known_covariates, dtype=torch.float32, device=x.device)
    if past is not None and past.shape[-1] != t:
        raise ValueError("past covariates must cover the context")
    rows, m, role_ids, kf, kfm, n_tgt = fc._assemble_rows(x, mask, int(horizon), past, known)
    bb, v, tt = rows.shape
    group_ids = torch.arange(bb, device=rows.device).repeat_interleave(v)
    if kf is not None:
        kf, kfm = kf[:, :hp * ps], kfm[:, :hp * ps]
    q = fc.model.predict_block(rows.reshape(bb * v, tt), m.reshape(bb * v, tt), group_ids, hp,
                               role_ids, kf, kfm)
    # Match the inference decoder's advancing frame beyond the first patch.
    # This private method is differentiable; forecast() itself is not.
    target_rows = None if role_ids is None else role_ids == ROLE_TARGET
    q = fc._refine_block(q, rows.reshape(bb * v, tt), m.reshape(bb * v, tt), target_rows)
    return q.reshape(bb, v, hp * ps, -1)[:, 0, :int(horizon)]


def export_checkpoint(model, out_dir: str | Path, *, base_dir: str | Path,
                      meta: dict) -> str:
    """Write a fine-tuned model as an exported Timestral checkpoint the
    library's strict loader reads: the BASE's ``config.json`` (the
    architecture is unchanged), ``model.safetensors`` (float32, LoRA merged,
    the base's exact key set) and ``marketcast_meta.json`` (recipe, base pin,
    covariates). Returns the sha256 of ``model.safetensors``."""
    import json
    import shutil

    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merge_lora(model)
    sd = {k: v.detach().to("cpu", torch.float32).contiguous()
          for k, v in model.state_dict().items()}
    save_file(sd, str(out / WEIGHTS_FILE))
    shutil.copyfile(Path(base_dir) / CONFIG_FILE, out / CONFIG_FILE)
    (out / META_FILE).write_text(json.dumps(meta, sort_keys=True, indent=1), encoding="utf-8")
    problems = same_structure(Path(base_dir) / WEIGHTS_FILE, out / WEIGHTS_FILE)
    if problems:
        raise ValueError(f"exported checkpoint differs from its base: {problems[:3]}")
    return sha256_file(out / WEIGHTS_FILE)


def read_meta(ckpt_dir: str | Path) -> dict:
    """The fine-tune's ``marketcast_meta.json`` (``{}`` when absent)."""
    import json

    p = Path(ckpt_dir) / META_FILE
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


__all__ = [
    "CONFIG_FILE", "META_FILE", "QuantileForecaster", "TIMESTRAL_LIB_COMMIT",
    "TIMESTRAL_LIB_VERSION", "TIMESTRAL_MODULE", "TimestralQuantileForecaster",
    "TimestralUnavailable", "WEIGHTS_FILE", "LoRALinear", "batched_forecasts",
    "check_base_sha256", "export_checkpoint", "forecast_windows", "forward_block",
    "load_pinned_base", "load_timestral", "merge_lora", "read_meta", "safetensors_all_finite",
    "safetensors_header", "same_structure", "select_trainable", "sha256_file",
    "verify_checkpoint",
]
