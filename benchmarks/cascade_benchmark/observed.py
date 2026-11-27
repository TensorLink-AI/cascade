"""GIFT-Eval / GluonTS context convention for ``cascade_pyramid`` checkpoints.

The reference (GIFT-Eval's harness on gluonts 0.15.1, the pin in this
sidecar's ``uv.lock``):

* ``gift_eval.data.Dataset`` hands the raw target to the model's predictor.
  NaN stays NaN: it does not fill, pad or drop anything.
* The predictor's input transform (``AddObservedValuesIndicator``,
  gluonts ``transform/feature.py``) sets ``observed_values`` to 1 where the
  target is observed and 0 where it is NaN. It replaces the NaN with
  ``DummyValueImputation(0.0)``.
* ``InstanceSplitter`` (gluonts ``transform/split.py``) LEFT-pads a context
  shorter than ``past_length`` with ``dummy_value = 0.0``, in the target and
  in every split time-series field. So a padded step's observed indicator is
  0 too. It also sets ``past_is_pad = 1`` there.
* The model reads ``past_target`` with its ``past_observed_values`` mask and
  ``past_is_pad`` (Moirai, from the GIFT-Eval authors: ``past_observed_target``
  plus ``past_is_pad``). Padding and missing points are unobserved and never
  enter the scaler.

This module applies that convention to the release's byte-pinned
``pyramid_wrapper`` WITHOUT editing it. A checkpoint's
``forecast_wrapper.py`` must be byte-identical to the release's copy
(``cascade.eval.checkpoint_guard``), and the validator's private evaluation
(lane A) runs that file. Editing it would refuse every existing checkpoint
and would change consensus scoring. :class:`ObservedMaskWrapper` takes the
loaded wrapper and replaces its private ``_prep`` / ``_parameters`` step:

* unobserved = left padding | non-finite history value | forecast horizon;
* unobserved values hold :data:`FILL_VALUE`, 0.0 as in gluonts. The value
  is inert: ``causal_standardize`` weights every moment by the observed
  mask, and the stem reads ``values * (1 - mask)``;
* the scaler (``causal_standardize``) counts observed points only;
* a patch (the wrapper's grid, aligned to the context's right edge) whose
  real points are MORE than half non-finite is unobserved as a whole.
  Training only ever masks whole patches (CPM spans,
  ``pyramid_trainer.prediction``), so this is the closest match to it.
  Scattered NaN (most of GIFT-Eval's gaps are isolated points) stays per
  point, as in GluonTS / Moirai, because masking its patch would drop up to
  31 observed points per NaN. Left padding never triggers the rule: a very
  short series (M4 yearly) keeps all its points;
* NaN / inf never raises. A history with fewer than :data:`MIN_OBS`
  observed points (all-missing included) is still forecast, as GIFT-Eval
  requires one forecast per series. The scaler's own fallback handles it
  (all-missing: loc 0, scale 1). Each such history is COUNTED and logged.

A full-length, all-finite history produces exactly what the wrapper
produces: the mask is the same, so the forward pass is the same.

Leading padding is never an attention key (``observed_mask/2``, owner
2026-11-27; TimesFM / Timestral / PatchTST-FM do the same). A patch that is
ENTIRELY left padding (no real point; it sits before the first real point)
is removed from the keys every other token attends to, on the time and the
variate axis. Each padding token still attends to itself, so no softmax row
is empty. Gaps inside the series (NaN, mostly-missing patches) and the
forecast horizon are unobserved but stay keys, exactly as before: the
padding mask is kept apart from the observed mask (:func:`leading_pad_points`
/ :func:`leading_pad_patches`). A batch without leading padding runs the
checkpoint's model untouched (bit-identical).

Every checkpoint carries its own byte-pinned ``model.py`` (the
``base_arch_digest`` and ``checkpoint_guard`` pin those bytes), so the rule is
applied here, from outside: :class:`LeadingPadKeys` replaces the ``forward``
of each attention module INSTANCE of the loaded model (never a file, never
the class). It installs only on a ``_GatedAttention`` whose source hashes to
one of :data:`KNOWN_ATTENTION_SHA256` (every pyramid ``model.py`` released so
far) and refuses any other. A future ``model.py`` whose forward takes a
``pad_mask`` is called with it natively instead.

Pure numpy at import time. torch is imported lazily (the wrapper already
holds it), so the core repo's tests can load this file by path.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger("cascade_benchmark.observed")

# Recorded in every report's ``methodology`` (results.METHODOLOGY).
CONVENTION = "gluonts-observed/1"
# ``bench_method``: stamped on every report (results.BenchmarkReport) and,
# cascade-side, on every published rows suite, so an auditor reproduces a
# report with the method it was swept with. LEGACY_METHOD is the pre-#972
# behaviour (the pinned wrapper as loaded + ``impute_history``); a report
# without a stamp was swept with it. OBSERVED_MASK_V1 is #972's observed mask
# without the leading-pad attention rule (owner 2026-11-27): under METHOD a
# patch that is entirely left padding is never an attention KEY.
METHOD = "observed_mask/2"
OBSERVED_MASK_V1 = "observed_mask/1"
LEGACY_METHOD = "impute/0"
METHODS = (METHOD, OBSERVED_MASK_V1, LEGACY_METHOD)
# The CLI's ``--bench-method`` lands here, so spawned sweep workers inherit it.
METHOD_ENV = "CASCADE_BENCH_METHOD"
# gluonts DummyValueImputation(0.0) and InstanceSplitter(dummy_value=0.0).
FILL_VALUE = 0.0
# ``causal_standardize``'s ``min_obs``: fewer observed points than this and
# the scaler falls back. Counted, never silent.
MIN_OBS = 8
# A patch whose real (non-padding) points are more than this fraction
# non-finite is masked as a whole patch.
MOSTLY_MISSING = 0.5
# The module name pyramid_wrapper loads the checkpoint's model.py under.
PYRAMID_MODULE = "cascade_pyramid_checkpoint"

COUNT_KEYS = ("histories", "padded", "with_missing", "patches_masked", "below_min_obs",
              "all_missing")


def is_pyramid_wrapper(wrapper) -> bool:
    """True for the release's ``cascade_pyramid`` wrapper (its model module,
    scaler, model and config). Other architectures keep the sidecar's
    imputation path unchanged."""
    m = getattr(wrapper, "m", None)
    return (getattr(m, "__name__", "") == PYRAMID_MODULE
            and callable(getattr(m, "causal_standardize", None))
            and hasattr(wrapper, "model") and hasattr(wrapper, "cfg"))


def observed_context(histories, context_length: int, patch_size: int = 32):
    """``(values (B, L) float64, observed (B, L) bool, counts)`` for raw 1-D
    histories at context ``L``. The last ``L`` points are kept. A shorter
    history is left-padded with :data:`FILL_VALUE`, and its padded steps are
    unobserved. A non-finite value becomes :data:`FILL_VALUE` and is
    unobserved. So is every point of a patch whose real points are more than
    :data:`MOSTLY_MISSING` non-finite."""
    size = int(context_length)
    ps = int(patch_size)
    if size % ps:
        raise ValueError("context_length must be a whole number of patches")
    values = np.full((len(histories), size), FILL_VALUE, dtype=np.float64)
    observed = np.zeros((len(histories), size), dtype=bool)
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for i, history in enumerate(histories):
        row = np.asarray(history, dtype=np.float64).reshape(-1)[-size:] if size else np.zeros(0)
        n = row.size
        finite = np.isfinite(row)
        if n:
            values[i, size - n:] = np.where(finite, row, FILL_VALUE)
            observed[i, size - n:] = finite
        if n and not finite.all():
            real = np.zeros(size, dtype=bool)
            real[size - n:] = True
            n_real = real.reshape(-1, ps).sum(1)
            n_bad = (real & ~observed[i]).reshape(-1, ps).sum(1)
            mostly = (n_bad > MOSTLY_MISSING * n_real) & (n_bad < n_real)
            if mostly.any():
                observed[i].reshape(-1, ps)[mostly] = False
                values[i].reshape(-1, ps)[mostly] = FILL_VALUE
                counts["patches_masked"] += int(mostly.sum())
        n_obs = int(observed[i].sum())
        counts["histories"] += 1
        counts["padded"] += int(n < size)
        counts["with_missing"] += int(n_obs < n)
        counts["below_min_obs"] += int(n_obs < MIN_OBS)
        counts["all_missing"] += int(n_obs == 0)
    return values, observed, counts


# ── leading padding: never an attention key (observed_mask/2) ────────────────

# sha256 of the ``class _GatedAttention`` source (its ``ast`` segment) in every
# released cascade_pyramid model.py. The hook's arithmetic is that class's
# masked branch; any other source is refused, never guessed at.
KNOWN_ATTENTION_SHA256 = {
    # 8d484e21 / 4f100495 (2026-10-27): forward(h), no attn_mask argument
    "9750e8bd3c03d21646bddc02f3526bf09c85bf4f0736b7555930ef475276fc7d": "pre-groups",
    # 5f81fcf4 (groups): forward(h, attn_mask=None), the release's current copy
    "70bb8cb8306b38d5b41b59f5f1fb21d771a7cc0323dfe2a1369d094ef2eccf76": "groups",
}


def leading_pad_points(histories, context_length: int) -> np.ndarray:
    """``(B, L)`` bool: True on the left padding :func:`observed_context`
    adds to a history shorter than ``L`` (no real point there). A real
    non-finite value is missing, NOT padding: it is False here."""
    size = int(context_length)
    pad = np.zeros((len(histories), size), dtype=bool)
    for i, history in enumerate(histories):
        n = min(int(np.asarray(history).reshape(-1).size), size)
        pad[i, :size - n] = True
    return pad


def leading_pad_patches(pad_points, patch_size: int):
    """``(..., P)`` bool from point-level padding ``(..., P * patch_size)``
    (numpy or torch): True for a patch that is ENTIRELY padding and has no
    non-padding patch before it. A partly padded boundary patch, and any
    padding-marked patch after real data (a gap), is False."""
    ps = int(patch_size)
    if isinstance(pad_points, np.ndarray):
        full = pad_points.reshape(*pad_points.shape[:-1], -1, ps).all(-1)
        return np.cumprod(full, axis=-1).astype(bool)
    full = pad_points.reshape(*pad_points.shape[:-1], -1, ps).all(-1)
    return full.long().cumprod(-1).bool()


def attention_source_sha256(source: str) -> str | None:
    """sha256 of the ``class _GatedAttention`` segment of a model.py source
    text, or ``None`` when it has none."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_GatedAttention":
            seg = ast.get_source_segment(source, node) or ""
            return hashlib.sha256(seg.encode("utf-8")).hexdigest()
    return None


def time_key_mask(pad):
    """``(N, T)`` leading-pad patches → ``(N, 1, T, T)`` bool SDPA mask (True
    = may attend): causal, and a padding key only for its own query."""
    import torch

    t = pad.shape[-1]
    eye = torch.eye(t, dtype=torch.bool, device=pad.device)
    causal = torch.ones(t, t, dtype=torch.bool, device=pad.device).tril()
    return (causal & (~pad[:, None, :] | eye))[:, None]


def variate_key_mask(pad, attn_mask=None):
    """``(N, C)`` leading-pad tokens → ``(N, 1, C, C)`` bool SDPA mask: a
    padding key only for its own query, AND the caller's own (group) mask."""
    import torch

    c = pad.shape[-1]
    eye = torch.eye(c, dtype=torch.bool, device=pad.device)
    allowed = (~pad[:, None, :] | eye)[:, None]
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError("leading-pad keys: the variate attn_mask must be boolean")
        allowed = allowed & attn_mask
    return allowed


def _masked_attention(module, attn, h, mask):
    """The released ``_GatedAttention.forward`` masked branch (hash-checked
    source), for the pre-groups copy whose forward takes no ``attn_mask``."""
    import torch
    import torch.nn.functional as F

    B, T, _ = h.shape
    H, hd = attn.cfg.num_heads, attn.cfg.head_dim
    q, k, v = attn.qkv(h).view(B, T, 3, H, hd).unbind(dim=2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if attn.axis == "time":
        q, k = module._xpos(q, k, attn.inv_freq, attn.zeta)
    q = q * (F.softplus(attn.per_dim_scale) / math.log(2.0))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / hd)
    a = a.transpose(1, 2).reshape(B, T, H * hd)
    if attn.gate is not None:
        a = a * (2 * torch.sigmoid(attn.gate(h)))
    return attn.proj(a)


def takes_pad_mask(model) -> bool:
    """The checkpoint's model applies the rule itself (``forward(pad_mask=)``)."""
    try:
        return "pad_mask" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


class LeadingPadKeys:
    """The leading-pad key rule on a loaded pyramid model that predates it.

    Replaces ``forward`` on each attention module INSTANCE of ``model`` (the
    checkpoint's file and class are untouched). Outside :meth:`active`, or
    inside it with no padding, every replaced forward calls the original
    with the original arguments: bit-identical. Inside it, each time block's
    keys exclude the leading-pad patches (causal kept) and each variate
    block's exclude the leading-pad channels at that patch, self kept.
    Raises ``TypeError`` for an attention source it does not know."""

    ATTR = "_cascade_leading_pad_keys"

    @classmethod
    def install(cls, module, model) -> LeadingPadKeys:
        have = getattr(model, cls.ATTR, None)
        return have if have is not None else cls(module, model)

    def __init__(self, module, model) -> None:
        path = getattr(module, "__file__", None)
        try:
            source = Path(path).read_text(encoding="utf-8") if path else ""
        except OSError:
            source = ""
        digest = attention_source_sha256(source)
        if digest not in KNOWN_ATTENTION_SHA256:
            raise TypeError(f"leading-pad keys: unknown _GatedAttention source "
                            f"(sha256 {digest}); refusing to hook this checkpoint")
        attn_cls = getattr(module, "_GatedAttention", None)
        blocks = list(getattr(model, "blocks", None) or [])
        if not blocks or any(not isinstance(getattr(b, "attn", None), attn_cls)
                             or getattr(b.attn, "axis", None) not in ("time", "variate")
                             for b in blocks):
            raise TypeError("leading-pad keys: the model's blocks are not the released layout")
        self.module = module
        self._pad = None
        self._time = None
        self._variate = None
        for block in blocks:
            self._wrap(block.attn)
        setattr(model, self.ATTR, self)

    def _wrap(self, attn) -> None:
        orig = attn.forward
        native = "attn_mask" in inspect.signature(orig).parameters
        hook = self

        def forward(h, attn_mask=None):
            if hook._pad is None:
                return orig(h, attn_mask=attn_mask) if native else orig(h)
            mask = hook._mask(attn.axis, h, attn_mask)
            return orig(h, attn_mask=mask) if native else _masked_attention(
                hook.module, attn, h, mask)

        attn.forward = forward

    def _mask(self, axis, h, attn_mask):
        B, C, P = self._pad.shape
        if axis == "time":
            if tuple(h.shape[:2]) != (B * C, P) or attn_mask is not None:
                raise RuntimeError(f"leading-pad keys: time block input {tuple(h.shape)} "
                                   f"is not ({B * C}, {P}, d)")
            if self._time is None:
                self._time = time_key_mask(self._pad.reshape(B * C, P))
            return self._time
        if tuple(h.shape[:2]) != (B * P, C):
            raise RuntimeError(f"leading-pad keys: variate block input {tuple(h.shape)} "
                               f"is not ({B * P}, {C}, d)")
        if attn_mask is not None:
            return variate_key_mask(self._variate_pad(), attn_mask)
        if self._variate is None:
            self._variate = variate_key_mask(self._variate_pad())
        return self._variate

    def _variate_pad(self):
        B, C, P = self._pad.shape
        return self._pad.transpose(1, 2).reshape(B * P, C)

    @contextlib.contextmanager
    def active(self, pad_patches):
        """``pad_patches`` ``(B, C, P)`` bool, the model input's patch grid
        (horizon patches included, never padding). No True ⇒ untouched."""
        # leading only: a padding-marked patch after a real one is a gap, not padding
        pad = pad_patches.bool().long().cumprod(-1).bool()
        self._pad = pad if bool(pad.any()) else None
        self._time = self._variate = None
        try:
            yield
        finally:
            self._pad = self._time = self._variate = None


class ObservedMaskWrapper:
    """The loaded ``pyramid_wrapper.Wrapper`` with the GluonTS observed mask.
    It has the same public surface (``quantile_levels``,
    ``forecast_quantiles_batch``, ``forecast_quantiles``, ``forecast_joint``,
    ``forecast``). It advertises ``handles_missing`` so the suites hand it
    the raw history (no ``impute_history``). ``pad_keys`` (:data:`METHOD`)
    adds the leading-pad attention rule; ``observed_mask/1`` runs without it."""

    handles_missing = True
    convention = CONVENTION

    def __init__(self, inner, *, pad_keys: bool = True) -> None:
        if not is_pyramid_wrapper(inner):
            raise TypeError("ObservedMaskWrapper needs the cascade_pyramid wrapper")
        self.pad_keys = bool(pad_keys)
        # the checkpoint's own model applies it natively, else the hook (refuses
        # an attention source it does not know, before anything is forecast)
        self.native_pad = self.pad_keys and takes_pad_mask(inner.model)
        self.pad_hook = (LeadingPadKeys.install(inner.m, inner.model)
                         if self.pad_keys and not self.native_pad else None)
        self.inner = inner
        self.m = inner.m
        self.cfg = inner.cfg
        self.model = inner.model
        self.device = inner.device
        self.quantile_levels = list(inner.quantile_levels)
        self.counts = dict.fromkeys(COUNT_KEYS, 0)

    # ── context + mask ───────────────────────────────────────────────────────

    def _prep(self, histories):
        import torch

        values, observed, counts = observed_context(histories, self.cfg.context_length,
                                                    self.cfg.patch_size)
        for k, v in counts.items():
            self.counts[k] += v
        if counts["below_min_obs"]:
            log.warning("observed-mask: %d of %d histories have < %d observed points "
                        "(%d all-missing); forecast from the scaler fallback",
                        counts["below_min_obs"], counts["histories"], MIN_OBS,
                        counts["all_missing"])
        return (torch.as_tensor(values, device=self.device, dtype=torch.float64),
                torch.as_tensor(observed, device=self.device, dtype=torch.bool))

    def _parameters(self, context, observed, horizon: int, pad=None):
        """``pyramid_wrapper.Wrapper._parameters`` with unobserved context
        points masked as well as the horizon. ``context``/``observed`` are
        ``(B, C, L)``; ``pad`` ``(B, C, L)`` bool is the left padding
        (:func:`leading_pad_points`), kept apart from ``observed``: under
        ``pad_keys`` its whole patches are never attention keys."""
        import torch

        b, c, length = context.shape
        ps = self.cfg.patch_size
        count = (horizon + ps - 1) // ps
        x = torch.cat([context, context.new_zeros(b, c, count * ps)], -1)
        mask = torch.cat([~observed, observed.new_ones(b, c, count * ps)], -1)
        flat = x.reshape(b * c, -1)
        flat_mask = mask.reshape_as(flat)
        z, loc, scale = self.m.causal_standardize(flat, flat_mask)
        z = z.masked_fill(flat_mask, 0).float().reshape(b, c, -1, ps)
        patch_mask = mask.reshape(b, c, -1, ps)
        pad_patches = None
        if self.pad_keys and pad is not None:
            # horizon patches are never padding
            points = torch.cat([pad.to(device=x.device, dtype=torch.bool),
                                pad.new_zeros(b, c, count * ps, dtype=torch.bool,
                                              device=x.device)], -1)
            pad_patches = leading_pad_patches(points, ps) & patch_mask.all(-1)
            if not bool(pad_patches.any()):
                pad_patches = None
        if pad_patches is None:
            params = self.model(z, mask=patch_mask, with_parameters=True)
        elif self.native_pad:
            params = self.model(z, mask=patch_mask, with_parameters=True, pad_mask=pad_patches)
        else:
            with self.pad_hook.active(pad_patches):
                params = self.model(z, mask=patch_mask, with_parameters=True)
        start = length // ps - 1
        params = params[:, :, start:start + count]
        loc = loc.reshape(b, c, -1, ps)[:, :, start:start + count, -1]
        scale = scale.reshape(b, c, -1, ps)[:, :, start:start + count, -1]
        return params, loc.float(), scale.float()

    # ── forecasts (same contract as the wrapper) ─────────────────────────────

    def forecast_quantiles_batch(self, histories, horizon):
        import torch

        if int(horizon) < 1:
            raise ValueError("horizon must be positive")
        with torch.no_grad():
            context, observed = self._prep(histories)
            pad = torch.as_tensor(leading_pad_points(histories, self.cfg.context_length),
                                  device=self.device)
            params, loc, scale = self._parameters(context[:, None], observed[:, None],
                                                  int(horizon), pad=pad[:, None])
            q = self.m.quantiles(params) * scale[..., None, None] + loc[..., None, None]
        if not torch.isfinite(q).all():
            raise FloatingPointError("non-finite pyramid quantiles")
        out = q[:, 0].reshape(len(histories), -1, self.cfg.num_quantiles)[:, :horizon]
        return out.cpu().numpy().astype(np.float64)

    def forecast_quantiles(self, history, horizon):
        return self.forecast_quantiles_batch([history], horizon)

    def forecast_joint(self, history, horizon, num_samples):
        import torch

        history = np.atleast_2d(np.asarray(history, dtype=np.float64))
        if history.ndim != 2 or horizon < 1 or num_samples < 1:
            raise ValueError("expected C,L history and positive horizon/sample count")
        with torch.no_grad():
            context, observed = self._prep(list(history))
            pad = torch.as_tensor(leading_pad_points(list(history), self.cfg.context_length),
                                  device=self.device)
            params, loc, scale = self._parameters(context[None], observed[None], int(horizon),
                                                  pad=pad[None])
            # The wrapper's seed rule: the raw window bytes (NaN included).
            seed_bytes = (history.tobytes() + int(horizon).to_bytes(8, "big")
                          + int(num_samples).to_bytes(8, "big"))
            seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big") & ((1 << 63) - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            samples = self.m.sample_copula_paths(params, int(num_samples), generator=generator)
            samples = samples * scale[..., None, None] + loc[..., None, None]
            out = samples[0].transpose(1, 2).reshape(history.shape[0], num_samples, -1)[..., :horizon]
        if not torch.isfinite(out).all():
            raise FloatingPointError("non-finite pyramid forecast")
        return out.cpu().numpy().astype(np.float64)

    def forecast(self, history, horizon, num_samples):
        return self.forecast_joint(np.asarray(history).reshape(1, -1), horizon, num_samples)


def current_method() -> str:
    """This process's bench method (``$CASCADE_BENCH_METHOD``, default
    :data:`METHOD`). An unknown value raises: never a silent fallback."""
    import os

    method = os.environ.get(METHOD_ENV, "") or METHOD
    if method not in METHODS:
        raise ValueError(f"{METHOD_ENV}={method!r}: expected one of {METHODS}")
    return method


def adapt(wrapper, method: str | None = None):
    """The wrapper every suite forecasts with. Under :data:`METHOD` a
    ``cascade_pyramid`` wrapper gets the observed mask and the leading-pad
    attention rule; under :data:`OBSERVED_MASK_V1` the observed mask only.
    Under :data:`LEGACY_METHOD`, and for any other architecture, the wrapper
    is returned unchanged."""
    method = current_method() if method is None else method
    if method not in METHODS:
        raise ValueError(f"bench method {method!r}: expected one of {METHODS}")
    if method == LEGACY_METHOD:
        return wrapper
    if not is_pyramid_wrapper(wrapper):
        return wrapper
    return ObservedMaskWrapper(wrapper, pad_keys=method == METHOD)
