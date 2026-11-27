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

Pure numpy at import time. torch is imported lazily (the wrapper already
holds it), so the core repo's tests can load this file by path.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

log = logging.getLogger("cascade_benchmark.observed")

# Recorded in every report's ``methodology`` (results.METHODOLOGY).
CONVENTION = "gluonts-observed/1"
# ``bench_method``: stamped on every report (results.BenchmarkReport) and,
# cascade-side, on every published rows suite, so an auditor reproduces a
# report with the method it was swept with. LEGACY_METHOD is the pre-#972
# behaviour (the pinned wrapper as loaded + ``impute_history``); a report
# without a stamp was swept with it.
METHOD = "observed_mask/1"
LEGACY_METHOD = "impute/0"
METHODS = (METHOD, LEGACY_METHOD)
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


class ObservedMaskWrapper:
    """The loaded ``pyramid_wrapper.Wrapper`` with the GluonTS observed mask.
    It has the same public surface (``quantile_levels``,
    ``forecast_quantiles_batch``, ``forecast_quantiles``, ``forecast_joint``,
    ``forecast``). It advertises ``handles_missing`` so the suites hand it
    the raw history (no ``impute_history``)."""

    handles_missing = True
    convention = CONVENTION

    def __init__(self, inner) -> None:
        if not is_pyramid_wrapper(inner):
            raise TypeError("ObservedMaskWrapper needs the cascade_pyramid wrapper")
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

    def _parameters(self, context, observed, horizon: int):
        """``pyramid_wrapper.Wrapper._parameters`` with unobserved context
        points masked as well as the horizon. ``context``/``observed`` are
        ``(B, C, L)``."""
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
        params = self.model(z, mask=mask.reshape(b, c, -1, ps), with_parameters=True)
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
            params, loc, scale = self._parameters(context[:, None], observed[:, None], int(horizon))
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
            params, loc, scale = self._parameters(context[None], observed[None], int(horizon))
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
    ``cascade_pyramid`` wrapper gets the observed mask. Under
    :data:`LEGACY_METHOD`, and for any other architecture, the wrapper is
    returned unchanged."""
    method = current_method() if method is None else method
    if method not in METHODS:
        raise ValueError(f"bench method {method!r}: expected one of {METHODS}")
    if method == LEGACY_METHOD:
        return wrapper
    return ObservedMaskWrapper(wrapper) if is_pyramid_wrapper(wrapper) else wrapper
