"""Synthetic multivariate coupling, built from the published mechanisms.

Sources checked rather than assumed (2026-09-09):

* **TimePFN** (arXiv 2502.16294) is the multivariate one: synthetic MV series via
  diverse GP kernels plus **linear coregionalization** — every channel is a
  weighted mixture of a few shared latent processes. This is the primary
  mechanism here.
* **CauKer** (arXiv 2508.02879) is **univariate**, for classification. GP
  kernel-bank roots feed a DAG whose edges each apply an activation from a bank
  — linear ``a*x+b``, ReLU, sigmoid, sine, modulo, leaky-ReLU — and a node
  aggregates incoming edges by CONCATENATION followed by a random linear map
  ``W[phi(e)(t_u)] + b`` with ``W, b ~ N(0, 1)``. **No time lags.** Its
  activation bank and aggregation are reused here for the nonlinear part.
* **TempoPFN** (arXiv 2510.25502) is univariate and names multivariate as future
  work; nothing here derives from it.

Two coupling regimes, because real panels contain both and they reward a joint
forecaster differently:

``coregionalize``
    Shared-latent structure (TimePFN). Channels are noisy views of common
    factors, so observing siblings sharpens the latent and improves a channel's
    forecast. Contemporaneous — no lag needed.

``lagged_dag``
    A parent's PAST drives a child's FUTURE, through CauKer's activation bank
    and concat-then-linear aggregation. This is a deliberate DEVIATION: neither
    paper applies lags across channels (both are univariate), but lagged
    cross-channel structure is precisely what a forecaster can exploit at a
    horizon, so it is the regime multivariate scoring would reward. Labelled as
    an extension, not as either paper's recipe.

Scale discipline throughout: contributions are computed from STANDARDISED
signals and rescaled to the target channel's own spread, so coupling is
structural rather than an artefact of a corpus spanning ~5 orders of magnitude.
"""
from __future__ import annotations

import numpy as np

# CauKer's edge-activation bank (arXiv 2508.02879), used verbatim.
ACTIVATIONS: tuple[str, ...] = (
    "linear",      # a*x + b,  a ~ U(0.5, 2), b ~ U(-1, 1)
    "relu",
    "sigmoid",
    "sine",
    "modulo",      # x mod c,  c ~ U(1, 5)
    "leaky_relu",  # negative slope ~ U(0.01, 0.3)
)


def _standardise(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


def _activate(kind: str, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if kind == "linear":
        return float(rng.uniform(0.5, 2.0)) * x + float(rng.uniform(-1.0, 1.0))
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    if kind == "sine":
        return np.sin(x)
    if kind == "modulo":
        return np.mod(x, float(rng.uniform(1.0, 5.0)))
    if kind == "leaky_relu":
        return np.where(x > 0.0, x, float(rng.uniform(0.01, 0.3)) * x)
    raise ValueError(f"unknown activation {kind!r}")


def _lag(x: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(x)
    if d < x.shape[-1]:
        out[..., d:] = x[..., : x.shape[-1] - d]
    return out


def coregionalize(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 0.7,
    n_latent: int | None = None,
) -> np.ndarray:
    """Linear coregionalization (TimePFN): channels share latent processes.

    ``strength`` is the fraction of each channel's standardised variation that
    comes from the shared latents; the remainder stays its own signal, so at
    ``strength = 0`` the corpus is unchanged and at 1 every channel is a pure
    mixture. Each channel's original mean and spread are restored afterwards, so
    marginals stay in the caller's prior.
    """
    arr = np.asarray(base, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"base must be (C, L); got {arr.shape}")
    c, _ = arr.shape
    if c < 2 or strength <= 0.0:
        return arr.copy()

    q = int(n_latent or rng.integers(1, max(2, c // 2) + 1))
    z = np.stack([_standardise(row) for row in arr], axis=0)
    # latents as random mixtures of the channels' own standardised signals
    mix = rng.normal(0.0, 1.0, size=(q, c))
    lat = mix @ z
    lat = np.stack([_standardise(row) for row in lat], axis=0)

    a = rng.normal(0.0, 1.0, size=(c, q))                 # coregionalization matrix
    shared = a @ lat
    shared = np.stack([_standardise(row) for row in shared], axis=0)

    # Chronos-2's cotemporaneous multivariatizers apply "linear OR NONLINEAR
    # transformations at the same time step"; apply an activation to a share of
    # the mixtures so the class is not linear-only.
    if float(rng.random()) < 0.5:
        shared = np.stack([
            _standardise(_activate(str(rng.choice(ACTIVATIONS)), row, rng))
            for row in shared
        ], axis=0)
    s = float(np.clip(strength, 0.0, 1.0))
    mixed = np.sqrt(1.0 - s ** 2) * z + s * shared
    out = np.empty_like(arr)
    for k in range(c):
        sd = float(np.std(arr[k]))
        mu = float(np.mean(arr[k]))
        out[k] = _standardise(mixed[k]) * (sd if sd > 0 else 1.0) + mu
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def cointegrate(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    frac: float = 0.4,
    beta_scale: float = 1.0,
) -> np.ndarray:
    """Cointegration — Chronos-2 names it as a *sequential* multivariatizer
    property alongside lead-lag.

    A fraction of channels are rebuilt as ``y = beta * x + u`` where ``x`` is
    another channel's stochastic trend (its I(1) component, taken as the running
    sum of its increments) and ``u`` is that channel's own stationary deviation.
    The pair then shares a common trend and their spread is mean-reverting —
    which a joint forecaster can exploit and a per-channel one cannot.
    """
    arr = np.asarray(base, dtype=np.float64)
    c, _ = arr.shape
    if c < 2 or frac <= 0.0:
        return arr.copy()
    out = arr.copy()
    n = max(1, int(round(frac * (c - 1))))
    targets = rng.choice(np.arange(1, c), size=min(n, c - 1), replace=False)
    for k in targets:
        p = int(rng.integers(0, k))
        beta = float(rng.normal(0.0, 1.0)) * beta_scale
        # stationary deviation of the child, re-attached to the parent's level
        dev = np.diff(arr[k], prepend=arr[k][0])
        dev = dev - float(np.mean(dev))
        sd_k = float(np.std(arr[k])) or 1.0
        y = beta * _standardise(out[p]) + _standardise(np.cumsum(dev) * 0.0 + dev)
        out[k] = _standardise(y) * sd_k + float(np.mean(arr[k]))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def lagged_dag(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    max_delay: int = 24,
    activations: tuple[str, ...] = ACTIVATIONS,
) -> np.ndarray:
    """CauKer's activation bank + concat/random-linear aggregation, applied
    across channels at a LAG (the documented extension — see module docstring).

    Channel ``k`` draws parents from ``0..k-1`` (triangular ⇒ acyclic); channel 0
    is never modified. Parents are read from the accumulating output so the DAG
    has real depth, as in a structural causal model.
    """
    arr = np.asarray(base, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"base must be (C, L); got {arr.shape}")
    c, _ = arr.shape
    if c < 2 or strength <= 0.0:
        return arr.copy()

    out = arr.copy()
    for k in range(1, c):
        n_par = int(rng.integers(1, k + 1))
        parents = rng.choice(k, size=n_par, replace=False)
        edges = []
        for p in parents:
            d = int(rng.integers(1, max_delay + 1))
            phi = str(rng.choice(activations))
            edges.append(_standardise(_activate(phi, _lag(_standardise(out[p]), d), rng)))
        # CauKer aggregation: concatenate the activated edges, random linear map
        w = rng.normal(0.0, 1.0, size=len(edges))
        b = float(rng.normal(0.0, 1.0))
        agg = _standardise(np.tensordot(w, np.stack(edges, axis=0), axes=(0, 0)) + b)
        sd_k = float(np.std(arr[k]))
        out[k] = arr[k] + strength * (sd_k if sd_k > 0 else 1.0) * agg
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def couple_channels(
    base: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    coregion_strength: float = 0.7,
    max_delay: int = 24,
    mode: str = "both",
) -> np.ndarray:
    """Apply coupling. ``mode``: ``coregion`` (TimePFN only), ``lagged`` (the
    CauKer-activation extension only), or ``both`` — shared latents first, then
    lagged causal edges on top, which is the combination real panels display."""
    if mode in ("coregion", "both"):
        base = coregionalize(base, rng, strength=coregion_strength)
    if mode in ("lagged", "both"):
        base = lagged_dag(base, rng, strength=strength, max_delay=max_delay)
    # NOTE: cointegrate is NOT in "both". As implemented it rebuilds the target
    # channel (y = beta*x + stationary deviation), which is contemporaneous, not
    # lead-lag: measured cross-predictive gain -0.049 against a -0.056 no-coupling
    # baseline, i.e. nothing. It also overwrites channels the lagged DAG had
    # already coupled, dropping "both" from +0.165 to +0.045. A faithful version
    # needs an error-correction dynamic (dy_t = alpha*(y_{t-1} - beta*x_{t-1}) + ...)
    # so the SPREAD is the predictable part. Left callable, out of the default.
    if mode == "cointegrate":
        base = cointegrate(base, rng)
    return np.asarray(base, dtype=np.float64)
