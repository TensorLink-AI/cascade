"""MarketCast training leg (DEC-CA-0064) — fine-tune the PINNED Timestral-28M
base from a miner's ``marketcast.json`` recipe (+ optional generator).

Runs on the miner's payer pod (worker ``--role mkt_finetune``, credential-
free: the market data is public, the base is a public Hub pin, the
checkpoint is left on disk for the orchestrator's harvest). Never the era
king, never lane A's lineage, never ``cascade/trainer/moe_model.py``.

Pipeline (:func:`run_mkt_leg`):

1. the recipe is re-validated with the shared rule
   (:func:`cascade.shared.marketcast_recipe.parse_marketcast_recipe`);
2. the base is resolved at its commit and its ``model.safetensors`` checked
   against ``[marketcast] base_sha256`` BEFORE it is loaded;
3. the TRAINING snapshot — the recipe's assets over ``[data.start,
   data.end]`` (which ends ≥ ``embargo_days`` before the submission) — is
   fetched from the classes' primary sources with the same adapters the eval
   snapshot uses, and digested (``data_digest``);
4. batches (:class:`MktSampler`, pure + seeded): per batch one suite (by
   ``objective.suite_weights``) and one horizon (``horizon_mix`` for the
   target suite), then per sample a class (``class_weights``), an asset, a
   random origin; with probability ``synthetic_fraction`` a sample is cut from
   the miner's generator series instead (the sandboxed ``generate()``
   contract, :func:`cascade.trainer.sandbox.stream_series`) — synthetic
   samples carry NaN (missing) covariate rows, unless the recipe names its
   generator's rows (``synthetic_channels``): then each ``(C, L)`` array is
   one sample whose past covariates are its own rows and whose known rows
   (calendar / session / events) come from its ``time`` row;
5. allow-listed augmentations (:func:`augment`, pure, seeded) per sample;
6. the fine-tune (:func:`finetune`): the recipe's trainable scope
   (:func:`cascade.eval.timestral_adapter.select_trainable`), AdamW at
   ``lr`` / ``weight_decay``, warmup + schedule, gradient clipping, optional
   EMA, ``steps`` × ``batch`` (bounded by ``[marketcast] max_steps`` /
   ``max_batch``) under ``[lane_c] leg_hours`` of wall clock; the loss is the
   knot-weighted pinball on each sample's scale-normalised target (+ the
   optional soft coverage penalty);
7. :func:`cascade.eval.timestral_adapter.export_checkpoint` writes the
   base's ``config.json`` + ``model.safetensors`` (+ ``marketcast_meta.json``)
   and returns its sha256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..eval.marketcast import MktWindow, window_geometry
from ..shared.config.marketcast import MarketCastConfig

log = logging.getLogger("cascade.trainer.marketcast")

MIN_CONTEXT = 64


class MktLegError(ValueError):
    """The MINER's recipe / generator made the leg impossible (miner fault)."""


class MktNonFiniteTraining(RuntimeError):
    """A non-finite training loss or gradient: OUR side (operator fault).

    The recipe's numbers are bounded when it is parsed (``optimizer.lr`` in
    ``(0, [marketcast] max_lr]``, ``clip``, ``steps``, ``batch``), the base
    and the adapter are ours, and the data is real bars + a sandboxed series
    stream. 2026-11-28: an exactly flat context NaN'd the refine frame's
    gradient in our adapter and was labelled the miner's. The worker exits
    1 (never the rc 3 "miner submission rejected"), and the trainer classes
    :data:`NONFINITE_TRAINING_MARKER` as an operator fault (requeued,
    unburned, make-good eligible) whatever the exit code."""


# The worker's message prefix the trainer's classifier keys on
# (cascade.trainer.faults.nonfinite_training).
NONFINITE_TRAINING_MARKER = "non-finite training"


class MktDataUnavailable(RuntimeError):
    """An upstream feed failed; retry the leg without charging a miner fault."""


# ── augmentations (pure, seeded) ─────────────────────────────────────────────

AUG_DEFAULTS: dict[str, dict[str, float]] = {
    "scale": {"sigma": 0.1},          # × exp(N(0, sigma)) on the whole sample
    "jitter": {"sigma": 0.01},        # + N(0, sigma · std(context)) on the context
    "crop": {"min_frac": 0.5},        # keep the last U[min_frac, 1] of the context
    "time_warp": {"width": 0.1},      # resample the context by a factor in [1 ± width]
    "mixup": {"alpha": 0.2},          # λ·x + (1−λ)·partner, λ ~ Beta(alpha, alpha)
}


def _p(params: dict, name: str, key: str) -> float:
    return float(dict(params).get(key, AUG_DEFAULTS[name][key]))


def augment(context: np.ndarray, target: np.ndarray, augs: Sequence, rng: np.random.Generator,
            partner: tuple[np.ndarray, np.ndarray] | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the recipe's augmentations (``(name, p, ((k, v), ...))``, in the
    recipe's order) to one sample. Every draw comes from ``rng``; NaN
    (missing) positions stay NaN. Unknown names raise ``ValueError``."""
    c = np.asarray(context, dtype=np.float64).copy()
    y = np.asarray(target, dtype=np.float64).copy()
    for name, p, params in augs:
        params = dict(params)
        if name not in AUG_DEFAULTS:
            raise ValueError(f"unknown augmentation {name!r}")
        if rng.random() >= float(p):
            continue
        if name == "scale":
            s = math.exp(rng.normal(0.0, _p(params, name, "sigma")))
            c, y = c * s, y * s
        elif name == "jitter":
            sd = float(np.nanstd(c)) if np.isfinite(c).any() else 0.0
            c = c + rng.normal(0.0, _p(params, name, "sigma") * sd, c.shape)
        elif name == "crop":
            lo = min(1.0, max(0.05, _p(params, name, "min_frac")))
            keep = max(MIN_CONTEXT, int(round(c.size * rng.uniform(lo, 1.0))))
            c = c[-keep:]
        elif name == "time_warp":
            w = min(0.9, abs(_p(params, name, "width")))
            fac = rng.uniform(1.0 - w, 1.0 + w)
            n = c.size
            src = np.linspace(0.0, n - 1.0, max(2, int(round(n * fac))))
            xs = np.arange(n, dtype=np.float64)
            ok = np.isfinite(c)
            if ok.sum() >= 2:
                warped = np.interp(src, xs[ok], c[ok])
                c = warped[-n:] if warped.size >= n else np.concatenate(
                    [np.full(n - warped.size, np.nan), warped])
        elif name == "mixup" and partner is not None:
            a = max(1e-3, _p(params, name, "alpha"))
            lam = float(rng.beta(a, a))
            pc, py = partner
            m = min(c.size, np.asarray(pc).size)
            c = np.concatenate([c[:-m], lam * c[-m:] + (1.0 - lam) * np.asarray(pc)[-m:]]) \
                if m < c.size else lam * c + (1.0 - lam) * np.asarray(pc)[-c.size:]
            y = lam * y + (1.0 - lam) * np.asarray(py)
    return c, y


# ── the sampler (pure, seeded) ───────────────────────────────────────────────


@dataclass
class Batch:
    suite: str
    horizon: str
    steps: int
    contexts: np.ndarray            # (B, T) NaN-left-padded
    targets: np.ndarray             # (B, steps)
    past: np.ndarray | None         # (B, C_p, T)
    known: np.ndarray | None        # (B, C_f, T + steps)
    synthetic: int
    freq_minutes: int = 0           # the series' step (the seasonal-naive season)


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = sorted(k for k, w in weights.items() if w > 0)
    w = np.array([weights[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


class MktSampler:
    """Training batches from a training snapshot (:class:`~cascade.pool.
    marketcast.MktSnapshot`) + an optional synthetic series iterator."""

    def __init__(self, recipe, mc: MarketCastConfig, snap, *, seed: int,
                 synthetic: Iterator[np.ndarray] | None = None) -> None:
        from ..pool.marketcast import asset_series

        self.r, self.mc, self.snap = recipe, mc, snap
        self.rng = np.random.default_rng(int(seed))
        self.synthetic = synthetic
        if recipe.synthetic_fraction > 0 and synthetic is None:
            raise MktLegError("synthetic_fraction > 0 but no generator series")
        freqs_ok = set(recipe.frequencies)
        self.suites: dict[str, float] = {}
        self.series: dict[tuple, np.ndarray] = {}
        self.geom: dict[tuple[str, str], tuple[int, int]] = {}
        assets = dict(recipe.assets)
        # owner 2026-11-16: a pure-synthetic recipe trains every weighted suite
        # on generator series alone (no real series needed)
        pure = float(recipe.synthetic_fraction) >= 1.0
        from ..shared.config.marketcast import SUITE_CLASSES, base_suite

        # the composite target (owner 2026-11-10) trains on every weighted
        # suite with the recipe's horizon_mix; ohlcv splits into one series
        # label per scored field (``ohlcv.<field>``), its weight shared
        composite = recipe.suite not in mc.horizons_map
        labels_of = getattr(mc, "series_labels", lambda s: (s,))
        for base, w in dict(recipe.suite_weights).items():
            if w <= 0 or base not in mc.horizons_map:
                continue
            labels = labels_of(base)
            freqs = [f for f in mc.frequencies_map[base] if f in freqs_ok]
            if base == recipe.suite or composite:
                hz = {h: x for h, x in dict(recipe.horizon_mix).items()
                      if h in mc.horizons_map[base]} or {h: 1.0 for h in mc.horizons_map[base]}
            else:
                hz = {h: 1.0 for h in mc.horizons_map[base]}
            for suite in labels:
                for h in hz:
                    try:
                        self.geom[(suite, h)] = window_geometry(freqs, h)
                    except ValueError:
                        continue
                have = False
                for cls, syms in assets.items():
                    classes = (mc.suite_classes(base_suite(suite))
                               if hasattr(mc, "suite_classes")
                               else SUITE_CLASSES.get(base_suite(suite), ()))
                    if cls not in classes:
                        continue
                    for a in syms:
                        if a not in snap.assets(cls):
                            continue
                        for (s2, _h), (f, _st) in self.geom.items():
                            if s2 != suite or (suite, cls, a, f) in self.series:
                                continue
                            x = asset_series(snap, cls, a, suite, f)
                            if x is not None and np.isfinite(x).sum() > MIN_CONTEXT:
                                self.series[(suite, cls, a, f)] = x
                                have = True
                if have or (pure and any(s2 == suite for s2, _h in self.geom)):
                    self.suites[suite] = float(w) / len(labels)
        # a horizon whose frequency has no real series (e.g. 15m in an
        # embargoed range: the training snapshot holds 1m + 1h only) trains
        # on generator series or not at all — never a batch that cannot be cut
        if self.synthetic is None:
            real = {(k[0], k[3]) for k in self.series}
            self.geom = {sh: g for sh, g in self.geom.items() if (sh[0], g[0]) in real}
            self.suites = {s: w for s, w in self.suites.items()
                           if any(s2 == s for s2, _h in self.geom)}
        if composite:
            if not self.suites:
                raise MktLegError("no training data for any suite of the composite target "
                                  "over the recipe's assets / frequencies / date range")
        elif not any(base_suite(s) == recipe.suite for s in self.suites):
            raise MktLegError(f"no training data for the target suite {recipe.suite!r} "
                              "over the recipe's assets / frequencies / date range")

    def _real(self, suite: str, f: int, steps: int):
        cw = {c: w for c, w in dict(self.r.class_weights).items()
              if any(k[0] == suite and k[1] == c and k[3] == f for k in self.series)}
        if not cw:
            return None
        cls = _pick(self.rng, cw)
        keys = sorted(k for k in self.series if k[:2] == (suite, cls) and k[3] == f)
        key = keys[int(self.rng.integers(len(keys)))]
        s = self.series[key]
        ctx = int(self.r.context_length)
        for _ in range(16):
            k = int(self.rng.integers(MIN_CONTEXT, s.size - steps + 1))
            c, y = s[max(0, k - ctx):k], s[k:k + steps]
            if np.isfinite(c).sum() >= MIN_CONTEXT and np.isfinite(y).any():
                return key, k, c, y
        return None

    # generator arrays held back for another bar size (per size, bounded)
    SYN_HOLD = 64
    # arrays drawn looking for one that fits a batch's bar size
    SYN_PULLS = 32

    def _syn_arrays(self) -> list[tuple[object, int | None]]:
        """The generator's next yield as ``[(array, bar minutes)]``. Without
        ``synthetic_channels`` that is the raw array, untagged (each row a
        target series). With them a ``(C, L)`` yield is cut into consecutive
        groups of ``len(synthetic_channels)`` rows — the sandbox fixes ``C``
        (``[generator] min_channels`` / ``max_channels``, 32 on mainnet), so
        one yield carries ``C // n`` samples and its last ``C % n`` rows are
        ignored — each ``{name: row}``, tagged by its ``time`` row when that
        is a clock (finite, a constant whole-minute step), else untagged with
        the row dropped (known rows missing)."""
        arr = np.asarray(next(self.synthetic), dtype=np.float64)
        chans = tuple(self.r.synthetic_channels)
        if not chans:
            return [(arr, None)]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n = len(chans)
        if arr.ndim != 2 or arr.shape[0] < n:
            raise MktLegError(f"generator series has shape {arr.shape}; synthetic_channels "
                              f"names {n} rows {list(chans)}")
        out = []
        for g in range(arr.shape[0] // n):
            named = dict(zip(chans, arr[g * n:(g + 1) * n], strict=True))
            out.append((named, self._clock_bar(named)))
        return out

    @staticmethod
    def _clock_bar(named: dict) -> int | None:
        t = named.get("time")
        if t is None:
            return None
        d = np.diff(t)
        if (d.size and np.isfinite(t).all() and d[0] > 0
                and np.allclose(d, d[0], rtol=0.0, atol=1e-3)
                and abs(d[0] / 60.0 - round(d[0] / 60.0)) < 1e-6):
            return int(round(d[0] / 60.0))
        named.pop("time")
        return None

    def _syn_take(self, f: int):
        """A sample usable at bar size ``f``: one tagged ``f`` or untagged,
        held back or newly drawn; None when ``SYN_PULLS`` yields gave none."""
        from collections import deque

        hold = self.__dict__.setdefault("_syn_hold", {})
        for _ in range(self.SYN_PULLS + 1):
            for key in (f, None):
                if hold.get(key):
                    return hold[key].popleft()
            try:
                got = self._syn_arrays()
            except StopIteration as e:
                raise MktLegError("the generator stopped producing series") from e
            for a, bar in got:
                hold.setdefault(bar, deque(maxlen=self.SYN_HOLD)).append(a)
        return None

    def _synthetic(self, steps: int, f: int = 0):
        """``(context, target, rows)`` cut from a generator array usable at
        bar size ``f``: ``rows`` None, or (with ``synthetic_channels``)
        ``{"past": {name: context-aligned row}, "time": context + horizon
        unix starts or None}``."""
        from collections import deque

        ctx = int(self.r.context_length)
        chans = tuple(self.r.synthetic_channels)
        if not hasattr(self, "_synthetic_channels"):
            self._synthetic_channels = deque()
        for _ in range(16):
            if chans:
                named = self._syn_take(f)
                if named is None:
                    return None
                named = dict(named)
                v = named.pop("target")
            else:
                if not self._synthetic_channels:
                    try:
                        ((arr, _bar),) = self._syn_arrays()
                    except StopIteration as e:
                        raise MktLegError("the generator stopped producing series") from e
                    self._synthetic_channels.extend(arr if arr.ndim == 2 else [arr.reshape(-1)])
                v = self._synthetic_channels.popleft()
            if v.size < MIN_CONTEXT + steps:
                continue
            k = int(self.rng.integers(MIN_CONTEXT, v.size - steps + 1))
            k0 = max(0, k - ctx)
            c, y = v[k0:k], v[k:k + steps]
            if not (np.isfinite(c).sum() >= MIN_CONTEXT and np.isfinite(y).any()):
                continue
            rows = None
            if chans:
                t = named.pop("time", None)
                rows = {"past": {n: np.where(np.isfinite(r[k0:k]), r[k0:k], np.nan)
                                 for n, r in named.items()},
                        "time": None if t is None else t[k0:k + steps]}
            return c, y, rows
        return None

    def batch(self, size: int) -> Batch:
        from ..eval.marketcast import PAST_COVARIATES, build_covariates
        from ..pool.marketcast import window_covariates
        from ..shared.config.marketcast import base_suite

        while True:
            suite = _pick(self.rng, self.suites)
            hz = [h for (s, h) in self.geom if s == suite]
            own = base_suite(suite) == self.r.suite or self.r.suite not in self.mc.horizons_map
            mix = dict(self.r.horizon_mix) if own else {h: 1.0 for h in hz}
            mix = {k: v for k, v in mix.items() if k in hz} or {h: 1.0 for h in hz}
            h = _pick(self.rng, mix)
            f, steps = self.geom[(suite, h)]
            samples, synth = [], 0
            for _ in range(int(size)):
                if self.synthetic is not None and self.rng.random() < self.r.synthetic_fraction:
                    got = self._synthetic(steps, f)
                    if got is not None:
                        c, y, rows = got
                        samples.append((None, rows, c, y))
                        synth += 1
                        continue
                got = self._real(suite, f, steps)
                if got is not None:
                    samples.append(got)
            if samples:
                break
            if (self.r.synthetic_channels
                    and not any(k[0] == suite and k[3] == f for k in self.series)):
                # the generator's clocks never give this bar size and no real
                # series has it: the horizon trains not at all, like a
                # horizon with neither (never a leg killed by it)
                log.warning("marketcast leg: no generator series at %dm and no real series — "
                            "%s/%s is not trained", f, suite, h)
                del self.geom[(suite, h)]
                if not any(s2 == suite for s2, _h in self.geom):
                    self.suites.pop(suite, None)
                if self.suites:
                    continue
            raise MktLegError(f"could not cut a {suite}/{h} training sample")
        out_c, out_y, past, known = [], [], [], []
        names = tuple(self.r.covariates)
        for i, (key, k, c, y) in enumerate(samples):
            partner = None
            if len(samples) > 1:
                j = int(self.rng.integers(len(samples) - 1))
                j += j >= i
                partner = (samples[j][2], samples[j][3])
            c2, y2 = augment(c, y, self.r.augmentations, self.rng, partner)
            out_c.append(c2)
            out_y.append(y2)
            if names:
                if key is None and k is None:
                    past.append(None)
                    known.append(None)
                elif key is None:
                    # synthetic with named rows: its own past rows (a covariate
                    # it does not carry is a NaN row), known rows from its clock
                    want = names if k["time"] is not None else tuple(
                        n for n in names if n in PAST_COVARIATES)
                    p, kn = build_covariates(
                        want, block_starts=k["time"] if k["time"] is not None else c[:0],
                        n_context=int(c.size), series=k["past"],
                        event_times=tuple(getattr(self.mc, "event_times", ())))
                    past.append(p)
                    known.append(kn)
                else:
                    w = MktWindow("", key[1], key[2], suite, h, f, steps, 0.0, 0.0, k, c, y)
                    p, kn = window_covariates(self.snap, w, names,
                                              context_length=int(self.r.context_length),
                                              event_times=tuple(getattr(self.mc, "event_times",
                                                                        ())))
                    past.append(p)
                    known.append(kn)
        T = max(c.size for c in out_c)
        C = np.full((len(out_c), T), np.nan)
        for i, c in enumerate(out_c):
            C[i, T - c.size:] = c[-T:]
        P = K = None
        cp = next((p for p in past if p is not None), None)
        ck = next((k for k in known if k is not None), None)
        if cp is not None:
            P = np.full((len(out_c), cp.shape[0], T), np.nan)
        if ck is not None:
            K = np.full((len(out_c), ck.shape[0], T + steps), np.nan)
        for i in range(len(out_c)):
            n = out_c[i].size
            if P is not None and past[i] is not None:
                P[i, :, T - n:] = past[i][:, -n:]
            if K is not None and known[i] is not None:
                K[i, :, T - n:] = known[i][:, -(n + steps):]
        return Batch(suite, h, steps, C, np.stack(out_y), P, K, synth, int(f))


# ── loss + the fine-tune loop (torch) ────────────────────────────────────────


def lr_at(step: int, *, total: int, lr: float, warmup_fraction: float, schedule: str) -> float:
    """Linear warmup over ``warmup_fraction · total`` steps, then ``cosine``
    (to 0), ``linear`` (to 0) or ``constant``."""
    w = int(round(float(warmup_fraction) * total))
    if w and step < w:
        return lr * (step + 1) / w
    rest = max(1, total - w)
    t = min(1.0, (step - w) / rest)
    if schedule == "cosine":
        return lr * 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "linear":
        return lr * (1.0 - t)
    return lr


def quantile_loss(pred, target, context, levels: Sequence[float], weights: Sequence[float],
                  coverage_penalty: float = 0.0, coverage_levels=(0.5, 0.8, 0.98),
                  freq_minutes: int | None = None):
    """Knot-weighted pinball on the scale-normalised target (scale = mean
    |Δ context| per sample, ≥ 1e-6), NaN targets masked, + the optional
    soft coverage penalty ``Σ relu(nominal − soft coverage)²``. With
    ``freq_minutes`` (``[marketcast] crps_scale = "seasonal_naive"``) the scale
    is the eval's own per-window divisor, the GIFT-Eval seasonal-naive error
    of the context (:func:`~cascade.eval.marketcast.seasonal_naive_scale`),
    so the leg trains on what the verdict scores; a sample with no such scale
    keeps the mean |Δ| one."""
    import torch

    q = torch.as_tensor(levels, dtype=pred.dtype, device=pred.device)
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    ctx = torch.as_tensor(context, dtype=pred.dtype, device=pred.device)
    y = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    d = torch.diff(ctx, dim=-1)
    ok = torch.isfinite(d)
    scale = torch.where(ok, d.abs(), torch.zeros_like(d)).sum(-1) / ok.sum(-1).clamp_min(1)
    if freq_minutes:
        from ..eval.marketcast import seasonal_naive_scale

        sn = torch.as_tensor([seasonal_naive_scale(r, int(freq_minutes))
                              for r in np.asarray(context, dtype=np.float64)],
                             dtype=pred.dtype, device=pred.device)
        scale = torch.where(torch.isfinite(sn), sn, scale)
    scale = scale.clamp_min(1e-6)[:, None, None]
    mask = torch.isfinite(y)
    yy = torch.where(mask, y, torch.zeros_like(y))[..., None]
    diff = (yy - pred) / scale
    pin = torch.maximum(q * diff, (q - 1.0) * diff) * w
    m = mask[..., None].to(pred.dtype)
    loss = (pin * m).sum() / (m.sum() * w.sum()).clamp_min(1.0)
    if coverage_penalty > 0:
        lv = list(levels)
        pen = pred.new_zeros(())
        for c in coverage_levels:
            lo_i = int(np.argmin([abs(x - (1 - c) / 2) for x in lv]))
            hi_i = int(np.argmin([abs(x - (1 + c) / 2) for x in lv]))
            tau = 0.1
            inside = torch.sigmoid((yy[..., 0] - pred[..., lo_i]) / scale[..., 0] / tau) * \
                torch.sigmoid((pred[..., hi_i] - yy[..., 0]) / scale[..., 0] / tau)
            cov = (inside * mask).sum() / mask.sum().clamp_min(1)
            pen = pen + torch.relu(torch.as_tensor(c, dtype=pred.dtype) - cov) ** 2
        loss = loss + float(coverage_penalty) * pen
    return loss


@dataclass
class FinetuneResult:
    steps: int
    final_loss: float
    seconds: float
    synthetic_samples: int
    real_samples: int


def finetune(model, forward: Callable, sampler: MktSampler, recipe, *, levels: Sequence[float],
             max_seconds: float, clock=time.monotonic, continuation=None) -> FinetuneResult:
    """The recipe's fine-tune of ``model`` (a Timestral-shaped module):
    ``forward(contexts, steps, past, known) -> (B, steps, K)`` torch.
    Raises :class:`MktNonFiniteTraining` on a non-finite loss or gradient
    norm (operator fault: the recipe is bounded at parse time)."""
    import torch

    from ..eval.timestral_adapter import select_trainable

    params = select_trainable(model, recipe.scope, n_blocks=recipe.n_blocks,
                              lora_rank=recipe.lora_rank)
    if not params:
        raise MktLegError("the trainable scope selects no parameter")
    opt = torch.optim.AdamW(params, lr=recipe.lr, weight_decay=recipe.weight_decay)
    ema = [p.detach().clone() for p in params] if recipe.ema_decay else None
    t0 = clock()
    total = int(recipe.steps)
    loss_v, syn, real, step = float("nan"), 0, 0, 0
    model.train()
    completed = 0
    input_hash = ""
    restored = continuation.store.load() if continuation is not None else None
    if restored:
        from .continuation import assert_training_runtime, batch_digest, restore_rng
        assert_training_runtime(model, restored)
        model.load_state_dict(restored['model'])
        opt.load_state_dict(restored['optimizer'])
        ema = [v.to(p) for v, p in zip(restored['ema'], params, strict=True)] if restored['ema'] is not None else None
        completed = int(restored['step'])
        if completed < 1 or completed > total or not math.isfinite(restored['elapsed']) or restored['elapsed'] < 0:
            raise ValueError('invalid MarketCast continuation cursor/budget')
        for _ in range(completed):
            batch = sampler.batch(int(recipe.batch))
            input_hash = hashlib.sha256((input_hash + batch_digest(batch)).encode()).hexdigest()
        if input_hash != restored['input_hash']:
            raise ValueError('MarketCast continuation input replay differs')
        restore_rng(restored['rng'])
        loss_v, syn, real = restored['loss'], restored['synthetic'], restored['real']
        t0 = clock() - restored['elapsed']
    for step in range(completed, total):
        if clock() - t0 > float(max_seconds):
            log.warning("marketcast leg: wall budget reached at step %d/%d", step, total)
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total=total, lr=recipe.lr,
                            warmup_fraction=recipe.warmup_fraction, schedule=recipe.schedule)
        b = sampler.batch(int(recipe.batch))
        if continuation is not None:
            from .continuation import batch_digest
            input_hash = hashlib.sha256((input_hash + batch_digest(b)).encode()).hexdigest()
        syn += b.synthetic
        real += b.contexts.shape[0] - b.synthetic
        sn = getattr(sampler.mc, "crps_scale", "seasonal_naive") == "seasonal_naive"
        n = int(b.contexts.shape[0])
        # gradient accumulation: one optimizer step per recipe batch, at most
        # micro_batch_tokens context steps per forward (one chunk = unchanged)
        mb = max(1, int(getattr(sampler.mc, "micro_batch_tokens", 0) or n * b.contexts.shape[1])
                 // max(1, int(b.contexts.shape[1])))
        opt.zero_grad(set_to_none=True)
        loss = None
        for i in range(0, n, mb):
            j = min(n, i + mb)
            pred = forward(b.contexts[i:j], b.steps,
                           None if b.past is None else b.past[i:j],
                           None if b.known is None else b.known[i:j])
            part = quantile_loss(pred, b.targets[i:j], b.contexts[i:j], levels,
                                 recipe.quantile_weights, recipe.coverage_penalty,
                                 freq_minutes=(b.freq_minutes or None) if sn else None)
            if not torch.isfinite(part):
                raise MktNonFiniteTraining(f"{NONFINITE_TRAINING_MARKER} loss at step {step}")
            part = part * ((j - i) / n) if (i, j) != (0, n) else part
            part.backward()
            loss = part.detach() if loss is None else loss + part.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(params, float(recipe.clip))
        if not torch.isfinite(gnorm):
            # a NaN/inf gradient would poison every weight on opt.step();
            # stop before it (the next loss would be NaN anyway)
            raise MktNonFiniteTraining(f"{NONFINITE_TRAINING_MARKER} gradient at step {step}")
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for e, p in zip(ema, params, strict=True):
                    e.mul_(recipe.ema_decay).add_(p.detach(), alpha=1.0 - recipe.ema_decay)
        loss_v = float(loss.detach())
        completed = step + 1
        if continuation is not None and continuation.due():
            from .continuation import rng_state, training_runtime
            checkpoint_start = clock()
            continuation.save({'runtime': training_runtime(model), 'step': completed, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'ema': ema, 'rng': rng_state(), 'input_hash': input_hash, 'elapsed': clock() - t0,
                'loss': loss_v, 'synthetic': syn, 'real': real,
                'data_files': getattr(continuation, 'data_files', {})})
            t0 += clock() - checkpoint_start
    if ema is not None:
        with torch.no_grad():
            for e, p in zip(ema, params, strict=True):
                p.copy_(e)
    model.eval()
    return FinetuneResult(completed, loss_v, clock() - t0, syn, real)


# ── the leg ──────────────────────────────────────────────────────────────────


def recipe_digest(recipe) -> str:
    d = asdict(recipe)
    if not d.get("synthetic_channels"):
        d.pop("synthetic_channels", None)   # recipes without it keep their digest
    return hashlib.sha256(json.dumps(d, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def training_span(recipe, not_after: float | None = None) -> tuple[int, int]:
    """``(t0, t1)`` of a recipe's real data: ``[data.start, data.end + 1
    day)``, clipped to the last whole hour before ``not_after``."""
    from datetime import UTC, date, datetime

    t0 = int(datetime.combine(date.fromisoformat(recipe.start), datetime.min.time(),
                              tzinfo=UTC).timestamp())
    t1 = int(datetime.combine(date.fromisoformat(recipe.end), datetime.min.time(),
                              tzinfo=UTC).timestamp()) + 86400
    if not_after is not None:
        t1 = max(t0, min(t1, int(float(not_after)) // 3600 * 3600))
    return t0, t1


def training_intervals(mc: MarketCastConfig) -> tuple[str, ...]:
    """The recording intervals a training snapshot reads: the minute bars,
    every native interval an enabled series is cut from, and 1h when a
    series uses the hourly bank."""
    from ..pool.marketcast import needed_banks, needs_coarse
    from ..shared.config.marketcast import duration_minutes

    want = needed_banks(mc)
    natives = tuple(iv for iv in mc.native_intervals if duration_minutes(iv) in want)
    return ("1m", *natives, *(("1h",) if needs_coarse(mc) else ()))


def training_relay_plan(store, prefix: str, mc: MarketCastConfig, recipe,
                        not_after: float | None = None) -> dict:
    """The trainer-side relay plan (:mod:`cascade.pool.marketcast_relay`)
    for one leg: GET links for the recording a recipe's training snapshot
    reads, Hyperliquid classes only (other sources are fetched on the pod)."""
    from ..pool.marketcast_relay import relay_plan, relay_rels

    t0, t1 = training_span(recipe, not_after)
    hl = [(c, a) for c, a in recipe.assets
          if mc.sources_for(c)[:1] and mc.sources_for(c)[0] in ("hyperliquid", "hyperliquid_xyz")]
    return relay_plan(store, prefix, relay_rels(hl, training_intervals(mc), t0, t1))


def clip_training_snapshot(snap, out_dir, mc: MarketCastConfig, recipe,
                           not_after: float | None = None) -> dict:
    """W-CMP2 (DEC-CA-0075): the recipe's real data cut from a competition's
    STANDARD POOL (``snap``, the MarketCast-runner view) — every bank sliced
    to :func:`training_span` (``[data.start, data.end + 1 day)`` clipped
    before the dispatch ``not_after``), written as an unsigned training
    snapshot (digested into the entry's ``data_digest``). Only the recipe's
    assets; nothing at or after ``not_after`` ever reaches the pod's
    training data."""
    import numpy as np

    from ..pool.marketcast import native_dir, write_snapshot

    t0, t1 = training_span(recipe, not_after)
    want = {(c, a) for c, syms in dict(recipe.assets).items() for a in syms}
    meta = snap.meta

    def _slice(bank_t0: int, step: int, cls: str, asset: str, bank: str):
        arrs = snap.bars(cls, asset, bank=bank)
        n_all = max((len(v) for v in arrs.values()), default=0)
        lo = max(0, -(-(t0 - bank_t0) // step))
        hi = max(lo, min(n_all, (t1 - bank_t0) // step))
        return bank_t0 + lo * step, {k: np.asarray(v, dtype=np.float64)[lo:hi]
                                     for k, v in arrs.items()}, hi - lo

    bars: dict = {}
    b_t0, b_n = int(t0), (int(t1) - int(t0)) // 60
    for cls in sorted(meta.get("assets", {})):
        for a in snap.assets(cls):
            if (cls, a) not in want:
                continue
            got_t0, got, n = _slice(int(meta["t0"]), 60, cls, a, "bars")
            if got and n:
                b_t0, b_n = got_t0, n
                bars[(cls, a)] = got
    coarse, c_t0, c_n = {}, 0, 0
    c = meta.get("coarse")
    if c:
        for cls in sorted(meta.get("assets", {})):
            for a in snap.assets(cls):
                if (cls, a) not in want:
                    continue
                got_t0, got, n = _slice(int(c["t0"]), 3600, cls, a, "coarse")
                if got and n:
                    c_t0, c_n = got_t0 - got_t0 % 86400, n
                    if got_t0 % 86400:
                        # write_snapshot's coarse grid starts on a UTC day: pad
                        pad = (got_t0 % 86400) // 3600
                        got = {k: np.concatenate([np.full(pad, np.nan), v])
                               for k, v in got.items()}
                        c_n = n + pad
                    coarse[(cls, a)] = got
    native: dict = {}
    for iv, rec in sorted((meta.get("native") or {}).items()):
        step = 60 * int(rec["bar_minutes"])
        part, nt0, nn = {}, 0, 0
        for cls in sorted(meta.get("assets", {})):
            for a in snap.assets(cls):
                if (cls, a) not in want:
                    continue
                got_t0, got, n = _slice(int(rec["t0"]), step, cls, a, native_dir(iv))
                if got and n:
                    pad = (got_t0 % 86400) // step
                    if pad:
                        got = {k: np.concatenate([np.full(pad, np.nan), v])
                               for k, v in got.items()}
                    part[(cls, a)], nt0, nn = got, got_t0 - got_t0 % 86400, n + pad
        if part:
            native[iv] = (nt0, nn, part)
    return write_snapshot(out_dir, t0=int(b_t0) // 60 * 60, minutes=int(max(0, b_n)),
                          bars=bars, sources=dict(meta.get("sources") or {}),
                          anchor_block=int(meta["anchor_block"]),
                          anchor_time=float(meta["anchor_time"]),
                          block_seconds=float(meta["block_seconds"]),
                          coarse=coarse or None, coarse_t0=int(c_t0), coarse_steps=int(c_n),
                          native=native or None)


def build_instance_training_snapshot(out_dir, mc: MarketCastConfig, recipe, *, corpus_dir,
                                     work, not_after: float | None = None,
                                     get_bytes=None) -> dict:  # pragma: no cover — a pod
    """W-CMP2: download the competition pool the trainer relayed
    (``mkt_pool.json``: one GET link + the tar's sha256), verify it, read it
    through the generic reader and clip it (:func:`clip_training_snapshot`)."""
    import json
    import urllib.request
    from pathlib import Path

    from ..pool.marketcast import load_snapshot
    from ..shared.hippius import tar_cid_digest, unpack_tar_to_dir

    p = Path(corpus_dir or "") / "mkt_pool.json"
    plan = json.loads(p.read_text(encoding="utf-8"))
    if int(plan.get("version", 0)) != 1:
        raise OSError("unsupported competition pool relay")
    if get_bytes is None:
        def get_bytes(url):
            with urllib.request.urlopen(url, timeout=600) as r:  # noqa: S310 — presigned GET
                return r.read()
    data = get_bytes(str(plan["url"]))
    if tar_cid_digest(data) != str(plan["sha256"]):
        raise OSError("competition pool tar digest mismatch")
    d = unpack_tar_to_dir(data, Path(work) / "_mkt_pool")
    snap = load_snapshot(d, verify_signature=False, select=dict(plan.get("select") or {}),
                         channels=dict(plan.get("channels") or {}))
    na = not_after if not_after is not None else (float(plan.get("not_after") or 0) or None)
    return clip_training_snapshot(snap, out_dir, mc, recipe, not_after=na)


def build_training_snapshot(out_dir, mc: MarketCastConfig, recipe, *, post, get,
                            not_after: float | None = None, store=None) -> dict:
    """The recipe's real data: its assets over ``[data.start, data.end + 1
    day)`` from the classes' primary sources (unsigned — digested into the
    entry's ``data_digest``), with the coarse hourly bank over the same span
    for the >= 1h series (review 2026-10-19: Hyperliquid serves only its most
    recent ~5000 candles per interval, so an embargoed range has no 1m bars
    there but does have hourly ones for ~208 days). ``not_after`` (the
    trainer-signed dispatch time) clips the range: no bar at or after it is
    ever fetched (review M12; the embargo already keeps ``data.end`` far
    before it — this is the pod-side backstop). Feed failures raise
    :class:`MktDataUnavailable`; a successful empty response is kept empty,
    so an unavailable feed cannot be mistaken for an invalid recipe.

    With ``store`` (a :class:`~cascade.pool.marketcast_store.MktBarStore` the
    trainer relayed from the forward recording) every Hyperliquid bank — the
    1m bars, each native interval (5m / 15m) and the hourly bank — is read
    from the recording ALONE, never the API: the API keeps only its last
    ~5000 candles per interval (5m ≈ 17 days), so an embargoed range has no
    sub-hour candles there, and a recording is fixed bytes, so the same
    recipe and ``not_after`` give the same ``data_digest``."""
    from ..pool.marketcast import (
        fetch_class_bars,
        fetch_coarse_bars,
        needed_banks,
        needs_coarse,
        write_snapshot,
    )
    from ..shared.config.marketcast import duration_minutes

    t0, t1 = training_span(recipe, not_after)
    minutes = (t1 - t0) // 60
    hours = (t1 - t0) // 3600
    want_coarse = needs_coarse(mc)
    want = needed_banks(mc)
    natives = [iv for iv in mc.native_intervals if duration_minutes(iv) in want]
    bars: dict = {}
    coarse: dict = {}
    native: dict = {iv: {} for iv in natives}

    def _hl(cls):
        return mc.sources_for(cls)[:1][0] in ("hyperliquid", "hyperliquid_xyz")

    def _banked(cls, a, interval, steps):
        if store is not None and _hl(cls):
            return store.read(cls, a, t0, steps, interval=interval)
        return fetch_coarse_bars(cls, a, mc.sources_for(cls)[:1], t0, steps, post=post,
                                 interval=interval)

    for cls, syms in recipe.assets:
        for a in syms:
            try:
                b, _rep = fetch_class_bars(cls, a, mc.sources_for(cls)[:1], t0, minutes,
                                           post=post, get=get,
                                           store=store if _hl(cls) else None)
            except Exception as e:  # noqa: BLE001 — feed errors are infrastructure faults
                raise MktDataUnavailable(f"{cls}/{a} training feed unavailable: {e}") from e
            if any(np.isfinite(v).any() for v in b.values()):
                bars[(cls, a)] = b
            for iv in natives:
                try:
                    nb = _banked(cls, a, iv, (t1 - t0) // (60 * duration_minutes(iv)))
                except Exception as e:  # noqa: BLE001 — same fault split for native banks
                    raise MktDataUnavailable(f"{cls}/{a} {iv} feed unavailable: {e}") from e
                if any(np.isfinite(nb[k]).any() for k in ("close",)):
                    native[iv][(cls, a)] = nb
            if want_coarse:
                try:
                    cb = _banked(cls, a, "1h", hours)
                except Exception as e:  # noqa: BLE001 — same fault split for coarse feeds
                    raise MktDataUnavailable(f"{cls}/{a} coarse feed unavailable: {e}") from e
                if any(np.isfinite(v).any() for v in cb.values()):
                    coarse[(cls, a)] = cb
    return write_snapshot(out_dir, t0=t0, minutes=minutes, bars=bars,
                          sources={c: mc.data_sources_map.get(c, "") for c, _ in recipe.assets},
                          anchor_block=0, anchor_time=float(t0), coarse=coarse or None,
                          coarse_t0=t0, coarse_steps=hours,
                          native={iv: (t0, (t1 - t0) // (60 * duration_minutes(iv)), d)
                                  for iv, d in native.items() if d} or None)


def load_init_weights(model, ckpt_dir, *, base_dir, sha256: str) -> None:
    """Owner 2026-11-18: warm-start ``model`` (the base, in place) from an
    earlier MarketCast checkpoint the trainer resolved: its weights' sha256
    must be ``sha256``, its architecture the base's and every weight finite;
    ``RuntimeError`` otherwise — an OPERATOR fault (the trainer's own
    lineage disagrees with the registry), never the miner's: never trained on."""
    from safetensors.torch import load_file

    from ..eval.timestral_adapter import (
        WEIGHTS_FILE,
        safetensors_all_finite,
        same_structure,
        sha256_file,
    )

    p = Path(ckpt_dir) / WEIGHTS_FILE
    if not p.is_file() or sha256_file(p) != str(sha256):
        raise RuntimeError("warm-start checkpoint is not the one the trainer resolved")
    diff = same_structure(Path(base_dir) / WEIGHTS_FILE, p)
    if diff:
        raise RuntimeError(f"warm-start checkpoint is not the base architecture: {diff[:3]}")
    if not safetensors_all_finite(p):
        raise RuntimeError("warm-start checkpoint weights are not finite")
    sd = load_file(str(p))
    dev = next(model.parameters()).device
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)


def run_mkt_leg(*, recipe, mc: MarketCastConfig, base_dir, base_fc_inner, train_snap,
                out_dir, seed: int, max_seconds: float, synthetic=None,
                meta_extra: dict | None = None, clock=time.monotonic, continuation=None) -> dict:
    """Fine-tune + export. ``base_fc_inner`` is the library forecaster of
    the (digest-verified) base; returns the receipt fields
    ``{checkpoint_sha256, data_digest, recipe_digest, steps, final_loss,
    ...}``."""
    import torch

    from ..eval.timestral_adapter import export_checkpoint, forward_block

    torch.manual_seed(int(seed))  # includes LoRA initialisation and training stochasticity

    if continuation is not None:
        from .continuation import marketcast_data_files
        continuation.data_files = marketcast_data_files(train_snap)
    sampler = MktSampler(recipe, mc, train_snap, seed=seed, synthetic=synthetic)

    def _fwd(ctx, steps, past, known):
        return forward_block(base_fc_inner, ctx, steps, past_covariates=past,
                             known_covariates=known)

    res = finetune(base_fc_inner.model, _fwd, sampler, recipe, levels=tuple(mc.quantile_knots),
                   max_seconds=max_seconds, clock=clock, continuation=continuation)
    meta = {"base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
            "target": recipe.target, "covariates": list(recipe.covariates),
            "recipe": asdict(recipe), "data_digest": str(train_snap.digest),
            **dict(meta_extra or {})}
    sha = export_checkpoint(base_fc_inner.model, out_dir, base_dir=base_dir, meta=meta)
    return {"checkpoint_sha256": sha, "data_digest": str(train_snap.digest),
            "recipe_digest": recipe_digest(recipe), "steps": int(res.steps),
            "final_loss": float(res.final_loss), "seconds": float(res.seconds),
            "synthetic_samples": int(res.synthetic_samples),
            "real_samples": int(res.real_samples)}


__all__ = [
    "AUG_DEFAULTS", "Batch", "FinetuneResult", "MktDataUnavailable", "MktLegError",
    "MktNonFiniteTraining", "MktSampler", "NONFINITE_TRAINING_MARKER", "augment",
    "build_training_snapshot", "finetune", "lr_at", "quantile_loss", "recipe_digest",
    "clip_training_snapshot", "build_instance_training_snapshot",
    "run_mkt_leg", "load_init_weights", "training_intervals", "training_relay_plan", "training_span",
]
