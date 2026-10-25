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
   samples carry NaN (missing) covariate rows;
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

import numpy as np

from ..eval.marketcast import MktWindow, window_geometry
from ..shared.config.marketcast import MarketCastConfig

log = logging.getLogger("cascade.trainer.marketcast")

MIN_CONTEXT = 64


class MktLegError(ValueError):
    """The MINER's recipe / generator made the leg impossible (miner fault)."""


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
        for suite, w in dict(recipe.suite_weights).items():
            if w <= 0 or suite not in mc.horizons_map:
                continue
            freqs = [f for f in mc.frequencies_map[suite] if f in freqs_ok]
            hz = (dict(recipe.horizon_mix) if suite == recipe.suite
                  else {h: 1.0 for h in mc.horizons_map[suite]})
            for h in hz:
                try:
                    self.geom[(suite, h)] = window_geometry(freqs, h)
                except ValueError:
                    continue
            from ..shared.config.marketcast import SUITE_CLASSES

            have = False
            for cls, syms in assets.items():
                if cls not in SUITE_CLASSES.get(suite, ()):
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
            if have:
                self.suites[suite] = float(w)
        if recipe.suite not in self.suites:
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

    def _synthetic(self, steps: int):
        from collections import deque

        ctx = int(self.r.context_length)
        if not hasattr(self, "_synthetic_channels"):
            self._synthetic_channels = deque()
        for _ in range(16):
            try:
                if not self._synthetic_channels:
                    arr = np.asarray(next(self.synthetic), dtype=np.float64)
                    self._synthetic_channels.extend(arr if arr.ndim == 2 else [arr.reshape(-1)])
                v = self._synthetic_channels.popleft()
            except StopIteration as e:
                raise MktLegError("the generator stopped producing series") from e
            if v.size < MIN_CONTEXT + steps:
                continue
            k = int(self.rng.integers(MIN_CONTEXT, v.size - steps + 1))
            c, y = v[max(0, k - ctx):k], v[k:k + steps]
            if np.isfinite(c).sum() >= MIN_CONTEXT and np.isfinite(y).any():
                return c, y
        return None

    def batch(self, size: int) -> Batch:
        from ..pool.marketcast import window_covariates

        suite = _pick(self.rng, self.suites)
        hz = [h for (s, h) in self.geom if s == suite]
        mix = dict(self.r.horizon_mix) if suite == self.r.suite else {h: 1.0 for h in hz}
        h = _pick(self.rng, {k: v for k, v in mix.items() if k in hz})
        f, steps = self.geom[(suite, h)]
        samples, synth = [], 0
        for _ in range(int(size)):
            if self.synthetic is not None and self.rng.random() < self.r.synthetic_fraction:
                got = self._synthetic(steps)
                if got is not None:
                    samples.append((None, None, *got))
                    synth += 1
                    continue
            got = self._real(suite, f, steps)
            if got is not None:
                samples.append(got)
        if not samples:
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
                if key is None:
                    past.append(None)
                    known.append(None)
                else:
                    w = MktWindow("", key[1], key[2], suite, h, f, steps, 0.0, 0.0, k, c, y)
                    p, kn = window_covariates(self.snap, w, names,
                                              context_length=int(self.r.context_length))
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
        return Batch(suite, h, steps, C, np.stack(out_y), P, K, synth)


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
                  coverage_penalty: float = 0.0, coverage_levels=(0.5, 0.8, 0.98)):
    """Knot-weighted pinball on the scale-normalised target (scale = mean
    |Δ context| per sample, ≥ 1e-6), NaN targets masked, + the optional
    soft coverage penalty ``Σ relu(nominal − soft coverage)²``."""
    import torch

    q = torch.as_tensor(levels, dtype=pred.dtype, device=pred.device)
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    ctx = torch.as_tensor(context, dtype=pred.dtype, device=pred.device)
    y = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    d = torch.diff(ctx, dim=-1)
    ok = torch.isfinite(d)
    scale = torch.where(ok, d.abs(), torch.zeros_like(d)).sum(-1) / ok.sum(-1).clamp_min(1)
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
    Raises :class:`MktLegError` on a non-finite loss (the miner's
    recipe / data diverged)."""
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
        pred = forward(b.contexts, b.steps, b.past, b.known)
        loss = quantile_loss(pred, b.targets, b.contexts, levels, recipe.quantile_weights,
                             recipe.coverage_penalty)
        if not torch.isfinite(loss):
            raise MktLegError(f"non-finite training loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(recipe.clip))
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
    return hashlib.sha256(json.dumps(asdict(recipe), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def build_training_snapshot(out_dir, mc: MarketCastConfig, recipe, *, post, get,
                            not_after: float | None = None) -> dict:
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
    so an unavailable feed cannot be mistaken for an invalid recipe."""
    from datetime import UTC, date, datetime

    from ..pool.marketcast import fetch_class_bars, fetch_coarse_bars, needs_coarse, write_snapshot

    t0 = int(datetime.combine(date.fromisoformat(recipe.start), datetime.min.time(),
                              tzinfo=UTC).timestamp())
    t1 = int(datetime.combine(date.fromisoformat(recipe.end), datetime.min.time(),
                              tzinfo=UTC).timestamp()) + 86400
    if not_after is not None:
        t1 = max(t0, min(t1, int(float(not_after)) // 3600 * 3600))
    minutes = (t1 - t0) // 60
    hours = (t1 - t0) // 3600
    want_coarse = needs_coarse(mc)
    bars: dict = {}
    coarse: dict = {}
    for cls, syms in recipe.assets:
        for a in syms:
            try:
                b, _rep = fetch_class_bars(cls, a, mc.sources_for(cls)[:1], t0, minutes,
                                           post=post, get=get)
            except Exception as e:  # noqa: BLE001 — feed errors are infrastructure faults
                raise MktDataUnavailable(f"{cls}/{a} training feed unavailable: {e}") from e
            if any(np.isfinite(v).any() for v in b.values()):
                bars[(cls, a)] = b
            if want_coarse:
                try:
                    cb = fetch_coarse_bars(cls, a, mc.sources_for(cls)[:1], t0, hours, post=post)
                except Exception as e:  # noqa: BLE001 — same fault split for coarse feeds
                    raise MktDataUnavailable(f"{cls}/{a} coarse feed unavailable: {e}") from e
                if any(np.isfinite(v).any() for v in cb.values()):
                    coarse[(cls, a)] = cb
    return write_snapshot(out_dir, t0=t0, minutes=minutes, bars=bars,
                          sources={c: mc.data_sources_map.get(c, "") for c, _ in recipe.assets},
                          anchor_block=0, anchor_time=float(t0), coarse=coarse or None,
                          coarse_t0=t0, coarse_steps=hours)


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
    "MktSampler", "augment",
    "build_training_snapshot", "finetune", "lr_at", "quantile_loss", "recipe_digest",
    "run_mkt_leg",
]
