from __future__ import annotations
import json
import math
import os
from functools import lru_cache
from collections.abc import Iterator
from queue import Full, Queue
from threading import Event, Thread
import numpy as np
from scipy.signal import lfilter
try:
    from cascade.interface import DataGenerator
except Exception:

    class DataGenerator:
        pass
_TARGET_WEIGHTS = {'ar2': 0.12825, 'integrated': 0.09975, 'regime_shift': 0.09025, 'trend_seasonal_ar': 0.09025, 'ou_stochastic_vol': 0.08075, 'threshold_ar': 0.06175, 'spectral_gp': 0.057, 'long_memory': 0.057, 'multiplicative': 0.057, 'chaotic': 0.0285, 'tidal_harmonic': 0.03325, 'flow_recession': 0.0285, 'step_level': 0.0285, 'vol_regime_switch': 0.02375, 'weekly_demand': 0.0475, 'seasonal_counts': 0.019, 'intermittent_demand': 0.019, 'policy_rate': 0.05}
_START_WEIGHTS = {'trend_seasonal_ar': 0.192, 'ar2': 0.1536, 'multiplicative': 0.096, 'spectral_gp': 0.096, 'weekly_demand': 0.096, 'integrated': 0.0768, 'tidal_harmonic': 0.0576, 'regime_shift': 0.048, 'seasonal_counts': 0.0384, 'threshold_ar': 0.0288, 'long_memory': 0.0288, 'ou_stochastic_vol': 0.0192, 'flow_recession': 0.0096, 'step_level': 0.0096, 'vol_regime_switch': 0.0048, 'intermittent_demand': 0.0048, 'chaotic': 0.0, 'policy_rate': 0.04}
_DEFAULTS = {'generate_length': 4096, 'batch_size': 2048, 'max_abs_value': 1000000.0, 'family_weights': _TARGET_WEIGHTS, 'curriculum': {'enabled': True, 'total_points': 2000000000.0, 'full_mixture_at': 0.7, 'start_family_weights': _START_WEIGHTS}, 'observation_layer': {'enabled': True, 'apply_frac': 0.35}, 'record_families': False}

def _merge(dst: dict, src: dict) -> dict:
    out = dict(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def _logu(rng, lo: float, hi: float, size) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))

def _smoothstep(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)
_CADENCES = np.array([4, 7, 12, 24, 30, 48, 52, 90, 96, 144, 168, 183, 288, 336, 365, 672, 730], dtype=np.float64)
_CADENCE_P = np.array([0.04, 0.12, 0.04, 0.16, 0.05, 0.06, 0.04, 0.03, 0.07, 0.03, 0.13, 0.04, 0.04, 0.06, 0.07, 0.04, 0.05])
_CADENCE_P = _CADENCE_P / _CADENCE_P.sum()

def _periods(rng, size, lo: float=8.0, hi: float=1200.0, cadence_frac: float=0.7) -> np.ndarray:
    cad = rng.choice(_CADENCES, size=size, p=_CADENCE_P)
    cont = _logu(rng, lo, hi, size)
    use_cad = rng.random(size) < cadence_frac
    return np.where(use_cad, cad, cont)

def _ar1_batch(e: np.ndarray, phi: np.ndarray) -> np.ndarray:
    k, L = e.shape
    x = np.empty_like(e)
    p = np.asarray(phi).reshape(k)
    for i in range(k):
        x[i] = lfilter([1.0], [1.0, -float(p[i])], e[i])
    return x

def _ar2_batch(e: np.ndarray, phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    k, L = e.shape
    x = np.empty_like(e)
    for i in range(k):
        x[i] = lfilter([1.0], [1.0, -float(phi1[i]), -float(phi2[i])], e[i])
    return x

def _ffill(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    k, L = x.shape
    idx = np.where(keep, np.arange(L)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return np.take_along_axis(x, idx, axis=1)
_H_GRID = np.linspace(0.55, 0.95, 33)

@lru_cache(maxsize=64)
def _dh_eigs(h_idx: int, L: int) -> np.ndarray:
    H = float(_H_GRID[h_idx])
    lags = np.arange(L + 1, dtype=np.float64)
    g = 0.5 * (np.abs(lags + 1) ** (2 * H) - 2 * np.abs(lags) ** (2 * H) + np.abs(lags - 1) ** (2 * H))
    row = np.concatenate([g, g[L - 1:0:-1]])
    lam = np.fft.fft(row).real
    return np.maximum(lam, 0.0)

def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-09)
_ps_SEASONAL_PERIODS = np.array([4, 7, 12, 15, 24, 30, 48, 52, 60, 90, 96, 144, 168, 183, 240, 288, 336, 365, 672, 730], dtype=np.float64)
_ps_SEASONAL_PROBS = np.array([0.01, 0.23, 0.02, 0.02, 0.07, 0.01, 0.06, 0.01, 0.08, 0.01, 0.14, 0.07, 0.04, 0.01, 0.08, 0.08, 0.02, 0.02, 0.01, 0.01], dtype=np.float64)
_ps_SEASONAL_PROBS /= _ps_SEASONAL_PROBS.sum()
_ps_SEASONAL_PAIRS = np.array([[15, 60], [60, 240], [24, 168], [48, 336], [96, 672], [7, 365], [12, 52]], dtype=np.float64)

def _ps_ar1_batch(innov: np.ndarray, phi: np.ndarray) -> np.ndarray:
    n, L = innov.shape
    x = np.empty((n, L), dtype=np.float64)
    p = phi.reshape(n)
    for i in range(n):
        x[i] = lfilter([1.0], [1.0, -float(p[i])], innov[i])
    return x

def _ps_prefix_mean_std(x: np.ndarray, *, calibration_points: int=512) -> tuple[np.ndarray, np.ndarray]:
    prefix = x[:, :min(x.shape[1], calibration_points)]
    mean = prefix.mean(axis=1, keepdims=True)
    std = prefix.std(axis=1, keepdims=True)
    return (mean, np.where(std < 1e-12, 1.0, std))

def _ps_prefix_standardize(x: np.ndarray, *, center: bool=True, calibration_points: int=512) -> np.ndarray:
    mean, std = _ps_prefix_mean_std(x, calibration_points=calibration_points)
    return (x - mean) / std if center else x / std

def _ps_seasonal_basis(L: int) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * np.arange(L, dtype=np.float64)[None, :] / _ps_SEASONAL_PERIODS[:, None]
    return (np.sin(angle), np.cos(angle))

def _ps_seasonal(rng: np.random.Generator, n: int, L: int, k_max: int=3) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    sin_basis, cos_basis = _ps_seasonal_basis(L)
    k = rng.integers(1, k_max + 1, size=n)
    pair = _ps_SEASONAL_PAIRS[rng.integers(0, len(_ps_SEASONAL_PAIRS), size=n)]
    use_pair = rng.random(n) < 0.35
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(k_max):
        active = np.nonzero(k > j)[0]
        per = rng.choice(_ps_SEASONAL_PERIODS, size=n, p=_ps_SEASONAL_PROBS)
        if j < 2:
            per = np.where(use_pair, pair[:, j], per)
        per = per[:, None]
        amp = rng.uniform(0.2, 2.0, size=n)[:, None]
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)[:, None]
        basis_idx = np.searchsorted(_ps_SEASONAL_PERIODS, per[active, 0])
        component = amp[active] * (sin_basis[basis_idx] * np.cos(phase[active]) + cos_basis[basis_idx] * np.sin(phase[active]))
        modulated = np.nonzero((k > j) & (rng.random(n) < 0.35))[0]
        if modulated.size:
            modulated_local = np.searchsorted(active, modulated)
            modulated_arg = 2.0 * np.pi * t / per[modulated] + phase[modulated]
            m_per = np.clip(per[modulated] * rng.uniform(4.0, 12.0, size=(modulated.size, 1)), 32.0, 2.0 * L)
            m_phase = rng.uniform(0.0, 2.0 * np.pi, size=(modulated.size, 1))
            slow = np.sin(2.0 * np.pi * t / m_per + m_phase)
            amp_mod = 1.0 + rng.uniform(0.05, 0.45, size=(modulated.size, 1)) * slow
            phase_mod = rng.uniform(0.05, 0.75, size=(modulated.size, 1)) * np.sin(2.0 * np.pi * t / (1.7 * m_per) - m_phase)
            component[modulated_local] = amp[modulated] * amp_mod * np.sin(modulated_arg + phase_mod)
        out[active] += component
    return out

def _ps_sparse_jumps(rng: np.random.Generator, n: int, L: int, rate: float, scale) -> np.ndarray:
    mask = rng.random((n, L)) < rate
    mask[:, 0] = False
    rows, cols = np.nonzero(mask)
    jumps = np.zeros((n, L), dtype=np.float64)
    if rows.size == 0:
        return jumps
    s = np.asarray(scale, dtype=np.float64)
    event_scale = s if s.ndim == 0 else s.reshape(n)[rows]
    jumps[rows, cols] = rng.normal(0.0, 1.0, size=rows.size) * event_scale
    return jumps

def _ps_spectral_gp(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    embed = 2 * L
    f = np.fft.rfftfreq(embed)[None, :]
    lengthscale = np.exp(rng.uniform(np.log(8.0), np.log(256.0), size=(n, 1)))
    envelope = np.exp(-0.5 * (2.0 * np.pi * lengthscale * f) ** 2)
    z = rng.standard_normal((n, f.shape[1])) + 1j * rng.standard_normal((n, f.shape[1]))
    z[:, 0] = 0.0
    x = np.fft.irfft(z * np.sqrt(envelope), n=embed, axis=1)[:, :L]
    return _ps_prefix_standardize(x)

def _ps_physical_sensors(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    seasonal = _ps_seasonal(rng, n, L, k_max=2)
    smooth = _ps_spectral_gp(rng, n, L)
    fronts = np.cumsum(_ps_sparse_jumps(rng, n, L, rate=5.0 / L, scale=1.0), axis=1)
    base = seasonal * rng.uniform(0.3, 2.0, size=(n, 1)) + smooth * rng.uniform(0.2, 1.2, size=(n, 1)) + fronts * rng.uniform(0.2, 1.0, size=(n, 1))
    kind = rng.integers(0, 4, size=n)
    out = base.copy()
    bounded = kind == 1
    if bounded.any():
        gain = rng.uniform(0.8, 3.5, size=(int(bounded.sum()), 1))
        midpoint = rng.uniform(-0.8, 0.8, size=(int(bounded.sum()), 1))
        out[bounded] = 100.0 / (1.0 + np.exp(-gain * (base[bounded] - midpoint)))
    pressure = kind == 2
    if pressure.any():
        count = int(pressure.sum())
        diffusion = np.exp(rng.uniform(np.log(0.03), np.log(0.2), size=(count, 1)))
        walk = np.cumsum(rng.standard_normal((count, L)) * diffusion, axis=1)
        level = rng.uniform(900.0, 1100.0, size=(count, 1))
        out[pressure] = level + walk + 2.0 * fronts[pressure] + 0.5 * seasonal[pressure]
    magnitude = kind == 3
    if magnitude.any():
        count = int(magnitude.sum())
        gusts = (rng.random((count, L)) < 8.0 / L) * rng.lognormal(0.0, 0.8, size=(count, L))
        power = rng.uniform(1.0, 1.6, size=(count, 1))
        out[magnitude] = np.abs(base[magnitude]) ** power + gusts
    return out
_ps_CS_CALM_LO = 192
_ps_CS_CALM_HI = 1024
_ps_CS_DYNAMIC_LO = 96
_ps_CS_DYNAMIC_HI = 512

def _ps_conditional_stability(rng: np.random.Generator, n: int, L: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    if L <= 0:
        return np.empty((n, 0), dtype=np.float64)
    seasonal = _ps_seasonal(rng, n, L, k_max=2)
    ar = _ps_ar1_batch(rng.normal(size=(n, L)) * rng.uniform(0.12, 0.55, size=(n, 1)), rng.uniform(0.35, 0.92, size=n))
    smooth = _ps_spectral_gp(rng, n, L)
    walk = np.cumsum(rng.normal(size=(n, L)) * rng.uniform(0.025, 0.16, size=(n, 1)), axis=1)
    ingredients = (seasonal, ar, smooth, walk)
    kind = rng.integers(0, len(ingredients), size=n)
    calm_kind = rng.integers(0, 4, size=n)
    level = rng.normal(0.0, 2.0, size=n)
    scale = np.exp(rng.uniform(np.log(0.4), np.log(12.0), size=n))
    dynamic_amp = rng.uniform(0.6, 2.2, size=n)
    start_calm = rng.random(n) < 0.65
    out = np.empty((n, L), dtype=np.float64)
    for row in range(n):
        dynamic = ingredients[int(kind[row])][row].copy()
        calibration = dynamic[:min(L, 512)]
        calibration_mean = float(calibration.mean())
        calibration_std = float(calibration.std())
        dynamic -= calibration_mean
        if calibration_std > 1e-12:
            dynamic /= calibration_std
        current = float(level[row])
        calm = bool(start_calm[row])
        pos = 0
        segment_index = 0
        while pos < L:
            if calm:
                seg_len = int(rng.integers(_ps_CS_CALM_LO, _ps_CS_CALM_HI + 1))
            else:
                seg_len = int(rng.integers(_ps_CS_DYNAMIC_LO, _ps_CS_DYNAMIC_HI + 1))
            if segment_index == 0 and L >= 2 * _ps_CS_DYNAMIC_LO:
                seg_len = min(seg_len, L - _ps_CS_DYNAMIC_LO)
            end = min(pos + max(seg_len, 1), L)
            span = end - pos
            if calm:
                mode = int(calm_kind[row])
                if mode == 0:
                    values = np.full(span, current)
                elif mode == 1:
                    drift = rng.normal(0.0, 0.0025, size=span).cumsum()
                    drift += np.linspace(0.0, float(rng.normal(0.0, 0.025)), span)
                    values = current + drift
                elif mode == 2:
                    count_level = max(0.0, float(np.rint(abs(current) * 8.0)))
                    updates = rng.random(span) < 0.025
                    changes = updates * rng.choice([-1.0, 1.0], size=span)
                    values = np.maximum(count_level + np.cumsum(changes), 0.0)
                else:
                    events = rng.random(span) < 0.012
                    values = events * rng.gamma(1.5, 0.35, size=span)
            else:
                piece = dynamic[pos:end] * float(dynamic_amp[row])
                values = piece - piece[0] + current
                if span > 1 and np.ptp(values) < 1e-10:
                    values = current + np.linspace(0.0, 1.0, span)
            out[row, pos:end] = values
            current = float(values[-1])
            pos = end
            calm = not calm
            segment_index += 1
        row_scale = 1.0 if int(calm_kind[row]) == 2 else float(scale[row])
        out[row] *= row_scale
        if L > 1 and np.ptp(out[row]) < 1e-10:
            out[row, -1] += max(0.001, 0.01 * row_scale)
    return out

class _F:

    @staticmethod
    def ar2(rng, k, L):
        phi2 = rng.uniform(-0.9, 0.9, k)
        phi1 = rng.uniform(-1.0, 1.0, k) * (1.0 - phi2) * 0.97
        sigma = _logu(rng, 0.2, 2.0, k)
        e = rng.normal(0.0, 1.0, (k, L)) * sigma[:, None]
        x = _ar2_batch(e, phi1, phi2)
        b = rng.normal(0.0, 1.5, k) * (rng.random(k) < 0.3)
        x = _standardize(x) + b[:, None] * (np.arange(L) / L)[None, :]
        return _F._scale_signed(rng, x, k)

    @staticmethod
    def integrated(rng, k, L):
        phi = rng.uniform(0.0, 0.6, k)
        sigma = _logu(rng, 0.3, 1.5, k)
        drift = rng.normal(0.0, 0.25, k) * sigma
        e = rng.normal(0.0, 1.0, (k, L)) * sigma[:, None]
        base = _ar1_batch(e, phi) + drift[:, None]
        x = np.cumsum(base, axis=1)
        dbl = rng.random(k) < 0.25
        if dbl.any():
            x[dbl] = np.cumsum(x[dbl], axis=1) / L
        rip = rng.random(k) < 0.3
        P = _logu(rng, 16, 512, k)
        amp = rng.uniform(0.0, 0.4, k) * rip
        ph = rng.uniform(0, 2 * np.pi, k)
        t = np.arange(L)[None, :]
        x = _standardize(x)
        x += amp[:, None] * np.sin(2 * np.pi * t / P[:, None] + ph[:, None])
        return _F._scale_signed(rng, x, k)

    @staticmethod
    def regime_shift(rng, k, L):
        t = np.arange(L)[None, :]
        p_mean = 1.0 / _logu(rng, 96, 1200, k)
        cp = rng.random((k, L)) < p_mean[:, None]
        cp[:, 0] = False
        jscale = _logu(rng, 0.5, 2.5, k)
        jumps = np.zeros((k, L))
        rr, cc = np.nonzero(cp)
        if rr.size:
            jumps[rr, cc] = rng.normal(0.0, 1.0, rr.size) * jscale[rr]
        mean = np.cumsum(jumps, axis=1)
        p_var = 1.0 / _logu(rng, 128, 1500, k)
        cpv = rng.random((k, L)) < p_var[:, None]
        vj = np.zeros((k, L))
        rr, cc = np.nonzero(cpv)
        if rr.size:
            vj[rr, cc] = rng.normal(0.0, 0.6, rr.size)
        lsig = np.clip(np.cumsum(vj, axis=1), -1.5, 1.5)
        phi = rng.uniform(0.0, 0.8, k)
        e = rng.normal(0.0, 1.0, (k, L)) * np.exp(lsig)
        noise = _ar1_batch(e, phi)
        x = mean + noise
        rec = rng.random(k) < 0.3
        if rec.any():
            kk = int(rec.sum())
            imp_rate = (1.0 / _logu(rng, 300, 2500, kk))[:, None]
            imp = np.where(rng.random((kk, L)) < imp_rate, rng.normal(0, 1, (kk, L)) * _logu(rng, 2.0, 8.0, kk)[:, None], 0.0)
            rho = rng.uniform(0.9, 0.995, kk)
            xr = x[rec]
            for i in range(kk):
                xr[i] += lfilter([1.0], [1.0, -float(rho[i])], imp[i])
            x[rec] = xr
        sl = rng.random(k) < 0.4
        if sl.any():
            cps = rng.random((sl.sum(), L)) < (1.0 / _logu(rng, 256, 2048, int(sl.sum())))[:, None]
            dslope = np.where(cps, rng.normal(0, 1.0, cps.shape), 0.0)
            slope = np.cumsum(dslope, axis=1)
            x[sl] += np.cumsum(slope, axis=1) / L * _logu(rng, 0.5, 3.0, int(sl.sum()))[:, None]
        _ = t
        return _F._scale_signed(rng, _standardize(x), k)

    @staticmethod
    def trend_seasonal_ar(rng, k, L):
        t1 = np.arange(L, dtype=np.float64)
        n_active = rng.integers(1, 4, k)
        Pm = _logu(rng, L / 8, 2 * L, k)
        m = rng.uniform(0.0, 0.6, k)
        phm = rng.uniform(0, 2 * np.pi, k)
        am = 1.0 + m[:, None] * np.sin(2 * np.pi * t1[None, :] / Pm[:, None] + phm[:, None])
        Pd = _logu(rng, L / 4, 4 * L, k)
        dph = rng.uniform(0.2, 1.0, k)
        drifted = rng.random(k) < 0.35
        seas = np.zeros((k, L))
        for j in range(3):
            Pj = _periods(rng, k)
            Aj = _logu(rng, 0.2, 2.0, k)
            phj = rng.uniform(0, 2 * np.pi, k)
            h2j = rng.uniform(0.1, 0.5, k)
            h2on = rng.random(k) < 0.5
            idx = np.flatnonzero(j < n_active)
            if idx.size == 0:
                continue
            arg = 2 * np.pi * t1[None, :] / Pj[idx, None] + phj[idx, None]
            dl = drifted[idx]
            if dl.any():
                ii = idx[dl]
                arg[dl] += dph[ii, None] * np.sin(2 * np.pi * t1[None, :] / Pd[ii, None])
            comp = np.sin(arg)
            hl = h2on[idx]
            if hl.any():
                comp[hl] += h2j[idx[hl], None] * np.sin(2 * arg[hl])
            seas[idx] += Aj[idx, None] * comp
        seas *= am
        tl = t1[None, :] / L
        b = rng.normal(0.0, 2.0, k)
        c = rng.normal(0.0, 1.0, k)
        trend = b[:, None] * tl + c[:, None] * tl * tl
        phi = rng.uniform(0.0, 0.9, k)
        clean = rng.random(k) < 0.4
        sig = np.where(clean, rng.uniform(0.02, 0.12, k), _logu(rng, 0.1, 0.6, k))
        noise = _ar1_batch(rng.normal(0, 1, (k, L)) * sig[:, None], phi)
        return _F._scale_signed(rng, _standardize(seas + trend + noise), k)

    @staticmethod
    def ou_stochastic_vol(rng, k, L):
        kappa = rng.uniform(0.005, 0.1, k)
        mu_h = rng.uniform(-1.0, 0.5, k)
        eta = rng.uniform(0.05, 0.4, k)
        xi = rng.normal(0.0, 1.0, (k, L))
        drive = kappa[:, None] * mu_h[:, None] + eta[:, None] * xi
        h = np.empty((k, L))
        for i in range(k):
            p = 1.0 - float(kappa[i])
            h[i] = lfilter([1.0], [1.0, -p], drive[i], zi=[p * float(mu_h[i])])[0]
        z = rng.normal(0.0, 1.0, (k, L))
        r = np.exp(np.clip(h, -6.0, 4.0)) * z
        mode = rng.random(k)
        x = np.where((mode < 0.75)[:, None], np.cumsum(r, axis=1), r)
        lvl = mode < 0.25
        if lvl.any():
            cs = np.cumsum(r[lvl], axis=1)
            cs = cs / np.maximum(np.abs(cs).max(axis=1, keepdims=True), 1e-09) * rng.uniform(1.0, 8.0, (int(lvl.sum()), 1))
            x[lvl] = np.exp(np.clip(cs, -12, 12))
            sc = _logu(rng, 0.1, 1000.0, int(lvl.sum()))
            x[lvl] *= sc[:, None]
            oth = ~lvl
            x[oth] = _F._scale_signed(rng, _standardize(x[oth]), int(oth.sum()))
            return x
        return _F._scale_signed(rng, _standardize(x), k)

    @staticmethod
    def threshold_ar(rng, k, L):
        phi_lo = rng.uniform(0.3, 0.98, k)
        phi_hi = rng.uniform(-0.5, 0.9, k)
        c_lo = rng.uniform(0.0, 0.5, k)
        c_hi = rng.uniform(-0.5, 0.0, k)
        tau = rng.normal(0.0, 0.5, k)
        sig = _logu(rng, 0.2, 1.5, k)
        e = rng.normal(0.0, 1.0, (k, L)) * sig[:, None]
        x = np.empty((k, L))
        prev = e[:, 0]
        x[:, 0] = prev
        for t in range(1, L):
            lo = prev < tau
            prev = np.where(lo, phi_lo * prev + c_lo, phi_hi * prev + c_hi) + e[:, t]
            x[:, t] = prev
        return _F._scale_signed(rng, _standardize(x), k)

    @staticmethod
    def spectral_gp(rng, k, L):
        F = L // 2 + 1
        f = np.fft.rfftfreq(L)
        f0 = f.copy()
        f0[0] = f0[1]
        alpha = rng.uniform(0.0, 2.0, k)
        psd = f0[None, :] ** (-alpha[:, None])
        psd[:, 0] = 0.0
        nb = rng.integers(1, 4, k)
        for j in range(3):
            on = (j < nb).astype(np.float64)
            fc = _logu(rng, 1.0 / 2048, 0.4, k)
            w = fc * rng.uniform(0.05, 0.5, k)
            h = _logu(rng, 0.1, 10.0, k) * psd.max(axis=1)
            psd += (on * h)[:, None] * np.exp(-0.5 * ((f[None, :] - fc[:, None]) / w[:, None]) ** 2)
        zr = rng.normal(0.0, 1.0, (k, F))
        zi = rng.normal(0.0, 1.0, (k, F))
        spec = (zr + 1j * zi) * np.sqrt(psd / 2.0)
        x = np.fft.irfft(spec, n=L, axis=1)
        x = _standardize(x)
        b = rng.normal(0.0, 1.5, k) * (rng.random(k) < 0.3)
        x += b[:, None] * (np.arange(L) / L)[None, :]
        return _F._scale_signed(rng, x, k)

    @staticmethod
    def long_memory(rng, k, L):
        h_idx = rng.integers(0, _H_GRID.size, k)
        lam = np.empty((k, 2 * L))
        for u in np.unique(h_idx):
            lam[h_idx == u] = _dh_eigs(int(u), L)[None, :]
        m = 2 * L
        zr = rng.normal(0.0, 1.0, (k, m))
        zi = rng.normal(0.0, 1.0, (k, m))
        w = np.fft.fft(np.sqrt(lam / (2.0 * m)) * (zr + 1j * zi), axis=1)
        fgn = w.real[:, :L] * math.sqrt(2.0)
        as_fbm = rng.random(k) < 0.5
        x = np.where(as_fbm[:, None], np.cumsum(fgn, axis=1), fgn)
        return _F._scale_signed(rng, _standardize(x), k)

    @staticmethod
    def multiplicative(rng, k, L):
        t = np.arange(L)[None, :] / L
        a = rng.normal(0.0, 1.0, k)
        rw = np.cumsum(rng.normal(0, 1, (k, L)), axis=1)
        rw = rw / np.maximum(np.abs(rw).max(axis=1, keepdims=True), 1e-09) * rng.uniform(0.0, 0.8, (k, 1))
        trend = np.exp(np.clip(a[:, None] * t + rw, -6, 6))
        P = _periods(rng, (k, 2), lo=8, hi=700)
        mamp = rng.uniform(0.05, 0.7, (k, 2))
        ph = rng.uniform(0, 2 * np.pi, (k, 2))
        seas = np.ones((k, L))
        for j in range(2):
            seas *= 1.0 + mamp[:, j:j + 1] * np.sin(2 * np.pi * np.arange(L)[None, :] / P[:, j:j + 1] + ph[:, j:j + 1])
        seas = np.maximum(seas, 0.05)
        phi = rng.uniform(0.0, 0.8, k)
        sig = rng.uniform(0.02, 0.25, k)
        noise = np.exp(np.clip(_ar1_batch(rng.normal(0, 1, (k, L)) * sig[:, None], phi), -3, 3))
        y = trend * seas * noise
        return _F._scale_positive(rng, y, k)

    @staticmethod
    def chaotic(rng, k, L):
        burn = 128
        T = L + burn
        which = rng.integers(0, 3, k)
        s0 = rng.uniform(0.2, 0.8, k)
        r = rng.uniform(3.9, 3.99, k)
        a_t = rng.uniform(1.8, 1.999, k)
        hx0 = rng.uniform(-0.5, 0.5, k)
        hy0 = rng.uniform(-0.2, 0.2, k)
        x = np.empty((k, T))
        m = which == 0
        if m.any():
            s = s0[m].copy()
            rr = r[m]
            for t in range(T):
                s = rr * s * (1.0 - s)
                x[np.flatnonzero(m), t] = s
        m = which == 1
        if m.any():
            s = s0[m].copy()
            aa = a_t[m]
            for t in range(T):
                s = aa * np.minimum(s, 1.0 - s)
                x[np.flatnonzero(m), t] = s
        m = which == 2
        if m.any():
            hx = hx0[m].copy()
            hy = hy0[m].copy()
            rows = np.flatnonzero(m)
            for t in range(T):
                nx = 1.0 - 1.4 * hx * hx + hy
                hy = 0.3 * hx
                hx = nx
                esc = np.abs(hx) > 2.0
                hx = np.where(esc, 0.1, hx)
                hy = np.where(esc, 0.1, hy)
                x[rows, t] = hx
        x = x[:, burn:]
        wlen = rng.choice([1, 2, 4, 8], k)
        for w in (2, 4, 8):
            m = wlen == w
            if m.any():
                c = np.cumsum(np.pad(x[m], ((0, 0), (w, 0))), axis=1)
                x[m] = (c[:, w:] - c[:, :-w]) / w
        b = rng.normal(0.0, 1.0, k) * (rng.random(k) < 0.3)
        x = _standardize(x) + b[:, None] * (np.arange(L) / L)[None, :]
        return _F._scale_signed(rng, x, k)

    @staticmethod
    def tidal_harmonic(rng, k, L):
        t = np.arange(L, dtype=np.float64)[None, :]
        P0 = _logu(rng, 16, 400, k)
        ratios = np.array([1.0, 0.9661, 1.0191, 1.927, 2.0787])
        n_c = rng.integers(2, 5, k)
        x = np.zeros((k, L))
        for j in range(5):
            use = (j < n_c).astype(np.float64)
            jitter = 1.0 + rng.uniform(-0.004, 0.004, k)
            Pj = P0 * ratios[j] * jitter
            A = np.where(j == 0, 1.0, rng.uniform(0.15, 0.7, k)) * use
            ph = rng.uniform(0, 2 * np.pi, k)
            x += A[:, None] * np.sin(2 * np.pi * t / Pj[:, None] + ph[:, None])
        Pm = P0 * rng.uniform(10, 40, k)
        m = rng.uniform(0.1, 0.5, k)
        x *= 1.0 + m[:, None] * np.sin(2 * np.pi * t / Pm[:, None] + rng.uniform(0, 2 * np.pi, k)[:, None])
        phi = rng.uniform(0.2, 0.9, k)
        sig = rng.uniform(0.03, 0.15, k)
        x += _ar1_batch(rng.normal(0, 1, (k, L)) * sig[:, None], phi)
        x += np.cumsum(rng.normal(0, 1, (k, L)), axis=1) * rng.uniform(0.0, 0.02, k)[:, None] / math.sqrt(L)
        x = _standardize(x)
        datum = rng.random(k) < 0.4
        off = _logu(rng, 2.0, 50.0, k) * datum
        return _F._scale_signed(rng, x, k, extra_offset=off)

    @staticmethod
    def flow_recession(rng, k, L):
        t = np.arange(L)[None, :]
        lam0 = rng.uniform(0.01, 0.15, k)
        Ps = _logu(rng, 200, 2000, k)
        amp = rng.uniform(0.0, 0.9, k)
        lam = lam0[:, None] * (1.0 + amp[:, None] * np.sin(2 * np.pi * t / Ps[:, None] + rng.uniform(0, 2 * np.pi, k)[:, None]))
        lam = np.maximum(lam, 0.005)
        occ = rng.random((k, L)) < lam
        marks = np.exp(rng.normal(rng.uniform(-0.5, 1.5, k)[:, None], rng.uniform(0.6, 1.2, k)[:, None], (k, L))) * occ
        rho_f = rng.uniform(0.55, 0.92, k)
        rho_s = rng.uniform(0.96, 0.998, k)
        beta = rng.uniform(0.1, 0.5, k)
        q = np.empty((k, L))
        for i in range(k):
            fast = lfilter([1.0 - float(beta[i])], [1.0, -float(rho_f[i])], marks[i])
            slow = lfilter([float(beta[i])], [1.0, -float(rho_s[i])], marks[i])
            q[i] = fast + slow
        base = _logu(rng, 0.02, 0.5, k)[:, None] * (1.0 + 0.3 * np.sin(2 * np.pi * t / Ps[:, None]))
        q = q + np.maximum(base, 0.005)
        logmode = rng.random(k) < 0.25
        sc = _logu(rng, 0.05, 500.0, k)
        y = q * sc[:, None]
        if logmode.any():
            y[logmode] = np.log(q[logmode] + 1e-06)
        return y

    @staticmethod
    def step_level(rng, k, L):
        p = 1.0 / _logu(rng, 48, 1024, k)
        cp = rng.random((k, L)) < p[:, None]
        cp[:, 0] = False
        jscale = _logu(rng, 0.3, 2.0, k)
        jumps = np.zeros((k, L))
        rr, cc = np.nonzero(cp)
        if rr.size:
            jumps[rr, cc] = rng.standard_t(3, rr.size) * jscale[rr]
        level = np.cumsum(jumps, axis=1)
        qz = rng.random(k) < 0.5
        if qz.any():
            span = np.maximum(np.abs(level[qz]).max(axis=1, keepdims=True), 1e-06)
            nlev = rng.integers(10, 200, int(qz.sum()))[:, None]
            g = span / nlev
            level[qz] = np.round(level[qz] / g) * g
        nur = rng.random(k) < 0.5
        if nur.any():
            phi = rng.uniform(0.985, 0.9995, int(nur.sum()))
            e = rng.normal(0, 1, (int(nur.sum()), L)) * rng.uniform(0.005, 0.05, int(nur.sum()))[:, None]
            level[nur] += _ar1_batch(e, phi) * np.maximum(np.abs(level[nur]).max(axis=1, keepdims=True), 1.0)
        micro = rng.uniform(0.001, 0.01, k)[:, None] * np.maximum(level.std(axis=1, keepdims=True), 1.0)
        level = level + rng.normal(0, 1, (k, L)) * micro
        return _F._scale_signed(rng, _standardize(level), k)

    @staticmethod
    def vol_regime_switch(rng, k, L):
        rate1 = np.exp(rng.uniform(math.log(0.0005), math.log(0.015), k))[:, None]
        rate2 = np.exp(rng.uniform(math.log(0.0002), math.log(0.005), k))[:, None]
        reg1 = np.bitwise_and(np.cumsum(rng.random((k, L)) < rate1, axis=1), 1)
        reg2 = np.bitwise_and(np.cumsum(rng.random((k, L)) < rate2, axis=1), 1)
        r1 = rng.uniform(2.0, 6.0, k)[:, None]
        r2 = rng.uniform(2.0, 6.0, k)[:, None]
        sig = r1 ** reg1 * np.where(rng.random(k)[:, None] < 0.5, r2 ** reg2, 1.0)
        df = rng.uniform(3.0, 8.0, k)
        innov = rng.standard_t(df[:, None], (k, L))
        r = sig * innov
        r *= _logu(rng, 0.002, 0.05, k)[:, None]
        mode = rng.random(k)
        x = np.where((mode < 0.7)[:, None], np.cumsum(r, axis=1), r)
        lvl = mode < 0.2
        if lvl.any():
            cs = np.clip(np.cumsum(r[lvl], axis=1), -12, 12)
            x[lvl] = np.exp(cs) * _logu(rng, 0.5, 5000.0, int(lvl.sum()))[:, None]
            oth = ~lvl
            x[oth] = _F._scale_signed(rng, _standardize(x[oth]), int(oth.sum()))
            return x
        return _F._scale_signed(rng, _standardize(x), k)

    @staticmethod
    def weekly_demand(rng, k, L):
        t = np.arange(L)
        hourly = rng.random(k) < 0.4
        Pw = np.where(hourly, 168, 7)
        Py = np.where(hourly, 168 * 52, 364)
        prof = np.ones((k, L))
        for grp, pw in ((~hourly, 7), (hourly, 168)):
            if not grp.any():
                continue
            kk = int(grp.sum())
            if pw == 7:
                base = np.ones((kk, 7))
                wk_mult = rng.uniform(0.15, 1.6, kk)
                base[:, 5:] *= wk_mult[:, None]
                base *= np.exp(rng.normal(0, 0.15, (kk, 7)))
            else:
                hod = np.arange(24)
                m1 = np.exp(-0.5 * ((hod - rng.uniform(7, 10, (kk, 1))) / rng.uniform(1.5, 3, (kk, 1))) ** 2)
                m2 = np.exp(-0.5 * ((hod - rng.uniform(17, 20, (kk, 1))) / rng.uniform(1.5, 3, (kk, 1))) ** 2)
                day = 0.15 + m1 + rng.uniform(0.3, 1.2, (kk, 1)) * m2
                base = np.tile(day, (1, 7))
                wk_mult = rng.uniform(0.15, 1.6, kk)
                base[:, 24 * 5:] *= wk_mult[:, None]
            prof[grp] = base[:, t % pw]
        ay = rng.uniform(0.0, 0.5, k)
        prof *= 1.0 + ay[:, None] * np.sin(2 * np.pi * t[None, :] / Py[:, None] + rng.uniform(0, 2 * np.pi, k)[:, None])
        n_h = rng.integers(6, 15, k)
        maxh = 14
        pos = rng.integers(0, 364 * 24, (k, maxh))
        eff_dip = rng.uniform(0.05, 0.6, (k, maxh))
        eff_spk = rng.uniform(1.5, 4.0, (k, maxh))
        is_spk = rng.random((k, maxh)) < 0.3
        width = rng.integers(1, 4, (k, maxh))
        eff = np.where(is_spk, eff_spk, eff_dip)
        doy = t[None, :] % Py[:, None]
        hol = np.ones((k, L))
        daily = ~hourly
        if daily.any():
            kk = int(daily.sum())
            prof_y = np.ones((kk, 364))
            rows = np.arange(kk)
            pd_ = pos[daily] % 364
            for j in range(maxh):
                on = j < n_h[daily]
                for wd in range(3):
                    hitw = on & (width[daily, j] > wd)
                    prof_y[rows[hitw], (pd_[hitw, j] + wd) % 364] = eff[daily, j][hitw]
            hol[daily] = prof_y[np.arange(kk)[:, None], t[None, :] % 364]
        if hourly.any():
            kk = np.flatnonzero(hourly)
            for j in range(maxh):
                on = j < n_h[kk]
                pj = (pos[kk, j] % Py[kk]).astype(np.int64)
                d = (doy[kk] - pj[:, None]) % Py[kk][:, None]
                hit = (d < width[kk, j][:, None]) & on[:, None]
                hol[kk] = np.where(hit, eff[kk, j][:, None], hol[kk])
        g = rng.uniform(-0.5, 1.5, k)
        trend = np.maximum(1.0 + g[:, None] * (t[None, :] / L), 0.1)
        intensity = prof * hol * trend
        nu = rng.uniform(4.0, 60.0, k)
        y = intensity * rng.gamma(nu[:, None], 1.0 / nu[:, None], (k, L))
        cnt = rng.random(k) < 0.3
        if cnt.any():
            mscale = _logu(rng, 2.0, 500.0, int(cnt.sum()))
            lam = intensity[cnt] / np.maximum(intensity[cnt].mean(axis=1, keepdims=True), 1e-09) * mscale[:, None]
            y[cnt] = rng.poisson(np.clip(lam, 0, 100000.0)).astype(np.float64)
            oth = ~cnt
            y[oth] = _F._scale_positive(rng, y[oth], int(oth.sum()))
            return y
        return _F._scale_positive(rng, y, k)

    @staticmethod
    def seasonal_counts(rng, k, L):
        t = np.arange(L)[None, :]
        a = rng.uniform(-1.5, 4.0, k)
        P1 = _logu(rng, 6, 500, k)
        P2 = _logu(rng, 6, 500, k)
        b1 = rng.uniform(0.2, 1.5, k)
        b2 = rng.uniform(0.0, 1.0, k)
        tr = rng.uniform(-1.0, 1.5, k)
        loglam = a[:, None] + b1[:, None] * np.sin(2 * np.pi * t / P1[:, None] + rng.uniform(0, 2 * np.pi, k)[:, None]) + b2[:, None] * np.sin(2 * np.pi * t / P2[:, None] + rng.uniform(0, 2 * np.pi, k)[:, None]) + tr[:, None] * t / L
        lam = np.exp(np.clip(loglam, -6, 9))
        over = rng.random(k) < 0.5
        rdisp = rng.uniform(1.0, 10.0, k)
        mix = np.where(over[:, None], rng.gamma(rdisp[:, None], 1.0 / rdisp[:, None], (k, L)), 1.0)
        return rng.poisson(np.clip(lam * mix, 0, 100000.0)).astype(np.float64)

    @staticmethod
    def intermittent_demand(rng, k, L):
        p_on = rng.uniform(0.01, 0.15, k)
        p_off = rng.uniform(0.2, 0.8, k)
        u = rng.random((k, L))
        state = np.zeros(k, dtype=bool)
        occ = np.empty((k, L), dtype=bool)
        for t in range(L):
            turn_on = ~state & (u[:, t] < p_on)
            turn_off = state & (u[:, t] < p_off)
            state = (state | turn_on) & ~turn_off
            occ[:, t] = state
        mu = rng.uniform(0.0, 2.5, k)
        sg = rng.uniform(0.3, 1.0, k)
        sizes = np.exp(rng.normal(mu[:, None], sg[:, None], (k, L)))
        rounded = rng.random(k) < 0.6
        y = np.where(occ, sizes, 0.0)
        if rounded.any():
            y[rounded] = np.round(y[rounded])
        sc = _logu(rng, 0.5, 50.0, k)
        return y * sc[:, None]

    @staticmethod
    def _scale_signed(rng, x, k, extra_offset=None):
        sc = _logu(rng, 0.02, 200.0, k)
        off = rng.normal(0.0, 3.0, k) * sc
        if extra_offset is not None:
            off = off + extra_offset * sc
        return x * sc[:, None] + off[:, None]

    @staticmethod
    def _scale_positive(rng, y, k):
        sc = _logu(rng, 0.05, 200.0, k)
        med = np.maximum(np.median(y, axis=1), 1e-09)
        return y / med[:, None] * sc[:, None]

    @staticmethod
    def policy_rate(rng, k, L):
        """Administered/policy rates: long EXACT flats between decisions, moves
        snapped to a coarse grid (quarter-point), persistent hike/cut runs, and
        a level that lives on its own scale rather than a standardized one.

        step_level already covers generic level shifts, but its jumps are
        t-distributed and continuous, so it never produces the "held at 4.25
        for eight months, then +0.25" structure that policy-rate and
        administered-price feeds are made of. Those series are trivially
        forecastable by persistence and punish any model that adds noise to a
        flat line, so the shape needs its own mass.
        """
        if k <= 0:
            return np.empty((0, L), dtype=np.float64)
        rate = rng.uniform(3.0, 16.0, k) / max(L, 1)
        moves = rng.random((k, L)) < rate[:, None]
        moves[:, 0] = False
        grid = rng.choice(np.array([0.25, 0.5, 1.0]), size=k)
        # persistent direction: hikes cluster with hikes
        drift = _ar1_batch(rng.normal(0, 1, (k, L)), rng.uniform(0.9, 0.99, k))
        mag = np.ceil(np.abs(rng.normal(0.0, 0.8, (k, L))))
        steps = np.sign(drift) * grid[:, None] * mag
        level = rng.uniform(-2.0, 8.0, k)[:, None] * grid[:, None] \
            + np.cumsum(np.where(moves, steps, 0.0), axis=1)
        # a minority drift slowly instead of holding exactly flat (yield curves)
        drifty = rng.random(k) < 0.3
        if drifty.any():
            kk = int(drifty.sum())
            slow = _ar1_batch(rng.normal(0.0, 0.02, (kk, L)),
                              rng.uniform(0.97, 0.999, kk))
            level[drifty] += slow
        sc = _logu(rng, 0.5, 50.0, k)
        return level * sc[:, None]

    @staticmethod
    def physical_sensors(rng, k, L):
        return _ps_physical_sensors(rng, k, L)

    @staticmethod
    def conditional_stability(rng, k, L):
        return _ps_conditional_stability(rng, k, L)
_FAMILY_ORDER = ('ar2', 'integrated', 'regime_shift', 'trend_seasonal_ar', 'ou_stochastic_vol', 'threshold_ar', 'spectral_gp', 'long_memory', 'multiplicative', 'chaotic', 'tidal_harmonic', 'flow_recession', 'step_level', 'vol_regime_switch', 'weekly_demand', 'seasonal_counts', 'intermittent_demand', 'physical_sensors', 'conditional_stability', 'policy_rate')
_NO_OBS = frozenset({'seasonal_counts', 'intermittent_demand', 'step_level', 'physical_sensors'})

class Generator(DataGenerator):

    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = os.path.join(config_dir, 'config.json')
        user_cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                user_cfg = json.load(f)
        self.cfg = _merge(_DEFAULTS, user_cfg)
        self.L = int(self.cfg['generate_length'])
        self.batch_size = int(self.cfg['batch_size'])
        self.max_abs = float(self.cfg['max_abs_value'])
        self.rng = np.random.default_rng(seed)
        self._points = 0
        tw = self.cfg['family_weights']
        cur = self.cfg['curriculum']
        sw = cur.get('start_family_weights', tw)
        self._target = np.array([float(tw.get(f, 0.0)) for f in _FAMILY_ORDER])
        self._start = np.array([float(sw.get(f, 0.0)) for f in _FAMILY_ORDER])
        self._target = self._target / self._target.sum()
        self._start = self._start / max(self._start.sum(), 1e-12)
        self._cur_on = bool(cur.get('enabled', True))
        self._cur_pts = float(cur.get('total_points', 2000000000.0))
        self._cur_full_at = float(cur.get('full_mixture_at', 0.7))
        obs = self.cfg['observation_layer']
        self._obs_on = bool(obs.get('enabled', True))
        self._obs_frac = float(obs.get('apply_frac', 0.35))
        self._record = bool(self.cfg.get('record_families', False))
        self.families_emitted: list[str] = []

    def _weights(self) -> np.ndarray:
        if not self._cur_on:
            return self._target
        horizon = max(self._cur_pts * self._cur_full_at, 1.0)
        s = _smoothstep(self._points / horizon)
        w = (1.0 - s) * self._start + s * self._target
        return w / w.sum()

    def _observe(self, x: np.ndarray, fam: str) -> np.ndarray:
        if not self._obs_on or fam in _NO_OBS:
            return x
        rng = self.rng
        k, L = x.shape
        apply = rng.random(k) < self._obs_frac
        m = apply & (rng.random(k) < 0.35)
        if m.any():
            lo = np.percentile(x[m], 5, axis=1, keepdims=True)
            hi = np.percentile(x[m], 95, axis=1, keepdims=True)
            n = _logu(rng, 20, 500, int(m.sum()))[:, None]
            g = np.maximum((hi - lo) / n, 1e-09)
            x[m] = np.round(x[m] / g) * g
        m = apply & (rng.random(k) < 0.3)
        if m.any():
            rho = rng.uniform(0.05, 0.5, int(m.sum()))[:, None]
            keep = rng.random((int(m.sum()), L)) >= rho
            keep[:, 0] = True
            x[m] = _ffill(x[m], keep)
        m = apply & (rng.random(k) < 0.2)
        if m.any():
            kk = int(m.sum())
            upper = rng.random(kk) < 0.5
            q = np.where(upper, rng.uniform(0.85, 0.99, kk), rng.uniform(0.01, 0.15, kk))
            xs = np.sort(x[m], axis=1)
            th = xs[np.arange(kk), (q * (L - 1)).astype(np.int64)][:, None]
            x[m] = np.where(upper[:, None], np.minimum(x[m], th), np.maximum(x[m], th))
        m = apply & (rng.random(k) < 0.25)
        if m.any():
            kk = int(m.sum())
            nsp = rng.integers(1, 6, kk)
            pos = rng.integers(0, L, (kk, 5))
            iqr = np.maximum(np.percentile(x[m], 75, axis=1) - np.percentile(x[m], 25, axis=1), 1e-09)
            mag = rng.normal(0, 1, (kk, 5)) * (iqr * rng.uniform(3, 10, kk))[:, None]
            xm = x[m]
            for j in range(5):
                on = j < nsp
                xm[np.arange(kk)[on], pos[on, j]] += mag[on, j]
            x[m] = xm
        return x

    def _build_batch(self, B: int) -> tuple[np.ndarray, list[str]]:
        L = self.L
        w = self._weights()
        counts = self.rng.multinomial(B, w)
        chunks: list[np.ndarray] = []
        fams: list[str] = []
        for fam, c in zip(_FAMILY_ORDER, counts):
            if c == 0:
                continue
            x = getattr(_F, fam)(self.rng, int(c), L)
            x = self._observe(x, fam)
            chunks.append(x)
            fams.extend([fam] * int(c))
        batch = np.concatenate(chunks, axis=0)
        np.nan_to_num(batch, copy=False, nan=0.0, posinf=self.max_abs, neginf=-self.max_abs)
        np.clip(batch, -self.max_abs, self.max_abs, out=batch)
        perm = self.rng.permutation(B)
        self._points += B * L
        return (batch[perm], [fams[i] for i in perm])

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        if n_series <= 0:
            return
        queue: Queue = Queue(maxsize=1)
        stop = Event()
        done = object()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    queue.put(item, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def produce() -> None:
            try:
                remaining = int(n_series)
                while remaining > 0 and (not stop.is_set()):
                    B = min(self.batch_size, remaining)
                    if not put(self._build_batch(B)):
                        return
                    remaining -= B
            except BaseException as exc:
                put(exc)
            finally:
                put(done)
        producer = Thread(target=produce, name='fulltail-producer', daemon=True)
        producer.start()
        try:
            while True:
                item = queue.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                batch, fams = item
                if self._record:
                    self.families_emitted.extend(fams)
                for row in batch:
                    yield row
        finally:
            stop.set()
            producer.join(timeout=1.0)

    @property
    def name(self) -> str:
        return 'cascade-fulltail-v2'
