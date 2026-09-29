from __future__ import annotations
import json
from collections.abc import Iterator
from functools import lru_cache, partial
from pathlib import Path
from queue import Full, Queue
from threading import Event, Lock, Thread
import numpy as np
from numba import njit
from scipy.signal import lfilter
from cascade.interface import DataGenerator
_qsv00 = False
_qsv01 = Lock()
_qsv02 = 2048
_qsv03 = 256
_qsv04 = 1024
_qsv05 = np.array([4, 7, 12, 15, 24, 30, 48, 52, 60, 90, 96, 144, 168, 183, 240, 288, 336, 365, 672, 730], dtype=np.float64)
_qsv06 = np.array([0.01, 0.23, 0.02, 0.02, 0.07, 0.01, 0.06, 0.01, 0.08, 0.01, 0.14, 0.07, 0.04, 0.01, 0.08, 0.08, 0.02, 0.02, 0.01, 0.01], dtype=np.float64)
_qsv06 /= _qsv06.sum()
_qsv07 = np.array([[15, 60], [60, 240], [24, 168], [48, 336], [96, 672], [7, 365], [12, 52]], dtype=np.float64)
_qsv08: tuple[str, ...] = ('k00', 'k01', 'k02', 'k03', 'k04', 'k05', 'k06', 'k07', 'k08', 'k09', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15', 'k16', 'k17', 'k18', 'k19', 'k20', 'k21', 'k22', 'k23', 'k24', 'k25', 'k26', 'k27', 'k28', 'k29', 'k30', 'k31')
_qsv09: dict[str, float] = {'k28': 0.0, 'k29': 0.0, 'k30': 0.0, 'k31': 0.0, 'k00': 0.095, 'k01': 0.095, 'k02': 0.06, 'k03': 0.105, 'k04': 0.095, 'k05': 0.06, 'k06': 0.02, 'k07': 0.07, 'k08': 0.07, 'k09': 0.08, 'k10': 0.08, 'k11': 0.06, 'k12': 0.02, 'k13': 0.02, 'k14': 0.07, 'k15': 0.0, 'k16': 0.0, 'k17': 0.0, 'k18': 0.0, 'k19': 0.0, 'k20': 0.0, 'k21': 0.0, 'k22': 0.0, 'k23': 0.0, 'k24': 0.0, 'k25': 0.0, 'k26': 0.0, 'k27': 0.0}
_qsv10: frozenset[str] = frozenset({'k21', 'k15', 'k18', 'k17', 'k19'})

def _k15(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv34 = _qsv33.uniform(1.0, 8.0, size=(n, 1))
    at = _qsv33.random((n, L)) < _qsv34 / max(L, 1)
    at[:, 0] = True
    size = _qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv33.uniform(0.4, 3.0, size=(n, 1))
    size *= at
    np.cumsum(size, axis=1, out=size)
    _qsv35 = size
    scale = np.exp(_qsv33.uniform(np.log(1.0), np.log(2000.0), size=(n, 1)))
    _qsv36 = _qsv33.random((n, 1)) < 0.6
    sd = np.where(_qsv36, 0.0, _qsv33.uniform(0.002, 0.03, size=(n, 1)))
    out = (_qsv33.uniform(-2.0, 2.0, size=(n, 1)) + _qsv35) * scale
    out = out + _qsv33.normal(0.0, 1.0, size=(n, L)) * sd * scale
    _qsv37 = _qsv33.random((n, 1)) < 0.65
    return np.where(_qsv37, np.rint(out), out)

def _k16(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv38 = _qsv13(_qsv33.normal(0.0, 1.0, size=(n, L)), _qsv33.uniform(0.99, 0.9995, size=(n, 1)))
    _qsv38 = (_qsv38 - _qsv38.mean(axis=1, keepdims=True)) / np.maximum(_qsv38.std(axis=1, keepdims=True), 1e-09)
    _qsv39 = _qsv38 > _qsv33.uniform(0.2, 1.2, size=(n, 1))
    lo = _qsv33.uniform(0.02, 0.2, size=(n, 1))
    _qsv40 = _qsv33.uniform(4.0, 25.0, size=(n, 1))
    sd = np.where(_qsv39, lo * _qsv40, lo)
    x = _qsv13(_qsv33.normal(0.0, 1.0, size=(n, L)) * sd, _qsv33.uniform(0.9, 0.999, size=(n, 1)))
    scale = np.exp(_qsv33.uniform(np.log(1.0), np.log(500.0), size=(n, 1)))
    return x * scale + _qsv33.uniform(-1.0, 1.0, size=(n, 1)) * scale

def _k17(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv41 = _qsv33.choice(np.array([24.0, 48.0, 96.0, 144.0]), size=(n, 1))
    _qsv42 = _qsv41 * 7.0
    _qsv43 = _qsv33.uniform(0.4, 1.6, size=(n, 1))
    _qsv44 = _qsv33.uniform(0.3, 1.4, size=(n, 1))
    y = _qsv43 * np.sin(2.0 * np.pi * t / _qsv42 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    y = y + _qsv44 * np.sin(2.0 * np.pi * t / _qsv41 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    y = y + 0.35 * _qsv44 * np.sin(4.0 * np.pi * t / _qsv41 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    _qsv45 = _qsv33.uniform(-0.3, 0.3, size=(n, 1)) * t / max(L - 1, 1)
    _qsv46 = _qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv33.uniform(0.02, 0.15, size=(n, 1))
    _qsv47 = np.exp(_qsv33.uniform(np.log(5.0), np.log(5000.0), size=(n, 1)))
    out = _qsv47 * np.exp(np.clip(y * 0.4 + _qsv45 + _qsv46, -6.0, 6.0))
    _qsv48 = _qsv33.random((n, 1)) < 0.45
    return np.where(_qsv48, np.rint(out), out)

def _k18(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    k = int(_qsv33.integers(3, 6))
    out = np.zeros((n, L), dtype=np.float64)
    _qsv47 = _qsv33.uniform(10.0, 400.0, size=(n, 1))
    for _ in range(k):
        _qsv49 = _qsv47 * _qsv33.uniform(0.31, 2.7, size=(n, 1))
        out += _qsv33.uniform(0.2, 1.0, size=(n, 1)) * np.sin(2.0 * np.pi * t / _qsv49 + _qsv33.uniform(0, 2 * np.pi, size=(n, 1)))
    sd = _qsv33.uniform(0.005, 0.05, size=(n, 1))
    scale = np.exp(_qsv33.uniform(np.log(1.0), np.log(1000.0), size=(n, 1)))
    out = out + _qsv33.normal(0.0, 1.0, size=(n, L)) * sd
    return out * scale + _qsv33.uniform(-1.0, 1.0, size=(n, 1)) * scale

def _k19(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv50 = _qsv33.uniform(1.0, 25.0, size=(n, 1)) / max(L, 1)
    _qsv51 = (_qsv33.random((n, L)) < _qsv50).astype(np.float64)
    _qsv52 = _qsv33.gamma(2.0, 1.0, size=(n, L)) * _qsv33.uniform(1.0, 12.0, size=(n, 1))
    _qsv53 = _qsv33.uniform(0.9, 0.998, size=(n, 1))
    _qsv54 = _qsv13(_qsv51 * _qsv52, _qsv53)
    _qsv55 = _qsv33.uniform(0.03, 0.6, size=(n, 1))
    scale = np.exp(_qsv33.uniform(np.log(1.0), np.log(800.0), size=(n, 1)))
    sd = _qsv33.uniform(0.0, 0.02, size=(n, 1))
    out = (_qsv54 + _qsv55) * scale
    return np.maximum(out * (1.0 + _qsv33.normal(0.0, 1.0, (n, L)) * sd), 0.0)

def _k20(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv56 = _qsv33.integers(4, 80, size=(n, 1)).astype(np.float64)
    _qsv57 = _qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv33.uniform(0.01, 0.12, size=(n, 1)) * _qsv56
    _qsv58 = np.cumsum(_qsv57, axis=1) + _qsv33.uniform(0.0, 1.0, size=(n, 1)) * _qsv56
    _qsv59 = 2.0 * _qsv56
    _qsv60 = _qsv56 - np.abs(np.mod(_qsv58, _qsv59) - _qsv56)
    _qsv61 = _qsv33.random((n, 1)) < 0.35
    _qsv60 = np.where(_qsv61, _qsv60, _qsv60 + _qsv33.normal(0.0, 0.35, size=(n, L)))
    return np.clip(np.rint(_qsv60), 0.0, _qsv56)

def _k21(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv47 = _qsv33.uniform(-2.0, 8.0, size=(n, 1))
    _qsv62 = _qsv33.choice(np.array([0.05, 0.1, 0.25, 0.5]), size=(n, 1))
    k = _qsv33.integers(0, 4, size=(n, 1))
    at = _qsv33.random((n, L)) < k / max(L, 1)
    at[:, 0] = False
    size = _qsv62 * _qsv33.choice(np.array([-2.0, -1.0, 1.0, 2.0]), size=(n, L))
    _qsv63 = _qsv47 + np.cumsum(at * size, axis=1)
    _qsv36 = _qsv33.random((n, 1)) < 0.8
    sd = np.where(_qsv36, 0.0, _qsv33.uniform(0.001, 0.01, size=(n, 1)))
    out = _qsv63 + _qsv33.normal(0.0, 1.0, size=(n, L)) * sd
    scale = np.exp(_qsv33.uniform(np.log(0.5), np.log(200.0), size=(n, 1)))
    return out * scale

def _k22(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv41 = _qsv33.choice(np.array([24.0, 48.0, 96.0]), size=(n, 1))
    _qsv64 = _qsv33.uniform(0.2, 1.0, size=(n, 1))
    _qsv65 = _qsv64 * np.sin(2 * np.pi * t / _qsv41 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    _qsv65 += 0.4 * _qsv64 * np.sin(4 * np.pi * t / _qsv41 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    _qsv42 = 0.3 * _qsv64 * np.sin(2 * np.pi * t / (_qsv41 * 7) + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    _qsv63 = _qsv13(_qsv33.normal(0.0, 0.05, size=(n, L)), _qsv33.uniform(0.995, 0.9999, size=(n, 1)))
    _qsv39 = _qsv13(_qsv33.normal(0.0, 1.0, size=(n, L)), _qsv33.uniform(0.95, 0.995, size=(n, 1)))
    _qsv39 = _qsv39 > np.quantile(_qsv39, _qsv33.uniform(0.7, 0.95), axis=1, keepdims=True)
    _qsv66 = _qsv33.uniform(0.002, 0.02, size=(n, 1)) * (1 + 6 * _qsv39)
    _qsv51 = _qsv33.random((n, L)) < _qsv66
    _qsv52 = _qsv33.standard_t(3, size=(n, L)) * _qsv33.uniform(0.5, 3.0, size=(n, 1))
    _qsv67 = _qsv13(_qsv51 * _qsv52, _qsv33.uniform(0.3, 0.8, size=(n, 1)))
    out = _qsv65 + _qsv42 + _qsv63 + _qsv67
    scale = np.exp(_qsv33.uniform(np.log(5.0), np.log(300.0), size=(n, 1)))
    _qsv68 = _qsv33.uniform(0.0, 2.0, size=(n, 1))
    return (out + _qsv68) * scale

def _k23(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    k = _qsv33.integers(2, 6)
    _qsv69 = np.sort(_qsv33.uniform(0, L, size=(n, k)), axis=1)
    _qsv70 = _qsv33.uniform(-0.004, 0.003, size=(n, k + 1))
    _qsv71 = np.zeros((n, L))
    _qsv72 = np.zeros((n, 1))
    for i in range(k + 1):
        lo = _qsv72
        hi = _qsv69[:, i:i + 1] if i < k else np.full((n, 1), float(L))
        _qsv77 = np.clip(t, lo, hi) - lo
        _qsv71 = _qsv71 + _qsv70[:, i:i + 1] * _qsv77
        _qsv72 = hi
    _qsv41 = _qsv33.choice(np.array([1.0, 24.0, 48.0]), size=(n, 1), p=[0.5, 0.3, 0.2])
    _qsv49 = np.where(_qsv41 == 1.0, 7.0, _qsv41 * 7)
    _qsv73 = _qsv33.uniform(0.1, 0.6, size=(n, 1)) * np.sin(2 * np.pi * t / _qsv49 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    _qsv47 = np.exp(_qsv33.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    _qsv74 = _qsv47 * np.exp(np.clip(_qsv71 + _qsv73, -8.0, 6.0))
    np.clip(_qsv74, 0.0, 5000000.0, out=_qsv74)
    _qsv75 = _qsv33.random((n, 1)) < 0.5
    shape = _qsv33.uniform(0.6, 4.0, size=(n, 1))
    _qsv76 = _qsv74 * _qsv33.gamma(shape, 1.0 / shape, size=(n, L))
    return _qsv33.poisson(np.where(_qsv75, _qsv76, _qsv74)).astype(np.float64)

def _k24(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv56 = np.exp(_qsv33.normal(np.log(18.0), 0.55, size=(n, 1)))
    _qsv56 = np.clip(np.rint(_qsv56), 4.0, 80.0)
    _qsv78 = _qsv33.beta(3.2, 1.0, size=(n, 1)) * 0.42 + 0.53
    _qsv79 = _qsv33.uniform(0.03, 0.15, size=(n, 1))
    _qsv80 = _qsv33.random((n, L)) >= _qsv78
    _qsv52 = np.where(_qsv33.random((n, L)) < _qsv79, 2.0, 1.0)
    _qsv81 = np.where(_qsv33.random((n, L)) < 0.5, -1.0, 1.0)
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv49 = _qsv33.choice(np.array([96.0, 144.0, 288.0]), size=(n, 1))
    _qsv82 = _qsv33.random((n, 1)) < 0.3
    _qsv83 = np.where(_qsv82, 0.35, 0.0) * np.sin(2 * np.pi * t / _qsv49 + _qsv33.uniform(0, 2 * np.pi, (n, 1)))
    _qsv81 = np.where(_qsv33.random((n, L)) < 0.5 + _qsv83, _qsv81, -_qsv81)
    _qsv84 = _qsv80 * _qsv52 * _qsv81
    _qsv58 = _qsv33.uniform(0.15, 0.85, size=(n, 1)) * _qsv56 + np.cumsum(_qsv84, axis=1)
    _qsv59 = 2.0 * _qsv56
    out = _qsv56 - np.abs(np.mod(_qsv58, _qsv59) - _qsv56)
    return np.rint(out)

def _k25(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv49 = _qsv33.choice(np.array([96.0, 96.0, 96.0, 24.0, 288.0, 48.0]), size=(n, 1))
    _qsv85 = np.zeros((n, L))
    for k in (1.0, 2.0, 3.0):
        _qsv85 += _qsv33.uniform(0.25, 1.0, size=(n, 1)) / k * np.sin(2 * np.pi * k * t / _qsv49 + _qsv33.uniform(0, 2 * np.pi, size=(n, 1)))
    _qsv42 = 0.25 * _qsv33.uniform(0.2, 1.0, size=(n, 1)) * np.sin(2 * np.pi * t / (_qsv49 * 7) + _qsv33.uniform(0, 2 * np.pi, size=(n, 1)))
    _qsv86 = _qsv13(_qsv33.normal(0.0, 1.0, size=(n, L)), _qsv33.uniform(0.995, 0.9999, size=(n, 1)))
    _qsv86 = _qsv86 / np.maximum(np.std(_qsv86, axis=1, keepdims=True), 1e-09)
    _qsv87 = _qsv33.random((n, L)) < _qsv33.uniform(0.004, 0.016, size=(n, 1))
    _qsv52 = np.clip(_qsv33.standard_t(4, size=(n, L)), -6.0, 6.0) * _qsv33.uniform(0.2, 0.75, size=(n, 1))
    _qsv67 = _qsv13(_qsv87 * _qsv52, _qsv33.uniform(0.25, 0.7, size=(n, 1)))
    _qsv64 = _qsv33.uniform(1.0, 2.4, size=(n, 1))
    y = _qsv64 * (_qsv85 + _qsv42) + _qsv33.uniform(0.15, 0.45, size=(n, 1)) * _qsv86 + _qsv67
    scale = np.exp(_qsv33.uniform(np.log(3.0), np.log(900.0), size=(n, 1)))
    _qsv88 = _qsv33.random((n, 1)) < 0.22
    _qsv35 = np.where(_qsv88, _qsv33.uniform(-1.2, 0.1, size=(n, 1)), _qsv33.uniform(0.4, 3.0, size=(n, 1)))
    return (y + _qsv35) * scale

def _k26(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv49 = _qsv33.choice(np.array([96.0, 96.0, 24.0, 288.0]), size=(n, 1))
    _qsv47 = _qsv33.uniform(0.4, 1.4, size=(n, 1)) * np.sin(2 * np.pi * t / _qsv49 + _qsv33.uniform(0, 2 * np.pi, size=(n, 1)))
    _qsv50 = _qsv33.uniform(0.003, 0.02, size=(n, 1))
    _qsv87 = (_qsv33.random((n, L)) < _qsv50).astype(np.float64)
    _qsv89 = np.where(_qsv33.random((n, L)) < 0.62, 1.0, -1.0)
    size = np.abs(np.clip(_qsv33.standard_t(4, size=(n, L)), -8, 8)) * _qsv33.uniform(0.5, 2.2, size=(n, 1))
    _qsv53 = _qsv33.uniform(0.55, 0.97, size=(n, 1))
    _qsv90 = _qsv13(_qsv87 * _qsv89 * size, _qsv53)
    _qsv86 = _qsv13(_qsv33.normal(0.0, 1.0, size=(n, L)), _qsv33.uniform(0.99, 0.9995, size=(n, 1)))
    _qsv86 = _qsv86 / np.maximum(np.std(_qsv86, axis=1, keepdims=True), 1e-09)
    y = _qsv47 + _qsv90 + _qsv33.uniform(0.15, 0.5, size=(n, 1)) * _qsv86
    scale = np.exp(_qsv33.uniform(np.log(2.0), np.log(600.0), size=(n, 1)))
    _qsv35 = np.where(_qsv33.random((n, 1)) < 0.18, _qsv33.uniform(-1.0, 0.2, size=(n, 1)), _qsv33.uniform(0.5, 3.0, size=(n, 1)))
    return (y + _qsv35) * scale

def _k27(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    t = np.arange(L, dtype=np.float64)[None, :]
    P0 = np.exp(_qsv33.uniform(np.log(20.0), np.log(60.0), size=(n, 1)))
    _qsv82 = np.sin(2.0 * np.pi * t / P0 + _qsv33.uniform(0, 2 * np.pi, size=(n, 1)))
    _qsv82 = _qsv82 + _qsv33.uniform(0.15, 0.7, size=(n, 1)) * np.sin(2.0 * np.pi * t / (P0 * 1.9323) + _qsv33.uniform(0, 2 * np.pi, size=(n, 1)))
    _qsv91 = P0 * _qsv33.uniform(12.0, 32.0, size=(n, 1))
    _qsv82 = _qsv82 * (1.0 + _qsv33.uniform(0.1, 0.55, size=(n, 1)) * np.sin(2.0 * np.pi * t / _qsv91 + _qsv33.uniform(0, 2 * np.pi, size=(n, 1))))
    _qsv92 = _qsv33.uniform(0.0, 1.0, size=(n, 1))
    _qsv93 = _qsv33.random((n, L)) < _qsv33.uniform(1.5, 8.0, size=(n, 1)) / max(L, 1)
    _qsv64 = np.exp(_qsv33.normal(_qsv33.uniform(-0.6, 0.9, size=(n, 1)), _qsv33.uniform(0.4, 1.0, size=(n, 1)), size=(n, L))) * _qsv93
    _qsv94 = _qsv33.uniform(0.55, 0.9, size=(n, 1))
    _qsv95 = _qsv33.uniform(0.97, 0.999, size=(n, 1))
    _qsv96 = _qsv13(0.6 * _qsv64, _qsv95) - 0.55 * _qsv13(_qsv64, _qsv94)
    _qsv97 = _qsv13(_qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv33.uniform(0.05, 0.3, size=(n, 1)), _qsv33.uniform(0.95, 0.999, size=(n, 1)))
    _qsv98 = np.exp(np.clip(_qsv97, -4.0, 4.0)) * _qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv33.uniform(0.05, 0.4, size=(n, 1))
    _qsv99 = np.cumsum(_qsv33.normal(0.0, 1.0, size=(n, L)), axis=1) * _qsv33.uniform(0.0, 0.02, size=(n, 1)) / np.sqrt(max(L, 1))
    x = _qsv92 * _qsv82 + _qsv33.uniform(0.3, 1.6, size=(n, 1)) * _qsv96 + _qsv98 + _qsv99
    x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-09)
    scale = np.exp(_qsv33.uniform(np.log(0.05), np.log(200.0), size=(n, 1)))
    _qsv100 = _qsv33.normal(0.0, 3.0, size=(n, 1)) * scale
    y = x * scale + _qsv100
    _qsv101 = _qsv33.random((n, 1)) < 0.45
    return np.where(_qsv101, np.abs(y) + 0.05 * scale, y)

def _qsv11(parameters: dict[str, float], _qsv102: str) -> None:
    if not all((np.isfinite(_qsv106) for _qsv106 in parameters.values())):
        raise ValueError(f'{_qsv102} must contain only finite values')
    _qsv103 = {'p15', 'p18', 'p19', *(_qsv107 for _qsv107 in parameters if _qsv107.startswith(('p20.', 'p30.')))}
    for name in _qsv103:
        if not 0.0 <= parameters[name] <= 1.0:
            raise ValueError(f'{_qsv102}.{name} must be in [0, 1]')
    for name in ('p11', 'p12', 'p13', 'p14', 'p16', 'p17'):
        if parameters[name] < 0.0:
            raise ValueError(f'{_qsv102}.{name} must be non-negative')
    for _qsv104, _qsv105 in (('p11', 'p12'), ('p13', 'p14'), ('p16', 'p17'), ('p20.p25', 'p20.p26'), ('p20.p28', 'p20.p29')):
        if parameters[_qsv104] > parameters[_qsv105]:
            raise ValueError(f'{_qsv102}.{_qsv104} must be <= {_qsv105}')

class Generator(DataGenerator):

    def __init__(self, config_dir: str, *, seed: int) -> None:
        _qsv334 = Path(config_dir) / 'config.json'
        _qsv335 = json.loads(_qsv334.read_text(encoding='utf-8')) if _qsv334.is_file() else {}
        self._cfg = _qsv335
        self._qsa12 = int(seed)
        self._qsa07 = int(_qsv335.get('p00', 64))
        self._qsa06 = int(_qsv335.get('p01', 4096))
        if self._qsa07 < 1 or self._qsa06 < self._qsa07:
            raise ValueError(f'invalid length band [{self._qsa07}, {self._qsa06}]')
        _qsv336 = dict(_qsv09)
        for k, v in dict(_qsv335.get('p03', {})).items():
            if k in _qsv336:
                _qsv336[k] = float(v)
        w = np.asarray([_qsv336[f] for f in _qsv08], dtype=np.float64)
        if not np.all(np.isfinite(w)) or w.min() < 0 or w.sum() <= 0:
            raise ValueError('p03 must be finite, non-negative, and not all zero')
        self._qsa16 = w / w.sum()
        p04 = dict(_qsv335.get('p04', {}))
        self._qsa01 = bool(p04.get('p05', False))
        self._qsa04 = float(p04.get('p06', 1.0))
        if not np.isfinite(self._qsa04) or not 0.0 < self._qsa04 <= 1.0:
            raise ValueError('p04.p06 must be in (0, 1]')
        self._qsa03 = float(p04.get('p07', 0.1))
        self._qsa02 = float(p04.get('p08', 0.7))
        if not 0.0 <= self._qsa03 < self._qsa02 <= 1.0:
            raise ValueError('p04 fractions must satisfy 0 <= p07 < p08 <= 1')
        _qsv337 = dict(_qsv336)
        for k, v in dict(p04.get('p09', {})).items():
            if k in _qsv337:
                _qsv337[k] = float(v)
        _qsv338 = np.asarray([_qsv337[f] for f in _qsv08], dtype=np.float64)
        if not np.all(np.isfinite(_qsv338)) or _qsv338.min() < 0 or _qsv338.sum() <= 0:
            raise ValueError('p04.p09 must be finite, non-negative, and not all zero')
        self._qsa14 = _qsv338 / _qsv338.sum()
        self._qsa15 = float(_qsv335.get('tr_hi_frac', 0.25))
        self._qsa10 = int(_qsv335.get('p02', 2))
        if not 1 <= self._qsa10 <= 4:
            raise ValueError('p02 must be in [1, 4]')
        self._qsa05 = float(_qsv335.get('p33', 0.0))
        if not np.isfinite(self._qsa05) or not 0.0 <= self._qsa05 <= 1.0:
            raise ValueError('p33 must be in [0, 1]')
        p30 = dict(_qsv335.get('p30', {}))
        p20 = dict(_qsv335.get('p20', {}))
        self._qsa08 = {'p11': float(_qsv335.get('p11', 0.4)), 'p12': float(_qsv335.get('p12', 3.0)), 'p13': float(_qsv335.get('p13', 0.3)), 'p14': float(_qsv335.get('p14', 2.0)), 'p15': float(_qsv335.get('p15', 0.4)), 'p16': float(_qsv335.get('p16', 0.02)), 'p17': float(_qsv335.get('p17', 0.12)), 'p18': float(_qsv335.get('p18', 0.25)), 'p19': float(_qsv335.get('p19', 0.3)), 'p20.p21': float(p20.get('p21', 0.06)), 'p20.p22': float(p20.get('p22', 0.07)), 'p20.p23': float(p20.get('p23', 0.04)), 'p20.p24': float(p20.get('p24', 0.0)), 'p20.p25': float(p20.get('p25', 0.01)), 'p20.p26': float(p20.get('p26', 0.1)), 'p20.p27': float(p20.get('p27', 0.0)), 'p20.p28': float(p20.get('p28', 0.001)), 'p20.p29': float(p20.get('p29', 0.015)), 'p30.p31': float(p30.get('p31', 0.0)), 'p30.p32': float(p30.get('p32', 0.0))}
        p10 = dict(p04.get('p10', {}))
        _qsv339 = dict(p10.pop('p20', {}))
        _qsv340 = dict(p10.pop('p30', {}))
        _qsv341 = {_qsv107 for _qsv107 in self._qsa08 if '.' not in _qsv107}
        unknown = set(p10) - _qsv341
        unknown.update((f'p20.{_qsv107}' for _qsv107 in _qsv339 if f'p20.{_qsv107}' not in self._qsa08))
        unknown.update((f'p30.{_qsv107}' for _qsv107 in _qsv340 if f'p30.{_qsv107}' not in self._qsa08))
        if unknown:
            _qsv342 = ', '.join(sorted(unknown))
            raise ValueError(f'unknown p04.p10: {_qsv342}')
        self._qsa13 = dict(self._qsa08)
        for _qsv107, _qsv106 in p10.items():
            self._qsa13[_qsv107] = float(_qsv106)
        for _qsv107, _qsv106 in _qsv339.items():
            self._qsa13[f'p20.{_qsv107}'] = float(_qsv106)
        for _qsv107, _qsv106 in _qsv340.items():
            self._qsa13[f'p30.{_qsv107}'] = float(_qsv106)
        _qsv11(self._qsa08, 'final parameters')
        _qsv11(self._qsa13, 'p04.p10')

    @property
    def name(self) -> str:
        return str(self._cfg.get('name', ''))

    def _qsa00(self, token_progress: float) -> float:
        if not self._qsa01:
            return 1.0
        _qsv343 = (token_progress - self._qsa03) / (self._qsa02 - self._qsa03)
        _qsv343 = float(np.clip(_qsv343, 0.0, 1.0))
        return _qsv343 * _qsv343 * (3.0 - 2.0 * _qsv343)

    def _qsa17(self, token_progress: float) -> np.ndarray:
        _qsv344 = self._qsa00(token_progress)
        if _qsv344 >= 1.0:
            return self._qsa16
        if _qsv344 <= 0.0:
            return self._qsa14
        return (1.0 - _qsv344) * self._qsa14 + _qsv344 * self._qsa16

    def _qsa09(self, token_progress: float) -> dict[str, float]:
        _qsv344 = self._qsa00(token_progress)
        if _qsv344 >= 1.0:
            return dict(self._qsa08)
        if _qsv344 <= 0.0:
            return dict(self._qsa13)
        return {_qsv107: (1.0 - _qsv344) * self._qsa13[_qsv107] + _qsv344 * _qsv345 for _qsv107, _qsv345 in self._qsa08.items()}

    def _qsa11(self, emitted_points: float, target_points: int) -> float:
        return emitted_points / (target_points * self._qsa04)

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        if n_series <= 0:
            return
        _qsv17()
        _qsv33 = np.random.default_rng(self._qsa12)
        _qsv346 = self._qsa06
        target_points = max(1, max(n_series - 2, 1) * self._qsa07)
        queue: Queue[object] = Queue(maxsize=self._qsa10)
        _qsv347 = Event()
        _qsv348 = object()

        def put(_qsv351: object) -> bool:
            while not _qsv347.is_set():
                try:
                    queue.put(_qsv351, timeout=0.1)
                    return True
                except Full:
                    continue
            return False

        def produce() -> None:
            try:
                _qsv350 = 0
                emitted_points = 0
                while _qsv350 < n_series and (not _qsv347.is_set()):
                    if _qsv350 == 0:
                        _qsv365 = _qsv03
                    elif _qsv350 == _qsv03:
                        _qsv365 = _qsv04
                    else:
                        _qsv365 = _qsv02
                    _qsv353 = _qsv33.integers(self._qsa07, _qsv346 + 1, size=_qsv365)
                    _qsv354 = min(_qsv365, n_series - _qsv350)
                    _qsv355 = int(_qsv353[:_qsv354].sum())
                    _qsv356 = self._qsa11(emitted_points + 0.5 * _qsv355, target_points)
                    p03 = self._qsa17(_qsv356)
                    parameters = self._qsa09(_qsv356)
                    _qsv357 = (partial(_k00, _qsv181=self._qsa15, _qsv182=parameters['p11'], _qsv183=parameters['p12'], _qsv184=parameters['p15'], _qsv185=parameters['p16'], _qsv186=parameters['p17']), _k01, partial(_k02, _qsv181=self._qsa15, _qsv182=parameters['p13'], _qsv183=parameters['p14']), _k03, partial(_k04, _qsv200=parameters['p18'], _qsv201=parameters['p19']), _k05, _k06, _k07, _k08, _k09, _k10, partial(_k11, _qsv256=self._qsa05), _k12, _k13, _k14, _k15, _k16, _k17, _k18, _k19, _k20, _k21, _k22, _k23, _k24, _k25, _k26, _k27, _k28, _k29, _k30, _k31)
                    _qsv358 = {_qsv107.removeprefix('p20.'): _qsv106 for _qsv107, _qsv106 in parameters.items() if _qsv107.startswith('p20.')}
                    _qsv359 = _qsv33.choice(len(_qsv08), size=_qsv365, p=p03)
                    _qsv360: list[np.ndarray | None] = [None] * _qsv365
                    for _qsv361 in range(len(_qsv08)):
                        _qsv180 = np.nonzero(_qsv359 == _qsv361)[0]
                        if _qsv180.size == 0:
                            continue
                        _qsv145 = _qsv357[_qsv361](_qsv33, int(_qsv180.size), _qsv346)
                        _qsv366 = _qsv08[_qsv361]
                        _qsv146 = _qsv366 in {'k24', 'k23', 'k02', 'k10', 'k11', 'k12', 'k25', 'k22', 'k26'}
                        if _qsv366 in _qsv10:
                            _qsv145 = _qsv28(_qsv145)
                        else:
                            _qsv145 = _qsv28(_qsv23(_qsv33, _qsv145, _qsv146=_qsv146, _qsv147=_qsv366 in {'k24', 'k23', 'k11', 'k12'}, _qsv148=_qsv366 in {'k00', 'k02', 'k07', 'k08'}, _qsv149=_qsv366 != 'k04', **_qsv358))
                        if not _qsv146:
                            _qsv145 = _qsv31(_qsv33, _qsv145)
                        for _qsv110, _qsv364 in enumerate(_qsv180):
                            length = int(_qsv353[_qsv364])
                            _qsv360[_qsv364] = np.ascontiguousarray(_qsv145[_qsv110, :length], dtype=np.float64)
                    if self._qsa07 == _qsv346:
                        _qsv367 = parameters['p30.p31']
                        _qsv76 = np.nonzero(_qsv33.random(_qsv365) < _qsv367)[0]
                        for _qsv364 in _qsv76:
                            _qsv328 = _qsv360[_qsv364]
                            if _qsv328 is None:
                                continue
                            _qsv369 = int(_qsv33.integers(1, 3))
                            _qsv370 = _qsv33.integers(0, _qsv365, size=_qsv369)
                            _qsv336 = _qsv33.dirichlet(np.ones(_qsv369 + 1))
                            _qsv371 = _qsv336[0] * _qsv328
                            _qsv372 = True
                            for j, _qsv373 in enumerate(_qsv370):
                                _qsv374 = _qsv360[int(_qsv373)]
                                if _qsv374 is None:
                                    _qsv372 = False
                                    break
                                _qsv371 = _qsv371 + _qsv336[j + 1] * _qsv374
                            if _qsv372:
                                _qsv360[_qsv364] = _qsv28(_qsv371)
                    _qsv362 = parameters['p30.p32']
                    _qsv363 = np.nonzero(_qsv33.random(_qsv365) < _qsv362)[0]
                    for _qsv364 in _qsv363:
                        series = _qsv360[_qsv364]
                        if series is None or series.size < 8:
                            continue
                        _qsv368 = int(_qsv33.integers(series.size // 8, 3 * series.size // 4))
                        series[:_qsv368] = series[_qsv368]
                    if not put((_qsv360, _qsv354)):
                        return
                    _qsv350 += _qsv354
                    emitted_points += _qsv355
            except BaseException as _qsv188:
                put(_qsv188)
            finally:
                put(_qsv348)
        _qsv349 = Thread(target=produce, name='', daemon=True)
        _qsv349.start()
        try:
            while True:
                _qsv351 = queue.get()
                if _qsv351 is _qsv348:
                    break
                if isinstance(_qsv351, BaseException):
                    raise _qsv351
                _qsv360, _qsv354 = _qsv351
                for _qsv352 in _qsv360[:_qsv354]:
                    if _qsv352 is None:
                        raise RuntimeError('internal: unfilled series slot')
                    yield _qsv352
        finally:
            _qsv347.set()
            _qsv349.join(timeout=1.0)

@njit(cache=False, fastmath=False)
def _qsv12(_qsv108: np.ndarray, _qsv109: np.ndarray) -> np.ndarray:
    n, length = _qsv108.shape
    out = np.empty((n, length), dtype=np.float64)
    for _qsv110 in range(n):
        _qsv111 = 0.0
        for t in range(length):
            _qsv106 = _qsv108[_qsv110, t] + _qsv111
            out[_qsv110, t] = _qsv106
            _qsv111 = _qsv109[_qsv110] * _qsv106
    return out

def _qsv13(_qsv108: np.ndarray, _qsv109: np.ndarray) -> np.ndarray:
    return _qsv12(np.ascontiguousarray(_qsv108, dtype=np.float64), np.asarray(_qsv109, dtype=np.float64).reshape(-1))

@njit(cache=False, fastmath=False)
def _qsv14(_qsv108: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    n, length = _qsv108.shape
    out = np.empty((n, length), dtype=np.float64)
    for _qsv110 in range(n):
        _qsv112 = 0.0
        _qsv113 = 0.0
        for t in range(length):
            _qsv106 = _qsv108[_qsv110, t] + _qsv112
            out[_qsv110, t] = _qsv106
            _qsv112 = a1[_qsv110] * _qsv106 + _qsv113
            _qsv113 = a2[_qsv110] * _qsv106
    return out

def _qsv15(_qsv108: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    return _qsv14(np.ascontiguousarray(_qsv108, dtype=np.float64), np.asarray(a1, dtype=np.float64).reshape(-1), np.asarray(a2, dtype=np.float64).reshape(-1))

@njit(cache=False, fastmath=False)
def _qsv16(_qsv108: np.ndarray, _qsv114: np.ndarray, _qsv115: np.ndarray, _qsv116: np.ndarray, _qsv117: np.ndarray) -> np.ndarray:
    n, length = _qsv108.shape
    out = np.empty((n, length), dtype=np.float64)
    for _qsv110 in range(n):
        out[_qsv110, 0] = _qsv108[_qsv110, 0]
        for t in range(1, length):
            _qsv72 = out[_qsv110, t - 1]
            if _qsv72 >= 0.0:
                _qsv106 = _qsv116[_qsv110] + _qsv114[_qsv110] * _qsv72 + _qsv108[_qsv110, t]
            else:
                _qsv106 = _qsv117[_qsv110] + _qsv115[_qsv110] * _qsv72 + _qsv108[_qsv110, t]
            out[_qsv110, t] = min(max(_qsv106, -1000000.0), 1000000.0)
    return out

def _qsv17() -> None:
    global _qsv00
    if _qsv00:
        return
    with _qsv01:
        if _qsv00:
            return
        zeros = np.zeros((2, 3), dtype=np.float64)
        _qsv118 = np.zeros(2, dtype=np.float64)
        _qsv12(zeros, _qsv118)
        _qsv14(zeros, _qsv118, _qsv118)
        _qsv16(zeros, _qsv118, _qsv118, _qsv118, _qsv118)
        _qsv00 = True

def _qsv18(x: np.ndarray, *, _qsv119: int=512) -> tuple[np.ndarray, np.ndarray]:
    _qsv120 = x[:, :min(x.shape[1], _qsv119)]
    mean = _qsv120.mean(axis=1, keepdims=True)
    std = _qsv120.std(axis=1, keepdims=True)
    return (mean, np.where(std < 1e-12, 1.0, std))

def _qsv19(x: np.ndarray, *, _qsv121: bool=True, _qsv119: int=512) -> np.ndarray:
    mean, std = _qsv18(x, _qsv119=_qsv119)
    return (x - mean) / std if _qsv121 else x / std

@lru_cache(maxsize=4)
def _qsv20(L: int) -> tuple[np.ndarray, np.ndarray]:
    _qsv122 = 2.0 * np.pi * np.arange(L, dtype=np.float64)[None, :] / _qsv05[:, None]
    return (np.sin(_qsv122), np.cos(_qsv122))

def _qsv21(_qsv33: np.random.Generator, n: int, L: int, _qsv123: int=3) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv126, _qsv127 = _qsv20(L)
    k = _qsv33.integers(1, _qsv123 + 1, size=n)
    _qsv124 = _qsv07[_qsv33.integers(0, len(_qsv07), size=n)]
    _qsv125 = _qsv33.random(n) < 0.35
    out = np.zeros((n, L), dtype=np.float64)
    for j in range(_qsv123):
        _qsv128 = np.nonzero(k > j)[0]
        _qsv129 = _qsv33.choice(_qsv05, size=n, p=_qsv06)
        if j < 2:
            _qsv129 = np.where(_qsv125, _qsv124[:, j], _qsv129)
        _qsv129 = _qsv129[:, None]
        _qsv64 = _qsv33.uniform(0.2, 2.0, size=n)[:, None]
        _qsv130 = _qsv33.uniform(0.0, 2.0 * np.pi, size=n)[:, None]
        _qsv131 = np.searchsorted(_qsv05, _qsv129[_qsv128, 0])
        _qsv132 = _qsv64[_qsv128] * (_qsv126[_qsv131] * np.cos(_qsv130[_qsv128]) + _qsv127[_qsv131] * np.sin(_qsv130[_qsv128]))
        _qsv133 = np.nonzero((k > j) & (_qsv33.random(n) < 0.35))[0]
        if _qsv133.size:
            _qsv134 = np.searchsorted(_qsv128, _qsv133)
            _qsv135 = 2.0 * np.pi * t / _qsv129[_qsv133] + _qsv130[_qsv133]
            _qsv136 = np.clip(_qsv129[_qsv133] * _qsv33.uniform(4.0, 12.0, size=(_qsv133.size, 1)), 32.0, 2.0 * L)
            _qsv137 = _qsv33.uniform(0.0, 2.0 * np.pi, size=(_qsv133.size, 1))
            _qsv138 = np.sin(2.0 * np.pi * t / _qsv136 + _qsv137)
            _qsv139 = 1.0 + _qsv33.uniform(0.05, 0.45, size=(_qsv133.size, 1)) * _qsv138
            _qsv140 = _qsv33.uniform(0.05, 0.75, size=(_qsv133.size, 1)) * np.sin(2.0 * np.pi * t / (1.7 * _qsv136) - _qsv137)
            _qsv132[_qsv134] = _qsv64[_qsv133] * _qsv139 * np.sin(_qsv135 + _qsv140)
        out[_qsv128] += _qsv132
    return out

def _qsv22(_qsv33: np.random.Generator, n: int, L: int, _qsv50: float, scale) -> np.ndarray:
    _qsv141 = _qsv33.random((n, L)) < _qsv50
    _qsv141[:, 0] = False
    _qsv143, _qsv144 = np.nonzero(_qsv141)
    _qsv34 = np.zeros((n, L), dtype=np.float64)
    if _qsv143.size == 0:
        return _qsv34
    s = np.asarray(scale, dtype=np.float64)
    _qsv142 = s if s.ndim == 0 else s.reshape(n)[_qsv143]
    _qsv34[_qsv143, _qsv144] = _qsv33.normal(0.0, 1.0, size=_qsv143.size) * _qsv142
    return _qsv34

def _qsv23(_qsv33: np.random.Generator, _qsv145: np.ndarray, *, _qsv146: bool, _qsv147: bool=False, _qsv148: bool=True, _qsv149: bool=True, p21: float=0.06, p22: float=0.07, p23: float=0.04, p24: float=0.0, p25: float=0.01, p26: float=0.1, p27: float=0.0, p28: float=0.001, p29: float=0.015, _qsv150: float=0.36, _qsv151: float=0.2, _qsv152: float=0.12, _qsv153: float=0.1) -> np.ndarray:
    _qsv154 = np.asarray(_qsv145, dtype=np.float64)
    out = _qsv32(_qsv33, _qsv154, _qsv150=_qsv150, _qsv151=_qsv151, _qsv152=_qsv152, _qsv153=_qsv153)
    n, L = out.shape
    _qsv155 = _qsv33.random(n) < 0.06 if _qsv148 else np.zeros(n, dtype=bool)
    out[_qsv155] = out[_qsv155, ::-1]
    if not _qsv146:
        _qsv162 = _qsv33.random(n) < 0.04
        out[_qsv162] *= -1.0
    _qsv156 = min(L, 512)
    _qsv157 = np.nonzero(_qsv33.random(n) < p27)[0]
    if _qsv157.size and L > 1:
        diff = np.diff(out[_qsv157, :_qsv156], axis=1)
        _qsv121 = np.median(diff, axis=1, keepdims=True)
        _qsv163 = 1.4826 * np.median(np.abs(diff - _qsv121), axis=1, keepdims=True)
        _qsv164 = np.maximum(np.std(diff, axis=1, keepdims=True), 1e-09)
        _qsv163 = np.where(_qsv163 > 1e-09, _qsv163, _qsv164)
        _qsv165 = _qsv33.uniform(p28, p29, size=(_qsv157.size, 1))
        _qsv172, _qsv173 = np.nonzero(_qsv33.random((_qsv157.size, L)) < _qsv165)
        if _qsv172.size:
            _qsv174 = _qsv33.choice([-1.0, 1.0], size=(_qsv157.size, 1))
            _qsv89 = np.where(_qsv33.random(_qsv172.size) < 0.75, _qsv174[_qsv172, 0], -_qsv174[_qsv172, 0])
            _qsv175 = _qsv33.lognormal(mean=np.log(4.0), sigma=0.6, size=_qsv172.size)
            out[_qsv157[_qsv172], _qsv173] += _qsv89 * _qsv175 * _qsv163[_qsv172, 0]
        if _qsv146:
            np.maximum(out, 0.0, out=out)
    for _qsv110 in np.nonzero(_qsv33.random(n) < p21)[0]:
        q = float(_qsv33.uniform(0.03, 0.18))
        _qsv166 = _qsv33.random() < 0.5
        if not _qsv149:
            continue
        _qsv167 = out[_qsv110, :_qsv156]
        if _qsv166:
            _qsv176 = np.quantile(_qsv167, 1.0 - q)
            out[_qsv110] = np.minimum(out[_qsv110], _qsv176)
        else:
            _qsv176 = np.quantile(_qsv167, q)
            out[_qsv110] = np.maximum(out[_qsv110], _qsv176)
    _qsv158 = np.nonzero(_qsv33.random(n) < p22)[0]
    if _qsv158.size:
        if _qsv149:
            for qi, _qsv110 in enumerate(_qsv158):
                x = out[_qsv110]
                _qsv167 = x[:_qsv156]
                lo = float(_qsv167.min())
                hi = float(_qsv167.max())
                if hi - lo < 1e-12:
                    continue
                _qsv177 = int(_qsv33.integers(0, 3))
                _qsv178 = int(_qsv33.integers(16, 257))
                if _qsv177 == 0:
                    _qsv57 = (hi - lo) / max(_qsv178 - 1, 1)
                    out[_qsv110] = lo + np.rint((np.clip(x, lo, hi) - lo) / _qsv57) * _qsv57
                elif _qsv177 == 1:
                    qs = np.linspace(0.0, 1.0, _qsv178)
                    _qsv179 = np.quantile(_qsv167, qs)
                    _qsv180 = np.searchsorted(_qsv179, x, side='left')
                    _qsv180 = np.clip(_qsv180, 0, _qsv178 - 1)
                    out[_qsv110] = _qsv179[_qsv180]
                else:
                    p = float(_qsv33.uniform(0.35, 0.85))
                    u = np.linspace(0.0, 1.0, _qsv178) ** p
                    _qsv179 = lo + (hi - lo) * u
                    _qsv180 = np.searchsorted(_qsv179, np.clip(x, lo, hi), side='left')
                    _qsv180 = np.clip(_qsv180, 0, _qsv178 - 1)
                    out[_qsv110] = _qsv179[_qsv180]
    _qsv159 = np.nonzero(_qsv33.random(n) < p23)[0]
    if _qsv159.size:
        _qsv168 = _qsv33.choice([2, 4, 8], size=_qsv159.size, p=[0.55, 0.3, 0.15])
        for _qsv169 in (2, 4, 8):
            _qsv143 = _qsv159[_qsv168 == _qsv169]
            if _qsv143.size:
                out[_qsv143] = np.repeat(out[_qsv143, ::_qsv169], _qsv169, axis=1)[:, :L]
    _qsv160 = np.nonzero(_qsv33.random(n) < p24)[0]
    if _qsv160.size and L > 1:
        _qsv170 = _qsv33.uniform(p25, p26, size=(_qsv160.size, 1))
        _qsv78 = _qsv33.random((_qsv160.size, L)) < _qsv170
        _qsv78[:, 0] = False
        _qsv171 = np.where(~_qsv78, np.arange(L, dtype=np.int64)[None, :], 0)
        np.maximum.accumulate(_qsv171, axis=1, out=_qsv171)
        out[_qsv160] = np.take_along_axis(out[_qsv160], _qsv171, axis=1)
    if _qsv147:
        out = np.maximum(np.rint(out), 0.0)
    _qsv161 = out[:, :_qsv156].std(axis=1) < 1e-09
    out[_qsv161] = _qsv154[_qsv161]
    return out

def _k00(_qsv33: np.random.Generator, n: int, L: int, *, _qsv181: float=0.25, _qsv182: float=0.4, _qsv183: float=3.0, _qsv184: float=0.4, _qsv185: float=0.02, _qsv186: float=0.12) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv35 = _qsv33.normal(0.0, 1.0, size=(n, 1))
    _qsv187 = _qsv33.random((n, 1)) < _qsv181
    _qsv188 = np.where(_qsv187, _qsv33.normal(0.0, _qsv183, size=(n, 1)), _qsv33.normal(0.0, _qsv182, size=(n, 1)))
    tn = t / max(L - 1, 1)
    series = _qsv35 + _qsv188 * tn + _qsv21(_qsv33, n, L)
    _qsv109 = _qsv33.uniform(0.0, 0.85, size=n)
    _qsv189 = _qsv33.random((n, 1)) < _qsv184
    sigma = np.where(_qsv189, _qsv33.uniform(_qsv185, _qsv186, size=(n, 1)), _qsv33.uniform(0.1, 0.6, size=(n, 1)))
    _qsv108 = _qsv33.normal(0.0, 1.0, size=(n, L)) * sigma
    return series + _qsv13(_qsv108, _qsv109)

def _k01(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv35 = np.cumsum(_qsv22(_qsv33, n, L, _qsv50=3.0 / L, scale=2.0), axis=1)
    _qsv190 = np.cumsum(_qsv22(_qsv33, n, L, _qsv50=3.0 / L, scale=0.5), axis=1)
    _qsv191 = np.exp(np.clip(_qsv190, -3.0, 3.0)) * _qsv33.uniform(0.1, 0.5, size=(n, 1))
    _qsv46 = _qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv191
    _qsv192 = _qsv21(_qsv33, n, L, _qsv123=2) * _qsv33.uniform(0.0, 1.0, size=(n, 1))
    _qsv193 = _qsv33.normal(0.0, 1.0 / L, size=(n, 1)) + np.cumsum(_qsv22(_qsv33, n, L, _qsv50=2.0 / L, scale=4.0 / L), axis=1)
    _qsv194 = np.cumsum(_qsv193, axis=1)
    return _qsv35 + _qsv194 + _qsv192 + _qsv46

def _k02(_qsv33: np.random.Generator, n: int, L: int, *, _qsv181: float=0.25, _qsv182: float=0.3, _qsv183: float=2.0) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv195 = _qsv33.random((n, 1)) < _qsv181
    _qsv196 = np.where(_qsv195, _qsv33.normal(0.0, _qsv183, size=(n, 1)), _qsv33.normal(0.0, _qsv182, size=(n, 1)))
    tn = t / max(L - 1, 1)
    _qsv197 = np.exp(_qsv196 * tn + _qsv33.normal(0.0, 0.3, size=(n, 1)))
    _qsv64 = _qsv33.uniform(0.1, 0.6, size=(n, 1))
    _qsv198 = _qsv21(_qsv33, n, L, _qsv123=1)
    _qsv198 = _qsv19(_qsv198, _qsv121=False)
    _qsv192 = 1.0 + _qsv64 * _qsv198
    _qsv46 = 1.0 + _qsv33.normal(0.0, 1.0, size=(n, L)) * _qsv33.uniform(0.02, 0.15, size=(n, 1))
    scale = _qsv33.uniform(1.0, 50.0, size=(n, 1))
    return scale * _qsv197 * np.clip(_qsv192, 0.05, None) * np.clip(_qsv46, 0.05, None)

def _k03(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    p1 = _qsv33.uniform(0.3, 0.98, size=n)
    p2 = _qsv33.uniform(-0.6, 0.6, size=n)
    a2 = p2
    a1 = p1 * (1.0 - p2)
    sigma = _qsv33.uniform(0.2, 0.8, size=(n, 1))
    _qsv199 = 512
    _qsv108 = _qsv33.normal(0.0, 1.0, size=(n, L + _qsv199)) * sigma
    return _qsv15(_qsv108, a1, a2)[:, _qsv199:]

def _k04(_qsv33: np.random.Generator, n: int, L: int, *, _qsv200: float=0.25, _qsv201: float=0.3) -> np.ndarray:
    _qsv202 = _qsv33.random(n) < 0.35
    _qsv45 = _qsv33.normal(0.0, 0.02, size=(n, 1))
    sigma = _qsv33.uniform(0.2, 1.0, size=(n, 1))
    _qsv203 = _qsv33.normal(0.0, 1.0, size=(n, L))
    _qsv204 = np.nonzero(_qsv33.random(n) < _qsv200)[0]
    if _qsv204.size:
        df = _qsv33.uniform(3.0, 12.0, size=(_qsv204.size, 1))
        _qsv203[_qsv204] = _qsv33.standard_t(df, size=(_qsv204.size, L)) / np.sqrt(df / (df - 2.0))
    _qsv205 = np.nonzero(_qsv33.random(n) < _qsv201)[0]
    if _qsv205.size:
        _qsv109 = 0.995
        _qsv199 = 256
        _qsv207 = _qsv33.standard_normal((_qsv205.size, L + _qsv199)) * np.sqrt(1.0 - _qsv109 * _qsv109)
        _qsv190 = lfilter([1.0], [1.0, -_qsv109], _qsv207, axis=1)[:, _qsv199:]
        _qsv190 *= _qsv33.uniform(0.1, 0.55, size=(_qsv205.size, 1))
        _qsv203[_qsv205] *= np.exp(np.clip(_qsv190, -2.0, 2.0))
    _qsv84 = _qsv203 * sigma + _qsv45
    _qsv58 = np.cumsum(_qsv84, axis=1)
    _qsv206 = np.cumsum(_qsv58, axis=1)
    o2 = _qsv202[:, None]
    return np.where(o2, _qsv206 / max(L, 1), _qsv58)

def _k05(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv114 = _qsv33.uniform(0.3, 0.9, size=n)
    _qsv115 = _qsv33.uniform(-0.9, 0.3, size=n)
    _qsv116 = _qsv33.normal(0.0, 0.3, size=n)
    _qsv117 = _qsv33.normal(0.0, 0.3, size=n)
    sigma = _qsv33.uniform(0.2, 0.7, size=(n, 1))
    _qsv199 = 256
    _qsv208 = L + _qsv199
    _qsv108 = _qsv33.normal(0.0, 1.0, size=(n, _qsv208)) * sigma
    x = _qsv16(np.ascontiguousarray(_qsv108, dtype=np.float64), np.asarray(_qsv114, dtype=np.float64), np.asarray(_qsv115, dtype=np.float64), np.asarray(_qsv116, dtype=np.float64), np.asarray(_qsv117, dtype=np.float64))
    return x[:, _qsv199:]

def _k06(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv209 = _qsv33.random(n) < 0.5
    _qsv210 = _qsv33.uniform(3.6, 4.0, size=n)
    _qsv211 = _qsv33.uniform(0.85, 1.0, size=n)
    x0 = _qsv33.uniform(0.05, 0.95, size=n)
    _qsv212 = x0.copy()
    for _ in range(64):
        _qsv213 = _qsv210 * _qsv212 * (1.0 - _qsv212)
        _qsv214 = _qsv211 * np.sin(np.pi * _qsv212)
        _qsv212 = np.clip(np.where(_qsv209, _qsv214, _qsv213), 0.0, 1.0)
    x = np.empty((n, L), dtype=np.float64)
    x[:, 0] = _qsv212
    for t in range(1, L):
        _qsv213 = _qsv210 * _qsv212 * (1.0 - _qsv212)
        _qsv214 = _qsv211 * np.sin(np.pi * _qsv212)
        _qsv212 = np.where(_qsv209, _qsv214, _qsv213)
        _qsv212 = np.clip(_qsv212, 0.0, 1.0)
        x[:, t] = _qsv212
    return _qsv19(x)

def _k07(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv215 = 2 * L
    f = np.fft.rfftfreq(_qsv215)[None, :]
    _qsv216 = np.exp(_qsv33.uniform(np.log(8.0), np.log(256.0), size=(n, 1)))
    _qsv217 = np.exp(-0.5 * (2.0 * np.pi * _qsv216 * f) ** 2)
    z = _qsv33.standard_normal((n, f.shape[1])) + 1j * _qsv33.standard_normal((n, f.shape[1]))
    z[:, 0] = 0.0
    x = np.fft.irfft(z * np.sqrt(_qsv217), n=_qsv215, axis=1)[:, :L]
    return _qsv19(x)

def _k08(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv215 = 2 * L
    f = np.fft.rfftfreq(_qsv215)
    _qsv218 = np.maximum(f, 1.0 / _qsv215)[None, :]
    beta = _qsv33.uniform(-0.6, 2.4, size=(n, 1))
    _qsv64 = _qsv218 ** (-0.5 * beta)
    _qsv219 = _qsv33.random((n, 1)) < 0.4
    _qsv220 = _qsv33.integers(8, max(9, f.size // 3), size=(n, 1))
    _qsv221 = np.maximum(_qsv220 / _qsv215, 1.0 / _qsv215)
    _qsv222 = _qsv33.uniform(-0.6, 2.8, size=(n, 1))
    _qsv223 = np.arange(f.size)[None, :] > _qsv220
    _qsv224 = _qsv221 ** (-0.5 * beta) * (_qsv218 / _qsv221) ** (-0.5 * _qsv222)
    _qsv64 = np.where(_qsv219 & _qsv223, _qsv224, _qsv64)
    _qsv64[:, 0] = 0.0
    z = _qsv33.standard_normal((n, f.size)) + 1j * _qsv33.standard_normal((n, f.size))
    x = np.fft.irfft(z * _qsv64, n=_qsv215, axis=1)[:, :L]
    _qsv225 = _qsv33.random(n) < 0.25
    if _qsv225.any():
        x[_qsv225] = np.cumsum(x[_qsv225], axis=1)
    return _qsv19(x)

def _k09(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv226 = np.exp(_qsv33.uniform(np.log(0.001), np.log(0.15), size=(n, 1)))
    _qsv227 = _qsv33.random((n, L)) < _qsv226
    _qsv227[:, 0] = _qsv33.random(n) < 0.5
    _qsv228 = np.bitwise_and(np.cumsum(_qsv227, axis=1), 1).astype(np.int8)
    _qsv138 = _qsv33.random((n, 1)) < 0.5
    _qsv109 = np.where(_qsv138, _qsv33.uniform(0.995, 0.9995, size=(n, 1)), _qsv33.uniform(0.9, 0.99, size=(n, 1)))
    _qsv229 = _qsv33.normal(-2.0, 1.0, size=(n, 1))
    _qsv230 = _qsv33.normal(2.0, 1.0, size=(n, 1))
    mean = np.where(_qsv228 == 0, _qsv229, _qsv230)
    _qsv231 = _qsv33.random((n, 1)) < 0.6
    mean += _qsv231 * _qsv21(_qsv33, n, L, _qsv123=3) * _qsv33.uniform(0.5, 3.0, size=(n, 1))
    _qsv232 = _qsv33.normal(np.log(0.3), 0.3, size=(n, 1))
    _qsv233 = _qsv33.normal(np.log(1.5), 0.5, size=(n, 1))
    _qsv234 = np.where(_qsv228 == 0, _qsv232, _qsv233)
    _qsv235 = _qsv33.uniform(0.951, 0.995, size=(n, 1))
    _qsv236 = _qsv33.uniform(0.03, 0.2, size=(n, 1))
    _qsv237 = _qsv33.standard_normal((n, L))
    _qsv238 = (1.0 - _qsv235) * _qsv234 + np.sqrt(1.0 - _qsv235 * _qsv235) * _qsv236 * _qsv237
    _qsv190 = np.empty((n, L), dtype=np.float64)
    _qsv190[:, 0] = _qsv234[:, 0]
    for i in range(n):
        _qsv241 = float(_qsv235[i, 0])
        _qsv190[i, 1:] = lfilter([1.0], [1.0, -_qsv241], _qsv238[i, 1:], zi=[_qsv241 * _qsv190[i, 0]])[0]
    _qsv191 = np.exp(np.clip(_qsv190, -5.0, 5.0))
    _qsv203 = _qsv33.standard_normal((n, L))
    _qsv204 = np.nonzero(_qsv33.random(n) < 0.35)[0]
    if _qsv204.size:
        _qsv203[_qsv204] = _qsv33.standard_t(4.0, size=(_qsv204.size, L)) / np.sqrt(2.0)
    _qsv239 = _qsv33.random((n, L)) < 3.0 / L
    _qsv242, _qsv243 = np.nonzero(_qsv239)
    _qsv203[_qsv242, _qsv243] += _qsv33.normal(0.0, 5.0, size=_qsv242.size)
    _qsv240 = np.sqrt(np.maximum(1.0 - _qsv109 * _qsv109, 1e-06))
    _qsv38 = (1.0 - _qsv109) * mean + _qsv240 * _qsv191 * _qsv203
    out = np.empty((n, L), dtype=np.float64)
    out[:, 0] = mean[:, 0] + _qsv191[:, 0] * _qsv203[:, 0]
    for i in range(n):
        p = float(_qsv109[i, 0])
        out[i, 1:] = lfilter([1.0], [1.0, -p], _qsv38[i, 1:], zi=[p * out[i, 0]])[0]
    scale = np.exp(_qsv33.uniform(np.log(0.1), np.log(50.0), size=(n, 1)))
    _qsv68 = _qsv33.uniform(-100.0, 100.0, size=(n, 1))
    return out * scale + _qsv68

def _k10(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv244 = _qsv21(_qsv33, n, L, _qsv123=2)
    _qsv245 = _k07(_qsv33, n, L)
    _qsv246 = np.cumsum(_qsv22(_qsv33, n, L, _qsv50=5.0 / L, scale=1.0), axis=1)
    _qsv47 = _qsv244 * _qsv33.uniform(0.3, 2.0, size=(n, 1)) + _qsv245 * _qsv33.uniform(0.2, 1.2, size=(n, 1)) + _qsv246 * _qsv33.uniform(0.2, 1.0, size=(n, 1))
    _qsv247 = _qsv33.integers(0, 4, size=n)
    out = _qsv47.copy()
    _qsv248 = _qsv247 == 1
    if _qsv248.any():
        _qsv250 = _qsv33.uniform(0.8, 3.5, size=(int(_qsv248.sum()), 1))
        _qsv251 = _qsv33.uniform(-0.8, 0.8, size=(int(_qsv248.sum()), 1))
        out[_qsv248] = 100.0 / (1.0 + np.exp(-_qsv250 * (_qsv47[_qsv248] - _qsv251)))
    _qsv249 = _qsv247 == 2
    if _qsv249.any():
        _qsv252 = int(_qsv249.sum())
        _qsv253 = np.exp(_qsv33.uniform(np.log(0.03), np.log(0.2), size=(_qsv252, 1)))
        _qsv58 = np.cumsum(_qsv33.standard_normal((_qsv252, L)) * _qsv253, axis=1)
        _qsv35 = _qsv33.uniform(900.0, 1100.0, size=(_qsv252, 1))
        out[_qsv249] = _qsv35 + _qsv58 + 2.0 * _qsv246[_qsv249] + 0.5 * _qsv244[_qsv249]
    _qsv175 = _qsv247 == 3
    if _qsv175.any():
        _qsv252 = int(_qsv175.sum())
        _qsv254 = (_qsv33.random((_qsv252, L)) < 8.0 / L) * _qsv33.lognormal(0.0, 0.8, size=(_qsv252, L))
        _qsv255 = _qsv33.uniform(1.0, 1.6, size=(_qsv252, 1))
        out[_qsv175] = np.abs(_qsv47[_qsv175]) ** _qsv255 + _qsv254
    return out

def _k11(_qsv33: np.random.Generator, n: int, L: int, *, _qsv256: float=0.0) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv49 = _qsv33.choice(_qsv05, size=(n, 1), p=_qsv06)
    _qsv130 = _qsv33.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    _qsv64 = _qsv33.uniform(0.15, 0.8, size=(n, 1))
    _qsv71 = _qsv64 * np.sin(2.0 * np.pi * t / _qsv49 + _qsv130)
    _qsv257 = _qsv33.random((n, 1)) < 0.55
    _qsv71 += _qsv257 * (0.5 * _qsv64) * np.sin(4.0 * np.pi * t / _qsv49 + _qsv33.uniform(0.0, 2.0 * np.pi, size=(n, 1)))
    _qsv258 = _qsv33.random((n, 1)) < 0.35
    _qsv259 = _qsv33.choice([24, 48, 96, 144], size=(n, 1))
    _qsv260 = (np.floor_divide(np.arange(L)[None, :], _qsv259) % 7).astype(np.int64)
    _qsv261 = _qsv33.normal(0.0, 0.12, size=(n, 7))
    _qsv261[:, 5:] += _qsv33.uniform(-0.8, 0.3, size=(n, 1))
    _qsv262 = np.take_along_axis(_qsv261, _qsv260, axis=1)
    _qsv71 += _qsv258 * _qsv262
    _qsv263 = _qsv33.uniform(-0.5, 0.5, size=(n, 1))
    _qsv71 += _qsv263 * t / max(L - 1, 1)
    _qsv264 = (_qsv33.random((n, L)) < 2.0 / L) * _qsv33.uniform(1.0, 10.0, size=(n, L))
    _qsv265 = _qsv13(_qsv264, _qsv33.uniform(0.85, 0.995, size=(n, 1)))
    _qsv47 = np.exp(_qsv33.uniform(np.log(3.0), np.log(3000.0), size=(n, 1)))
    _qsv74 = _qsv47 * np.exp(np.clip(_qsv71, -5.0, 5.0)) * (1.0 + _qsv265)
    np.clip(_qsv74, 0.0, 10000000.0, out=_qsv74)
    _qsv266 = _qsv33.random((n, 1)) < 0.5
    shape = _qsv33.uniform(0.5, 4.0, size=(n, 1))
    _qsv76 = _qsv74 * _qsv33.gamma(shape, 1.0 / shape, size=(n, L))
    out = _qsv33.poisson(np.where(_qsv266, _qsv76, _qsv74)).astype(np.float64)
    if _qsv256 > 0.0:
        _qsv267 = np.nonzero(_qsv33.random(n) < _qsv256)[0]
        if _qsv267.size:
            _qsv268 = _qsv33.choice(np.array([7, 7, 7, 14, 24, 48, 168], dtype=np.int64), size=_qsv267.size)
            _qsv269 = np.array([int(_qsv33.integers(0, int(_qsv49))) for _qsv49 in _qsv268], dtype=np.int64)
            _qsv270 = np.arange(L, dtype=np.int64)
            for _qsv110, _qsv271, _qsv272 in zip(_qsv267, _qsv268, _qsv269):
                _qsv273 = (_qsv270 - _qsv272) % _qsv271 == 0
                _qsv273[0] = True
                _qsv274 = np.maximum.accumulate(np.where(_qsv273, _qsv270, 0))
                out[_qsv110] = out[_qsv110, _qsv274]
    return out

def _k12(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    t = np.arange(L, dtype=np.float64)[None, :]
    _qsv275 = _qsv33.uniform(0.03, 0.35, size=(n, 1))
    _qsv49 = _qsv33.choice([7.0, 12.0, 24.0, 48.0, 168.0], size=(n, 1))
    _qsv276 = _qsv33.uniform(0.2, 1.2, size=(n, 1)) * np.sin(2.0 * np.pi * t / _qsv49 + _qsv33.uniform(0.0, 2.0 * np.pi, size=(n, 1)))
    _qsv277 = np.log(_qsv275 / (1.0 - _qsv275)) + _qsv276
    p = 1.0 / (1.0 + np.exp(-_qsv277))
    _qsv278 = (_qsv33.random((n, L)) < p).astype(np.float64)
    _qsv175 = np.maximum(1.0, np.rint(_qsv33.gamma(shape=2.0, scale=1.0, size=(n, L)) * _qsv33.uniform(1.0, 10.0, size=(n, 1)) * np.exp(0.25 * _qsv276)))
    return _qsv278 * _qsv175

def _k13(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    _qsv47 = _k07(_qsv33, n, L) * _qsv33.uniform(0.5, 2.0, size=(n, 1))
    _qsv47 += _qsv21(_qsv33, n, L, _qsv123=1) * _qsv33.uniform(0.0, 1.0, size=(n, 1))
    _qsv279 = _qsv22(_qsv33, n, L, _qsv50=3.0 / L, scale=_qsv33.uniform(3.0, 8.0, size=n))
    _qsv264 = _qsv22(_qsv33, n, L, _qsv50=2.0 / L, scale=_qsv33.uniform(2.0, 7.0, size=n))
    _qsv280 = _qsv13(_qsv264, _qsv33.uniform(0.75, 0.995, size=n))
    series = _qsv47 + _qsv279 + _qsv280
    _qsv281 = _qsv33.random((n, L)) < 2.0 / L
    _qsv281[:, 0] = False
    for _qsv110 in range(n):
        for start in np.nonzero(_qsv281[_qsv110])[0]:
            _qsv282 = int(_qsv33.integers(3, 65))
            _qsv283 = min(int(start) + _qsv282, L)
            series[_qsv110, start:_qsv283] = series[_qsv110, start - 1]
    return series
_qsv24 = 192
_qsv25 = 1024
_qsv26 = 192
_qsv27 = 1024

def _k14(_qsv33: np.random.Generator, n: int, L: int, *, _qsv284: int | None=None) -> np.ndarray:
    if n <= 0:
        return np.empty((0, L), dtype=np.float64)
    if L <= 0:
        return np.empty((n, 0), dtype=np.float64)
    _qsv247 = _qsv33.integers(0, 4, size=n)
    _qsv285 = np.empty((n, L), dtype=np.float64)
    for _k in range(4):
        _qsv290 = np.nonzero(_qsv247 == _k)[0]
        if _qsv290.size == 0:
            continue
        _m = int(_qsv290.size)
        if _k == 0:
            _v = _qsv21(_qsv33, _m, L, _qsv123=2)
        elif _k == 1:
            _v = _qsv13(_qsv33.normal(size=(_m, L)) * _qsv33.uniform(0.12, 0.55, size=(_m, 1)), _qsv33.uniform(0.35, 0.92, size=_m))
        elif _k == 2:
            _v = _k07(_qsv33, _m, L)
        else:
            _v = np.cumsum(_qsv33.normal(size=(_m, L)) * _qsv33.uniform(0.025, 0.16, size=(_m, 1)), axis=1)
        _qsv285[_qsv290] = _v
    _qsv286 = None
    _qsv287 = _qsv33.integers(0, 4, size=n)
    if _qsv284 is not None:
        _qsv287 = np.full(n, int(_qsv284), dtype=_qsv287.dtype)
    _qsv35 = _qsv33.normal(0.0, 2.0, size=n)
    scale = np.exp(_qsv33.uniform(np.log(0.4), np.log(12.0), size=n))
    _qsv288 = _qsv33.uniform(0.6, 2.2, size=n)
    _qsv289 = _qsv33.random(n) < 0.65
    out = np.empty((n, L), dtype=np.float64)
    for _qsv110 in range(n):
        _qsv291 = _qsv285[_qsv110]
        _qsv167 = _qsv291[:min(L, 512)]
        _qsv292 = float(_qsv167.mean())
        _qsv293 = float(_qsv167.std())
        _qsv291 -= _qsv292
        if _qsv293 > 1e-12:
            _qsv291 /= _qsv293
        _qsv294 = float(_qsv35[_qsv110])
        _qsv295 = bool(_qsv289[_qsv110])
        _qsv296 = 0
        _qsv297 = 0
        _qsv298 = False
        while _qsv296 < L:
            if _qsv295:
                _qsv300 = int(_qsv33.integers(_qsv24, _qsv25 + 1))
            else:
                _qsv300 = int(_qsv33.integers(_qsv26, _qsv27 + 1))
            if _qsv297 == 0 and L >= 2 * _qsv26:
                _qsv300 = min(_qsv300, L - _qsv26)
            _qsv283 = min(_qsv296 + max(_qsv300, 1), L)
            _qsv59 = _qsv283 - _qsv296
            if _qsv295:
                _qsv177 = int(_qsv287[_qsv110])
                if _qsv177 == 0:
                    out[_qsv110, _qsv296:_qsv283] = _qsv294
                    values = None
                elif _qsv177 == 1:
                    _qsv45 = _qsv33.normal(0.0, 0.0025, size=_qsv59).cumsum()
                    _qsv45 += np.linspace(0.0, float(_qsv33.normal(0.0, 0.025)), _qsv59)
                    values = _qsv294 + _qsv45
                elif _qsv177 == 2:
                    if not _qsv298:
                        _qsv305 = max(0.0, float(np.rint(abs(_qsv294) * 8.0)))
                        _qsv298 = True
                    else:
                        _qsv305 = max(0.0, float(np.rint(_qsv294)))
                    _qsv302 = _qsv33.random(_qsv59) < 0.025
                    _qsv303 = _qsv302 * _qsv33.choice([-1.0, 1.0], size=_qsv59)
                    values = np.maximum(_qsv305 + np.cumsum(_qsv303), 0.0)
                else:
                    _qsv304 = _qsv33.random(_qsv59) < 0.012
                    values = _qsv304 * _qsv33.gamma(1.5, 0.35, size=_qsv59)
            else:
                _qsv301 = _qsv291[_qsv296:_qsv283] * float(_qsv288[_qsv110])
                values = _qsv301 - _qsv301[0] + _qsv294
                if _qsv59 > 1 and np.ptp(values) < 1e-10:
                    values = _qsv294 + np.linspace(0.0, 1.0, _qsv59)
            if values is not None:
                out[_qsv110, _qsv296:_qsv283] = values
                _qsv294 = float(values[-1])
            _qsv296 = _qsv283
            _qsv295 = not _qsv295
            _qsv297 += 1
        _qsv299 = 1.0 if int(_qsv287[_qsv110]) == 2 else float(scale[_qsv110])
        out[_qsv110] *= _qsv299
        if L > 1 and np.ptp(out[_qsv110]) < 1e-10:
            out[_qsv110, -1] += max(0.001, 0.01 * _qsv299)
    return out

def _qsv28(_qsv145: np.ndarray) -> np.ndarray:
    x = np.asarray(_qsv145, dtype=np.float64)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    if x.ndim == 1:
        _qsv306 = float(np.max(np.abs(x)))
        if _qsv306 > 1000000.0:
            x *= 1000000.0 / _qsv306
    else:
        _qsv306 = np.max(np.abs(x), axis=1, keepdims=True)
        scale = np.where(_qsv306 > 1000000.0, 1000000.0 / np.maximum(_qsv306, 1e-12), 1.0)
        x *= scale
    return x

def _k28(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    return _k14(_qsv33, n, L, _qsv284=0)

def _k29(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    return _k14(_qsv33, n, L, _qsv284=1)

def _k30(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    return _k14(_qsv33, n, L, _qsv284=2)

def _k31(_qsv33: np.random.Generator, n: int, L: int) -> np.ndarray:
    return _k14(_qsv33, n, L, _qsv284=3)
_qsv29 = 0.85
_qsv30 = 0.7

def _qsv31(_qsv33: np.random.Generator, _qsv145: np.ndarray) -> np.ndarray:
    n = _qsv145.shape[0]
    if n == 0:
        return _qsv145
    _qsv307 = _qsv33.random(n) < _qsv29
    if not _qsv307.any():
        return _qsv145
    _qsv308 = np.nonzero(_qsv307)[0]
    _qsv143 = _qsv145[_qsv308]
    _qsv309 = _qsv143.min(axis=1, keepdims=True)
    _qsv299 = np.maximum(_qsv143.std(axis=1, keepdims=True), 1e-09)
    floor = _qsv299 * _qsv33.uniform(0.0, 0.15, size=(_qsv308.size, 1))
    _qsv68 = np.where(_qsv309 < floor, floor - _qsv309, 0.0)
    _qsv143 = _qsv143 + _qsv68
    _qsv310 = _qsv33.random(_qsv308.size) < _qsv30
    if _qsv310.any():
        _qsv311 = _qsv308[_qsv310]
        _qsv312 = np.rint(_qsv143[_qsv310])
        _qsv145[_qsv311] = _qsv312
        _qsv313 = ~_qsv310
        if _qsv313.any():
            _qsv145[_qsv308[_qsv313]] = _qsv143[_qsv313]
    else:
        _qsv145[_qsv308] = _qsv143
    return _qsv145

def _qsv32(_qsv33: np.random.Generator, _qsv145: np.ndarray, *, _qsv150: float=0.0, _qsv151: float=0.0, _qsv152: float=0.0, _qsv153: float=0.0) -> np.ndarray:
    out = np.asarray(_qsv145, dtype=np.float64)
    if out.ndim != 2:
        return out
    n, L = out.shape
    if n == 0 or L < 8:
        return out
    t = np.arange(L, dtype=np.float64)
    _qsv314 = np.nonzero(_qsv33.random(n) < _qsv150)[0]
    for i in _qsv314:
        _qsv318 = int(_qsv33.integers(2, 6))
        _qsv319 = np.sort(_qsv33.choice(L, size=_qsv318, replace=False).astype(np.float64))
        _qsv319[0] = 0.0
        _qsv319[-1] = float(L - 1)
        _qsv320 = np.exp(_qsv33.normal(0.0, 0.45, size=_qsv318))
        _qsv217 = np.interp(t, _qsv319, _qsv320)
        _qsv321 = float(np.median(out[i]))
        out[i] = _qsv321 + (out[i] - _qsv321) * _qsv217
    _qsv315 = np.nonzero(_qsv33.random(n) < _qsv151)[0]
    if _qsv315.size:
        _qsv322 = out[_qsv315, :min(L, 512)]
        _qsv121 = np.median(_qsv322, axis=1, keepdims=True)
        _qsv323 = 1.4826 * np.median(np.abs(_qsv322 - _qsv121), axis=1, keepdims=True)
        _qsv164 = np.maximum(np.std(_qsv322, axis=1, keepdims=True), 1e-09)
        _qsv323 = np.where(_qsv323 > 1e-09, _qsv323, _qsv164)
        for k, i in enumerate(_qsv315):
            _qsv329 = int(_qsv33.integers(1, 5))
            _qsv330 = float(_qsv33.uniform(1.5, max(2.0, 0.03 * L)))
            _qsv52 = float(_qsv33.lognormal(np.log(3.0), 0.55)) * float(_qsv323[k, 0])
            _qsv89 = float(_qsv33.choice([-1.0, 1.0]))
            _qsv331 = _qsv33.integers(0, L, size=_qsv329)
            _qsv332 = int(_qsv33.integers(0, 3))
            for c in _qsv331:
                if _qsv332 == 0:
                    _qsv333 = np.exp(-0.5 * ((t - c) / _qsv330) ** 2)
                elif _qsv332 == 1:
                    _qsv333 = np.maximum(0.0, 1.0 - np.abs(t - c) / (_qsv330 * 2.0))
                else:
                    _qsv333 = (np.abs(t - c) <= _qsv330).astype(np.float64)
                out[i] = out[i] + _qsv89 * _qsv52 * _qsv333
    _qsv316 = np.nonzero(_qsv33.random(n) < _qsv152)[0]
    for i in _qsv316:
        _qsv84 = _qsv33.normal(0.0, 1.0, size=L).cumsum()
        _qsv324 = _qsv84 - t / max(L - 1, 1) * _qsv84[-1]
        _qsv324 = _qsv324 - _qsv324.mean()
        _qsv325 = float(_qsv324.std()) or 1.0
        _qsv64 = float(_qsv33.uniform(0.5, 3.0))
        _qsv326 = _qsv64 * (_qsv324 / _qsv325)
        _qsv327 = np.clip(t + _qsv326, 0.0, float(L - 1))
        lo = np.floor(_qsv327).astype(np.int64)
        hi = np.minimum(lo + 1, L - 1)
        w = _qsv327 - lo
        out[i] = (1.0 - w) * out[i, lo] + w * out[i, hi]
    _qsv317 = np.nonzero(_qsv33.random(n) < _qsv153)[0]
    for i in _qsv317:
        _qsv49 = int(_qsv33.choice([4, 8, 12, 16, 24, 48]))
        on = int(_qsv33.integers(1, max(2, _qsv49)))
        _qsv130 = int(_qsv33.integers(0, _qsv49))
        _qsv128 = (t.astype(np.int64) + _qsv130) % _qsv49 < on
        _qsv328 = np.where(_qsv128, np.arange(L, dtype=np.int64), 0)
        np.maximum.accumulate(_qsv328, out=_qsv328)
        out[i] = out[i, _qsv328]
    return out