"""Fake datasets + checkpoints for the parallel-sweep equality tests.

Importing this module (in the test process AND in every spawned sweep worker —
the scheduler imports suites by name) replaces only the DATA side of the real
suites: GIFT-Eval's dataset build / gluonts scoring, TIME's timebench task,
FEV's fev task list. The real ``plan`` / ``run_unit`` / ``assemble`` code runs
unchanged, and so does the real wrapper loading (``load_wrapper_cached``) — the
stub checkpoints below are genuine ``forecast_wrapper.py`` files whose
forecasts depend on the history and on the checkpoint's own ``scale``.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

WRAPPER = textwrap.dedent(
    """
    import json, os
    from pathlib import Path

    import numpy as np

    class Wrapper:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        def __init__(self, checkpoint_dir, device="cpu"):
            d = Path(checkpoint_dir)
            cfg = json.loads((d / "config.json").read_text())
            if cfg.get("broken"):
                raise RuntimeError("size mismatch for head.weight")
            self.scale = float(cfg["scale"])
            with open(d / "loads.txt", "a") as f:
                f.write(f"{os.getpid()}\\n")

        def forecast_quantiles_batch(self, histories, horizon):
            out = []
            for h in histories:
                h = np.asarray(h, dtype=np.float64)
                mu, sd = float(np.mean(h[-8:])), float(np.std(h)) + 1e-3
                z = np.asarray(self.quantile_levels) - 0.5
                row = mu * self.scale + sd * z * (1.0 + 0.01 * len(histories))
                out.append(np.tile(row, (horizon, 1)))
            return np.stack(out)
    """
)


def make_checkpoint(root: Path, name: str, *, scale: float = 1.0, broken: bool = False) -> str:
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "forecast_wrapper.py").write_text(WRAPPER, encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"scale": scale, "broken": broken}),
                                   encoding="utf-8")
    return str(d)


def loads(checkpoint: str) -> list[str]:
    p = Path(checkpoint) / "loads.txt"
    return p.read_text().split() if p.is_file() else []


def _series(key: str, n: int, length: int = 48) -> list[np.ndarray]:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal(length)) + 10.0 for _ in range(n)]


def _metrics(wrapper, hist: list[np.ndarray], truth: list[np.ndarray], horizon: int,
             batch_size: int) -> tuple[float, float]:
    """A pinball-loss "CRPS" and an abs-error "MASE" off the wrapper's batched
    forecasts — enough float arithmetic that any batching or ordering change
    would move the low bits."""
    qs = np.concatenate([wrapper.forecast_quantiles_batch(hist[i:i + batch_size], horizon)
                         for i in range(0, len(hist), batch_size)])
    lv = np.asarray(LEVELS)
    crps = mase = 0.0
    for q, y in zip(qs, truth, strict=True):
        diff = y[:, None] - q
        crps += float(np.mean(np.maximum(lv * diff, (lv - 1) * diff)))
        mase += float(np.mean(np.abs(y - q[:, 4])))
    return crps / len(hist), mase / len(hist)


# ── GIFT-Eval: fake dataset build + gluonts scoring ─────────────────────────


class _DS:
    def __init__(self, name: str, term: str) -> None:
        self.name, self.term = name, term


def _gift_load(name: str, term: str, **_):
    if name.startswith("electricity"):
        return None, "FileNotFoundError: electricity/W/data.arrow"
    return _DS(name, term), ""


def _gift_score(ds, checkpoint_dir: str, *, num_samples: int, device: str,
                batch_size: int = 64) -> dict:
    from cascade_benchmark.predictor import load_wrapper_cached

    if ds.name == "m4_daily":
        raise MemoryError("Unable to allocate 4.2 GiB")
    crash = os.environ.get("CASCADE_FAKE_CRASH", "")
    if ds.name == "m4_weekly" and crash and multiprocessing.parent_process() is not None:
        # a hard worker death (OOM-kill / CUDA fatal): "always", or "once:<marker>"
        marker = Path(crash.partition(":")[2]) if crash.startswith("once:") else None
        if marker is None or not marker.exists():
            if marker is not None:
                marker.write_text("crashed")
            os._exit(9)
    w = load_wrapper_cached(checkpoint_dir, device)
    series = _series(f"gift/{ds.name}/{ds.term}", 7, 56)
    crps, mase = _metrics(w, [s[:-8] for s in series], [s[-8:] for s in series], 8, batch_size)
    return {"MASE": mase, "MAE": mase * 1.5, "CRPS": crps}


# ── TIME: fake timebench task ────────────────────────────────────────────────


def _time_tasks(_config, max_tasks):
    tasks = [("a/H", "short"), ("b/D", "short"), ("c/D", "long"), ("d/W", "short"),
             ("e/M", "medium")]
    return tasks[:max_tasks] if max_tasks else tasks


def _time_prepare(name: str, term: str, _config) -> dict:
    if name == "b/D":
        raise ValueError("Forecast contains NaN values")
    series = _series(f"time/{name}/{term}", 5, 40)
    return {"name": name, "term": term, "series": series,
            "info": {"num_variates": 1, "prediction_length": 6}}


def _time_score(wrapper, prep, out_dir, num_samples, batch_size=64, *, normalize=True,
                snaive_memo=None):
    s = prep["series"]
    crps, mase = _metrics(wrapper, [x[:-6] for x in s], [x[-6:] for x in s], 6, batch_size)
    if prep["name"] == "c/D":
        crps = float("nan")
    if snaive_memo is not None and "snaive" in snaive_memo:
        snaive = dict(snaive_memo["snaive"])
    else:
        snaive = {"CRPS": float(np.mean([np.std(x) for x in s])), "MASE": 1.0 + len(s) / 7}
        if snaive_memo is not None:
            snaive_memo["snaive"] = dict(snaive)
    return {"CRPS": crps, "MASE": mase}, snaive, prep["info"]


# ── FEV: fake fev tasks (picklable, real score_task_multi runs over them) ───


class FakeWindow:
    def __init__(self, key: str, n: int, horizon: int) -> None:
        self.key, self.n, self.horizon = key, n, horizon

    def _split(self):
        s = _series(self.key, self.n, 30 + self.horizon)
        return [x[:-self.horizon] for x in s], [x[-self.horizon:] for x in s]

    def get_input_data(self):
        return {"y": self._split()[0]}, None

    def get_ground_truth(self):
        return {"y": self._split()[1]}


class FakeTask:
    def __init__(self, name: str, *, horizon: int = 4, windows: int = 2, n: int = 5,
                 targets=("y",)) -> None:
        self.task_name, self.horizon, self.seasonality = name, horizon, 1
        self.target_columns = list(targets)
        self._windows, self._n = windows, n

    def iter_windows(self):
        for k in range(self._windows):
            yield FakeWindow(f"fev/{self.task_name}/{k}", self._n, self.horizon)


FEV_TASKS = [FakeTask("t1"), FakeTask("t2", horizon=3, windows=3),
             FakeTask("mv", targets=("y", "z")), FakeTask("t3", n=9)]


def _stubs() -> dict:
    """``timebench`` / ``fev`` stand-ins (only for the ones not installed)."""
    out: dict = {}
    try:
        import timebench  # noqa: F401
    except ImportError:
        data = types.ModuleType("timebench.evaluation.data")
        data.load_dataset_config = lambda _p: {"datasets": {}}
        out.update({"timebench": types.ModuleType("timebench"),
                    "timebench.evaluation": types.ModuleType("timebench.evaluation"),
                    "timebench.evaluation.data": data})
    fev = types.ModuleType("fev")
    fev.Benchmark = SimpleNamespace(from_yaml=lambda _src: SimpleNamespace(tasks=FEV_TASKS))
    out["fev"] = fev
    return out


def apply(monkeypatch=None) -> None:
    """Install the fakes: with ``monkeypatch`` in the test process (undone
    after the test), permanently in a spawned worker."""
    from cascade_benchmark.suites import boom, fev_bench, gifteval, time_bench

    def put(obj, name, value):
        if monkeypatch is not None:
            monkeypatch.setattr(obj, name, value)
        else:
            setattr(obj, name, value)

    for name, mod in _stubs().items():
        if monkeypatch is not None:
            monkeypatch.setitem(sys.modules, name, mod)
        else:
            sys.modules[name] = mod
    put(gifteval, "build_dataset_or_reason", _gift_load)
    put(gifteval, "score_dataset", _gift_score)
    put(boom, "build_dataset_or_reason", _gift_load)
    put(boom, "score_dataset", _gift_score)
    put(time_bench, "_tasks", _time_tasks)
    put(time_bench, "_prepare_task", _time_prepare)
    put(time_bench, "_score_prepared", _time_score)
    fev_bench._TASKS_MEMO.clear()


# A spawned sweep worker imports this module while unpickling its arguments
# (before multiprocessing marks it a child); the test fixture exports the flag.
if os.environ.get("CASCADE_FAKE_SUITES") == "1":
    apply()

from cascade_benchmark.suites import boom, fev_bench, gifteval, time_bench  # noqa: E402

GIFT = gifteval
BOOM = boom
TIME = time_bench
FEV = fev_bench
MODULES = {"gift-eval": "_fake_suites:GIFT", "boom": "_fake_suites:BOOM", "time": "_fake_suites:TIME",
           "fev": "_fake_suites:FEV"}
ENV = {"CASCADE_BENCH_GIFTEVAL_DATASETS": "m4_yearly m4_daily electricity/W m4_weekly "
                                          "m4_monthly m4_quarterly",
       "BOOM": "/fake/boom", "CASCADE_BENCH_BOOM_DATASETS": "ds-0-T,ds-1-T",
       "CASCADE_BENCH_TIME_DATASET": "/fake/time"}
