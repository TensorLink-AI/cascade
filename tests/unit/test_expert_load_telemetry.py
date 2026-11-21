"""Log-only routed-expert load telemetry (cascade/trainer/expert_load.py):
the per-layer stats, window/phase accumulation, the trainer wiring, and —
the property that matters — that it changes no byte of any checkpoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.shared.config import parse_chain_config  # noqa: E402
from cascade.trainer import expert_load as el  # noqa: E402
from cascade.trainer.toto2_trainer import Toto2Trainer  # noqa: E402
from scripts.pyramid_config import local_config  # noqa: E402


# ── stats ──────────────────────────────────────────────────────────────────

def test_balanced_load_has_every_expert_effective():
    s = el.load_stats(torch.full((64,), 10, dtype=torch.int64))
    assert s == {"eff": 64.0, "dead": 0, "max_mean": 1.0}


def test_collapsed_load_has_one_effective_expert():
    c = torch.zeros(64, dtype=torch.int64)
    c[3] = 1000
    s = el.load_stats(c)
    assert s["eff"] == 1.0 and s["dead"] == 63 and s["max_mean"] == 64.0


def test_empty_load_is_all_dead():
    assert el.load_stats(torch.zeros(8, dtype=torch.int64)) == {"eff": 0.0, "dead": 8,
                                                                 "max_mean": 0.0}


# ── accumulation ───────────────────────────────────────────────────────────

def _load(*hot):
    out = []
    for h in hot:
        c = torch.zeros(8, dtype=torch.int64)
        c[h] = 4
        out.append(c)
    return out


def test_window_resets_and_phases_split_at_the_trunk():
    t = el.ExpertLoadTelemetry(trunk_steps=2)
    t.record(0, _load(0, 1))
    t.record(1, _load(1, 1))
    w = t.window()
    assert [x["dead"] for x in w] == [6, 7]          # layer 0 saw experts 0,1
    assert t.window() is None                        # reset after a read
    t.record(2, _load(2, 2))                         # step 2 ≥ trunk → cooldown
    s = t.summary()
    assert set(s) == {"trunk", "cooldown"}
    assert s["trunk"][0]["dead"] == 6 and s["cooldown"][0]["dead"] == 7


def test_record_never_mutates_the_training_counts():
    t = el.ExpertLoadTelemetry(trunk_steps=10)
    load = _load(0, 1)
    before = [c.clone() for c in load]
    t.record(0, load)
    t.record(1, load)
    assert all(torch.equal(a, b) for a, b in zip(load, before, strict=True))


def test_no_counts_means_no_telemetry():
    t = el.ExpertLoadTelemetry(trunk_steps=1)
    t.record(0, [])
    t.record(1, None)
    assert t.window() is None and t.summary() is None


# ── the trainer ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def contract():
    c = parse_chain_config(local_config("2m")).training
    return replace(c, training_steps=3, batch_size=8, max_train_seconds=90)


def _train(path, contract, **kwargs):
    rng = np.random.default_rng(84)
    data = [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]
    return Toto2Trainer(device="cpu").train(iter(data), contract, training_seed=1,
                                            token_budget=contract.train_tokens,
                                            out_dir=path, **kwargs)


def _bytes(path):
    return {f.name: f.read_bytes() for f in sorted(Path(path).iterdir())}


def test_trainer_reports_load_by_step_and_phase(tmp_path, contract):
    records: list[dict] = []
    leg = _train(tmp_path / "leg", contract, logger=records.append)
    tel = leg.metrics["expert_load_telemetry"]
    # 3 steps: trunk = 3 - max(1, round(3/11)) = 2 steps, then 1 cooldown step
    assert set(tel) == {"trunk", "cooldown"}
    assert tel["trunk"] and all({"eff", "dead", "max_mean"} == set(x) for x in tel["trunk"])
    first = [r for r in records if r.get("event") == "step"][0]
    assert first["step"] == 1 and len(first["expert_load"]) == len(tel["trunk"])


def test_telemetry_changes_no_checkpoint_byte(tmp_path, contract, monkeypatch):
    on = _train(tmp_path / "on", contract, logger=lambda r: None)
    with monkeypatch.context() as m:
        m.setattr(el.ExpertLoadTelemetry, "record", lambda self, step, load: None)
        off = _train(tmp_path / "off", contract, logger=lambda r: None)
    assert "expert_load_telemetry" not in off.metrics
    assert _bytes(on.local_dir) == _bytes(off.local_dir)
