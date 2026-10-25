"""The concurrent sweep is the sequential sweep, byte for byte.

Real ``plan`` / ``run_unit`` / ``assemble`` code and real wrapper loading over
fake data (``_fake_suites``): 1 worker vs 2-4 spawned workers, a paired pass vs
solo sweeps, with failing units in every suite (a GIFT config that fails to
load, one that fails to score, a TIME task that raises, one with a non-finite
metric, an FEV multivariate task, and a checkpoint whose wrapper cannot load).
"""

from __future__ import annotations

import _fake_suites as fakes
import pytest
from cascade_benchmark import cli, sizing
from cascade_benchmark.parallel import WorkerCrashed, choose_plan, run_sweep
from cascade_benchmark.predictor import clear_wrapper_cache, load_wrapper_cached

SUITES = ["gift-eval", "boom", "time", "fev"]


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    for k, v in fakes.ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CASCADE_BENCH_NO_CACHE", "1")
    monkeypatch.setenv("CASCADE_FAKE_SUITES", "1")      # spawned workers install the fakes
    fakes.apply(monkeypatch)
    clear_wrapper_cache()
    yield tmp_path
    clear_wrapper_cache()


def _sweep(ckpts, workers, **kw):
    return run_sweep(SUITES, ckpts, num_samples=8, max_series=None, device="cpu",
                     batch_size=3, workers=workers, modules=fakes.MODULES, **kw)


def _texts(ckpts, per_ckpt):
    return [cli.report_json(c, r, {"gift-eval": "rev1"}) for c, r in zip(ckpts, per_ckpt,
                                                                       strict=True)]


def test_parallel_reports_are_byte_identical_to_sequential(fake_env):
    a = fakes.make_checkpoint(fake_env, "a", scale=1.0)
    results = _sweep([a], 1)
    seq = _texts([a], results)
    # the fixture really exercises failure isolation and every suite
    gift, boom, time_, fev = results[0]
    assert boom.status == "ok" and boom.n_series == boom.n_expected == 6
    assert gift.status == "ok" and [s["stage"] for s in gift.skipped] == ["score", "load"]
    assert time_.status == "ok" and {s["stage"] for s in time_.skipped} == {"score", "metric"}
    assert fev.status == "ok" and fev.skipped[0]["stage"] == "scope"
    for n in (2, 3, 4):
        assert _texts([a], _sweep([a], n)) == seq, f"workers={n}"


def test_paired_pass_equals_solo_sweeps(fake_env):
    a = fakes.make_checkpoint(fake_env, "a", scale=1.0)
    b = fakes.make_checkpoint(fake_env, "b", scale=0.7)
    solo = _texts([a], _sweep([a], 1)) + _texts([b], _sweep([b], 1))
    assert solo[0] != solo[1]            # the checkpoints really differ
    assert _texts([a, b], _sweep([a, b], 1)) == solo
    assert _texts([a, b], _sweep([a, b], 3)) == solo


def test_broken_checkpoint_in_a_pair_never_touches_the_other(fake_env):
    good = fakes.make_checkpoint(fake_env, "good")
    bad = fakes.make_checkpoint(fake_env, "bad", broken=True)
    solo_good = _texts([good], _sweep([good], 1))
    solo_bad = _texts([bad], _sweep([bad], 1))
    res_bad = _sweep([bad], 1)[0]
    # GIFT loads per config: every config fails at scoring; TIME/FEV load once: suite error
    gift = res_bad[0]
    assert gift.status == "error" and all(s["stage"] in ("score", "load") for s in gift.skipped)
    assert res_bad[1].status == "error" and len(res_bad[1].skipped) == 6
    assert res_bad[2].status == "error" and "size mismatch" in res_bad[2].detail
    assert res_bad[3].status == "error" and "size mismatch" in res_bad[3].detail
    for n in (1, 2, 4):
        assert _texts([good, bad], _sweep([good, bad], n)) == solo_good + solo_bad, n


def test_a_dead_worker_is_replaced_and_its_unit_rerun_once(fake_env, monkeypatch):
    a = fakes.make_checkpoint(fake_env, "a")
    seq = _texts([a], _sweep([a], 1))
    monkeypatch.setenv("CASCADE_FAKE_CRASH", f"once:{fake_env / 'crashed'}")
    assert _texts([a], _sweep([a], 3)) == seq
    assert (fake_env / "crashed").exists()


def test_a_unit_that_keeps_killing_workers_ends_the_sweep(fake_env, monkeypatch):
    a = fakes.make_checkpoint(fake_env, "a")
    monkeypatch.setenv("CASCADE_FAKE_CRASH", "always")
    with pytest.raises(WorkerCrashed, match="m4_weekly"):
        _sweep([a], 2)


def test_each_worker_loads_each_checkpoint_once(fake_env):
    a = fakes.make_checkpoint(fake_env, "a")
    _sweep([a], 1)
    assert len(fakes.loads(a)) == 1      # in-process: once for all three suites
    b = fakes.make_checkpoint(fake_env, "b")
    _sweep([b], 3)
    pids = fakes.loads(b)
    assert 1 <= len(pids) <= 3 and len(pids) == len(set(pids))   # once per worker


def test_wrapper_cache_loads_once_per_checkpoint_and_device(fake_env):
    a = fakes.make_checkpoint(fake_env, "a")
    w1 = load_wrapper_cached(a, "cpu")
    assert load_wrapper_cached(a + "/", "cpu") is w1        # same resolved path
    assert len(fakes.loads(a)) == 1
    clear_wrapper_cache()
    assert load_wrapper_cached(a, "cpu") is not w1
    assert len(fakes.loads(a)) == 2


def test_choose_plan_sequential_and_cpu_sizing(monkeypatch):
    monkeypatch.setattr("cascade_benchmark.parallel.host_limits", lambda: (128, 256 * sizing.GIB))
    assert choose_plan("1", checkpoints=[], device="cuda", batch_size=8, n_units=50).workers == 1
    assert choose_plan("auto", checkpoints=[], device="cpu", batch_size=8, n_units=1).workers == 1
    p = choose_plan("auto", checkpoints=[], device="cpu", batch_size=8, n_units=50, cpu_count=6)
    assert p.workers == 5 and p.devices == [None] * 5


def test_choose_plan_gpu_uses_the_measured_peak(monkeypatch):
    monkeypatch.setattr("cascade_benchmark.parallel.host_limits", lambda: (128, 256 * sizing.GIB))
    g = sizing.GIB

    def probe(ckpts, device, batch_size, *, measure):
        assert measure
        return {"gpus": [("0", 20 * g, 24 * g, "L4"), ("1", 10 * g, 24 * g, "L4")],
                "peak": 2 * g}

    p = choose_plan("auto", checkpoints=["x"], device="cuda", batch_size=512, n_units=200,
                    probe_fn=probe, cpu_count=64)
    per = sizing.per_worker_from_peak(2 * g)            # 2.5 GiB + 0.5 GiB context
    assert p.per_worker_bytes == per == 3 * g
    assert p.workers == (18 // 3) + (8 // 3) == 8       # (free - 2 GiB headroom) // per
    assert p.devices.count("0") == 6 and p.devices.count("1") == 2
    # cuda:1 narrows the sweep to that one device
    p1 = choose_plan("auto", checkpoints=["x"], device="cuda:1", batch_size=512, n_units=200,
                     probe_fn=probe, cpu_count=64)
    assert p1.workers == 2 and set(p1.devices) == {"1"}
    # a failed probe never guesses: sequential
    bad = choose_plan("auto", checkpoints=["x"], device="cuda", batch_size=512, n_units=200,
                      probe_fn=lambda *a, **k: {"error": "no driver"}, cpu_count=64)
    assert bad.workers == 1 and "probe failed" in bad.reason
