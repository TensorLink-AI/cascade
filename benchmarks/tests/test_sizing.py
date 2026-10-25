"""``--workers auto`` arithmetic on fake VRAM / CPU numbers."""

from __future__ import annotations

import pytest
from cascade_benchmark import sizing
from cascade_benchmark.sizing import GIB, GpuInfo, distribute, parse_workers, plan_workers


def test_parse_workers():
    assert parse_workers("auto") == "auto" and parse_workers(" AUTO ") == "auto"
    assert parse_workers("3") == 3 and parse_workers(2) == 2
    for bad in ("0", -1, "x", True, 0):
        with pytest.raises(ValueError):
            parse_workers(bad)


def test_auto_is_the_min_of_gpu_cpu_cap_and_units():
    gpu = [GpuInfo("0", 40 * GIB, 80 * GIB)]
    kw = dict(device="cuda", gpus=gpu, per_worker_bytes=3 * GIB, headroom_bytes=4 * GIB)
    assert plan_workers("auto", cpu_count=64, n_units=500, cap=99, **kw).workers == 12  # gpu
    assert plan_workers("auto", cpu_count=5, n_units=500, cap=99, **kw).workers == 4    # cpu-1
    assert plan_workers("auto", cpu_count=64, n_units=500, cap=7, **kw).workers == 7    # cap
    assert plan_workers("auto", cpu_count=64, n_units=3, cap=99, **kw).workers == 3     # units


def test_multi_gpu_round_robin_respects_each_device_capacity():
    gpus = [GpuInfo("0", 14 * GIB, 24 * GIB), GpuInfo("1", 8 * GIB, 24 * GIB),
            GpuInfo("GPU-abc", 5 * GIB, 24 * GIB)]
    p = plan_workers("auto", device="cuda", gpus=gpus, cpu_count=64, per_worker_bytes=3 * GIB,
                     n_units=100, headroom_bytes=2 * GIB, cap=99)
    # capacities (14-2)//3=4, (8-2)//3=2, (5-2)//3=1
    assert p.workers == 7
    assert p.devices == ["0", "1", "GPU-abc", "0", "1", "0", "0"]
    # fewer workers than capacity still alternate devices
    p2 = plan_workers("auto", device="cuda", gpus=gpus, cpu_count=4, per_worker_bytes=3 * GIB,
                      n_units=100, headroom_bytes=2 * GIB, cap=99)
    assert p2.devices == ["0", "1", "GPU-abc"]


def test_nothing_fits_runs_one_worker_on_the_roomiest_device():
    gpus = [GpuInfo("0", 3 * GIB, 24 * GIB), GpuInfo("1", 4 * GIB, 24 * GIB)]
    p = plan_workers("auto", device="cuda", gpus=gpus, cpu_count=8, per_worker_bytes=3 * GIB,
                     n_units=10, headroom_bytes=2 * GIB)
    assert p.workers == 1 and p.devices == ["1"] and "0 fit" in p.reason


def test_explicit_n_and_cpu_device():
    gpus = [GpuInfo("0", GIB, GIB), GpuInfo("1", GIB, GIB)]
    p = plan_workers(5, device="cuda", gpus=gpus, cpu_count=2, per_worker_bytes=None, n_units=9)
    assert p.workers == 5 and p.devices == ["0", "1", "0", "1", "0"]   # taken as given
    assert plan_workers(5, device="cuda", gpus=gpus, cpu_count=2, per_worker_bytes=None,
                        n_units=2).workers == 2
    c = plan_workers("auto", device="cpu", gpus=[], cpu_count=8, per_worker_bytes=None,
                     n_units=100)
    assert c.workers == 7 and c.devices == [None] * 7 and c.per_worker_bytes is None
    assert plan_workers("auto", device="cuda", gpus=[], cpu_count=8, per_worker_bytes=GIB,
                        n_units=9).workers == 1


def test_unknown_per_worker_vram_is_one_worker_and_estimates_sum_per_checkpoint():
    gpus = [GpuInfo("0", 80 * GIB, 80 * GIB)]
    assert plan_workers("auto", device="cuda", gpus=gpus, cpu_count=8, per_worker_bytes=None,
                        n_units=9).workers == 1
    assert sizing.estimate_per_worker(["cascade_moe", "toto2"]) == 5 * GIB
    assert sizing.estimate_per_worker(["?"]) == sizing.DEFAULT_ESTIMATE_BYTES
    assert sizing.per_worker_from_peak(4 * GIB) == 5 * GIB + sizing.CUDA_CONTEXT_BYTES


def test_distribute_refuses_an_impossible_fit():
    with pytest.raises(ValueError):
        distribute(3, ["0", "1"], [1, 1])
    assert distribute(4, ["a", "b"]) == ["a", "b", "a", "b"]
