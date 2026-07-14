"""Per-lane CPU fairness for generator sandboxes on multi-GPU pods.

The dispatcher (the only party that knows a pod's lane fan-out) stamps
CASCADE_LANE_INDEX/CASCADE_LANE_COUNT into each lane's env; the sandbox slices
its cores off that geometry — affinity in the preexec closures, BLAS thread
caps in the child env, a right-sized cumulative CPU rlimit, and container-mode
parity via --cpuset-cpus. Absent env ⇒ everything behaves exactly as before.
"""

from __future__ import annotations

import os

from cascade.trainer import sandbox as sandbox_mod
from cascade.trainer.remote import RemoteHost, build_remote_command, pod_lane_count
from cascade.trainer.sandbox import _child_env, _lane_cpu_slice
from cascade.trainer.sandbox_container import container_argv

_BLAS_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _set_lane(monkeypatch, idx: int | str, cnt: int | str) -> None:
    monkeypatch.setenv("CASCADE_LANE_INDEX", str(idx))
    monkeypatch.setenv("CASCADE_LANE_COUNT", str(cnt))


def _fake_affinity(monkeypatch, cores) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(cores))


# ── _lane_cpu_slice: geometry from env ────────────────────────────────────────


def test_lane_slice_none_without_env(monkeypatch):
    monkeypatch.delenv("CASCADE_LANE_INDEX", raising=False)
    monkeypatch.delenv("CASCADE_LANE_COUNT", raising=False)
    assert _lane_cpu_slice() is None


def test_lane_slice_equal_contiguous_split(monkeypatch):
    _fake_affinity(monkeypatch, range(16))
    for idx, want in [(0, set(range(0, 4))), (1, set(range(4, 8))),
                      (2, set(range(8, 12))), (3, set(range(12, 16)))]:
        _set_lane(monkeypatch, idx, 4)
        assert _lane_cpu_slice() == (want, 4)


def test_lane_slice_respects_preexisting_cpuset(monkeypatch):
    # lium docker-template pods: os.cpu_count() reports the HOST's cores; the
    # affinity set is the container's real allowance, and the slice must be
    # carved out of those allowed ids (not absolute 0..N indices).
    _fake_affinity(monkeypatch, {8, 9, 10, 11})
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    _set_lane(monkeypatch, 1, 2)
    assert _lane_cpu_slice() == ({10, 11}, 2)


def test_lane_slice_more_lanes_than_cores_still_one_valid_core(monkeypatch):
    _fake_affinity(monkeypatch, range(4))
    _set_lane(monkeypatch, 6, 8)          # k=1, wraps: core 6 % 4 = 2
    cores, size = _lane_cpu_slice()
    assert size == 1 and cores == {2}


def test_lane_slice_single_lane_or_bad_geometry_is_none(monkeypatch):
    _fake_affinity(monkeypatch, range(8))
    _set_lane(monkeypatch, 0, 1)          # single-lane pod
    assert _lane_cpu_slice() is None
    _set_lane(monkeypatch, 5, 2)          # index outside the count
    assert _lane_cpu_slice() is None
    _set_lane(monkeypatch, "x", 4)        # malformed
    assert _lane_cpu_slice() is None


# ── _child_env: thread caps track the slice ───────────────────────────────────


def test_child_env_caps_blas_threads_at_slice_size(monkeypatch):
    _fake_affinity(monkeypatch, range(16))
    _set_lane(monkeypatch, 1, 4)
    for gpu in (False, True):             # both profiles: fairness is uniform
        env = _child_env(gpu=gpu)
        for key in _BLAS_KEYS:
            assert env[key] == "4"


def test_child_env_without_lane_has_no_caps_and_scrub_intact(monkeypatch):
    monkeypatch.delenv("CASCADE_LANE_INDEX", raising=False)
    monkeypatch.delenv("CASCADE_LANE_COUNT", raising=False)
    monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "secret")
    env = _child_env()
    assert not any(k in env for k in _BLAS_KEYS)   # legacy behavior untouched
    assert "HIPPIUS_HUB_TOKEN" not in env          # secrets still scrubbed
    assert "PYTHONPATH" in env


# ── the streaming child's CPU rlimit budget ───────────────────────────────────


def test_stream_series_cpu_cap_uses_slice_size(
    monkeypatch, small_cfg, example_generator_dir
):
    # Lane 0 of N where N = core count ⇒ a 1-core slice: the child's cumulative
    # CPU budget must be sized off 1 core, and the frame pipe must still work
    # with the preexec affinity actually applied (integration: real subprocess).
    if (os.cpu_count() or 1) <= 1:
        return  # cannot express >1 lanes on a 1-core box
    _set_lane(monkeypatch, 0, os.cpu_count() or 1)
    seen: dict[str, int] = {}
    real = sandbox_mod.stream_cpu_rlimit

    def spy(max_generate_seconds, max_wall_seconds, nproc):
        seen["nproc"] = nproc
        return real(max_generate_seconds, max_wall_seconds, nproc)

    monkeypatch.setattr(sandbox_mod, "stream_cpu_rlimit", spy)
    with sandbox_mod.stream_series(
        example_generator_dir, 0, small_cfg.generator, token_budget=256,
        blocked=small_cfg.static_guard.blocked, allow_netns=False,
        max_wall_seconds=2700,
    ) as frames:
        assert next(frames).size > 0      # frames round-trip on a 1-core slice
    assert seen["nproc"] == 1


def test_stream_series_cpu_cap_whole_box_without_lane(
    monkeypatch, small_cfg, example_generator_dir
):
    monkeypatch.delenv("CASCADE_LANE_INDEX", raising=False)
    monkeypatch.delenv("CASCADE_LANE_COUNT", raising=False)
    seen: dict[str, int] = {}
    real = sandbox_mod.stream_cpu_rlimit

    def spy(max_generate_seconds, max_wall_seconds, nproc):
        seen["nproc"] = nproc
        return real(max_generate_seconds, max_wall_seconds, nproc)

    monkeypatch.setattr(sandbox_mod, "stream_cpu_rlimit", spy)
    with sandbox_mod.stream_series(
        example_generator_dir, 0, small_cfg.generator, token_budget=256,
        blocked=small_cfg.static_guard.blocked, allow_netns=False,
        max_wall_seconds=2700,
    ) as frames:
        next(frames)
    assert seen["nproc"] == (os.cpu_count() or 1)


# ── dispatch: lane geometry is stamped by the orchestrator ────────────────────


def _host(name, cuda=None, host="1.2.3.4", port=22):
    return RemoteHost(name=name, host=host, port=port, cuda_device=cuda)


def test_pod_lane_count_counts_shared_endpoints():
    hosts = [_host("a-0", "0"), _host("a-1", "1"),
             _host("b-0", "0", host="5.6.7.8"),
             _host("a-2", "2", port=22)]
    assert pod_lane_count(hosts[0], hosts) == 3   # a-0/a-1/a-2 share (host, port)
    assert pod_lane_count(hosts[2], hosts) == 1
    assert pod_lane_count(hosts[0], None) == 1


def test_build_remote_command_stamps_lane_env_quoted():
    cmd = build_remote_command(_host("a-1", cuda="1"), ["python", "-m", "x"],
                               {}, lane_count=4)
    assert "CASCADE_LANE_INDEX=1" in cmd
    assert "CASCADE_LANE_COUNT=4" in cmd
    assert "CUDA_VISIBLE_DEVICES=1" in cmd


def test_build_remote_command_omits_lane_env_when_unavailable():
    # no lane_count (local/tests), single-lane pods, and non-ordinal masks all
    # keep today's command exactly.
    for kwargs in ({}, {"lane_count": None}, {"lane_count": 1}):
        cmd = build_remote_command(_host("a", cuda="0"), ["python"], {}, **kwargs)
        assert "CASCADE_LANE" not in cmd
    cmd = build_remote_command(_host("a", cuda="0,1"), ["python"], {}, lane_count=2)
    assert "CASCADE_LANE" not in cmd      # multi-device mask is not a lane
    cmd = build_remote_command(_host("a"), ["python"], {}, lane_count=2)
    assert "CASCADE_LANE" not in cmd      # no cuda_device at all


# ── container-mode parity ─────────────────────────────────────────────────────


def _container_cfg(small_cfg):
    from dataclasses import replace

    return replace(small_cfg.generator, sandbox_mode="container",
                   sandbox_image="example/worker@sha256:" + "a" * 64,
                   sandbox_python="/venv/bin/python")


def test_container_argv_cpuset_and_thread_caps_iff_slice(small_cfg, tmp_path):
    cfg = _container_cfg(small_cfg)
    child = ["--stream", "/sandbox/repo", "0", "{}", "8"]
    with_lane = container_argv(cfg, runtime="docker", name="s", repo=tmp_path,
                               child_args=child, lane_cores=(4, 5, 6, 7))
    joined = " ".join(with_lane)
    assert "--cpuset-cpus 4,5,6,7" in joined
    for key in _BLAS_KEYS:
        assert f"{key}=4" in with_lane
    assert "--cpus" in with_lane          # the rate cap stays regardless
    without = container_argv(cfg, runtime="docker", name="s", repo=tmp_path,
                             child_args=child, lane_cores=None)
    assert "--cpuset-cpus" not in without
    assert not any(a.startswith(k) for a in without for k in _BLAS_KEYS)
