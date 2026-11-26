"""2026-11-26 22:34 UTC (lane B expert leg 5GMzsP…, energy, operator 8xRTX4090,
``cuda_device=None``): the leg trained on GPU 0 while GPUs 1-7 idled at P8 and
the throughput watchdog aborted it at step 1/1475. The pod ran the pinned
worker image (rc15, main 5e3fade), whose ``train_pyramid`` still read
``world = gang_size(trainer.device) if slices is None else 1`` — every expert
leg trained on one rank, whatever the dispatch mask.

Here: one rank rule for every leg — every visible GPU, for a whole-pod
(operator, no ``CUDA_VISIBLE_DEVICES``) and an explicit-list (funded,
``0,1,…,7``) dispatch alike — and a fail-fast guard when a multi-GPU pod's
worker builds fewer ranks than it sees. No torch: the rank rule is
``cascade.trainer.gang_ranks``; ``pyramid_dp`` is checked by source."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import shlex

import pytest

from . import helpers as h

MOD = "cascade.trainer.gang_ranks"
SKU = "8xRTX4090"
SELECTION = {0: [1, 2], 3: [5]}


def _ranks():
    # Fails by assertion (not ImportError) on a base without the fix.
    assert importlib.util.find_spec(MOD) is not None, \
        "cascade.trainer.gang_ranks (the torch-free rank rule + guard) is missing"
    with h.torch_blocked():
        m = importlib.import_module(MOD)
    for name in ("visible_gpu_count", "leg_ranks", "check_full_gang", "GangShortfall"):
        assert hasattr(m, name), f"gang_ranks lacks {name}"
    return m


def _expert_dispatch_env(cuda_device):
    """The pod-side env of a real role=expert dispatch command for a host
    dispatched with ``cuda_device`` (None = the operator's whole pod)."""
    with h.torch_blocked():
        remote = importlib.import_module("cascade.trainer.remote")
    host = remote.RemoteHost(name="leg", host="10.0.0.1", cuda_device=cuda_device,
                             remote_python="python", workdir="/root/cascade")
    argv = remote.worker_argv(host, gen_ref=h.REF, uid=7, hotkey=h.HK, role="expert",
                              base_seed=1, block=2, trainer_spec="t", domain="energy",
                              expert_selection=SELECTION, warm_start_ref="r@sha256:" + "b" * 64,
                              local_only=True)
    command, _ = remote.build_remote_command(host, argv, {}, lane_count=1)
    # the worker's own inline assignments (before `python -m cascade.trainer.worker`)
    inner = shlex.split(command.split("bash -c ", 1)[1])[0]
    worker = inner.split("set -m; ", 1)[1].split(" python -m ", 1)[0]
    env = dict(tok.split("=", 1) for tok in shlex.split(worker) if "=" in tok)
    assert "--role" in argv and argv[argv.index("--role") + 1] == "expert"
    return env


@pytest.mark.parametrize("dispatch", ["whole_pod", "explicit_list"])
def test_expert_leg_ranks_every_gpu_of_its_pod(dispatch):
    m = _ranks()
    with h.torch_blocked():
        layout = importlib.import_module("cascade.provision.layout")
    gpus = layout.layout_gpus(SKU)
    cuda_device = None if dispatch == "whole_pod" else layout.visible_devices(SKU)
    env = _expert_dispatch_env(cuda_device)
    if dispatch == "whole_pod":
        assert "CUDA_VISIBLE_DEVICES" not in env, env
    else:
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7", env
    visible = m.visible_gpu_count(gpus, env)
    assert visible == 8
    assert m.leg_ranks("cuda", visible, env) == 8, \
        f"an expert leg dispatched {dispatch} must train on all 8 GPUs"
    m.check_full_gang(8, visible, env)          # a full gang passes the guard


def test_rank_rule_edges():
    m = _ranks()
    assert m.visible_gpu_count(8, {}) == 8
    assert m.visible_gpu_count(8, {"CUDA_VISIBLE_DEVICES": ""}) == 0
    assert m.visible_gpu_count(8, {"CUDA_VISIBLE_DEVICES": "3"}) == 1
    assert m.leg_ranks("cuda", 8, {"CASCADE_TRAIN_GPUS": "4"}) == 4
    assert m.leg_ranks("cuda", 8, {"CASCADE_TRAIN_GPUS": "x"}) == 8
    assert m.leg_ranks("cuda", 8, {"CASCADE_TRAIN_GPUS": "0"}) == 1
    assert m.leg_ranks("cpu", 8, {}) == 1
    assert m.leg_ranks("cuda", 0, {}) == 1


def test_short_gang_fails_fast_with_a_named_error():
    m = _ranks()
    with pytest.raises(m.GangShortfall, match=r"1 training rank\(s\) on a pod with 8 visible GPUs"):
        m.check_full_gang(1, 8, {})
    with pytest.raises(m.GangShortfall):
        m.check_full_gang(1, 8, {}, device="cuda:0")
    m.check_full_gang(1, 1, {})                                   # single-GPU pod
    m.check_full_gang(4, 8, {"CASCADE_TRAIN_GPUS": "4"})          # deliberate gang
    m.check_full_gang(1, 8, {}, device="cpu")                     # CPU (CI) legs


def _fn(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_pyramid_uses_the_rule_and_the_guard():
    _ranks()
    dp = h.REPO / "cascade" / "trainer" / "pyramid_dp.py"
    trainer = h.REPO / "cascade" / "trainer" / "pyramid_trainer.py"
    assert "leg_ranks" in ast.unparse(_fn(dp, "gang_size")), \
        "pyramid_dp.gang_size does not use gang_ranks.leg_ranks"
    assert "check_full_gang" in ast.unparse(_fn(dp, "assert_full_gang"))
    src = ast.unparse(_fn(trainer, "train_pyramid"))
    assert "world = gang_size(trainer.device)" in src, "world is not gang_size(trainer.device)"
    assert "assert_full_gang(world, trainer.device)" in src, \
        "train_pyramid never refuses a short gang"
    assert src.index("assert_full_gang(world") < src.index("gang = Gang("), \
        "the short-gang guard must run before the gang is built"
