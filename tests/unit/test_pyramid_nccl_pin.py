"""The pyramid gang pins NCCL's all-reduce so the gradient summation order
depends on the world size only, never on the pod's interconnect topology."""

from __future__ import annotations

import os
import queue

import pytest

from cascade.trainer import pyramid_dp
from cascade.trainer.pyramid_dp import NCCL_PIN, pin_nccl_env


def test_pin_sets_every_key_on_a_clean_env():
    env = {"PATH": "/bin"}
    assert pin_nccl_env(env) == {}
    assert env == {"PATH": "/bin", **NCCL_PIN}


def test_pin_overrides_and_reports_a_host_value(caplog):
    env = {"NCCL_ALGO": "Tree", "NCCL_PROTO": "Simple", "NCCL_DEBUG": "INFO"}
    with caplog.at_level("WARNING"):
        replaced = pin_nccl_env(env)
    assert replaced == {"NCCL_ALGO": "Tree"}          # equal values are not "replaced"
    assert env["NCCL_ALGO"] == "Ring" and env["NCCL_DEBUG"] == "INFO"
    assert "overriding host NCCL settings" in caplog.text


def test_pin_values_are_the_measured_set():
    # 2026-11-20 measurement: Ring / Simple / NVLS off / 4 channels gave
    # bit-identical weights across two NCCL topologies. Changing this set
    # changes every leg's numerics: an image repin at a seed-epoch boundary.
    assert NCCL_PIN == {"NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple", "NCCL_NVLS_ENABLE": "0",
                        "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}


class _Stop(Exception):
    pass


def test_follower_pins_before_its_process_group(monkeypatch):
    for k in NCCL_PIN:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    seen = {}

    def init_process_group(*a, **k):
        seen.update({key: os.environ.get(key) for key in NCCL_PIN})
        raise _Stop("stop after init")

    import torch.distributed as dist

    monkeypatch.setattr(pyramid_dp.torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    errors = queue.Queue()
    with pytest.raises(_Stop):            # the helper reports, then re-raises
        pyramid_dp._follower(1, 2, 0, {}, True, 0, errors)
    assert seen == NCCL_PIN
    assert errors.get_nowait()[0] == "_Stop"


def test_rank0_pins_before_spawning_helpers(monkeypatch):
    for k in NCCL_PIN:
        monkeypatch.delenv(k, raising=False)
    seen = {}

    class _Ctx:
        def Queue(self):
            return queue.Queue()

        def Process(self, **_):
            seen.update({key: os.environ.get(key) for key in NCCL_PIN})
            raise _Stop("stop at spawn")

    import torch.multiprocessing as mp

    class _Cfg:
        def to_dict(self):
            return {}

    monkeypatch.setattr(mp, "get_context", lambda _: _Ctx())
    with pytest.raises(_Stop):
        pyramid_dp.Gang(2, _Cfg(), deterministic=True, seed=0)
    assert seen == NCCL_PIN
