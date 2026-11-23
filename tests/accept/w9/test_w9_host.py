"""W9 (plan rev6.18 B5/B7, owner 2026-11-23): the audit checks the signed
scoring host. Contract: README.md "Scoring host (rev6.18)".

A report carrying ``host`` PASSes only when ``host.env`` is in the qualified
scoring environment list at the report's boundary (block-scoped
``from_block`` / ``until_block``), with a GPU, driver and torch build that
entry allows, and an ``identity`` that hashes the host's fields. The list
defaults to the release's committed ``deploy/laned/scoring_envs.toml``.

Every test's first statement asserts ``check_lane_d`` takes
``scoring_envs`` (absent on main): a failure there is an assertion.
"""

from __future__ import annotations

import copy
import hashlib
import inspect

import pytest

from . import helpers as h

RTX4090 = "NVIDIA GeForce RTX 4090"
CU124 = "2.4.1+cu124/cuda-12.4"


def _takes_envs():
    check_lane_d = h._check_lane_d()
    assert "scoring_envs" in inspect.signature(check_lane_d).parameters, \
        "rev6.18 B7: check_lane_d(..., scoring_envs=None) checks report.host"
    return check_lane_d


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return h.crown_chain(tmp_path_factory.mktemp("w9-host"))


def _host(gpu=RTX4090, driver="550.127.05", env="cu124-torch241", torch_build=CU124):
    from cascade.shared.config.lane_d import host_identity

    fp = hashlib.sha256(b"GPU-1").hexdigest()
    return {"env": env, "gpu_model": gpu, "driver": driver, "torch_build": torch_build,
            "fingerprint": fp, "identity": host_identity(env, gpu, driver, torch_build, fp)}


def _envs(**kw):
    from cascade.shared.config.lane_d import parse_scoring_envs

    e = {"name": "cu124-torch241", "status": "qualified", "from_block": 0, "until_block": 0,
         "lock": "uv.lock", "lock_sha256": "c" * 64, "torch_build": CU124,
         "gpu_models": [RTX4090], "driver_min": "550.54.14", "driver_max": "",
         "evidence": "reference"}
    e.update(kw)
    return parse_scoring_envs({"env": [e]})


def _audit(report, cfg, **kw):
    check_lane_d = _takes_envs()
    return check_lane_d(copy.deepcopy(report), cfg, signer_hotkey=h.signer().ss58_address, **kw)


def _with_host(report, host):
    r = copy.deepcopy(report)
    r["host"] = host
    return h.resign(r)


def test_a_qualified_host_passes_against_the_committed_list(run):
    _takes_envs()
    r = _with_host(run["s1"].report, _host())
    res = _audit(r, run["cfg"])
    assert res.status == "PASS", res.detail
    res = _audit(r, run["cfg"], scoring_envs=_envs())
    assert res.status == "PASS", res.detail


@pytest.mark.parametrize("host_kw,env_kw,why", [
    ({}, {"from_block": 10**9}, "not yet qualified at the boundary"),
    ({}, {"until_block": 1}, "retired before the boundary"),
    ({}, {"status": "candidate", "evidence": "", "lock": "", "lock_sha256": ""}, "candidate"),
    ({"env": "unknown-env"}, {}, "env not listed"),
    ({"gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"}, {}, "GPU not allowed"),
    ({"driver": "535.104.05"}, {}, "driver below the floor"),
    ({"torch_build": "2.12.1/cuda-13.0"}, {}, "torch build differs"),
])
def test_an_unqualified_host_fails(run, host_kw, env_kw, why):
    _takes_envs()
    r = _with_host(run["s1"].report, _host(**host_kw))
    res = _audit(r, run["cfg"], scoring_envs=_envs(**env_kw))
    assert res.status == "FAIL", f"{why}: {res.status} {res.detail}"
    assert "scoring host" in res.detail


def test_a_report_without_host_is_unchanged(run):
    _takes_envs()
    res = _audit(run["s1"].report, run["cfg"], scoring_envs=_envs(from_block=10**9))
    assert res.status == "PASS", res.detail
