"""W15 (plan rev6.18 B6, owner 2026-11-23): same machine, same run.
Contract: README.md "Score cache (rev6.18 B6)".

Every model of a cohort (entries, KING_D, house) is forecast on one host in
one run. The score cache key is ``(digest, context, horizon, window_id)``
inside a namespace of the HOST IDENTITY (env, GPU model, driver, torch build,
fingerprint) and the code identity; a host change misses for every model and
the whole cohort is re-scored on the new host.

The source rig is W8c's (numpy only, no pod, no torch). Every test's first
statement imports ``cascade.validator.lane_d_score_cache`` (absent on main).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.accept.w8c import helpers as w8c

H = 64
RTX4090 = "NVIDIA GeForce RTX 4090"
CU124 = "2.4.1+cu124/cuda-12.4"
DG = "d" * 64


def _ns(host="1" * 64, code="2" * 64):
    from cascade.validator.lane_d_score_cache import namespace

    return namespace(host, code)


def _q(n, seed=0, horizon=H):
    return np.random.default_rng(seed).normal(size=(n, horizon, 9))


# ── the cache itself ────────────────────────────────────────────────────────


def test_a_put_is_returned_only_for_the_same_key_and_host(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    c = ScoreCache(tmp_path / "sc")
    ids = ["w1", "w2", "w3"]
    q = _q(3)
    c.put(_ns(), DG, context=4096, horizon=H, window_ids=ids, quantiles=q)
    got = c.get(_ns(), DG, context=4096, horizon=H, window_ids=ids)
    assert got is not None and np.array_equal(got, q), "exact float64 round trip"
    sub = c.get(_ns(), DG, context=4096, horizon=H, window_ids=["w3", "w1"])
    assert np.array_equal(sub, q[[2, 0]]), "rows follow the requested window order"
    assert c.get(_ns(host="3" * 64), DG, context=4096, horizon=H, window_ids=ids) is None, \
        "B6: another host identity never sees this host's forecasts"
    assert c.get(_ns(code="4" * 64), DG, context=4096, horizon=H, window_ids=ids) is None, \
        "other code never sees them either"
    assert c.get(_ns(), DG, context=8192, horizon=H, window_ids=ids) is None, "context is keyed"
    assert c.get(_ns(), DG, context=4096, horizon=32, window_ids=ids) is None, "horizon is keyed"
    assert c.get(_ns(), DG, context=4096, horizon=H, window_ids=ids + ["w4"]) is None, \
        "a missing window is a miss for the whole model (never a mixed answer)"


def test_non_finite_or_malformed_forecasts_are_never_cached(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    c = ScoreCache(tmp_path / "sc")
    bad = _q(2)
    bad[0, 0, 0] = np.nan
    c.put(_ns(), DG, context=4096, horizon=H, window_ids=["a", "b"], quantiles=bad)
    c.put(_ns(), DG, context=8192, horizon=H, window_ids=["a", "b"], quantiles=_q(3))
    assert c.get(_ns(), DG, context=4096, horizon=H, window_ids=["a", "b"]) is None
    assert c.get(_ns(), DG, context=8192, horizon=H, window_ids=["a", "b"]) is None


def test_a_corrupt_file_is_a_miss_and_is_dropped(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    c = ScoreCache(tmp_path / "sc")
    c.put(_ns(), DG, context=4096, horizon=H, window_ids=["a"], quantiles=_q(1))
    (f,) = (tmp_path / "sc" / _ns()).iterdir()
    f.write_bytes(b"not an npz")
    assert c.get(_ns(), DG, context=4096, horizon=H, window_ids=["a"]) is None
    assert not f.exists()


def test_retain_drops_other_hosts_and_unused_keys(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    c = ScoreCache(tmp_path / "sc")
    old, new = _ns(host="5" * 64), _ns(host="6" * 64)
    c.put(old, DG, context=4096, horizon=H, window_ids=["a"], quantiles=_q(1))
    c.put(new, DG, context=4096, horizon=H, window_ids=["a"], quantiles=_q(1))
    c.put(new, "e" * 64, context=4096, horizon=H, window_ids=["a"], quantiles=_q(1))
    c.retain(new, [(DG, 4096, H)])
    assert sorted(p.name for p in (tmp_path / "sc").iterdir()) == [new]
    assert c.get(new, DG, context=4096, horizon=H, window_ids=["a"]) is not None
    assert c.get(new, "e" * 64, context=4096, horizon=H, window_ids=["a"]) is None


def test_namespace_refuses_a_non_digest():
    from cascade.validator.lane_d_score_cache import namespace

    with pytest.raises(ValueError):
        namespace("", "2" * 64)
    with pytest.raises(ValueError):
        namespace("1" * 64, "XYZ")


# ── the settlement source: one host per cohort, host change re-scores ───────


def _envs():
    from cascade.shared.config.lane_d import parse_scoring_envs

    return parse_scoring_envs({"env": [{
        "name": "cu124-torch241", "status": "qualified", "from_block": 0, "until_block": 0,
        "lock": "uv.lock", "lock_sha256": "c" * 64, "torch_build": CU124,
        "gpu_models": [RTX4090], "driver_min": "550.54.14", "driver_max": "",
        "evidence": "reference"}]})


def _host_hello(uuid="GPU-1", driver="550.127.05") -> dict:
    return w8c.hello(env="cu124-torch241", torch_build=CU124, gpu_model=RTX4090,
                     driver=driver, fingerprint=hashlib.sha256(uuid.encode()).hexdigest())


def _rig(tmp_path, cache):
    rig = w8c.Rig(tmp_path)
    rig.eval_host.hello_doc = _host_hello()
    rig.source = rig.build(scoring_envs=_envs(), score_cache=cache)
    return rig


def test_a_second_run_on_the_same_host_reuses_the_cache(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    rig = _rig(tmp_path, ScoreCache(tmp_path / "score_cache"))
    first = rig.source(w8c.B1)
    n = len(rig.eval_host.calls)
    assert n >= 2, "the cohort (X and the house) is forecast"
    again = rig.source(w8c.B1)
    assert len(rig.eval_host.calls) == n, "same host, same code: nothing is re-forecast"
    assert w8c.forecasts_equal(again.forecasts, first.forecasts)
    assert again.host == first.host and first.host is not None


def test_a_host_change_forces_a_rescore_of_the_whole_cohort(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    rig = _rig(tmp_path, ScoreCache(tmp_path / "score_cache"))
    first = rig.source(w8c.B1)
    cohort = sorted(set(rig.eval_host.digests()))
    n = len(rig.eval_host.calls)
    for hello in (_host_hello(uuid="GPU-2"), _host_hello(driver="560.35.03")):
        rig.eval_host.hello_doc = hello
        got = rig.source(w8c.B1)
        redone = sorted(set(rig.eval_host.digests()[n:]))
        assert redone == cohort, "B6: every model of the cohort is re-scored on the new host"
        assert got.host["identity"] != first.host["identity"]
        n = len(rig.eval_host.calls)
    assert len(list((tmp_path / "score_cache").iterdir())) == 1, \
        "only the current host's forecasts are kept"


def test_every_forecast_of_a_run_comes_from_the_host_that_answered_hello(tmp_path):
    """One hello per run; the report's host is that host."""
    from cascade.validator.lane_d_score_cache import ScoreCache

    rig = _rig(tmp_path, ScoreCache(tmp_path / "score_cache"))
    inp = rig.source(w8c.B1)
    assert rig.eval_host.hellos == 1
    assert inp.host["env"] == "cu124-torch241" and inp.host["gpu_model"] == RTX4090


def test_an_unqualified_host_scores_nothing(tmp_path):
    from cascade.validator.lane_d_score_cache import ScoreCache

    rig = _rig(tmp_path, ScoreCache(tmp_path / "score_cache"))
    rig.eval_host.hello_doc = _host_hello(driver="535.104.05")
    with pytest.raises(Exception) as e:
        rig.source(w8c.B1)
    w8c.assert_inputs_error(e.value, "env_unqualified")
    assert rig.eval_host.calls == [], "nothing is forecast"
    rig.eval_host.hello_doc = w8c.hello()          # a pre-B7 hello: no environment
    with pytest.raises(Exception) as e:
        rig.source(w8c.B1)
    w8c.assert_inputs_error(e.value, "env_unqualified")
