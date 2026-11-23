"""W3b6a sidecar settings (tests-correction, coordinator 2026-11-23): the
lane D bench sweep runs the sidecar with exactly the settings lane A's
DEC-CA-0063 sweep uses (``validator_bench._vbench_run_sidecar_many``:
``num_samples = [eval] benchmark_num_samples or num_samples``, ``max_series
= [eval] cascade_bench_max_series``, ``data_dir = [eval] gift_gate_data_dir``,
``workers = [eval] bench_workers``, the sidecar's batch 512), never a
hard-coded copy, so a lane D and a lane A bench of the same checkpoint match.
The box derives them from its config and sends them in the request; the pod
passes them to the sidecar unchanged. Contract: README.md "Sidecar
settings"."""

from __future__ import annotations

import dataclasses

import pytest

from . import helpers as h

A = "a" * 64


def _cfg(tmp_path, **eval_over):
    from tests.accept.w8b import helpers as w8b

    cfg = w8b.chain_cfg(tmp_path / "v")
    return dataclasses.replace(cfg, eval=dataclasses.replace(cfg.eval, **eval_over)) \
        if eval_over else cfg


def _settings_fn():
    m = h.host_mod()
    assert hasattr(m, "bench_settings"), "lane_d_bench_host.bench_settings(cfg) (README)"
    return m.bench_settings


@pytest.mark.parametrize("over", [
    {},
    {"benchmark_num_samples": 0, "num_samples": 64},
    {"benchmark_num_samples": 250, "num_samples": 64},
    {"cascade_bench_max_series": 40, "gift_gate_data_dir": "/data/pinned"},
    {"bench_workers": "auto"},
])
def test_settings_are_lane_as(tmp_path, over):
    cfg = _cfg(tmp_path, **over)
    got = _settings_fn()(cfg)
    want = h.lane_a_settings(cfg)
    assert {k: got.get(k) for k in want} == want, \
        "num_samples / max_series / batch_size / workers / data_dir as lane A's sweep"
    assert got.get("timeout_s_per_checkpoint") == h.lane_a_timeout_per_checkpoint(cfg), \
        "lane A's per-checkpoint allowance"
    assert set(got) == set(want) | {"timeout_s_per_checkpoint"}


def test_box_sends_the_settings_and_scales_the_timeout(tmp_path):
    import inspect

    m = h.host_mod()
    assert "settings" in inspect.signature(m.SSHBenchHost).parameters
    run = h.Run(h.reply({A: h.rows_entry(1), "b" * 64: h.rows_entry(2)}))
    s = dict(h.SETTINGS, timeout_s_per_checkpoint=900)
    host = m.SSHBenchHost(h.hosts_file(tmp_path), cache_dir="/c", budget_bytes=10, run=run,
                          settings=s)
    host.set_contexts({A: 4096, "b" * 64: 4096})
    host((A, "b" * 64), h.SUITES)
    req = run.request()
    assert req.get("settings") == h.SETTINGS, \
        "the sidecar settings travel in the request (the timeout stays on the box)"
    assert run.calls[0]["timeout"] == 900 * 2, "the allowance is per checkpoint swept"


def test_inputs_file_bench_host_carries_lane_as_settings(tmp_path):
    m = __import__("cascade.validator.lane_d_inputs", fromlist=["x"])
    keys = {"intake_state_dir": "/i", "pool_pins": "/p.json", "house_pointer": "/h.json",
            "house_signer_ss58": "5x", "eval_hosts": "/e.toml", "eval_cache_dir": "/c",
            "readiness": "/r.json", "t7": "/t.json",
            "bench_hosts": str(h.hosts_file(tmp_path))}
    p = tmp_path / "inputs.toml"
    p.write_text("\n".join([f'{k} = "{v}"' for k, v in keys.items()]
                           + ["eval_cache_budget_bytes = 10"]) + "\n", encoding="utf-8")
    cfg = _cfg(tmp_path, benchmark_num_samples=123, cascade_bench_max_series=7)
    bench = m.load_inputs_services(cfg, p)["bench_host"]
    assert getattr(bench, "settings", None) == _settings_fn()(cfg), \
        "the production bench host is built with bench_settings(cfg)"


def test_pod_passes_the_request_settings_to_the_sidecar(tmp_path):
    e = h.Entry(21)
    cache = h.cache(tmp_path, e)
    sidecar = h.Sidecar({e.digest: h.report_for(21)})
    h.sweep_mod().serve_sweep(h.request([e.digest]), cache=cache, work_dir=tmp_path / "w",
                              sidecar=sidecar, fingerprint=dict(h.FINGERPRINT))
    assert sidecar.calls[0]["settings"] == h.SETTINGS, \
        "sidecar(dirs, suites, settings): the request's settings, unchanged"


@pytest.mark.parametrize("damage", [
    "missing", "not_object", "extra_key", "num_samples_zero", "max_series_negative",
    "batch_bool", "workers_int", "data_dir_int"])
def test_pod_refuses_malformed_settings(tmp_path, damage):
    e = h.Entry(22)
    cache = h.cache(tmp_path, e)
    req = h.request([e.digest])
    s = dict(h.SETTINGS)
    if damage == "missing":
        req.pop("settings")
    elif damage == "not_object":
        req["settings"] = [1, 2]
    else:
        if damage == "extra_key":
            s["device"] = "cpu"
        elif damage == "num_samples_zero":
            s["num_samples"] = 0
        elif damage == "max_series_negative":
            s["max_series"] = -1
        elif damage == "batch_bool":
            s["batch_size"] = True
        elif damage == "workers_int":
            s["workers"] = 3
        else:
            s["data_dir"] = 5
        req["settings"] = s
    sidecar = h.Sidecar({e.digest: h.report_for(22)})
    try:
        h.sweep_mod().serve_sweep(req, cache=cache, work_dir=tmp_path / "w", sidecar=sidecar,
                                  fingerprint=dict(h.FINGERPRINT))
    except ValueError:
        pass
    else:
        raise AssertionError(f"settings damage {damage!r} must be refused (ValueError)")
    assert sidecar.calls == []
