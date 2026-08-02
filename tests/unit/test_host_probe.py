"""Host-side run telemetry — the probes, the fixed calibration bench, the
per-run stderr line, and the round roll-up's fleet-uniformity clause.

The point of these fields is separating pod-attributable slowness from
generator-attributable slowness, so the properties that matter are: every probe
degrades to missing keys instead of raising, the ids group runs at the right
scope (pod vs physical machine), and the bench is comparable — same workload
everywhere, and it never perturbs the run that follows it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.trainer import host_probe
from cascade.trainer.host_probe import (
    HOST_BENCH_SPEC,
    HOST_BENCH_TOKENS_PER_STEP,
    cpu_facts,
    gpu_facts,
    host_ids,
    host_snapshot,
    host_summary_line,
    lane_geometry,
    resolve_device,
)
from cascade.trainer.loop import host_spread_part, telemetry_rollup_line

# ── identity: the two scopes ──────────────────────────────────────────────────


def test_host_id_is_opaque_and_stable(tmp_path, monkeypatch):
    mid = tmp_path / "machine-id"
    mid.write_text("3f2a91c4e5d64b0aa1c2d3e4f5061728\n", encoding="utf-8")
    monkeypatch.setattr(host_probe, "_MACHINE_ID_PATHS", (str(mid),))
    monkeypatch.setattr(host_probe, "_BOOT_ID_PATH", str(tmp_path / "absent"))

    first, second = host_ids(), host_ids()
    assert first == second                          # stable: grouping works ACROSS rounds
    assert first["host_id_source"] == "machine-id"
    # Opaque: the raw machine-id must not be recoverable from the published id.
    assert "3f2a91c4" not in first["host_id"]
    assert len(first["host_id"]) == 12


def test_boot_id_is_a_second_scope_not_a_fallback(tmp_path, monkeypatch):
    """Two pods on one machine: different machine-id, SAME boot_id.

    That pairing is the only co-tenancy signal available pod-side, so the two ids
    must be independent fields — collapsing them would erase it.
    """
    boot = tmp_path / "boot_id"
    boot.write_text("11112222-3333-4444-5555-666677778888\n", encoding="utf-8")
    monkeypatch.setattr(host_probe, "_BOOT_ID_PATH", str(boot))

    ids = []
    for name in ("pod-a", "pod-b"):
        mid = tmp_path / name
        mid.write_text(name, encoding="utf-8")
        monkeypatch.setattr(host_probe, "_MACHINE_ID_PATHS", (str(mid),))
        ids.append(host_ids())

    assert ids[0]["host_id"] != ids[1]["host_id"]            # distinct pods
    assert ids[0]["host_boot_id"] == ids[1]["host_boot_id"]  # one machine


def test_missing_machine_id_degrades_and_labels_the_coarser_source(tmp_path, monkeypatch):
    boot = tmp_path / "boot_id"
    boot.write_text("aaaabbbb-cccc-dddd-eeee-ffff00001111\n", encoding="utf-8")
    monkeypatch.setattr(host_probe, "_MACHINE_ID_PATHS", (str(tmp_path / "absent"),))
    monkeypatch.setattr(host_probe, "_BOOT_ID_PATH", str(boot))

    ids = host_ids()
    # No pod id available: the machine id stands in, but SAYS it is coarser — a
    # consumer must not read a boot-scoped grouping as per-pod.
    assert ids["host_id_source"] == "boot-id"
    assert ids["host_id"] == ids["host_boot_id"]


def test_no_identity_files_at_all_still_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(host_probe, "_MACHINE_ID_PATHS", (str(tmp_path / "no"),))
    monkeypatch.setattr(host_probe, "_BOOT_ID_PATH", str(tmp_path / "nope"))
    ids = host_ids()
    assert ids["host_id_source"] == "hostname"
    assert len(ids["host_id"]) == 12


# ── lane geometry + cpu ───────────────────────────────────────────────────────


def test_lane_geometry_reads_dispatch_env(monkeypatch):
    monkeypatch.setenv("CASCADE_LANE_INDEX", "3")
    monkeypatch.setenv("CASCADE_LANE_COUNT", "8")
    geo = lane_geometry()
    assert geo["host_lane_index"] == 3
    assert geo["host_lane_count"] == 8
    # The generator's core slice comes from the sandbox's own rule, so the number
    # reported is the number actually enforced.
    assert geo["host_gen_cpu_slice"] >= 1


def test_absent_lane_env_reads_as_a_single_lane(monkeypatch):
    monkeypatch.delenv("CASCADE_LANE_INDEX", raising=False)
    monkeypatch.delenv("CASCADE_LANE_COUNT", raising=False)
    geo = lane_geometry()
    assert (geo["host_lane_index"], geo["host_lane_count"]) == (0, 1)
    # No slicing on a single lane — the key is absent, not a bogus core count.
    assert "host_gen_cpu_slice" not in geo


def test_cpu_facts_report_affinity_and_the_box_total(monkeypatch):
    facts = cpu_facts()
    # Affinity, not cpu_count: a cpuset-limited pod on a big machine must report
    # what the run GOT. Both are present so the gap between them is visible.
    assert facts["host_cpu_count"] >= 1
    assert facts["host_cpu_count_total"] >= 1


def test_cpu_facts_survive_a_missing_proc_cpuinfo(monkeypatch):
    monkeypatch.setattr(
        host_probe.Path, "read_text",
        lambda self, **kw: (_ for _ in ()).throw(OSError("no /proc here")),
    )
    facts = cpu_facts()
    assert "host_cpu_model" not in facts       # omitted, not guessed
    assert facts["host_cpu_count"] >= 1


# ── nvidia-smi parsing ────────────────────────────────────────────────────────

_TWO_GPUS = (
    "0, NVIDIA GeForce RTX 4090, 550.90.07, 2640, 10501, 4, 16\n"
    "1, NVIDIA GeForce RTX 4090, 550.90.07, 2640, 10501, 1, 8\n"
)


def test_gpu_facts_pick_this_lanes_device_by_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    facts = gpu_facts(run_fn=lambda argv: _TWO_GPUS)
    assert facts["host_gpu_count"] == 2         # the BOX's fan-out, not the lane's
    assert facts["host_gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert facts["host_driver_version"] == "550.90.07"
    # A downshifted PCIe link is a host explanation for a slow run that the SKU
    # name hides — so it must come from the masked device's own row.
    assert facts["host_pcie_gen"] == 1
    assert facts["host_pcie_width"] == 8


def test_gpu_facts_omit_per_device_fields_when_the_mask_is_ambiguous(monkeypatch):
    # A multi-device mask over a multi-GPU box: attributing one row's nameplate
    # to the run would be a wrong number, which is worse than a missing one.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    facts = gpu_facts(run_fn=lambda argv: _TWO_GPUS)
    assert facts == {"host_gpu_count": 2}


def test_gpu_facts_use_the_only_row_when_unmasked(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    one = "0, NVIDIA L40S, 550.90.07, 2520, 9001, 4, 16\n"
    facts = gpu_facts(run_fn=lambda argv: one)
    assert facts["host_gpu_name"] == "NVIDIA L40S"
    assert facts["host_gpu_clock_max_mhz"] == 2520


def test_gpu_facts_tolerate_na_fields(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    na = "0, NVIDIA L40S, 550.90.07, [N/A], [N/A], [N/A], [N/A]\n"
    facts = gpu_facts(run_fn=lambda argv: na)
    assert facts["host_gpu_name"] == "NVIDIA L40S"
    assert "host_gpu_clock_max_mhz" not in facts   # virtualised GPU: absent, not 0


def test_gpu_facts_on_a_box_with_no_nvidia_smi():
    def boom(argv):
        raise FileNotFoundError("no nvidia-smi")

    with pytest.raises(FileNotFoundError):
        gpu_facts(run_fn=boom)
    # …and the assembled snapshot swallows it: a CPU-only box still reports the
    # rest (this is the contract host_snapshot promises, verified below).


# ── the fixed calibration bench ───────────────────────────────────────────────


def test_cpu_bench_reports_a_positive_token_rate():
    rate = host_probe._cpu_bench(iters=2)
    assert rate > 0.0


def test_host_bench_on_cpu_is_comparable_and_labelled():
    torch = pytest.importorskip("torch")
    del torch
    facts = host_probe.host_bench("cpu")
    assert facts["host_bench_tokens_per_s"] > 0.0
    assert facts["host_bench_cpu_tokens_per_s"] > 0.0
    assert facts["host_bench_device"] == "cpu"
    # Every measurement carries the workload it came from: two code versions must
    # never have their numbers silently pooled.
    assert facts["host_bench_spec"] == HOST_BENCH_SPEC
    # No CUDA ⇒ no device-bandwidth leg, rather than a meaningless zero.
    assert "host_bench_d2d_gb_s" not in facts
    assert facts["host_bench_seconds"] >= 0.0


def test_bench_does_not_disturb_the_global_seeds_the_contract_pins():
    """The contract pins init + data order through the global RNGs. A telemetry
    probe that consumed from them would shift the run it precedes."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(1234)
    np.random.seed(1234)
    before_t = torch.randn(4)
    before_n = np.random.rand(4)

    torch.manual_seed(1234)
    np.random.seed(1234)
    host_probe.host_bench("cpu")
    after_t = torch.randn(4)
    after_n = np.random.rand(4)

    assert torch.equal(before_t, after_t)
    assert np.array_equal(before_n, after_n)


def test_bench_falls_back_when_the_named_device_is_absent():
    # A custom BaseTrainer can name a device this box does not have. Losing every
    # leg over one bad device string would drop the telemetry precisely on the
    # hosts worth inspecting, so the CPU legs must survive.
    pytest.importorskip("torch")
    facts = host_probe.host_bench("cuda:7")
    assert facts["host_bench_device"] == "cpu"
    assert facts["host_bench_cpu_tokens_per_s"] > 0.0


def test_bench_token_unit_matches_a_training_step():
    # The tokens/s unit is only interpretable if one bench "step" moves the same
    # number of elements a real training step does (~262k).
    assert HOST_BENCH_TOKENS_PER_STEP == 262_144


# ── assembly: partial answers are the normal degraded mode ────────────────────


def test_host_snapshot_survives_every_probe_failing(monkeypatch):
    for name in ("host_ids", "lane_geometry", "cpu_facts", "gpu_facts", "torch_facts"):
        monkeypatch.setattr(
            host_probe, name,
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe down")),
        )
    monkeypatch.setattr(
        host_probe, "host_bench",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bench down")),
    )
    assert host_snapshot() == {}         # empty, never an exception


def test_host_snapshot_omits_the_bench_when_asked(monkeypatch):
    monkeypatch.setattr(
        host_probe, "host_bench",
        lambda *a, **k: pytest.fail("bench must not run when disabled"),
    )
    facts = host_snapshot(device="cpu", run_bench=False)
    assert facts["host_id"]
    assert not any(k.startswith("host_bench") for k in facts)


def test_host_snapshot_is_flat_json_scalars():
    # Both sinks require it: the S3 log is one JSON object per line and wandb
    # charts scalars only.
    facts = host_snapshot(device="cpu", run_bench=False)
    assert facts
    assert all(isinstance(v, (int, float, str, bool)) for v in facts.values())


def test_resolve_device_prefers_the_backends_own_device():
    assert resolve_device("cuda:3") == "cuda:3"
    assert resolve_device(None) in ("cpu", "cuda")


# ── the stderr line (the remote telemetry channel) ────────────────────────────


def test_host_summary_line_is_parseable_key_values():
    line = host_summary_line({
        "host_id": "abc123abc123", "host_lane_index": 2, "host_lane_count": 4,
        "host_cpu_count": 32, "host_gen_cpu_slice": 8,
        "host_gpu_name": "NVIDIA GeForce RTX 4090",
        "host_bench_tokens_per_s": 3_141_592.0,
    })
    assert "host_id=abc123abc123" in line
    assert "host_lane_count=4" in line
    assert "host_bench_tokens_per_s=3141592.0" in line
    assert "host_boot_id" not in line          # absent keys are not emitted empty


def test_host_summary_line_with_nothing_to_say():
    assert host_summary_line({}) == "no host facts available"


# ── round roll-up: fleet uniformity ───────────────────────────────────────────


def _run(bench=None, pod="p1", machine="m1", gpu="NVIDIA GeForce RTX 4090"):
    m = {"deadline_hit": False, "data_wait_frac": 0.1}
    if bench is not None:
        m |= {"host_bench_tokens_per_s": bench, "host_id": pod,
              "host_boot_id": machine, "host_gpu_name": gpu}
    return m


def test_host_spread_reports_the_bound_on_host_attributable_variance():
    part = host_spread_part([
        _run(1_000_000, pod="a", machine="m1"),
        _run(1_500_000, pod="b", machine="m1"),
        _run(2_000_000, pod="c", machine="m2"),
    ])
    assert "spread=2.00x" in part           # max/min: what the pods alone could explain
    assert "3 pods" in part
    assert "2 machines" in part             # a and b are co-tenants
    assert "1 skus" in part


def test_host_spread_is_silent_when_nothing_benched():
    assert host_spread_part([_run(), _run()]) == ""


def test_rollup_keeps_its_previous_text_without_host_facts():
    line = telemetry_rollup_line(7, [_run(), _run()], [_run()])
    assert "deadline_hit 0/2 heats + 0/1 finals" in line
    assert "host_bench" not in line


def test_rollup_appends_the_host_clause_when_present():
    line = telemetry_rollup_line(7, [_run(1e6, pod="a"), _run(4e6, pod="b")], [])
    assert "data_wait_frac p50=" in line
    assert "host_bench min=1,000,000" in line
    assert "spread=4.00x" in line


def test_host_clause_counts_runs_that_lack_backend_metrics():
    """Host facts come from the pod probe, backend metrics from the trainer. A
    custom BaseTrainer emits no deadline_hit — its run still occupied a pod, so
    excluding it would compute uniformity over the wrong denominator."""
    bare = {"host_bench_tokens_per_s": 5e6, "host_id": "z", "host_boot_id": "m9"}
    line = telemetry_rollup_line(9, [_run(1e6, pod="a"), bare], [])
    assert "2 benched" in line
    assert "spread=5.00x" in line


# ── wiring: the round actually collects and publishes this ─────────────────────


def test_shipped_chain_toml_enables_the_probe(cfg):
    # The section is optional and defaults on, so an operator who never adds it
    # still gets the telemetry — but the shipped template must not disagree.
    assert cfg.telemetry.host_probe is True
    assert cfg.telemetry.host_bench is True


def _round_records(cfg, tmp_path, monkeypatch, caplog):
    """Run one fixture round with a fake stream/trainer/registry and return
    ``(emitted log records, log messages)``."""
    from cascade.shared.chain import Commitment
    from cascade.shared.hippius import HubRef, HubUpload
    from cascade.trainer import loop as loop_mod
    from cascade.trainer.contract import TrainResult
    from cascade.trainer.loop import TrainerRunner

    records: list[dict] = []

    class _Sink:
        def __init__(self, *a, **k):
            pass

        def emit(self, record):
            records.append(dict(record))

        def flush(self):
            return None

    class _Stream:
        digest, n_series, total_points = "corpusdigest", 3, 192

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def series(self):
            yield np.ones((1, 64))

    class _Trainer:
        device = "cpu"          # the seam host_snapshot reads for its bench device

        def train(self, stream, contract, *, training_seed, token_budget,
                  out_dir, logger=None):
            for _ in stream:
                pass
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "weights.safetensors").write_bytes(b"x")
            return TrainResult(
                local_dir=out_dir, param_count=4, train_seconds=2.0,
                metrics={"deadline_hit": False, "tokens_frac": 1.0,
                         "data_wait_frac": 0.05, "throughput_tokens_per_s": 3.9e6},
            )

    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", lambda *a, **k: _Stream())
    monkeypatch.setattr(loop_mod, "LogSink", _Sink)
    monkeypatch.setattr(TrainerRunner, "logs_store", lambda self: object())
    monkeypatch.setattr(
        loop_mod, "upload_dir_to_hub_or_hf",
        lambda local_dir, repo, hub=None, *, hf_repo=None: HubUpload(
            ref=HubRef.parse("cascade/ckpt-out@sha256:" + "e" * 64), size_bytes=1),
    )
    runner = TrainerRunner(cfg=cfg, base_trainer=_Trainer(), work_root=tmp_path,
                           use_sandbox=False)
    commits = [
        Commitment(uid=0, hotkey="a", coldkey=None,
                   payload="metro-v1:gen:hippius:alice/gen-a@sha256:" + "a" * 64,
                   commit_block=5),
        Commitment(uid=1, hotkey="b", coldkey=None,
                   payload="metro-v1:gen:hippius:bob/gen-b@sha256:" + "b" * 64,
                   commit_block=6),
    ]
    with caplog.at_level("INFO", logger="cascade.trainer"):
        runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    return records, [r.message for r in caplog.records]


@pytest.fixture()
def probe_cfg(cfg):
    """Probe on, bench off — the wiring is what's under test here, and the bench
    would spend real GPU/CPU seconds per fixture run to prove nothing extra."""
    from dataclasses import replace

    from cascade.shared.config import TelemetryConfig

    return replace(cfg, telemetry=TelemetryConfig(host_probe=True, host_bench=False))


def test_round_emits_a_host_record_and_repeats_it_on_the_summary(
    probe_cfg, tmp_path, monkeypatch, caplog
):
    records, msgs = _round_records(probe_cfg, tmp_path, monkeypatch, caplog)

    hosts = [r for r in records if r.get("event") == "host"]
    assert hosts, "every run must publish its host facts"
    assert all(h.get("host_id") for h in hosts)
    assert all("role" in h for h in hosts)      # which run this host served

    # Repeated on the summary row: the regression of realized throughput on host
    # capability is only a one-liner when both sit on the same row.
    summaries = [r for r in records if r.get("event") == "summary"]
    assert summaries
    for s in summaries:
        assert s["host_id"] == hosts[0]["host_id"]
        assert "throughput_tokens_per_s" in s

    # …and on stderr, the only channel that survives a pod with no S3 access.
    assert any("host: host_id=" in m for m in msgs)


def test_host_facts_never_overwrite_backend_metrics(
    probe_cfg, tmp_path, monkeypatch, caplog
):
    """The summary merges host facts UNDER the run's metrics. A host key can only
    ever add context, never shadow a scored-adjacent number like throughput."""
    records, _ = _round_records(probe_cfg, tmp_path, monkeypatch, caplog)
    summaries = [r for r in records if r.get("event") == "summary"]
    assert summaries and all(s["throughput_tokens_per_s"] == 3.9e6 for s in summaries)


def test_disabling_the_probe_restores_the_previous_records(
    cfg, tmp_path, monkeypatch, caplog
):
    from dataclasses import replace

    from cascade.shared.config import TelemetryConfig

    off = replace(cfg, telemetry=TelemetryConfig(host_probe=False))
    records, msgs = _round_records(off, tmp_path, monkeypatch, caplog)
    assert not [r for r in records if r.get("event") == "host"]
    assert not any(k.startswith("host_") for r in records for k in r)
    assert not any("host: host_id=" in m for m in msgs)


def test_a_broken_probe_does_not_fail_the_round(
    probe_cfg, tmp_path, monkeypatch, caplog
):
    from cascade.trainer import loop as loop_mod

    monkeypatch.setattr(
        loop_mod, "host_snapshot",
        lambda **k: (_ for _ in ()).throw(RuntimeError("probe exploded")),
    )
    records, _ = _round_records(probe_cfg, tmp_path, monkeypatch, caplog)
    # The round still trained and still published its summaries.
    assert [r for r in records if r.get("event") == "summary"]
