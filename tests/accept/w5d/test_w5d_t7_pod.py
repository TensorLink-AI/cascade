"""W5d T7 pod command (plan §4 T7, §1.1 gate 2): ``python -P -m
cascade.eval.lane_d_t7_pod`` measures every context variant on the eval host
at 1000 windows (``[lane_d] min_windows``) and 5000 windows
(``max_windows_per_settlement``) and writes the ``laned_t7/1`` document the
box-side ``ops_jobs evidence`` job installs.

* the ROW that gates a variant is its 5000-window measurement (the most a
  settlement scores); both measurements are kept under ``window_sweep``;
* a variant is never timed on fewer or shorter windows than asked (RV3);
* the document parses with the scorer's own ``armed_record`` and passes the
  evidence job's pod check;
* only a CUDA run is pod evidence (``--device cpu`` writes ``source:
  "local"``, which the evidence job refuses).

The scorer is replaced by a recording fake of ``measure_variant`` /
``short_window_check`` built on the real ``variant_row`` (no torch, no
checkpoint is loaded). Every PATHS import is inside a test body.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def pod():
    return importlib.import_module("cascade.eval.lane_d_t7_pod")


class FakeKit:
    """``measure_variant`` / ``short_window_check`` stand-ins: wall time
    linear in windows and context, a CUDA peak, every row through the real
    ``variant_row``."""

    def __init__(self, *, mem_per_window_gb: float = 0.002):
        self.calls: list[tuple[int, int, int]] = []
        self.mem = mem_per_window_gb

    def measure(self, checkpoint_dir, windows, *, context, horizon, host, gpu, n_models,
                budget, device):
        from cascade.eval.lane_d_t7 import variant_row

        lengths = [w.size for w in windows]
        self.calls.append((context, len(windows), min(lengths)))
        per_model = 1e-4 * len(windows) * context / 4096
        return variant_row(host=host, gpu=gpu, variant=context, input_len=context,
                           history_min=min(lengths), history_max=max(lengths),
                           wall_s=per_model * n_models,
                           peak_mem_gb=self.mem * len(windows) * context / 4096,
                           peak_mem_source="cuda_max_allocated", budget=budget,
                           n_models=n_models, n_windows=len(windows))

    def short(self, checkpoint_dir, histories, *, horizon, device):
        assert all(h.size <= 4096 for h in histories)
        return {"base_context": 4096, "declared_context": 131072, "n_windows": len(histories),
                "max_abs_diff": 0.0, "identical": True}


def _args(tmp: Path, *extra: str):
    return pod().build_arg_parser().parse_args(
        ["--checkpoint-dir", str(tmp / "ckpt"), "--synthetic-seed", "7", "--host", "eval-1",
         "--gpu", "H100 80GB", "--mem-gb", "80", "--tool-git-sha", "a" * 40,
         "--out", str(tmp / "t7.json"), *extra])


def test_each_variant_is_measured_at_1000_and_5000_windows(tmp_path):
    kit = FakeKit()
    doc = pod().run(_args(tmp_path), measure=kit.measure, short=kit.short,
                    torch_version="2.4.1+cu124")
    assert sorted(kit.calls) == sorted((v, n, max(4096, 16384)) for v in (4096, 8192, 16384)
                                       for n in (1000, 5000))
    assert doc["schema"] == "laned_t7/1" and doc["source"] == "pod"
    assert [r["variant"] for r in doc["rows"]] == [4096, 8192, 16384]
    assert all(r["n_windows"] == 5000 for r in doc["rows"]), "the 5000-window row gates"
    assert {v: [r["n_windows"] for r in rows] for v, rows in doc["window_sweep"].items()} == {
        "4096": [1000, 5000], "8192": [1000, 5000], "16384": [1000, 5000]}
    assert doc["tool"]["window_counts"] == [1000, 5000]
    assert doc["tool"]["gate_window_count"] == 5000
    assert doc["short_window"]["identical"] is True and doc["short_window"]["n_windows"] == 5000


def test_the_document_arms_through_the_scorer_and_the_evidence_job(tmp_path):
    from cascade.eval.lane_d_scorer import armed_record

    kit = FakeKit(mem_per_window_gb=0.0045)      # 16384 at 5000 windows: 90 GB > 80 GB
    doc = pod().run(_args(tmp_path), measure=kit.measure, short=kit.short,
                    torch_version="2.4.1+cu124")
    raw = json.dumps(doc).encode()
    rec = armed_record([4096, 8192, 16384], readiness=None, t7=raw, min_long_series=0)
    fits = {v["context"]: v["t7_fits"] for v in rec["evidence"]["variants"]}
    assert fits == {4096: True, 8192: True, 16384: False}
    one_k = doc["window_sweep"]["16384"][0]
    assert one_k["n_windows"] == 1000 and one_k["fits"] is True, (
        "the 1000-window run alone would have armed it: only the 5000 row gates")


def test_main_writes_the_document_atomically(tmp_path, monkeypatch, capsys):
    kit = FakeKit()
    from cascade.eval import lane_d_t7

    monkeypatch.setattr(lane_d_t7, "measure_variant", kit.measure)
    monkeypatch.setattr(lane_d_t7, "short_window_check", kit.short)
    monkeypatch.setitem(__import__("sys").modules, "torch",
                        type("T", (), {"__version__": "2.4.1+cu124"}))
    argv = ["--checkpoint-dir", str(tmp_path / "ckpt"), "--synthetic-seed", "3", "--host",
            "eval-1", "--gpu", "H100", "--mem-gb", "80", "--tool-git-sha", "b" * 40,
            "--variants", "4096", "--out", str(tmp_path / "ev" / "t7.json")]
    assert pod().main(argv) == 0
    doc = json.loads((tmp_path / "ev" / "t7.json").read_text())
    assert doc["armed"] == [4096] and doc["torch"] == "2.4.1+cu124"
    assert not list((tmp_path / "ev").glob(".*"))


def test_too_few_long_windows_is_refused_and_nothing_written(tmp_path, capsys):
    hist = {f"w{i:04d}": np.arange(4096, dtype=np.float64) for i in range(1100)}
    hist.update({f"s{i:04d}": np.arange(100, dtype=np.float64) for i in range(400)})
    np.savez(tmp_path / "pool.npz", **hist)
    with pytest.raises(pod().T7RunError):
        pod().pick_windows(pod().load_histories(tmp_path / "pool.npz"), variant=4096, count=5000)
    got = pod().pick_windows(pod().load_histories(tmp_path / "pool.npz"), variant=4096,
                             count=1000)
    assert len(got) == 1000 and min(h.size for h in got) == 4096
    argv = ["--checkpoint-dir", str(tmp_path / "ckpt"), "--histories", str(tmp_path / "pool.npz"),
            "--host", "eval-1", "--gpu", "H100", "--mem-gb", "80", "--tool-git-sha", "c" * 40,
            "--variants", "4096", "--out", str(tmp_path / "t7.json")]
    assert pod().main(argv) == 1
    assert not (tmp_path / "t7.json").exists()


def test_a_cpu_run_is_local_evidence_the_evidence_job_refuses(tmp_path, capsys):
    import cascade.lane_d.ops_jobs as jobs

    kit = FakeKit()
    doc = pod().run(_args(tmp_path, "--device", "cpu", "--variants", "4096"),
                    measure=kit.measure, short=kit.short, torch_version="2.4.1")
    assert doc["source"] == "local"
    staged = tmp_path / "t7-local.json"
    staged.write_text(json.dumps(doc))
    rc = jobs.main(["evidence", "--chain-toml", str(REPO / "chain.toml"), "--t7", str(staged),
                    "--pinned-readiness", str(tmp_path / "r.json"),
                    "--pinned-t7", str(tmp_path / "t7.json")])
    assert rc == 1 and not (tmp_path / "t7.json").exists()
