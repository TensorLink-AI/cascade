"""W16 box tests: the reference fine-tune script's CLI, config and refusal
contract, checked without torch (every subprocess runs with ``import torch``
blocked; the in-process tests only parse arguments)."""

from __future__ import annotations

import json
import math

import pytest

from .helpers import (
    ALLOWED,
    REFUSAL_CODES,
    REFUSAL_EXIT,
    REFUSED_PREFIX,
    config_json,
    fixture_weights,
    last_json_line,
    refusal_codes,
    run_script,
    snapshot,
    write_base,
    write_data,
)

FLAGS = ("--base", "--data", "--out", "--steps", "--lr", "--seed", "--batch-size",
         "--device", "--context-length", "--dry-run")


def _args(base, data, out, *extra):
    return ["--base", str(base), "--data", str(data), "--out", str(out), *extra]


# ── module surface ───────────────────────────────────────────────────────────


def test_refusal_codes_are_the_documented_set():
    import cascade.miner.lane_d_finetune as ft

    assert tuple(ft.REFUSAL_CODES) == REFUSAL_CODES
    err = ft.FinetuneRefused("data_bad", "line 3")
    assert err.code == "data_bad"
    assert isinstance(err, Exception)


def test_parse_args_defaults_and_types(tmp_path):
    import cascade.miner.lane_d_finetune as ft

    cfg = ft.parse_args(_args(tmp_path / "b", tmp_path / "d.jsonl", tmp_path / "o"))
    assert cfg.base == tmp_path / "b" and cfg.data == tmp_path / "d.jsonl"
    assert cfg.out == tmp_path / "o"
    assert isinstance(cfg.steps, int) and cfg.steps >= 1
    assert isinstance(cfg.lr, float) and math.isfinite(cfg.lr) and cfg.lr > 0
    assert cfg.seed == 0
    assert isinstance(cfg.batch_size, int) and cfg.batch_size >= 1
    assert cfg.device == "auto"
    assert cfg.context_length is None
    assert cfg.dry_run is False
    cfg2 = ft.parse_args(_args(tmp_path / "b", tmp_path / "d.jsonl", tmp_path / "o",
                               "--steps", "7", "--lr", "0.003", "--seed", "11",
                               "--batch-size", "3", "--device", "cpu",
                               "--context-length", "256", "--dry-run"))
    assert (cfg2.steps, cfg2.lr, cfg2.seed, cfg2.batch_size, cfg2.device,
            cfg2.context_length, cfg2.dry_run) == (7, 0.003, 11, 3, "cpu", 256, True)


@pytest.mark.parametrize("extra", [
    ["--steps", "0"], ["--steps", "-1"], ["--steps", "1.5"],
    ["--lr", "0"], ["--lr", "-0.1"], ["--lr", "nan"], ["--lr", "inf"],
    ["--batch-size", "0"], ["--device", "tpu"], ["--context-length", "0"],
], ids=lambda e: "_".join(e).strip("-"))
def test_parse_args_rejects_bad_values(tmp_path, extra):
    import cascade.miner.lane_d_finetune as ft

    with pytest.raises(SystemExit) as e:
        ft.parse_args(_args(tmp_path / "b", tmp_path / "d.jsonl", tmp_path / "o", *extra))
    assert e.value.code == 2


@pytest.mark.parametrize("drop", ["--base", "--data", "--out"])
def test_parse_args_requires_base_data_out(tmp_path, drop):
    import cascade.miner.lane_d_finetune as ft

    argv = _args(tmp_path / "b", tmp_path / "d.jsonl", tmp_path / "o")
    i = argv.index(drop)
    with pytest.raises(SystemExit) as e:
        ft.parse_args(argv[:i] + argv[i + 2:])
    assert e.value.code == 2


# ── entry point without torch ────────────────────────────────────────────────


def test_help_without_torch(tmp_path):
    proc = run_script(tmp_path, "--help")
    assert proc.returncode == 0, proc.stderr[-3000:]
    for flag in FLAGS:
        assert flag in proc.stdout, flag


def test_bad_argument_exits_2_without_torch(tmp_path):
    base = write_base(tmp_path / "base")
    data = write_data(tmp_path / "data.jsonl")
    proc = run_script(tmp_path, *_args(base, data, tmp_path / "out", "--steps", "0"))
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-3000:])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, dtype):
    base = write_base(tmp_path / "base", dtype=dtype)
    data = write_data(tmp_path / "data.jsonl")
    before = snapshot(tmp_path / "base")
    out = tmp_path / "out"
    proc = run_script(tmp_path, *_args(base, data, out, "--steps", "5", "--lr", "0.001",
                                       "--seed", "9", "--batch-size", "2",
                                       "--device", "cpu", "--context-length", "256",
                                       "--dry-run"))
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert refusal_codes(proc.stderr) == []
    plan = last_json_line(proc.stdout)
    assert plan["base"] == str(base.resolve())
    assert plan["data"] == str(data.resolve())
    assert plan["out"] == str(out.resolve())
    assert plan["base_arch"] == "cascade_pyramid"
    assert plan["n_tensors"] == 4
    assert plan["n_series"] == 4
    assert (plan["steps"], plan["lr"], plan["seed"], plan["batch_size"], plan["device"],
            plan["context_length"]) == (5, 0.001, 9, 2, "cpu", 256)
    assert plan["out_dtype"] == "BF16"
    assert plan["files"] == list(ALLOWED)
    assert not out.exists()
    assert snapshot(tmp_path / "base") == before


def test_dry_run_accepts_an_empty_out_dir(tmp_path):
    base = write_base(tmp_path / "base")
    data = write_data(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    proc = run_script(tmp_path, *_args(base, data, out, "--dry-run"))
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert list(out.iterdir()) == []


def test_valid_inputs_without_torch_refuse_torch_unavailable(tmp_path):
    """Every input check runs before torch is imported; a box without torch
    gets the typed refusal, not a traceback, and no output dir."""
    base = write_base(tmp_path / "base")
    data = write_data(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    proc = run_script(tmp_path, *_args(base, data, out, "--steps", "1"))
    assert proc.returncode == REFUSAL_EXIT, (proc.returncode, proc.stderr[-3000:])
    assert refusal_codes(proc.stderr) == ["torch_unavailable"]
    assert "Traceback" not in proc.stderr
    assert not out.exists()


# ── typed refusals (README "Refusals") ───────────────────────────────────────


def _mutate_base(case: str, base, tmp_path):
    marker = tmp_path / "w16_marker"
    if case == "extra_py":
        (base / "evil.py").write_text(f"import os\nos.mkdir({str(marker)!r})\n")
    elif case == "pickle_pt":
        (base / "model.pt").write_bytes(b"\x80\x04K\x01.")
    elif case == "missing_weights":
        (base / "weights.safetensors").unlink()
    elif case == "nested_dir":
        (base / "pkg").mkdir()
        (base / "pkg" / "__init__.py").write_text(f"import os\nos.mkdir({str(marker)!r})\n")
    elif case == "edited_model_py":
        (base / "model.py").write_bytes(
            f"import os\nos.mkdir({str(marker)!r})\n".encode() + (base / "model.py").read_bytes())
    elif case == "edited_wrapper_py":
        (base / "forecast_wrapper.py").write_bytes(
            (base / "forecast_wrapper.py").read_bytes() + b"\n# edited\n")
    elif case == "truncated_config":
        (base / "config.json").write_bytes(config_json()[:20])
    elif case == "config_not_object":
        (base / "config.json").write_text("[1, 2]")
    elif case == "config_no_moe":
        (base / "config.json").write_bytes(config_json(drop_moe=True))
    elif case == "config_wrong_arch":
        (base / "config.json").write_bytes(config_json(base_arch="toto2"))
    elif case == "weights_garbage":
        (base / "weights.safetensors").write_bytes(b"\x80\x04K\x01." * 4)
    elif case == "weights_bad_offsets":
        (base / "weights.safetensors").write_bytes(
            fixture_weights(offsets={"head.bias": [0, 3]}))
    elif case == "weights_bad_dtype":
        (base / "weights.safetensors").write_bytes(
            fixture_weights().replace(b'"F32"', b'"X32"', 1))
    else:
        raise AssertionError(case)
    return marker


BASE_CASES = {
    "extra_py": "base_files", "pickle_pt": "base_files", "missing_weights": "base_files",
    "nested_dir": "base_files",
    "edited_model_py": "base_code", "edited_wrapper_py": "base_code",
    "truncated_config": "base_config", "config_not_object": "base_config",
    "config_no_moe": "base_config", "config_wrong_arch": "base_config",
    "weights_garbage": "base_header", "weights_bad_offsets": "base_header",
    "weights_bad_dtype": "base_header",
}


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "run"])
@pytest.mark.parametrize("case", sorted(BASE_CASES))
def test_bad_base_refused_with_typed_code(tmp_path, case, dry_run):
    base = write_base(tmp_path / "base")
    marker = _mutate_base(case, base, tmp_path)
    data = write_data(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    proc = run_script(tmp_path, *_args(base, data, out, *(["--dry-run"] if dry_run else [])))
    assert proc.returncode == REFUSAL_EXIT, (proc.returncode, proc.stderr[-3000:])
    assert refusal_codes(proc.stderr) == [BASE_CASES[case]]
    assert "Traceback" not in proc.stderr
    assert not marker.exists(), "base bytes were executed"
    assert not out.exists()


def _write_bad_data(case: str, path):
    good = json.dumps({"target": [0.5] * 640})
    if case == "missing":
        return path
    if case == "not_json":
        path.write_text(good + "\n{not json\n")
    elif case == "no_target":
        path.write_text(good + "\n" + json.dumps({"values": [1.0, 2.0]}) + "\n")
    elif case == "target_not_list":
        path.write_text(json.dumps({"target": "1,2,3"}) + "\n")
    elif case == "non_numeric":
        path.write_text(json.dumps({"target": [1.0, "x", 3.0]}) + "\n")
    elif case == "nan":
        path.write_text('{"target": [1.0, NaN, 3.0]}\n')
    elif case == "inf":
        path.write_text('{"target": [1.0, Infinity, 3.0]}\n')
    elif case == "bool":
        path.write_text(json.dumps({"target": [1.0, True, 3.0]}) + "\n")
    elif case == "empty":
        path.write_text("")
    elif case == "blank_lines":
        path.write_text("\n\n  \n")
    else:
        raise AssertionError(case)
    return path


DATA_CASES = {
    "missing": "data_missing",
    "not_json": "data_bad", "no_target": "data_bad", "target_not_list": "data_bad",
    "non_numeric": "data_bad", "nan": "data_bad", "inf": "data_bad", "bool": "data_bad",
    "empty": "data_empty", "blank_lines": "data_empty",
}


@pytest.mark.parametrize("case", sorted(DATA_CASES))
def test_bad_data_refused_with_typed_code(tmp_path, case):
    base = write_base(tmp_path / "base")
    data = _write_bad_data(case, tmp_path / "data.jsonl")
    out = tmp_path / "out"
    proc = run_script(tmp_path, *_args(base, data, out, "--dry-run"))
    assert proc.returncode == REFUSAL_EXIT, (proc.returncode, proc.stderr[-3000:])
    assert refusal_codes(proc.stderr) == [DATA_CASES[case]]
    assert not out.exists()


def test_non_empty_out_refused_and_left_untouched(tmp_path):
    base = write_base(tmp_path / "base")
    data = write_data(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("miner's own file")
    before = snapshot(out)
    for extra in ([], ["--dry-run"]):
        proc = run_script(tmp_path, *_args(base, data, out, *extra))
        assert proc.returncode == REFUSAL_EXIT, (proc.returncode, proc.stderr[-3000:])
        assert refusal_codes(proc.stderr) == ["out_not_empty"]
        assert snapshot(out) == before


def test_out_equal_to_base_refused(tmp_path):
    base = write_base(tmp_path / "base")
    data = write_data(tmp_path / "data.jsonl")
    before = snapshot(base)
    proc = run_script(tmp_path, *_args(base, data, base))
    assert proc.returncode == REFUSAL_EXIT, (proc.returncode, proc.stderr[-3000:])
    assert refusal_codes(proc.stderr) == ["out_not_empty"]
    assert snapshot(base) == before


def test_first_failing_check_is_reported(tmp_path):
    """Checks run in order (base, data, out, torch); only the first failure
    is reported."""
    base = write_base(tmp_path / "base")
    (base / "evil.py").write_text("x = 1\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    proc = run_script(tmp_path, *_args(base, tmp_path / "missing.jsonl", out))
    assert proc.returncode == REFUSAL_EXIT
    assert refusal_codes(proc.stderr) == ["base_files"]
    (base / "evil.py").unlink()
    proc = run_script(tmp_path, *_args(base, tmp_path / "missing.jsonl", out))
    assert refusal_codes(proc.stderr) == ["data_missing"]
    data = write_data(tmp_path / "data.jsonl")
    proc = run_script(tmp_path, *_args(base, data, out))
    assert refusal_codes(proc.stderr) == ["out_not_empty"]


def test_refusal_line_format(tmp_path):
    base = write_base(tmp_path / "base")
    proc = run_script(tmp_path, *_args(base, tmp_path / "nope.jsonl", tmp_path / "out"))
    lines = [ln for ln in proc.stderr.splitlines() if ln.startswith(REFUSED_PREFIX)]
    assert len(lines) == 1, proc.stderr[-3000:]
    assert lines[0].startswith(f"{REFUSED_PREFIX} data_missing: ")
    assert "nope.jsonl" in lines[0]
