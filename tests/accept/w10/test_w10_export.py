"""W10: `cascade lane-d export` writes a lane D entry directory from a trained
cascade_pyramid checkpoint (plan §1.1: exactly the four allowed files, the
repo's pyramid code bytes, BF16 weights). Contract: README.md "export"."""

from __future__ import annotations

import json

import pytest

from .helpers import (
    ALLOWED_FILES,
    TENSORS,
    cli_env,
    digest_of,
    read_header,
    repo_code,
    run_cli,
    safetensors_bytes,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cli_env(monkeypatch, tmp_path)
    return tmp_path


def _trained(d, *, dtype="BF16", junk=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"moe": {"d_model": 16}, "base_arch":
                                               "cascade_pyramid"}))
    tensors = {k: (dtype, shape) for k, (_, shape) in TENSORS.items()}
    (d / "weights.safetensors").write_bytes(safetensors_bytes(tensors))
    if junk:
        (d / "optimizer.pt").write_bytes(b"\x80\x04K\x01.")
        (d / "notes.txt").write_text("training notes")
    return d


def test_export_writes_exactly_the_allowed_files(env, capsys, caplog):
    src, out = _trained(env / "trained"), env / "entry"
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    assert res.rc == 0, res
    assert sorted(p.name for p in out.iterdir()) == sorted(ALLOWED_FILES)


def test_export_carries_the_repo_code_bytes_and_the_bf16_weights(env, capsys, caplog):
    src, out = _trained(env / "trained"), env / "entry"
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    assert res.rc == 0, res
    for name, blob in repo_code().items():
        assert (out / name).read_bytes() == blob, f"{name} is not the repo's pyramid bytes"
    assert (out / "weights.safetensors").read_bytes() == \
        (src / "weights.safetensors").read_bytes(), "BF16 weights must be copied unchanged"
    assert json.loads((out / "config.json").read_text()) == \
        json.loads((src / "config.json").read_text())


def test_export_reports_the_entry_digest(env, capsys, caplog):
    src, out = _trained(env / "trained"), env / "entry"
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    assert res.rc == 0, res
    assert res.data["digest"] == digest_of(out)
    assert sorted(res.data["files"]) == sorted(ALLOWED_FILES)


def test_export_refuses_a_non_empty_out_dir(env, capsys, caplog):
    src, out = _trained(env / "trained"), env / "entry"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    assert res.rc != 0 and res.code, res
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_export_refuses_a_source_without_weights(env, capsys, caplog):
    src, out = _trained(env / "trained"), env / "entry"
    (src / "weights.safetensors").unlink()
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    assert res.rc != 0 and res.code, res
    assert not out.exists() or not any(out.iterdir())


def test_export_refuses_modified_code_in_the_source(env, capsys, caplog):
    """A source that carries its own model.py must carry the repo's bytes:
    lane D is "anything goes" on OUR architecture (§1.1)."""
    src, out = _trained(env / "trained"), env / "entry"
    (src / "model.py").write_bytes(b"# edited\n" + repo_code()["model.py"])
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    assert res.rc == 1 and res.code == "code_mismatch", res
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.torch
def test_export_of_f32_weights_never_writes_a_non_bf16_file(env, capsys, caplog):
    """F32 (lane A's dtype) in: either converted to BF16 or refused with
    nothing written. Lane D is BF16 only (D14)."""
    src, out = _trained(env / "trained", dtype="F32"), env / "entry"
    res = run_cli(capsys, caplog, "lane-d", "export", str(src), "--out", str(out))
    if res.rc == 0:
        header = read_header(out / "weights.safetensors")
        dts = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
        assert dts == {"BF16"}, dts
        assert {k for k in header if k != "__metadata__"} == set(TENSORS)
    else:
        assert res.code, res
        assert not out.exists() or not any(out.iterdir())
