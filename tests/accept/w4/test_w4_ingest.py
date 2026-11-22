"""W4 ingest guards without torch: the T5 malformed set refused with exact
``IngestCode`` values on both transports (the storage adapter and presigned
GETs), the exact-filename allowlist and the size budget enforced before any
object is fetched, and invariant 3 (no miner bytes execute; only repo code
imports) — including a fresh interpreter that never loads torch."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest

from .helpers import (
    MANIFEST,
    REPO,
    T5_CASES,
    VaultServer,
    build_case,
    counting_storage,
    dir_is_clean,
    expectation_kwargs,
    foreign_modules,
    planted_python,
    write_good_checkpoint,
)

EXACT_CODES = {
    "DIGEST_MISMATCH": "digest_mismatch",
    "FILE_NOT_ALLOWED": "file_not_allowed",
    "STRAY_OBJECT": "stray_object",
    "MISSING_OBJECT": "missing_object",
    "SIZE_MISMATCH": "size_mismatch",
    "MISSING_MANIFEST": "missing_manifest",
    "BAD_MANIFEST": "bad_manifest",
    "BAD_IDENTITY": "bad_identity",
    "OVERSIZE": "oversize",
    "CODE_MISMATCH": "code_mismatch",
    "CONFIG_MISMATCH": "config_mismatch",
    "BAD_HEADER": "bad_header",
    "DTYPE_MISMATCH": "dtype_mismatch",
    "SHAPE_MISMATCH": "shape_mismatch",
    "TENSOR_SET_MISMATCH": "tensor_set_mismatch",
    "NON_FINITE": "non_finite",
    "FETCH_FAILED": "fetch_failed",
}


@pytest.fixture
def vault(monkeypatch):
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
                 "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = VaultServer()
    yield server
    server.close()


def _refusal_logged(caplog, code: str) -> bool:
    return any(r.levelno >= logging.WARNING and r.getMessage().startswith("lane_d_ingest_refused")
               and code in r.getMessage() for r in caplog.records)


def test_ingest_code_enum_is_exact():
    from cascade.eval.lane_d_ingest import INGEST_CODES, IngestCode, IngestError

    for member, value in EXACT_CODES.items():
        assert getattr(IngestCode, member).value == value, member
        assert IngestCode(value) is getattr(IngestCode, member)
        assert IngestCode(value) == value          # a str enum: compares as its value
    assert set(INGEST_CODES) == {c.value for c in IngestCode}
    err = IngestError(IngestCode.OVERSIZE, "weights.safetensors over budget")
    assert err.code is IngestCode.OVERSIZE and isinstance(err, Exception)


@pytest.mark.parametrize("case", sorted(T5_CASES))
def test_t5_malformed_refused_with_exact_code_storage(case, tmp_path, caplog):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_entry  # noqa: I001
    from cascade.lane_d.intake import upload_checkpoint

    want, policy = T5_CASES[case]
    caplog.set_level(logging.INFO)
    expect = LaneDExpectation(**expectation_kwargs())
    ckpt, marker = build_case(case, tmp_path / "src")
    store = counting_storage(tmp_path / "vault")
    hotkey = f"w4-hotkey-{case}"
    digest = upload_checkpoint(store, hotkey, ckpt)
    store.gets.clear()
    dest = tmp_path / "ingest"
    entry = {"hotkey": hotkey, "digest": digest}
    with pytest.raises(IngestError) as e:
        ingest_entry(store, entry, dest, contract=None, expect=expect)
    assert isinstance(e.value.code, IngestCode)
    assert e.value.code == IngestCode(want), f"{case}: got {e.value.code!r}, want {want!r}"
    prefix = f"lane-d/{hotkey}/{digest}/"
    if policy == "manifest_only":
        assert set(store.gets) <= {prefix + MANIFEST}, store.gets
    if policy == "no_weights":
        assert prefix + "weights.safetensors" not in store.gets, store.gets
    assert not marker.exists(), f"{case}: miner bytes executed"
    assert foreign_modules(tmp_path) == []
    assert dir_is_clean(dest), sorted(p.name for p in dest.iterdir())
    assert _refusal_logged(caplog, want), [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("case", sorted(T5_CASES))
def test_t5_malformed_refused_with_exact_code_presigned(case, tmp_path, vault, caplog):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_presigned  # noqa: I001

    want, policy = T5_CASES[case]
    caplog.set_level(logging.INFO)
    expect = LaneDExpectation(**expectation_kwargs())
    ckpt, marker = build_case(case, tmp_path / "src")
    grant = vault.publish(f"w4-hotkey-{case}", ckpt)
    dest = tmp_path / "ingest"
    with pytest.raises(IngestError) as e:
        ingest_presigned(grant, dest, contract=None, expect=expect)
    assert e.value.code == IngestCode(want), f"{case}: got {e.value.code!r}, want {want!r}"
    prefix = f"lane-d/{grant['hotkey']}/{grant['digest']}/"
    got = vault.got()
    if policy == "manifest_only":
        assert got == [prefix + MANIFEST], got
    if policy == "no_weights":
        assert prefix + "weights.safetensors" not in got, got
    assert not marker.exists(), f"{case}: miner bytes executed"
    assert foreign_modules(tmp_path) == []
    assert dir_is_clean(dest)
    assert _refusal_logged(caplog, want)


def test_storage_stray_object_and_overwrite_refused(tmp_path):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_entry  # noqa: I001
    from cascade.lane_d.intake import upload_checkpoint

    expect = LaneDExpectation(**expectation_kwargs())
    ckpt = write_good_checkpoint(tmp_path / "src")
    store = counting_storage(tmp_path / "vault")
    digest = upload_checkpoint(store, "w4-hk", ckpt)
    prefix = f"lane-d/w4-hk/{digest}/"
    marker = tmp_path / "markers" / "stray"
    store.put(prefix + "sitecustomize.py", planted_python(marker))
    with pytest.raises(IngestError) as e:
        ingest_entry(store, {"hotkey": "w4-hk", "digest": digest}, tmp_path / "a",
                     contract=None, expect=expect)
    assert e.value.code == IngestCode.STRAY_OBJECT
    assert not marker.exists() and dir_is_clean(tmp_path / "a")

    # Content addressing: an object overwritten after ready (same size) fails
    # the re-hash, the miner's fault.
    digest2 = upload_checkpoint(store, "w4-hk2", ckpt)
    key = f"lane-d/w4-hk2/{digest2}/weights.safetensors"
    data = bytearray(store.get(key))
    data[-1] ^= 0x01
    store.put(key, bytes(data))
    with pytest.raises(IngestError) as e:
        ingest_entry(store, {"hotkey": "w4-hk2", "digest": digest2}, tmp_path / "b",
                     contract=None, expect=expect)
    assert e.value.code == IngestCode.DIGEST_MISMATCH
    assert dir_is_clean(tmp_path / "b")


def test_presigned_overwritten_object_refused(tmp_path, vault):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_presigned  # noqa: I001

    expect = LaneDExpectation(**expectation_kwargs())
    ckpt = write_good_checkpoint(tmp_path / "src")
    grant = vault.publish("w4-hk", ckpt)
    key = f"lane-d/w4-hk/{grant['digest']}/weights.safetensors"
    same = bytearray(vault.objects[key])
    same[-1] ^= 0x01
    vault.objects[key] = bytes(same)                  # same size, other bytes
    with pytest.raises(IngestError) as e:
        ingest_presigned(grant, tmp_path / "a", contract=None, expect=expect)
    assert e.value.code == IngestCode.DIGEST_MISMATCH
    assert dir_is_clean(tmp_path / "a")
    vault.objects[key] = bytes(same) + b"\0" * 4096   # larger than the manifest says
    with pytest.raises(IngestError) as e:
        ingest_presigned(grant, tmp_path / "b", contract=None, expect=expect)
    assert e.value.code == IngestCode.OVERSIZE
    assert dir_is_clean(tmp_path / "b")


def test_presigned_manifest_must_hash_to_digest(tmp_path, vault):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_presigned  # noqa: I001

    expect = LaneDExpectation(**expectation_kwargs())
    grant = vault.publish("w4-hk", write_good_checkpoint(tmp_path / "src"))
    key = f"lane-d/w4-hk/{grant['digest']}/{MANIFEST}"
    manifest = json.loads(vault.objects[key])
    manifest["files"]["evil.py"] = {"sha256": manifest["files"]["model.py"]["sha256"], "size": 1}
    vault.objects[key] = json.dumps(manifest, sort_keys=True).encode()
    with pytest.raises(IngestError) as e:
        ingest_presigned(grant, tmp_path / "a", contract=None, expect=expect)
    assert e.value.code == IngestCode.DIGEST_MISMATCH
    assert vault.got() == [key]
    assert dir_is_clean(tmp_path / "a")


def test_presigned_forged_url_fails_closed(tmp_path, vault):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_presigned  # noqa: I001

    expect = LaneDExpectation(**expectation_kwargs())
    grant = vault.publish("w4-hk", write_good_checkpoint(tmp_path / "src"))
    key = f"lane-d/w4-hk/{grant['digest']}/weights.safetensors"
    grant["files"]["weights.safetensors"] = vault.url(key, valid=False)
    with pytest.raises(IngestError) as e:
        ingest_presigned(grant, tmp_path / "a", contract=None, expect=expect)
    assert e.value.code == IngestCode.FETCH_FAILED
    assert dir_is_clean(tmp_path / "a")


_SUBPROCESS = r"""
import json, sys
from pathlib import Path
from cascade.eval.lane_d_ingest import IngestError, LaneDExpectation, ingest_entry
from cascade.lane_d.storage import FilesystemStorage

args = json.loads(sys.argv[1])
exp = args["expect"]
expect = LaneDExpectation(
    code={k: Path(v).read_bytes() for k, v in exp["code"].items()},
    config=exp["config"],
    tensors={k: (d, tuple(s)) for k, (d, s) in exp["tensors"].items()},
    max_weights_bytes=int(exp["max_weights_bytes"]))
store = FilesystemStorage(args["vault"])
codes = {}
for case, entry in sorted(args["entries"].items()):
    try:
        ingest_entry(store, entry, Path(args["dest"]) / case, contract=None, expect=expect)
        codes[case] = "accepted"
    except IngestError as e:
        codes[case] = str(e.code.value)
base = str(Path(args["tmp"]).resolve())
foreign = sorted(n for n, m in list(sys.modules.items())
                 if str(Path(getattr(m, "__file__", None) or "/").resolve()).startswith(base))
print(json.dumps({"codes": codes, "torch": "torch" in sys.modules, "foreign": foreign}))
"""


def test_inv3_malformed_set_refused_in_fresh_interpreter_without_torch(tmp_path):
    """Invariant 3 end to end in a fresh interpreter: every T5 case is
    refused with its exact code, no marker appears, no module is imported
    from the miner's bytes, and torch is never loaded (the guards before the
    finite check are torch-free, so the 7 GB box can run them)."""
    from cascade.eval.lane_d_ingest import IngestCode, LaneDExpectation  # noqa: F401, I001
    from cascade.lane_d.intake import upload_checkpoint
    from cascade.lane_d.storage import FilesystemStorage

    store = FilesystemStorage(tmp_path / "vault")
    entries, markers = {}, []
    for case in sorted(T5_CASES):
        ckpt, marker = build_case(case, tmp_path / "src" / case)
        markers.append(marker)
        hotkey = f"w4-hotkey-{case}"
        entries[case] = {"hotkey": hotkey, "digest": upload_checkpoint(store, hotkey, ckpt)}
    kw = expectation_kwargs()
    code_paths = {"model.py": str(REPO / "cascade" / "trainer" / "pyramid_model.py"),
                  "forecast_wrapper.py": str(REPO / "cascade" / "trainer" / "pyramid_wrapper.py")}
    args = {"vault": str(tmp_path / "vault"), "dest": str(tmp_path / "ingest"),
            "tmp": str(tmp_path), "entries": entries,
            "expect": {"code": code_paths, "config": kw["config"],
                       "tensors": {k: [d, list(s)] for k, (d, s) in kw["tensors"].items()},
                       "max_weights_bytes": kw["max_weights_bytes"]}}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run([sys.executable, "-P", "-c", _SUBPROCESS, json.dumps(args)],
                          cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["codes"] == {case: T5_CASES[case][0] for case in T5_CASES}
    assert out["torch"] is False, "the pre-finite ingest guards imported torch"
    assert out["foreign"] == []
    assert [m for m in markers if m.exists()] == []
    assert dir_is_clean(tmp_path / "ingest")
