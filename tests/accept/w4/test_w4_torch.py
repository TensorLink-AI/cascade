"""W4 checks that need torch (marked ``torch``: CI only, never the 7 GB box):
the positive control for the T5 fixture, NaN/Inf refused by
``verify_weights_finite`` with the exact code, the lane D expectation derived
from the contract, and the invariant 1 pair for lane A's F32 guard."""

from __future__ import annotations

import logging

import pytest

from .helpers import (
    ALLOWED,
    T5_TORCH_CASES,
    VaultServer,
    build_case,
    counting_storage,
    dir_is_clean,
    expectation_kwargs,
    foreign_modules,
    pyramid_contract,
    read_header,
    tiny_checkpoint,
    write_good_checkpoint,
)

pytestmark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)


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


@pytest.fixture
def two_threads():
    """Keeps torch to 2 threads; imports torch only when a test asks."""
    state = {}
    yield state
    if "torch" in state:
        state["torch"].set_num_threads(state["threads"])


def _limit(state):
    import torch

    state.setdefault("threads", torch.get_num_threads())
    state["torch"] = torch
    torch.set_num_threads(2)


def _files(directory) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


def test_t5_fixture_positive_control_ingests_on_both_transports(tmp_path, vault):
    from cascade.eval.lane_d_ingest import LaneDExpectation, ingest_entry, ingest_presigned  # noqa: I001
    from cascade.lane_d.intake import upload_checkpoint

    expect = LaneDExpectation(**expectation_kwargs())
    src = write_good_checkpoint(tmp_path / "src")
    store = counting_storage(tmp_path / "vault")
    digest = upload_checkpoint(store, "w4-hk", src)
    out = ingest_entry(store, {"hotkey": "w4-hk", "digest": digest}, tmp_path / "a",
                       contract=None, expect=expect)
    assert out == tmp_path / "a" / digest
    assert tuple(sorted(_files(out))) == ALLOWED and _files(out) == _files(src)
    grant = vault.publish("w4-hk", src)
    out2 = ingest_presigned(grant, tmp_path / "b", contract=None, expect=expect)
    assert out2 == tmp_path / "b" / grant["digest"] and _files(out2) == _files(src)
    assert foreign_modules(tmp_path) == []


@pytest.mark.parametrize("case", sorted(T5_TORCH_CASES))
def test_t5_non_finite_refused_with_exact_code(case, tmp_path, vault, caplog):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, LaneDExpectation, ingest_entry, ingest_presigned  # noqa: I001
    from cascade.lane_d.intake import upload_checkpoint

    caplog.set_level(logging.INFO)
    want = T5_TORCH_CASES[case][0]
    expect = LaneDExpectation(**expectation_kwargs())
    ckpt, marker = build_case(case, tmp_path / "src")
    store = counting_storage(tmp_path / "vault")
    digest = upload_checkpoint(store, "w4-hk", ckpt)
    with pytest.raises(IngestError) as e:
        ingest_entry(store, {"hotkey": "w4-hk", "digest": digest}, tmp_path / "a",
                     contract=None, expect=expect)
    assert e.value.code == IngestCode(want)
    with pytest.raises(IngestError) as e:
        ingest_presigned(vault.publish("w4-hk", ckpt), tmp_path / "b", contract=None,
                         expect=expect)
    assert e.value.code == IngestCode(want)
    assert dir_is_clean(tmp_path / "a") and dir_is_clean(tmp_path / "b")
    assert not marker.exists()


def test_lane_d_expectation_is_lane_a_spec_with_bf16(tmp_path, two_threads):
    from cascade.eval.lane_d_ingest import expectation_for  # noqa: I001
    from cascade.eval.checkpoint_guard import expected_checkpoint_code
    import json

    _limit(two_threads)
    contract = pyramid_contract()
    f32 = tiny_checkpoint(tmp_path / "f32", contract, seed=3, bf16=False)
    bf16 = tiny_checkpoint(tmp_path / "bf16", contract, seed=3, bf16=True)
    exp = expectation_for(contract)
    assert dict(exp.code) == expected_checkpoint_code("cascade_pyramid")
    assert dict(exp.config) == json.loads((f32 / "config.json").read_text())["moe"]
    header = read_header(f32 / "weights.safetensors")
    assert set(exp.tensors) == set(header)
    for name, (dtype, shape) in exp.tensors.items():
        assert tuple(shape) == tuple(header[name]["shape"]), name
        assert dtype == ("BF16" if header[name]["dtype"] == "F32" else header[name]["dtype"]), name
    # Budget: at least the F32 file (so lane A's dtype is refused as
    # dtype_mismatch, the exact T5 code) and at most lane A's own F32 budget.
    f32_size = (f32 / "weights.safetensors").stat().st_size
    assert (bf16 / "weights.safetensors").stat().st_size <= exp.max_weights_bytes
    assert f32_size <= exp.max_weights_bytes <= 3 * f32_size


def test_real_tiny_checkpoint_ingest_and_non_finite(tmp_path, vault, two_threads):
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, ingest_entry, ingest_presigned  # noqa: I001
    from cascade.lane_d.intake import upload_checkpoint

    _limit(two_threads)
    contract = pyramid_contract()
    good = tiny_checkpoint(tmp_path / "good", contract, seed=5, bf16=True)
    store = counting_storage(tmp_path / "vault")
    digest = upload_checkpoint(store, "w4-hk", good)
    out = ingest_entry(store, {"hotkey": "w4-hk", "digest": digest}, tmp_path / "a",
                       contract=contract)
    assert _files(out) == _files(good)
    out2 = ingest_presigned(vault.publish("w4-hk", good), tmp_path / "b", contract=contract)
    assert _files(out2) == _files(good)
    for i, value in enumerate((float("nan"), float("inf"))):
        bad = tiny_checkpoint(tmp_path / f"bad{i}", contract, seed=5, bf16=True, poison=value)
        hk = f"w4-bad{i}"
        d = upload_checkpoint(store, hk, bad)
        with pytest.raises(IngestError) as e:
            ingest_entry(store, {"hotkey": hk, "digest": d}, tmp_path / f"c{i}", contract=contract)
        assert e.value.code == IngestCode.NON_FINITE
        assert dir_is_clean(tmp_path / f"c{i}")


def test_inv1_lane_a_f32_guard_unchanged_when_unarmed(tmp_path, two_threads):
    """Lane A's guard is untouched by W4: an F32 pyramid checkpoint verifies
    with the same report and its bytes stay identical; a BF16 one is still
    refused by lane A (BF16 is a lane D-only path)."""
    from cascade.eval.checkpoint_guard import CheckpointTampered, verify_checkpoint  # noqa: I001

    _limit(two_threads)
    contract = pyramid_contract()
    f32 = tiny_checkpoint(tmp_path / "f32", contract, seed=7, bf16=False)
    before = _files(f32)
    header = read_header(f32 / "weights.safetensors")
    r1 = verify_checkpoint(f32, contract)
    r2 = verify_checkpoint(f32, contract)
    assert r1 == r2
    assert r1.n_tensors == len(header) and r1.weights_bytes == len(before["weights.safetensors"])
    assert r1.n_params > 0
    assert _files(f32) == before
    assert {m["dtype"] for m in header.values()} == {"F32"}
    bf16 = tiny_checkpoint(tmp_path / "bf16", contract, seed=7, bf16=True)
    with pytest.raises(CheckpointTampered):
        verify_checkpoint(bf16, contract)


def test_inv1_lane_a_f32_guard_armed_nonempty_lane_d_effect(tmp_path, two_threads):
    """The lane D path is the non-empty effect: it accepts the BF16 entry
    lane A refuses and refuses lane A's F32 dtype with ``dtype_mismatch``."""
    from cascade.eval.lane_d_ingest import IngestCode, IngestError, ingest_entry  # noqa: I001
    from cascade.lane_d.intake import upload_checkpoint

    _limit(two_threads)
    contract = pyramid_contract()
    store = counting_storage(tmp_path / "vault")
    bf16 = tiny_checkpoint(tmp_path / "bf16", contract, seed=7, bf16=True)
    d = upload_checkpoint(store, "w4-bf16", bf16)
    out = ingest_entry(store, {"hotkey": "w4-bf16", "digest": d}, tmp_path / "a",
                       contract=contract)
    assert _files(out) == _files(bf16)
    f32 = tiny_checkpoint(tmp_path / "f32", contract, seed=7, bf16=False)
    d2 = upload_checkpoint(store, "w4-f32", f32)
    with pytest.raises(IngestError) as e:
        ingest_entry(store, {"hotkey": "w4-f32", "digest": d2}, tmp_path / "b",
                     contract=contract)
    assert e.value.code == IngestCode.DTYPE_MISMATCH
    assert dir_is_clean(tmp_path / "b")
