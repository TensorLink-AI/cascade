"""W1.5 skeleton pieces that need no torch: ``make laned-e2e``, the storage
interface and its filesystem adapter, intake through storage, lane D's
emission part, and ``LaneDConfig.active()`` not being a constant."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import shutil
import subprocess

import pytest

from .helpers import (
    ARM_BLOCK,
    BOUNDARY_1,
    BOUNDARY_2,
    BURN_UID,
    CHALLENGER_UID,
    CONTEXT,
    ERA,
    LANE_A_COURT,
    LANE_A_KING_DECAY,
    N_UIDS,
    READY_BLOCK,
    REPO,
    SETTLEMENT,
    fixture_hotkey,
    lane_d_cfg,
)


def test_makefile_defines_laned_e2e():
    mf = REPO / "Makefile"
    assert mf.is_file(), "no top-level Makefile (W1.5 adds it with target laned-e2e)"
    assert re.search(r"^laned-e2e\s*:", mf.read_text(), re.M), "Makefile has no laned-e2e target"
    assert shutil.which("make"), "make is not installed"
    out = subprocess.run(["make", "-n", "laned-e2e"], cwd=str(REPO), capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "uv run --frozen" in out.stdout, out.stdout
    assert "pytest" in out.stdout and "tests/accept/w1_5" in out.stdout, out.stdout


def test_storage_interface_and_filesystem_adapter(tmp_path):
    from cascade.lane_d.storage import FilesystemStorage, LaneDStorage

    assert inspect.isabstract(LaneDStorage)
    assert {"put", "get", "list", "size"} <= set(LaneDStorage.__abstractmethods__)
    store = FilesystemStorage(tmp_path / "vault")
    assert isinstance(store, LaneDStorage)
    store.put("lane-d/hk/dg/a.bin", b"abc")
    store.put("lane-d/hk/dg/b.bin", b"de")
    store.put("lane-d/other/dg/x.bin", b"z")
    assert store.get("lane-d/hk/dg/a.bin") == b"abc"
    assert store.size("lane-d/hk/dg/b.bin") == 2
    assert store.list("lane-d/hk/dg/") == ["lane-d/hk/dg/a.bin", "lane-d/hk/dg/b.bin"]
    with pytest.raises(KeyError):
        store.get("lane-d/hk/dg/missing.bin")


def test_intake_ready_reads_through_storage(tmp_path):
    from cascade.lane_d.intake import checkpoint_manifest, manifest_digest, ready, upload_checkpoint
    from cascade.lane_d.storage import FilesystemStorage

    hotkey, coldkey = fixture_hotkey("intake-hotkey"), fixture_hotkey("intake-coldkey")
    src = tmp_path / "ckpt"
    src.mkdir()
    files = {"config.json": b'{"moe": {}}', "model.py": b"# model\n",
             "forecast_wrapper.py": b"# wrapper\n", "weights.safetensors": os.urandom(4096)}
    for name, data in files.items():
        (src / name).write_bytes(data)
    manifest = checkpoint_manifest(src)
    assert set(manifest["files"]) == set(files)
    for name, data in files.items():
        assert manifest["files"][name]["sha256"] == hashlib.sha256(data).hexdigest()
        assert manifest["files"][name]["size"] == len(data)
    store = FilesystemStorage(tmp_path / "vault")
    digest = upload_checkpoint(store, hotkey, src)
    assert digest == manifest_digest(manifest)
    prefix = f"lane-d/{hotkey}/{digest}/"
    assert {prefix + name for name in files} <= set(store.list(prefix))
    shutil.rmtree(src)                   # ready() sees only the storage
    rec = ready(store, hotkey=hotkey, digest=digest, context=CONTEXT, coldkey=coldkey,
                ready_block=READY_BLOCK, cfg=lane_d_cfg())
    assert rec["hotkey"] == hotkey and rec["digest"] == digest
    assert rec["context"] == CONTEXT and rec["ready_block"] == READY_BLOCK
    assert rec["coldkey_at_ready"] == coldkey


# rev6.23 (owner 2026-11-24): lane D's court weights (KING_D, prior 1, prior 2).
CW = (0.80, 0.15, 0.05)


def test_lane_d_emission_part_taken_from_lane_a():
    from cascade.shared.emission import lane_d_vector, lane_parts

    king = lane_d_vector(N_UIDS, share=0.25, court_uids=[CHALLENGER_UID], court_weights=CW,
                         burn_uid=BURN_UID)
    assert math.isclose(king[CHALLENGER_UID], 0.25) and math.isclose(sum(king), 0.25)
    # D21 (decided: burn): while the house holds KING_D its slot burns.
    house = lane_d_vector(N_UIDS, share=0.25, court_uids=[None], court_weights=CW,
                          burn_uid=BURN_UID)
    assert math.isclose(house[BURN_UID], 0.25) and math.isclose(sum(house), 0.25)
    assert lane_d_vector(N_UIDS, share=0.0, court_uids=[CHALLENGER_UID], court_weights=CW,
                         burn_uid=BURN_UID) == [0.0] * N_UIDS
    # D1/D48: lane D's 0.25 comes out of lane A's pool.
    a, b = lane_parts(list(LANE_A_COURT), N_UIDS, king_decay=LANE_A_KING_DECAY,
                      burn_uid=BURN_UID, lane_d_cfg=lane_d_cfg(),
                      d_court_uids=(CHALLENGER_UID,))
    assert math.isclose(sum(a), 0.75) and a[CHALLENGER_UID] == 0.0
    assert math.isclose(b[CHALLENGER_UID], 0.25) and math.isclose(sum(a) + sum(b), 1.0)


def test_lane_d_config_active_is_not_a_constant():
    armed, unarmed = lane_d_cfg(), lane_d_cfg(from_block=0)
    bounded = lane_d_cfg(until_block=ARM_BLOCK + 2 * ERA)
    probes = (ARM_BLOCK - SETTLEMENT, ARM_BLOCK, BOUNDARY_1, BOUNDARY_2)
    assert [armed.active(b) for b in probes] == [False, False, True, True]
    assert [unarmed.active(b) for b in probes] == [False] * len(probes)
    assert bounded.active(BOUNDARY_1) and not bounded.active(ARM_BLOCK + 3 * ERA)
