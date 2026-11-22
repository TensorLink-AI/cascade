"""R12: the on-rent check (plan §10.2 "an on-rent check (nvidia-smi -L, VRAM,
driver, disk) before wiring", §10.5 S4).

``evalhost.py on-rent-check`` parses CAPTURED ``nvidia-smi -L``, the
``--query-gpu`` CSV and ``df`` output against a pinned profile and refuses a
pod that cannot share the gate: a different GPU model (Ada differs from
Blackwell at 1e-8), too little VRAM, a driver below the minimum (Crusoe 535 /
Warsaw 570 could not bench), or less free disk than the two-checkpoint
headroom (the 11-15 and 11-22 disk-full failures). Typed exit code + JSON
reasons. Contract: README.md.
"""

from __future__ import annotations

import pytest

from .helpers import (
    ADA,
    ADA_MIB,
    BLACKWELL,
    BLACKWELL_MIB,
    EXIT,
    ON_RENT_ORDER,
    RTX4090,
    SMI_QUERY,
    TWO_CHECKPOINTS_BYTES,
    Kit,
    as_json,
    output,
    reason_codes,
)


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _check(kit, profile=None, **caps):
    profile = profile or kit.profile()
    proc = kit.on_rent(profile, kit.captures(**caps))
    out = as_json(proc)
    assert kit.shim_calls() == [], f"the check ran an external command: {kit.shim_calls()}"
    return proc, out


def test_on_rent_accepts_the_pinned_gpu(kit):
    proc, out = _check(kit)
    assert proc.returncode == 0, output(proc)
    assert out["ok"] is True, out
    assert reason_codes(out) == []
    obs = out["observed"]
    assert obs["gpu_models"] == [BLACKWELL], obs
    assert obs["vram_mib"] == [BLACKWELL_MIB], obs
    assert obs["driver"] == "595.99.02", obs
    assert obs["disk_free_bytes"] == 14_800_000_000, obs
    req = out["required"]
    assert req["gpu_model"] == BLACKWELL and req["min_vram_mib"] == 97000, req
    assert req["min_driver"] == "580" and req["min_disk_free_bytes"] == TWO_CHECKPOINTS_BYTES, req


@pytest.mark.parametrize(("caps", "want"), [
    # The Ada fallback pod (48 GB): wrong model AND too little VRAM.
    ({"model": ADA, "vram": ADA_MIB}, ["wrong_gpu", "vram_low"]),
    # A near-miss name is still not the pinned model: exact equality.
    ({"model": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"}, ["wrong_gpu"]),
    ({"model": RTX4090, "vram": 24564, "driver": "550.127.05"},
     ["wrong_gpu", "vram_low", "driver_low"]),
    ({"vram": 49152}, ["vram_low"]),
    ({"driver": "570.195.03"}, ["driver_low"]),
    ({"driver": "535.230.02"}, ["driver_low"]),
    ({"avail": 5_700_000_000}, ["disk_low"]),
    ({"driver": "570.195.03", "avail": 5_700_000_000}, ["driver_low", "disk_low"]),
], ids=["ada", "near_miss_name", "rtx4090", "vram", "driver570", "driver535", "disk",
        "driver_and_disk"])
def test_on_rent_refuses_with_typed_exit_and_reasons(kit, caps, want):
    proc, out = _check(kit, **caps)
    assert out["ok"] is False, out
    assert reason_codes(out) == want, out
    assert proc.returncode == EXIT[want[0]], (
        f"exit code must be the first refusal in {ON_RENT_ORDER}: {want[0]}\n{output(proc)}")


def test_on_rent_every_gpu_must_be_the_pinned_model(kit):
    proc, out = _check(kit, models=[BLACKWELL, ADA], vram=[BLACKWELL_MIB, BLACKWELL_MIB])
    assert proc.returncode == EXIT["wrong_gpu"], output(proc)
    assert reason_codes(out) == ["wrong_gpu"], out
    assert out["observed"]["gpu_models"] == [BLACKWELL, ADA], out


def test_on_rent_every_gpu_needs_the_vram(kit):
    proc, out = _check(kit, n=2, vram=[BLACKWELL_MIB, 49152])
    assert proc.returncode == EXIT["vram_low"], output(proc)
    assert reason_codes(out) == ["vram_low"], out


def test_on_rent_list_and_query_must_agree(kit):
    proc, out = _check(kit, models=[BLACKWELL], query_models=[ADA])
    assert proc.returncode == EXIT["wrong_gpu"], output(proc)
    assert "wrong_gpu" in reason_codes(out), out


def test_on_rent_no_gpu_is_refused(kit):
    profile = kit.profile()
    caps = {"L": kit.write("e/L.txt", ""), "query": kit.write("e/q.txt", ""),
            "df": kit.write("e/df.txt", "          Avail Mounted on\n14800000000 /data\n")}
    proc = kit.on_rent(profile, caps)
    assert proc.returncode == EXIT["wrong_gpu"], output(proc)
    out = as_json(proc)
    assert out["ok"] is False and reason_codes(out)[0] == "wrong_gpu", out


def test_on_rent_driver_compares_numerically(kit):
    # "580.173.02" < "580.95" as strings, > as versions.
    profile = kit.profile(min_driver="580.95")
    proc, out = _check(kit, profile=profile, driver="580.173.02")
    assert proc.returncode == 0, output(proc)
    proc, out = _check(kit, profile=profile, driver="580.9.1")
    assert proc.returncode == EXIT["driver_low"], output(proc)
    assert reason_codes(out) == ["driver_low"], out


def test_on_rent_limits_are_inclusive(kit):
    profile = kit.profile(min_vram_mib=BLACKWELL_MIB, min_driver="595.99.02",
                          min_disk_free_bytes=14_800_000_000)
    proc, out = _check(kit, profile=profile)
    assert proc.returncode == 0, output(proc)
    proc, out = _check(kit, profile=profile, avail=14_799_999_999)
    assert proc.returncode == EXIT["disk_low"], output(proc)
    assert reason_codes(out) == ["disk_low"], out


@pytest.mark.parametrize("which", ["query", "df"])
def test_on_rent_unparseable_capture_is_a_usage_error(kit, which):
    profile = kit.profile()
    caps = kit.captures()
    caps[which].write_text("NVIDIA-SMI has failed because it couldn't communicate with the driver\n")
    proc = kit.on_rent(profile, caps)
    assert proc.returncode == EXIT["usage"], (
        f"an unparseable capture is refused as an input error (2), never passed\n{output(proc)}")


def test_on_rent_bad_profile_is_a_usage_error(kit):
    profile = kit.profile(schema="laned_evalhost_profile/0")
    proc = kit.on_rent(profile, kit.captures())
    assert proc.returncode == EXIT["usage"], output(proc)


def test_on_rent_dry_run_prints_the_capture_commands_and_runs_none(kit):
    profile = kit.profile(disk_path="/data")
    proc = kit.tool("on-rent-check", "--profile", profile, "--dry-run")
    assert proc.returncode == 0, output(proc)
    out = as_json(proc)
    assert out.get("dry_run") is True, out
    cmds = out.get("would_run")
    assert isinstance(cmds, list) and all(isinstance(c, str) for c in cmds), out
    assert "nvidia-smi -L" in cmds, cmds
    assert SMI_QUERY in cmds, cmds
    assert "df -B1 --output=avail,target /data" in cmds, cmds
    assert kit.shim_calls() == [], kit.shim_calls()
