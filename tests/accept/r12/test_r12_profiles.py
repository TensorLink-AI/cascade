"""R12: the shipped host profiles pin the real eval-host history.

* ``profiles/bench_gate.toml``: the lane D candidate's bench gate host
  (valeval-blackwell4 class, §10.4 "drawn suite on valeval-blackwell4").
  targon-eval.toml history: RTX PRO 6000 Blackwell 96 GB is required — the
  Ada fallback differs at 1e-8 and "must not share a gate"; drivers 580.173.02
  and 595.99.02 worked, Crusoe 535 / Warsaw 570 could not bench; the four
  suite identities and 17747 bench files; ≥ 9.2 GB free for two checkpoints
  (OPSLOG 2026-11-22: 5.7 GB free was the failure class).
* ``profiles/launch_eval.toml``: D11's launch eval host, valpriv-4090 (1x RTX
  4090 24 GB, pinned torch 2.4.1+cu124; not Blackwell: sm_120 cannot run the
  pinned torch). It carries no suite pins (it serves forecasts, not the
  bench), so its handshake is the W4 code handshake.

Contract: README.md.
"""

from __future__ import annotations

import tomllib

import pytest

from .helpers import (
    ADA,
    ADA_MIB,
    BENCH_FILES,
    BLACKWELL_MIB,
    EXIT,
    GATE_PROFILE,
    LAUNCH_PROFILE,
    PINNED_IDENTITIES,
    PROFILE_SCHEMA,
    RTX4090,
    RTX4090_MIB,
    TWO_CHECKPOINTS_BYTES,
    Kit,
    as_json,
    output,
    reason_codes,
    require,
)


def _vt(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def _load(path):
    require(path)
    data = tomllib.loads(path.read_text())
    assert data.get("schema") == PROFILE_SCHEMA, data.get("schema")
    for key in ("name", "gpu_model", "min_vram_mib", "min_driver", "disk_path",
                "min_disk_free_bytes"):
        assert key in data, f"{path.name}: missing {key}"
    return data


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def test_gate_profile_pins_the_blackwell_gate_host():
    p = _load(GATE_PROFILE)
    assert "RTX PRO 6000 Blackwell" in p["gpu_model"], p["gpu_model"]
    assert 90_000 <= p["min_vram_mib"] <= BLACKWELL_MIB, p["min_vram_mib"]
    assert (580,) <= _vt(p["min_driver"]) <= (580, 173, 2), p["min_driver"]
    assert TWO_CHECKPOINTS_BYTES <= p["min_disk_free_bytes"] <= 14_800_000_000, (
        p["min_disk_free_bytes"])
    assert p["suite_identities"] == PINNED_IDENTITIES, p.get("suite_identities")
    assert p["bench_files"] == BENCH_FILES, p.get("bench_files")
    assert p["disk_path"].startswith("/"), p["disk_path"]


def test_launch_profile_pins_the_pinned_torch_4090():
    p = _load(LAUNCH_PROFILE)
    assert "RTX 4090" in p["gpu_model"] and "Blackwell" not in p["gpu_model"], p["gpu_model"]
    assert 24_000 <= p["min_vram_mib"] <= RTX4090_MIB, p["min_vram_mib"]
    assert p["min_disk_free_bytes"] >= TWO_CHECKPOINTS_BYTES, p["min_disk_free_bytes"]
    assert not p.get("suite_identities"), "the launch eval host is not a bench gate"


@pytest.mark.parametrize("driver", ["580.173.02", "595.99.02"],
                         ids=["blackwell3", "blackwell4"])
def test_gate_profile_accepts_the_known_good_pods(kit, driver):
    model = _load(GATE_PROFILE)["gpu_model"]
    proc = kit.on_rent(GATE_PROFILE, kit.captures(model=model, vram=BLACKWELL_MIB,
                                                  driver=driver, avail=14_800_000_000))
    assert proc.returncode == 0, output(proc)
    assert as_json(proc)["ok"] is True


@pytest.mark.parametrize(("caps", "first"), [
    ({"model": ADA, "vram": ADA_MIB, "driver": "580.95.05"}, "wrong_gpu"),
    ({"model": RTX4090, "vram": RTX4090_MIB, "driver": "580.95.05"}, "wrong_gpu"),
    ({"driver": "570.195.03"}, "driver_low"),
    ({"driver": "535.230.02"}, "driver_low"),
    ({"avail": 5_700_000_000}, "disk_low"),
], ids=["ada_fallback", "rtx4090", "warsaw_570", "crusoe_535", "disk_5_7gb"])
def test_gate_profile_refuses_the_known_bad_pods(kit, caps, first):
    model = _load(GATE_PROFILE)["gpu_model"]
    caps = {"model": model, "vram": BLACKWELL_MIB, "driver": "595.99.02",
            "avail": 14_800_000_000, **caps}
    proc = kit.on_rent(GATE_PROFILE, kit.captures(**caps))
    assert proc.returncode == EXIT[first], output(proc)
    assert reason_codes(as_json(proc))[0] == first


def test_launch_profile_refuses_a_blackwell_pod(kit):
    gate_model = _load(GATE_PROFILE)["gpu_model"]
    _load(LAUNCH_PROFILE)
    proc = kit.on_rent(LAUNCH_PROFILE, kit.captures(model=gate_model, vram=BLACKWELL_MIB,
                                                    driver="595.99.02"))
    assert proc.returncode == EXIT["wrong_gpu"], (
        f"sm_120 cannot run the pinned torch 2.4.1+cu124 (D11)\n{output(proc)}")
