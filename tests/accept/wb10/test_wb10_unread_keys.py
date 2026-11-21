"""WB10: build rule 4 for lane B. ``scripts/laned/unread_keys.py`` checks the
keys of a ``[lane_d]`` table only, so the WB10 key names are fed to it under
that table in a scratch config: each must be read by cascade/shared/config."""

from __future__ import annotations

import pytest

from .helpers import SHIPPED, WB10_KEYS, run_unread_keys, shipped_raw


def test_wb10_unread_keys_every_wb10_key_is_read(tmp_path):
    cfg = tmp_path / "lane_b_keys.toml"
    cfg.write_text("[lane_d]\n" + "".join(f"{k} = 0\n" for k in WB10_KEYS), encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode == 0, f"unread [lane_b] keys:\n{proc.stdout}{proc.stderr}"


@pytest.mark.parametrize("name", SHIPPED)
def test_wb10_unread_keys_every_shipped_lane_b_key_is_read(name, tmp_path):
    """Every key of each shipped [lane_b] table (old and WB10) is read."""
    keys = sorted(shipped_raw(name).get("lane_b") or {})
    assert set(WB10_KEYS) <= set(keys), f"{name} [lane_b] lacks {sorted(set(WB10_KEYS) - set(keys))}"
    cfg = tmp_path / "shipped_lane_b_keys.toml"
    cfg.write_text("[lane_d]\n" + "".join(f"{k} = 0\n" for k in keys), encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode == 0, f"{name}: unread [lane_b] keys:\n{proc.stdout}{proc.stderr}"
