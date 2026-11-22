"""W2b: scripts/laned/unread_keys.py passes for ``[lane_d_ops]`` (build
rule 4): every key is read by a parser under cascade/shared/config."""

from __future__ import annotations

import pytest

from .helpers import OPS_KEYS, SHIPPED, run_unread_keys, shipped_raw


def test_w2b_unread_keys_every_ops_key_is_read(tmp_path):
    cfg = tmp_path / "lane_d_ops_keys.toml"
    cfg.write_text("[lane_d_ops]\n" + "".join(f"{k} = 0\n" for k in OPS_KEYS), encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode == 0, f"unread [lane_d_ops] keys:\n{proc.stdout}{proc.stderr}"


def test_w2b_unread_keys_catches_an_unread_ops_key(tmp_path):
    """The tool really checks this table (a key no parser reads fails)."""
    cfg = tmp_path / "lane_d_ops_unread.toml"
    cfg.write_text("[lane_d_ops]\nw2b_never_read_key = 0\n", encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode != 0
    assert "unread lane_d_ops.w2b_never_read_key" in proc.stdout, proc.stdout


@pytest.mark.parametrize("name", SHIPPED)
def test_w2b_unread_keys_passes_on_the_shipped_file(name):
    assert "lane_d_ops" in shipped_raw(name), f"{name} has no [lane_d_ops] table"
    proc = run_unread_keys(name)
    assert proc.returncode == 0, f"{name}: unread keys:\n{proc.stdout}{proc.stderr}"
