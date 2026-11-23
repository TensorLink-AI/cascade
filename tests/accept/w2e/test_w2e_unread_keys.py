"""W2e: scripts/laned/unread_keys.py passes over the parser for every new key
(build rule 4), and on both shipped files."""

from __future__ import annotations

import pytest

from .helpers import SHIPPED, W2E_KEYS, run_unread_keys


def test_w2e_unread_keys_every_new_key_is_read(tmp_path):
    cfg = tmp_path / "lane_d_w2e_keys.toml"
    cfg.write_text("[lane_d]\n" + "".join(f"{k} = 0\n" for k in W2E_KEYS), encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode == 0, f"unread [lane_d] keys:\n{proc.stdout}{proc.stderr}"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2e_unread_keys_passes_on_the_shipped_file(name):
    proc = run_unread_keys(name)
    assert proc.returncode == 0, f"{name}: unread [lane_d] keys:\n{proc.stdout}{proc.stderr}"
