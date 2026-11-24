"""W2: scripts/laned/unread_keys.py passes over the new parser — every
[lane_d] W2 key is read by cascade/shared/config (build rule 4). The retired
keys (rev6.18 B11) are read by nothing and the shipped files carry none."""

from __future__ import annotations

import pytest

from .helpers import SHIPPED, W2_KEYS, run_unread_keys


def test_w2_unread_keys_every_w2_key_is_read(tmp_path):
    cfg = tmp_path / "lane_d_keys.toml"
    cfg.write_text("[lane_d]\n" + "".join(f"{k} = 0\n" for k in W2_KEYS),
                   encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode == 0, f"unread [lane_d] keys:\n{proc.stdout}{proc.stderr}"


@pytest.mark.parametrize("name", SHIPPED)
def test_w2_unread_keys_passes_on_the_shipped_file(name):
    proc = run_unread_keys(name)
    assert proc.returncode == 0, f"{name}: unread [lane_d] keys:\n{proc.stdout}{proc.stderr}"
