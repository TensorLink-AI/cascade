"""W-S1: build rule 4 for the sentinel keys. ``scripts/laned/unread_keys.py``
checks ``[lane_d]`` tables only, so the ``[v2]`` key names are fed to it
under that table in a scratch file: each must be read by
cascade/shared/config."""

from __future__ import annotations

from .helpers import WS1_KEYS, run_unread_keys


def test_ws1_unread_keys_every_sentinel_key_is_read(tmp_path):
    cfg = tmp_path / "v2_sentinel_keys.toml"
    cfg.write_text("[lane_d]\n" + "".join(f"{k} = 0\n" for k in WS1_KEYS), encoding="utf-8")
    proc = run_unread_keys(cfg)
    assert proc.returncode == 0, f"unread [v2] sentinel keys:\n{proc.stdout}{proc.stderr}"
