"""W0c: scripts/laned/unread_keys.py fails when a [lane_d]/[lane_d_ops] key is
read by no parser (build rule 4) and passes when every key is read."""

from __future__ import annotations

import re

from .helpers import FIXTURES, copy_fixture, output, run_tool

FX = FIXTURES / "unread_keys"


def _repo(tmp_path, parser: str):
    root = tmp_path / "repo"
    copy_fixture(FX / "chain.toml", root, "chain.toml")
    copy_fixture(FX / f"{parser}.py.txt", root, "lanedpkg/config/lane_d.py")
    return root


def _run(root):
    return run_tool("unread_keys.py", "--root", root, "--config", "chain.toml", "--source", "lanedpkg")


def test_unread_keys_passes_when_every_key_is_read(tmp_path):
    proc = _run(_repo(tmp_path, "parser_clean"))
    assert proc.returncode == 0, output(proc)


def test_unread_keys_ignores_other_tables(tmp_path):
    # [lane_a] unrelated_knob is read by no fixture parser and is not reported.
    proc = _run(_repo(tmp_path, "parser_clean"))
    assert "unrelated_knob" not in output(proc)


def test_unread_keys_mentions_in_allowed_docstring_or_comment_do_not_count(tmp_path):
    proc = _run(_repo(tmp_path, "parser_missing_remainder"))
    assert proc.returncode == 1, output(proc)
    assert re.search(r"^unread lane_d\.remainder$", proc.stdout, re.M), proc.stdout
    assert "lane_d.share" not in proc.stdout and "lane_d.from_block" not in proc.stdout


def test_unread_keys_covers_lane_d_ops(tmp_path):
    proc = _run(_repo(tmp_path, "parser_missing_ops"))
    assert proc.returncode == 1, output(proc)
    assert re.search(r"^unread lane_d_ops\.poll_seconds$", proc.stdout, re.M), proc.stdout
