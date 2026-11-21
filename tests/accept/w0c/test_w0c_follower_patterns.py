"""W0c: scripts/laned/follower_patterns.py fails when a lane D log call has no
cascade-ops follower pattern (build rule 5) and passes when each one does."""

from __future__ import annotations

import re

from .helpers import FIXTURES, copy_fixture, output, run_tool

FX = FIXTURES / "follower_patterns"


def _repo(tmp_path, lane_d_module: str, patterns: str, extra_lane_d: str | None = None):
    root = tmp_path / "repo"
    copy_fixture(FX / f"{lane_d_module}.py.txt", root, "lanedpkg/lane_d/crown.py")
    copy_fixture(FX / "other_module.py.txt", root, "lanedpkg/validator/poll.py")
    if extra_lane_d:
        copy_fixture(FX / f"{extra_lane_d}.py.txt", root, "lanedpkg/validator/lane_d_mixin.py")
    copy_fixture(FX / f"{patterns}.py.txt", root, "ops/cascade_ops/events.py")
    return root


def _run(root):
    return run_tool("follower_patterns.py", "--root", root, "--source", "lanedpkg",
                    "--patterns", "ops/cascade_ops/events.py")


def test_follower_patterns_pass_when_every_lane_d_log_call_is_followed(tmp_path):
    proc = _run(_repo(tmp_path, "lane_d_clean", "patterns_clean"))
    assert proc.returncode == 0, output(proc)


def test_follower_patterns_fail_on_uncovered_tag(tmp_path):
    # lane_d_hold appears only in a comment of the patterns file: not a pattern.
    proc = _run(_repo(tmp_path, "lane_d_clean", "patterns_missing_hold"))
    assert proc.returncode == 1, output(proc)
    assert re.search(r"^uncovered lanedpkg/lane_d/crown\.py:\d+ lane_d_hold$", proc.stdout, re.M), proc.stdout
    assert "lane_d_crowned" not in proc.stdout, proc.stdout


def test_follower_patterns_fail_on_untagged_lane_d_message(tmp_path):
    root = _repo(tmp_path, "lane_d_clean", "patterns_clean", extra_lane_d="lane_d_untagged")
    proc = _run(root)
    assert proc.returncode == 1, output(proc)
    assert re.search(r"^untagged lanedpkg/validator/lane_d_mixin\.py:\d+", proc.stdout, re.M), proc.stdout


def test_follower_patterns_fail_on_non_literal_lane_d_message(tmp_path):
    root = _repo(tmp_path, "lane_d_clean", "patterns_clean", extra_lane_d="lane_d_nonliteral")
    proc = _run(root)
    assert proc.returncode == 1, output(proc)
    assert re.search(r"^untagged lanedpkg/validator/lane_d_mixin\.py:\d+", proc.stdout, re.M), proc.stdout
