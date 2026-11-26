"""R5 (§10.8 Rollback: "keep the previous release worktree and restart script
for every service"): the dry-run shows release and previous-release worktree
creation or keep, refuses stale or unknown refs, and never removes one."""

from __future__ import annotations

import pytest

from .helpers import SERVICES, build_box, git, output, plan_lines

ALL = tuple(SERVICES)


@pytest.fixture
def box(tmp_path):
    return build_box(tmp_path)


@pytest.mark.parametrize("svc", ALL)
def test_missing_release_dir_is_planned_as_worktree_add(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    wt = plan_lines(proc.stdout, "worktree")
    want = f"git -C {box.repo} worktree add --detach {box.release_dir} {box.release_ref}"
    assert any(want in ln for ln in wt), f"missing `{want}`:\n" + "\n".join(wt)


@pytest.mark.parametrize("svc", ALL)
def test_existing_previous_release_is_kept(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    wt = plan_lines(proc.stdout, "worktree")
    assert any("keep" in ln and str(box.previous_dir) in ln for ln in wt), "\n".join(wt)
    assert not any(f"worktree add --detach {box.previous_dir}" in ln for ln in wt)


def test_missing_previous_release_is_planned_as_worktree_add(box):
    git(box.repo, "worktree", "remove", "--force", str(box.previous_dir))
    proc = box.run("validator")
    assert proc.returncode == 0, output(proc)
    wt = plan_lines(proc.stdout, "worktree")
    want = f"git -C {box.repo} worktree add --detach {box.previous_dir} {box.previous_ref}"
    assert any(want in ln for ln in wt), f"missing `{want}`:\n" + "\n".join(wt)
    assert not box.previous_dir.exists()


def test_existing_release_dir_at_the_right_commit_is_kept(box):
    git(box.repo, "worktree", "add", "-q", "--detach", str(box.release_dir), box.release_ref)
    proc = box.run("trainer")
    assert proc.returncode == 0, output(proc)
    wt = plan_lines(proc.stdout, "worktree")
    assert any("keep" in ln and str(box.release_dir) in ln for ln in wt), "\n".join(wt)


def test_existing_release_dir_at_a_stale_commit_is_refused(box):
    git(box.repo, "worktree", "add", "-q", "--detach", str(box.release_dir), box.previous_ref)
    proc = box.run("trainer")
    text = output(proc)
    assert proc.returncode != 0, text
    assert "REFUSE" in text and str(box.release_dir) in text, text
    assert plan_lines(proc.stdout, "restart") == [], proc.stdout
    assert box.shim_calls() == []


@pytest.mark.parametrize("flag", ["--release-ref", "--previous-ref"])
def test_unknown_ref_is_refused(box, flag):
    args = box.args("validator")
    args[args.index(flag) + 1] = "rc-does-not-exist"
    proc = box.run("validator", args=args)
    assert proc.returncode != 0, output(proc)
    assert plan_lines(proc.stdout, "restart") == [], proc.stdout
    assert not box.release_dir.exists()


@pytest.mark.parametrize("svc", ALL)
def test_no_worktree_is_ever_removed(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    assert "worktree remove" not in output(proc)
    assert "worktree prune" not in output(proc)


def test_rollback_is_the_same_script_with_the_previous_release(box):
    """A rollback swaps the refs: the target is the kept previous worktree."""
    args = box.args("validator")
    args[args.index("--release-ref") + 1] = box.previous_ref
    args[args.index("--release-dir") + 1] = str(box.previous_dir)
    args[args.index("--previous-ref") + 1] = box.release_ref
    args[args.index("--previous-dir") + 1] = str(box.release_dir)
    proc = box.run("validator", args=args)
    assert proc.returncode == 0, output(proc)
    restart = "\n".join(plan_lines(proc.stdout, "restart"))
    assert f"PYTHONPATH={box.previous_dir}" in restart, restart
    wt = plan_lines(proc.stdout, "worktree")
    assert any("keep" in ln and str(box.previous_dir) in ln for ln in wt), "\n".join(wt)
