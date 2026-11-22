"""R5 (plan §5 row 10 "Snapshot all state (§10.8)", §10.8 Rollback; RV2
BLOCKING R5): the PLAN snapshot lines copy every state file byte-identically
with ``cp -a`` into ``<root>/_state_backups/<ts>/``, report every absent fixed
name, and never use ``sqlite3`` (``validator_state.db`` is JSON text on the
deployment, and the box has no sqlite3 binary).

The copy is checked by replaying the printed snapshot lines (each line's text
is the argv it runs) against the fixture root only, after the dry-run."""

from __future__ import annotations

import json
import re
import shlex

import pytest

from .helpers import (
    ABSENT_STATE,
    LANED,
    NOT_STATE,
    OPTIONAL_STATE,
    PRESENT_STATE,
    QUIET_TS,
    SERVICES,
    SNAPSHOT_FIXED,
    build_box,
    output,
    plan_lines,
)

ALL = tuple(SERVICES)
SQLITE = re.compile(r"\bsqlite3\b")


@pytest.fixture
def box(tmp_path):
    return build_box(tmp_path)


def snapshot_lines(box, svc, path=None):
    proc = box.run(svc, path=path)
    assert proc.returncode == 0, output(proc)
    lines = plan_lines(proc.stdout, "snapshot")
    assert lines, f"no `PLAN snapshot` lines:\n{proc.stdout}"
    return lines


def copy_argvs(lines):
    out = []
    for ln in lines:
        argv = shlex.split(ln)[2:]
        if len(argv) >= 2 and argv[1] == "absent":
            continue
        if argv and argv[0] != "mkdir":
            out.append(argv)
    return out


def assert_byte_identical(box, names):
    backup = box.root / "_state_backups" / QUIET_TS
    for name in names:
        src, dst = box.root / name, backup / name
        assert dst.is_file(), f"{name} was not copied to {dst}"
        assert dst.read_bytes() == src.read_bytes(), f"{name}: copy is not byte-identical"
        s, d = src.stat(), dst.stat()
        assert (d.st_mode & 0o7777) == (s.st_mode & 0o7777), f"{name}: mode not preserved (cp -a)"
        assert d.st_mtime_ns == s.st_mtime_ns, f"{name}: mtime not preserved (cp -a)"


def test_fixture_validator_state_is_json(box):
    """The fixture matches the deployment: validator_state.db is JSON text."""
    for name in ("validator_state.db", "validator_state.db.bak"):
        json.loads((box.root / name).read_text())


@pytest.mark.parametrize("svc", ALL)
def test_snapshot_copies_every_state_file_with_cp_a(box, svc):
    lines = snapshot_lines(box, svc)
    backup = box.root / "_state_backups" / QUIET_TS
    argvs = copy_argvs(lines)
    for name in PRESENT_STATE:
        want = ["cp", "-a", str(box.root / name), str(backup / name)]
        assert want in argvs, (
            f"no `{shlex.join(want)}` snapshot line for {name} "
            "(every state file, its .bak and the lane B/C sidecars, copied with cp -a):\n"
            + "\n".join(lines))


@pytest.mark.parametrize("svc", ALL)
def test_snapshot_commands_are_plain_copies(box, svc):
    """Every snapshot command is `mkdir` or `cp -a <root>/<name> <backup>/<name>`."""
    lines = snapshot_lines(box, svc)
    backup = box.root / "_state_backups" / QUIET_TS
    for argv in copy_argvs(lines):
        assert argv[0] == "cp" and argv[1:2] == ["-a"] and len(argv) == 4, (
            f"snapshot command is not `cp -a SRC DST`: {shlex.join(argv)}")
        src, dst = argv[2], argv[3]
        assert src.startswith(f"{box.root}/"), shlex.join(argv)
        assert dst == f"{backup}/{src.rsplit('/', 1)[1]}", shlex.join(argv)


@pytest.mark.parametrize("svc", ALL)
def test_snapshot_reports_absent_fixed_names(box, svc):
    box.remove_state(OPTIONAL_STATE)
    lines = snapshot_lines(box, svc)
    copied = {a[2].rsplit("/", 1)[1] for a in copy_argvs(lines) if len(a) == 4}
    absent = set(ABSENT_STATE) | {n for n in OPTIONAL_STATE if n in SNAPSHOT_FIXED}
    for name in sorted(absent):
        assert any(shlex.split(ln)[2:4] == [name, "absent"] for ln in lines), (
            f"no `PLAN snapshot {name} absent` line:\n" + "\n".join(lines))
        assert name not in copied, f"absent {name} is copied:\n" + "\n".join(lines)
    for name in PRESENT_STATE:
        if name not in OPTIONAL_STATE:
            assert name in copied, f"{name} is not copied:\n" + "\n".join(lines)


@pytest.mark.parametrize("svc", ALL)
def test_snapshot_never_copies_non_state(box, svc):
    lines = snapshot_lines(box, svc)
    copied = {a[-1].rsplit("/", 1)[1] for a in copy_argvs(lines)}
    for name in NOT_STATE:
        assert name not in copied, f"{name} is not §10.8 state:\n" + "\n".join(lines)


@pytest.mark.parametrize("svc", ALL)
def test_snapshot_replay_is_byte_identical(box, svc):
    lines = snapshot_lines(box, svc)
    box.replay_snapshot(lines, path=box.env("")["PATH"])
    assert box.shim_calls() == [], f"the snapshot called shimmed commands: {box.shim_calls()}"
    assert_byte_identical(box, PRESENT_STATE)
    backup = box.root / "_state_backups" / QUIET_TS
    assert sorted(p.name for p in backup.iterdir()) == sorted(PRESENT_STATE), (
        "the backup dir holds exactly the present state files")


@pytest.mark.parametrize("svc", ALL)
def test_missing_sqlite3_does_not_block_snapshot(box, svc):
    """The box has no sqlite3: the dry-run is GO and the printed snapshot,
    run as printed, still copies validator_state.db byte-identically."""
    path = box.path_without_sqlite3()
    lines = snapshot_lines(box, svc, path=path)
    box.replay_snapshot(lines, path=path, tools=None)
    assert_byte_identical(box, ("validator_state.db", "validator_state.db.bak"))


@pytest.mark.parametrize("svc", ALL)
def test_plan_never_invokes_sqlite3(box, svc):
    proc = box.run(svc)
    assert proc.returncode == 0, output(proc)
    hits = [ln for ln in plan_lines(proc.stdout) if SQLITE.search(ln)]
    assert not hits, "the plan uses sqlite3 (validator_state.db is JSON):\n" + "\n".join(hits)


def test_scripts_never_invoke_sqlite3():
    hits = []
    for p in sorted(LANED.rglob("*")):
        if not p.is_file():
            continue
        for i, ln in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if SQLITE.search(ln) and not ln.lstrip().startswith("#"):
                hits.append(f"{p.relative_to(LANED.parents[1])}:{i}: {ln.strip()}")
    assert not hits, ("deploy/laned/ uses sqlite3; validator_state.db is JSON and the box "
                      "has no sqlite3:\n" + "\n".join(hits))
