"""R12: the bring-up handshake (plan §10.2 "with the worker handshake",
§10.4 "handshake first", T19; targon-eval.toml history: every eval-host
rotation checked the four suite identities gift 30841734 / boom 69325b54 /
time 83e3d0b3 / fev 6de245af and /data/bench 17747 files).

``evalhost.py handshake`` checks the pod's captured suite identities against
the profile's four pins and, given the pod's captured worker hello, the W4
code handshake (``cascade.eval.lane_d_handshake``) against the release.
Contract: README.md.
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    BENCH_FILES,
    EXIT,
    PINNED_IDENTITIES,
    Kit,
    as_json,
    make_git_release,
    output,
)

FULL = {k: v + "9c1e7f2a" * 7 for k, v in PINNED_IDENTITIES.items()}


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def _ids(kit, suites, bench_files=BENCH_FILES, name="identities.json"):
    data = {"suites": suites}
    if bench_files is not None:
        data["bench_files"] = bench_files
    return kit.write(name, json.dumps(data))


def _handshake(kit, *args):
    proc = kit.tool("handshake", *args)
    assert kit.shim_calls() == [], f"handshake ran an external command: {kit.shim_calls()}"
    return proc


def test_identities_matching_the_pins_pass(kit):
    for suites in (dict(PINNED_IDENTITIES), FULL):
        proc = _handshake(kit, "--profile", kit.profile(), "--identities", _ids(kit, suites))
        assert proc.returncode == 0, output(proc)
        out = as_json(proc)
        assert out["ok"] is True and out["mismatched"] == [], out
        for suite, pin in PINNED_IDENTITIES.items():
            row = out["suites"][suite]
            assert row["pinned"] == pin and row["ok"] is True, row
            assert row["observed"] == suites[suite], row
        assert out["code"] is None, out


@pytest.mark.parametrize(("mutate", "want"), [
    (lambda s: s.update(boom="69325b55"), ["boom"]),
    (lambda s: s.update(gift="0" + s["gift"]), ["gift"]),
    (lambda s: s.pop("fev"), ["fev"]),
    (lambda s: s.update(time="83e3d0"), ["time"]),
    (lambda s: s.update(gift="3084173", fev="6de245ae"), ["fev", "gift"]),
], ids=["one_digit", "shifted", "missing", "too_short", "two"])
def test_suite_identity_mismatch_is_refused(kit, mutate, want):
    suites = dict(FULL)
    mutate(suites)
    proc = _handshake(kit, "--profile", kit.profile(), "--identities", _ids(kit, suites))
    assert proc.returncode == EXIT["suite_identity_mismatch"], output(proc)
    out = as_json(proc)
    assert out["ok"] is False and out["mismatched"] == want, out


@pytest.mark.parametrize("bench_files", [17746, None])
def test_bench_file_count_must_match(kit, bench_files):
    proc = _handshake(kit, "--profile", kit.profile(), "--identities",
                      _ids(kit, FULL, bench_files=bench_files))
    assert proc.returncode == EXIT["suite_identity_mismatch"], output(proc)
    out = as_json(proc)
    assert out["mismatched"] == ["bench_files"], out


def test_a_profile_with_pins_needs_the_identities(kit):
    proc = _handshake(kit, "--profile", kit.profile())
    assert proc.returncode == EXIT["usage"], output(proc)


def test_a_profile_without_pins_needs_a_hello(kit):
    profile = kit.profile(suite_identities=None, bench_files=None)
    proc = _handshake(kit, "--profile", profile)
    assert proc.returncode == EXIT["usage"], output(proc)


def _hello(release):
    from cascade.eval.lane_d_handshake import worker_hello

    return worker_hello(root=release)


def test_code_handshake_matches_the_release(kit):
    release = make_git_release(kit.tmp / "release")
    hello = kit.write("hello.json", json.dumps(_hello(release)))
    profile = kit.profile(suite_identities=None, bench_files=None)
    proc = _handshake(kit, "--profile", profile, "--hello", hello, "--release-dir", release)
    assert proc.returncode == 0, output(proc)
    out = as_json(proc)
    assert out["ok"] is True and out["code"]["ok"] is True, out
    assert out["code"]["fields"] == [], out


@pytest.mark.parametrize(("field", "value"), [
    ("package_hash", "0" * 63 + "1"),
    ("git_head", "1" * 40),
    ("git_head", ""),
], ids=["stale_package", "other_head", "empty_head"])
def test_stale_worker_code_is_refused(kit, field, value):
    release = make_git_release(kit.tmp / "release")
    hello = _hello(release)
    hello[field] = value
    path = kit.write("hello.json", json.dumps(hello))
    profile = kit.profile(suite_identities=None, bench_files=None)
    proc = _handshake(kit, "--profile", profile, "--hello", path, "--release-dir", release)
    assert proc.returncode == EXIT["code_mismatch"], output(proc)
    out = as_json(proc)
    assert out["ok"] is False and out["code"]["ok"] is False, out
    assert out["code"]["fields"] == [field], out


def test_both_checks_run_and_suite_mismatch_decides_the_exit(kit):
    release = make_git_release(kit.tmp / "release")
    hello = _hello(release)
    hello["package_hash"] = "f" * 64
    path = kit.write("hello.json", json.dumps(hello))
    suites = dict(FULL, boom="00000001")
    proc = _handshake(kit, "--profile", kit.profile(), "--identities", _ids(kit, suites),
                      "--hello", path, "--release-dir", release)
    assert proc.returncode == EXIT["suite_identity_mismatch"], output(proc)
    out = as_json(proc)
    assert out["mismatched"] == ["boom"] and out["code"]["fields"] == ["package_hash"], out


def test_non_object_hello_fails_closed(kit):
    release = make_git_release(kit.tmp / "release")
    path = kit.write("hello.json", json.dumps(["not", "a", "hello"]))
    profile = kit.profile(suite_identities=None, bench_files=None)
    proc = _handshake(kit, "--profile", profile, "--hello", path, "--release-dir", release)
    assert proc.returncode == EXIT["code_mismatch"], output(proc)


def test_handshake_dry_run_runs_nothing(kit):
    proc = _handshake(kit, "--profile", kit.profile(), "--dry-run")
    assert proc.returncode == 0, output(proc)
    out = as_json(proc)
    assert out.get("dry_run") is True and isinstance(out.get("would_run"), list), out
