"""W4 worker handshake (git head + package hash), T19 and T28: the eval host
and the validator compare code identity before anything is scored; a
mismatch fails closed with a CRITICAL ``lane_d_handshake_mismatch`` line."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess

import pytest

from .helpers import REPO

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=w4", "-c", "user.email=w4@localhost",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True, timeout=60).stdout.strip()


def _tree(root, files: dict[str, str]):
    for rel, text in files.items():
        p = root / "cascade" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _release_repo(tmp_path):
    """A release repo with two commits (A = old release, B = new release) and
    one checkout of each: the hosts' release dirs."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    _tree(origin, {"__init__.py": "", "eval/__init__.py": "", "eval/scorer.py": "VERSION = 1\n"})
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "release A")
    head_a = _git(origin, "rev-parse", "HEAD")
    _tree(origin, {"eval/scorer.py": "VERSION = 2\n"})
    _git(origin, "add", "-A")
    _git(origin, "commit", "-q", "-m", "release B")
    head_b = _git(origin, "rev-parse", "HEAD")
    old, new = tmp_path / "rel-old", tmp_path / "rel-new"
    subprocess.run(["git", "clone", "-q", str(origin), str(old)], check=True, timeout=60)
    subprocess.run(["git", "clone", "-q", str(origin), str(new)], check=True, timeout=60)
    _git(old, "checkout", "-q", head_a)
    return old, new, head_a, head_b


def test_package_hash_tracks_source_bytes_only(tmp_path):
    from cascade.eval.lane_d_handshake import package_hash

    a = _tree(tmp_path / "a", {"__init__.py": "", "x.py": "X = 1\n", "sub/y.py": "Y = 2\n"})
    h = package_hash(a)
    assert HEX64.match(h)
    assert package_hash(a) == h                                  # deterministic
    b = tmp_path / "b"
    shutil.copytree(a, b)
    assert package_hash(b) == h                                  # independent of location
    os.utime(a / "cascade" / "x.py", (1, 1))
    assert package_hash(a) == h                                  # mtimes never count
    pyc = a / "cascade" / "__pycache__" / "x.cpython-311.pyc"
    pyc.parent.mkdir()
    pyc.write_bytes(b"\x00compiled")
    assert package_hash(a) == h                                  # bytecode never counts
    (a / "cascade" / "sub" / "y.py").write_text("Y = 3\n")
    assert package_hash(a) != h                                  # a source edit does
    (a / "cascade" / "sub" / "y.py").write_text("Y = 2\n")
    assert package_hash(a) == h
    (a / "cascade" / "z.py").write_text("")
    assert package_hash(a) != h                                  # a new module does
    (a / "cascade" / "z.py").unlink()
    (a / "cascade" / "x.py").rename(a / "cascade" / "w.py")
    assert package_hash(a) != h                                  # a rename does


def test_local_identity_reads_git_head_of_the_release_dir_only(tmp_path):
    from cascade.eval.lane_d_handshake import CodeIdentity, local_identity, package_hash

    old, _new, head_a, _head_b = _release_repo(tmp_path)
    ident = local_identity(old)
    assert isinstance(ident, CodeIdentity)
    assert ident.git_head == head_a and HEX40.match(ident.git_head)
    assert ident.package_hash == package_hash(old)
    plain = tmp_path / "plain"
    shutil.copytree(old / "cascade", plain / "cascade")
    assert local_identity(plain).git_head == ""                  # not a checkout
    nested = _tree(old / "vendored", {"__init__.py": ""})
    assert local_identity(nested).git_head == ""                 # inside one, not its top


def test_local_identity_default_is_the_imported_release():
    from cascade.eval.lane_d_handshake import local_identity  # noqa: I001
    import cascade

    root = os.path.dirname(os.path.dirname(os.path.realpath(cascade.__file__)))
    assert local_identity() == local_identity(root)
    assert os.path.realpath(root) == os.path.realpath(str(REPO))


def test_worker_hello_carries_the_identity(tmp_path):
    from cascade.eval.lane_d_handshake import HANDSHAKE_SCHEMA, local_identity, worker_hello

    old, *_ = _release_repo(tmp_path)
    hello = worker_hello(old)
    ident = local_identity(old)
    assert HANDSHAKE_SCHEMA == "lane-d-handshake/1"
    assert hello == {"schema": HANDSHAKE_SCHEMA, "git_head": ident.git_head,
                     "package_hash": ident.package_hash}


def test_handshake_match_is_ok(tmp_path, caplog):
    from cascade.eval.lane_d_handshake import check_handshake, local_identity, worker_hello

    caplog.set_level(logging.INFO)
    _old, new, *_ = _release_repo(tmp_path)
    ident = local_identity(new)
    assert check_handshake(worker_hello(new), expected=ident) == ident
    assert any(r.getMessage().startswith("lane_d_handshake_ok") for r in caplog.records)
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_t19_stale_worker_is_crit_and_nothing_scored(tmp_path, caplog):
    from cascade.eval.lane_d_handshake import HandshakeMismatch, local_identity, with_handshake, worker_hello  # noqa: I001

    caplog.set_level(logging.INFO)
    old, new, *_ = _release_repo(tmp_path)
    calls = []

    def score():
        calls.append(1)
        return "scored"

    with pytest.raises(HandshakeMismatch) as e:
        with_handshake(worker_hello(old), score, expected=local_identity(new))
    assert calls == [], "work ran on a stale eval worker"
    assert e.value.fields == ("git_head", "package_hash")
    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert crit and crit[0].getMessage().startswith("lane_d_handshake_mismatch")
    assert "git_head" in crit[0].getMessage()
    # Matching identity: the work runs exactly once and its result is returned.
    assert with_handshake(worker_hello(new), score, expected=local_identity(new)) == "scored"
    assert calls == [1]


def test_handshake_package_drift_on_same_head_fails_closed(tmp_path):
    from cascade.eval.lane_d_handshake import HandshakeMismatch, check_handshake, local_identity, worker_hello  # noqa: I001

    _old, new, *_ = _release_repo(tmp_path)
    expected = local_identity(new)
    drifted = tmp_path / "drifted"
    subprocess.run(["git", "clone", "-q", str(new), str(drifted)], check=True, timeout=60)
    (drifted / "cascade" / "eval" / "scorer.py").write_text("VERSION = 2  # hot-patched\n")
    with pytest.raises(HandshakeMismatch) as e:
        check_handshake(worker_hello(drifted), expected=expected)
    assert e.value.fields == ("package_hash",)


@pytest.mark.parametrize("bad", [
    None, "lane-d-handshake/1", {},
    {"schema": "lane-d-handshake/0"},
    {"schema": "lane-d-handshake/1", "git_head": ""},
    {"schema": "lane-d-handshake/1", "package_hash": ""},
])
def test_handshake_malformed_or_unknown_hello_fails_closed(tmp_path, bad, caplog):
    from cascade.eval.lane_d_handshake import HandshakeMismatch, check_handshake, local_identity  # noqa: I001

    _old, new, *_ = _release_repo(tmp_path)
    expected = local_identity(new)
    hello = bad
    if isinstance(bad, dict) and bad.get("schema") == "lane-d-handshake/1":
        hello = {"schema": bad["schema"], "git_head": expected.git_head,
                 "package_hash": expected.package_hash, **bad}
    with pytest.raises(HandshakeMismatch) as e:
        check_handshake(hello, expected=expected)
    assert e.value.fields
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_handshake_unknown_git_head_on_both_sides_fails_closed(tmp_path):
    """Two plain (non-checkout) trees with equal bytes still fail: an unknown
    head is never a match."""
    from cascade.eval.lane_d_handshake import HandshakeMismatch, check_handshake, local_identity, worker_hello  # noqa: I001

    a = _tree(tmp_path / "a", {"__init__.py": "", "x.py": "X = 1\n"})
    b = tmp_path / "b"
    shutil.copytree(a, b)
    with pytest.raises(HandshakeMismatch) as e:
        check_handshake(worker_hello(b), expected=local_identity(a))
    assert e.value.fields == ("git_head",)


def test_t28_documented_upgrade_order(tmp_path):
    """§10.5 S5 order: eval host first, then the validator. Every mixed step
    fails closed (nothing scored); the end state works. The reverse order
    fails closed too and recovers once both hosts match."""
    from cascade.eval.lane_d_handshake import HandshakeMismatch, local_identity, with_handshake, worker_hello  # noqa: I001

    old, new, *_ = _release_repo(tmp_path)
    scored = []

    def step(eval_root, validator_root) -> bool:
        try:
            with_handshake(worker_hello(eval_root), lambda: scored.append(eval_root),
                           expected=local_identity(validator_root))
        except HandshakeMismatch:
            return False
        return True

    # documented order: eval host updated first, then the validator
    assert step(old, old) is True
    assert step(new, old) is False
    assert step(new, new) is True
    # reverse order (validator first) fails closed until the eval host follows
    assert step(old, new) is False
    assert step(new, new) is True
    assert scored == [old, new, new]
