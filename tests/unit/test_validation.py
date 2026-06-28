"""Submission validation: generator commit format, requirements, repo layout."""

from __future__ import annotations

import pytest

from metronome.interface.validation import (
    check_repo_layout,
    check_repo_size,
    check_requirements_hash_locked,
    format_commit,
    parse_commit,
)

# A valid CIDv0 (base58btc) and CIDv1 (base32) for the Hippius registry pointer.
CID_V0 = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
CID_V1 = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"


def test_parse_commit_round_trip():
    payload = f"metro-v1:gen:hippius:{CID_V0}"
    parsed = parse_commit(payload)
    assert parsed is not None
    assert parsed.cid == CID_V0
    assert format_commit(CID_V0) == payload
    # CIDv1 too
    assert parse_commit(f"metro-v1:gen:hippius:{CID_V1}").cid == CID_V1


@pytest.mark.parametrize(
    "payload",
    [
        "",
        f"metro-v0:gen:hippius:{CID_V0}",
        f"metro-v1:trained:hippius:{CID_V0}",        # trained tag is not a gen commit
        f"metro-v1:gen:hf:{CID_V0}",                  # old backend tag is gone
        "metro-v1:gen:hippius:not-a-cid",
        "metro-v1:gen:hippius:Qmtooshort",
        f"metro-v1:gen:hippius: {CID_V0}",
        f"metro-v1:gen:hippius:{CID_V0}extra!",
        "metro-v1:gen:hippius:",
    ],
)
def test_parse_commit_rejects_malformed(payload):
    assert parse_commit(payload) is None


def test_format_commit_refuses_invalid_inputs():
    with pytest.raises(ValueError):
        format_commit("not-a-cid")
    with pytest.raises(ValueError):
        format_commit("Qmshort")


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_repo_layout_accepts_generator_repo(tmp_path):
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "generator.py", "x = 1\n")
    _write(tmp_path, "requirements.txt", "numpy==1.26.4 --hash=sha256:" + "a" * 64 + "\n")
    assert check_repo_layout(tmp_path).ok


def test_repo_layout_accepts_safetensors_weights(tmp_path):
    # A generator may BE a model: safetensors weights are allowed.
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "generator.py", "x = 1\n")
    _write(tmp_path, "requirements.txt", "")
    _write(tmp_path, "model.safetensors", "binary")
    assert check_repo_layout(tmp_path).ok


def test_repo_layout_rejects_pickle_weights(tmp_path):
    # Pickle checkpoints execute code on load — rejected (ship safetensors).
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "generator.py", "x = 1\n")
    _write(tmp_path, "requirements.txt", "")
    _write(tmp_path, "model.pt", "binary")
    r = check_repo_layout(tmp_path)
    assert not r.ok
    assert r.reason == "pickle_weights_forbidden"


def test_check_repo_size_caps_total_bytes(tmp_path):
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "generator.py", "x = 1\n")
    _write(tmp_path, "weights.safetensors", "z" * 4096)
    assert check_repo_size(tmp_path, max_repo_mb=1).ok       # ~4 KB <= 1 MB
    over = check_repo_size(tmp_path, max_repo_mb=0)          # 0-byte cap
    assert not over.ok
    assert over.reason == "repo_too_large"


def test_repo_layout_rejects_missing_files(tmp_path):
    _write(tmp_path, "config.json", "{}")
    r = check_repo_layout(tmp_path)
    assert not r.ok
    assert r.reason == "missing_files"


def test_requirements_hash_locked(tmp_path):
    ok = _write(tmp_path, "ok.txt", f"numpy==1.26.4 --hash=sha256:{'a' * 64}\n")
    assert check_requirements_hash_locked(ok, allowed=("numpy",), max_packages=5).ok

    unpinned = _write(tmp_path, "bad.txt", "numpy>=1.0\n")
    r = check_requirements_hash_locked(unpinned, allowed=("numpy",), max_packages=5)
    assert not r.ok and r.reason == "requirement_not_hash_locked"

    bad_pkg = _write(tmp_path, "bad2.txt", f"evil==1.0 --hash=sha256:{'b' * 64}\n")
    r = check_requirements_hash_locked(bad_pkg, allowed=("numpy",), max_packages=5)
    assert not r.ok and r.reason == "requirement_not_allowlisted"
