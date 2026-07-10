"""HuggingFace submission fallback for `cascade deploy` — a miner can still submit
a generator when the Hippius Hub OCI registry is down. Covers ``upload_dir_to_hf``
(produces a ``repo@hf:<sha>`` ref the chain + trainer already understand) and the
deploy dispatch (Hub-first, HF fallback), all over a fake HfApi (no network)."""

from __future__ import annotations

import types

import pytest

from cascade.interface.validation import format_commit, parse_commit
from cascade.miner import cli
from cascade.shared.hippius import (
    HubRef,
    HubUpload,
    StorageError,
    is_hub_ref,
    upload_dir_to_hf,
)


class _FakeCommit:
    def __init__(self, oid):
        self.oid = oid


class _FakeHfApi:
    """Records create/upload calls; returns a full-sha commit by default."""

    last: _FakeHfApi | None = None

    def __init__(self, token=None):
        self.token = token
        self.created = None
        self.uploaded = None
        _FakeHfApi.last = self

    def create_repo(self, repo, **k):
        self.created = (repo, k)

    def upload_folder(self, *, repo_id, folder_path, repo_type, allow_patterns, commit_message):
        self.uploaded = (repo_id, folder_path, repo_type, tuple(allow_patterns))
        return _FakeCommit("ab" * 20)  # a 40-hex commit sha


def _gen(tmp_path):
    d = tmp_path / "gen"
    d.mkdir()
    (d / "generator.py").write_text("x")
    (d / "config.json").write_text("{}")
    return d


# ── upload_dir_to_hf ─────────────────────────────────────────────────────────


def test_upload_to_hf_returns_hf_ref_that_commits(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)

    up = upload_dir_to_hf(_gen(tmp_path), "me/gen")
    assert up.ref.repo == "me/gen"
    assert up.ref.digest == "hf:" + "ab" * 20
    assert is_hub_ref(up.ref.immutable_ref)

    # the ref round-trips as an on-chain commit (grammar accepts hf: digests)
    payload = format_commit(up.ref.immutable_ref)
    assert parse_commit(payload).ref == up.ref.immutable_ref

    # must push to a MODEL repo so fetch_from_hub's hf: snapshot_download matches
    assert _FakeHfApi.last.uploaded[2] == "model"
    assert _FakeHfApi.last.created[0] == "me/gen"


def test_upload_to_hf_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    with pytest.raises(StorageError, match="requires an HF token"):
        upload_dir_to_hf(_gen(tmp_path), "me/gen")


def test_upload_to_hf_resolves_full_sha_when_oid_is_a_tag(tmp_path, monkeypatch):
    full = "cd" * 20

    class _TagApi(_FakeHfApi):
        def upload_folder(self, **k):
            return _FakeCommit("main")  # a branch name, not a sha

        def list_repo_refs(self, repo, **k):
            b = types.SimpleNamespace(name="main", target_commit=full)
            return types.SimpleNamespace(branches=[b])

    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setattr("huggingface_hub.HfApi", _TagApi)
    up = upload_dir_to_hf(_gen(tmp_path), "me/gen")
    assert up.ref.digest == "hf:" + full


def test_upload_to_hf_rejects_unusable_oid(tmp_path, monkeypatch):
    class _BadApi(_FakeHfApi):
        def upload_folder(self, **k):
            return _FakeCommit("main")

        def list_repo_refs(self, repo, **k):
            return types.SimpleNamespace(branches=[])  # can't resolve → hard fail

    monkeypatch.setenv("HF_TOKEN", "tok")
    monkeypatch.setattr("huggingface_hub.HfApi", _BadApi)
    with pytest.raises(StorageError, match="no usable commit sha"):
        upload_dir_to_hf(_gen(tmp_path), "me/gen")


# ── deploy dispatch (Hub-first, HF fallback) ─────────────────────────────────


def _args(hub_repo=None, hf_repo=None):
    return types.SimpleNamespace(repo_dir="d", hub_repo=hub_repo, hf_repo=hf_repo)


def _hub_up(repo):
    return HubUpload(ref=HubRef(repo, "sha256:" + "aa" * 32), size_bytes=10)


def _hf_up(repo, sha="bb"):
    return HubUpload(ref=HubRef(repo, "hf:" + sha * 20), size_bytes=20)


def test_dispatch_hub_success_skips_hf(monkeypatch, cfg):
    import cascade.shared.hippius as hip

    def _boom_hf(*a, **k):
        raise AssertionError("HF must not be touched when the Hub succeeds")

    monkeypatch.setattr(hip, "upload_dir_to_hub", lambda rd, repo, hub: _hub_up(repo))
    monkeypatch.setattr(hip, "upload_dir_to_hf", _boom_hf)
    rc, ref = cli._upload_generator(_args(hub_repo="me/gen"), cfg)
    assert rc == 0 and ref == "me/gen@sha256:" + "aa" * 32


def test_dispatch_hub_fail_falls_back_to_hf(monkeypatch, cfg, capsys):
    import cascade.shared.hippius as hip

    def _down(rd, repo, hub):
        raise StorageError("hub 503")

    monkeypatch.setattr(hip, "upload_dir_to_hub", _down)
    monkeypatch.setattr(hip, "upload_dir_to_hf", lambda rd, repo: _hf_up(repo))
    rc, ref = cli._upload_generator(_args(hub_repo="me/gen", hf_repo="me/hf"), cfg)
    assert rc == 0 and ref == "me/hf@hf:" + "bb" * 20
    assert "falling back to HuggingFace" in capsys.readouterr().err


def test_dispatch_hub_fail_without_hf_errors(monkeypatch, cfg):
    import cascade.shared.hippius as hip

    def _down(rd, repo, hub):
        raise StorageError("hub down")

    monkeypatch.setattr(hip, "upload_dir_to_hub", _down)
    rc, ref = cli._upload_generator(_args(hub_repo="me/gen"), cfg)
    assert rc == 4 and ref is None


def test_dispatch_hf_only_bypasses_hub(monkeypatch, cfg):
    import cascade.shared.hippius as hip

    def _boom_hub(*a, **k):
        raise AssertionError("Hub must not be touched when only --hf-repo is given")

    monkeypatch.setattr(hip, "upload_dir_to_hub", _boom_hub)
    monkeypatch.setattr(hip, "upload_dir_to_hf", lambda rd, repo: _hf_up(repo, sha="cc"))
    rc, ref = cli._upload_generator(_args(hf_repo="me/hf"), cfg)
    assert rc == 0 and ref == "me/hf@hf:" + "cc" * 20
