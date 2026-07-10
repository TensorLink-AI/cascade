"""Checkpoint-upload Hub→HF fallback — the training path's counterpart to the
miner's ``cascade deploy --hf-repo``. When the Hippius Hub is down, the trainer
mirrors a trained checkpoint to a HuggingFace **model** repo and returns an
``hf:`` ref the validator's ``fetch_from_hub`` already handles, instead of
aborting the round on the checkpoint upload."""

from __future__ import annotations

import types

import pytest

from cascade.shared.hippius import (
    HubRef,
    HubUpload,
    StorageError,
    upload_dir_to_hub_or_hf,
)
from cascade.trainer.loop import TrainerRunner


def _hub_up(repo):
    return HubUpload(ref=HubRef(repo, "sha256:" + "aa" * 32), size_bytes=10)


def _hf_up(repo, sha="bb"):
    return HubUpload(ref=HubRef(repo, "hf:" + sha * 20), size_bytes=20)


# ── upload_dir_to_hub_or_hf: Hub-first, HF only on outage ────────────────────


def test_hub_success_never_touches_hf(monkeypatch):
    import cascade.shared.hippius as hip

    def _boom_hf(*a, **k):
        raise AssertionError("HF must not be touched when the Hub succeeds")

    monkeypatch.setattr(hip, "upload_dir_to_hub", lambda ld, repo, hub: _hub_up(repo))
    monkeypatch.setattr(hip, "upload_dir_to_hf", _boom_hf)
    up = upload_dir_to_hub_or_hf("d", "cascade/ckpt-r1-king-toto2-4m", None,
                                 hf_repo="ns/ckpt-r1-king-toto2-4m")
    assert up.ref.digest == "sha256:" + "aa" * 32          # the content-addressed Hub ref


def test_hub_outage_falls_back_to_hf(monkeypatch):
    import cascade.shared.hippius as hip

    def _down(ld, repo, hub):
        raise StorageError("upload of d to cascade/ckpt failed after 4 attempt(s): hub 503")

    seen = {}

    def _hf(ld, repo, *, token=None):
        seen["repo"], seen["token"] = repo, token
        return _hf_up(repo)

    monkeypatch.setattr(hip, "upload_dir_to_hub", _down)
    monkeypatch.setattr(hip, "upload_dir_to_hf", _hf)
    up = upload_dir_to_hub_or_hf("d", "cascade/ckpt-r1-king-toto2-4m", None,
                                 hf_repo="ns/ckpt-r1-king-toto2-4m", hf_token="tok")
    assert up.ref.digest == "hf:" + "bb" * 20              # the hf: fallback ref
    assert seen["repo"] == "ns/ckpt-r1-king-toto2-4m"      # mirrored to the given HF repo
    assert seen["token"] == "tok"


def test_hub_outage_without_hf_repo_reraises(monkeypatch):
    import cascade.shared.hippius as hip

    def _down(ld, repo, hub):
        raise StorageError("hub down")

    monkeypatch.setattr(hip, "upload_dir_to_hub", _down)
    monkeypatch.setattr(hip, "upload_dir_to_hf",
                        lambda *a, **k: pytest.fail("no hf_repo ⇒ must not mirror off-Hub"))
    with pytest.raises(StorageError, match="hub down"):
        upload_dir_to_hub_or_hf("d", "cascade/ckpt-r1-king-toto2-4m", None, hf_repo=None)


# ── TrainerRunner._hf_ckpt_repo: derive the fallback repo from hf_backup_repo ─


def _stub(hf_backup_repo):
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(storage=types.SimpleNamespace(hf_backup_repo=hf_backup_repo))
    )


def test_hf_ckpt_repo_reuses_backup_namespace_keeps_basename():
    got = TrainerRunner._hf_ckpt_repo(
        _stub("tensorlink-dev/cascade-testnet-mirror"), "cascade/ckpt-r99-king-toto2-4m"
    )
    assert got == "tensorlink-dev/ckpt-r99-king-toto2-4m"


def test_hf_ckpt_repo_none_when_backup_unset_or_bare():
    assert TrainerRunner._hf_ckpt_repo(_stub(""), "cascade/ckpt-r1-king-toto2-4m") is None
    assert TrainerRunner._hf_ckpt_repo(_stub("no-slash"), "cascade/ckpt-r1-king-toto2-4m") is None
