"""W13 production wiring of king publication (plan §11.1, rev5 B13).

* :func:`cascade.lane_d.publish.king_entry` builds the §11.1 index entry of a
  crowned report from the signed report alone (plus the intake's ready
  licence and the era's sentinel record), on a REAL signed crowned report
  (the W17 crowned scenario: ``settle_lane_d`` on a synthetic chain).
* ``python -P -m cascade.lane_d.ops_jobs king-publish`` takes every crowned
  persisted report not yet indexed, copies the king from the vault to the
  Hub, R2 and HF mirrors (``publish_king``), re-signs
  ``champions/lane_d/index.json`` with the pinned signer, puts it on the
  public store and writes the local copy; after it ``king-check`` (the full
  vault + mirror re-hash) is green on that crowned report.
* ``takedown`` unpublishes the Hub and HF copies and re-signs the index.
* :mod:`cascade.lane_d.mirrors` adapts the existing Hub / HF (HF-shaped API)
  and R2 (boto3-shaped) clients to the five-method mirror protocol;
  ``publish_king`` and ``check_one_digest`` run unchanged over them.

The filesystem mode (``--vault-dir``, ``--mirror NAME=DIR``,
``--index-store-dir``) is what testnet and these tests run; the live clients
are tested with fakes of the API objects. No network. Every PATHS import is
inside a test body.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from .helpers import MIRRORS, fixture_key, no_network, vault_with_king  # noqa: F401


def jobs():
    return importlib.import_module("cascade.lane_d.ops_jobs")


def _main(argv: list[str]) -> int:
    try:
        return jobs().main(argv)
    except SystemExit as e:              # argparse usage error (a job this base lacks)
        return int(e.code or 0)


def _last(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def crowned_run(tmp_path_factory):
    """The W17 crowned scenario, run once: a real signed crowned report."""
    from tests.accept.w17 import helpers as w17

    return w17.crowned(tmp_path_factory.mktemp("w13-crowned"))


def _testnet_report(tmp: Path, run: dict) -> dict:
    """The crowned scenario re-homed under ``tmp`` with its crowned digest
    replaced by a real vault entry's (``vault_with_king``) and re-signed by
    the scenario's own signer: a crowned report whose king is in a vault."""
    from cascade.validator.lane_d_report import rows_sha256, sign_report, verify_report
    from cascade.validator.lane_d_state import report_path

    signer = run["fx"].signer
    report = run["report"]
    old = report["crowned"]
    hotkey = next(e["hotkey"] for e in report["entries"] if e["digest"] == old)
    vault, digest, files = vault_with_king(tmp, hotkey)
    body = json.loads(json.dumps({k: v for k, v in report.items() if k != "signature"})
                      .replace(old, digest))
    body["rows_sha256"] = rows_sha256(body["rows"])
    signed = sign_report(body, signer)
    assert verify_report(signed, signer.ss58_address)
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True)
    state.write_text("{}")
    for b, rep in ((r["boundary"], r) for r in run["reports"]):
        p = report_path(state, b)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(signed if rep["crowned"] is not None else rep))
    intake = tmp / "intake"
    intake.mkdir()
    (intake / "lane_d_intake_queue.json").write_text(json.dumps({"entries": [
        {"hotkey": hotkey, "digest": digest, "state": "revealed", "licence": "MIT"}]}))
    return {"signer": signer, "report": signed, "digest": digest, "hotkey": hotkey,
            "vault": tmp / "vault", "state": state, "intake": intake, "files": files,
            "tmp": tmp}


@pytest.fixture
def testnet(tmp_path, monkeypatch, crowned_run):
    import cascade.shared.chain as chain

    t = _testnet_report(tmp_path, crowned_run)
    monkeypatch.setattr(chain, "open_wallet", lambda *a, **k: t["signer"])
    return t


def _fs(t) -> list[str]:
    return ["--vault-dir", str(t["vault"]),
            *(a for m in MIRRORS for a in ("--mirror", f"{m}={t['tmp'] / 'mirrors' / m}")),
            "--index-store-dir", str(t["tmp"] / "public")]


def _publish(t, *extra: str) -> int:
    return _main(["king-publish", "--state", str(t["state"]), "--intake-state-dir",
                        str(t["intake"]), "--index", str(t["tmp"] / "index.json"),
                        "--signer-ss58", t["signer"].ss58_address, *_fs(t), *extra])


def _check(t) -> int:
    return _main(["king-check", "--state", str(t["state"]), "--index",
                        str(t["tmp"] / "index.json"), "--signer-ss58", t["signer"].ss58_address,
                        "--vault-dir", str(t["vault"]),
                        *(a for m in MIRRORS
                          for a in ("--mirror", f"{m}={t['tmp'] / 'mirrors' / m}"))])


# ── the index entry of a real crowned report ─────────────────────────────────


def test_king_entry_comes_from_the_signed_crowned_report(crowned_run):
    from cascade.lane_d.publish import KING_FIELDS, king_entry, sentinel_entry

    report = crowned_run["report"]
    sentinel = sentinel_entry(report, None)
    assert sentinel == {"suite": "none", "status": "not_armed"}, "gate.sentinel is null here"
    king = king_entry(report, licence="MIT", sentinel=sentinel)
    entry = next(e for e in report["entries"] if e["digest"] == report["crowned"])
    verdict = next(v for v in report["verdicts"] if v["outcome"] == "crowned")
    assert set(KING_FIELDS) <= set(king)
    assert king["digest"] == report["crowned"] and king["crown_block"] == report["boundary"]
    assert (king["hotkey"], king["context"]) == (entry["hotkey"], entry["context"])
    assert king["lcb"] == verdict["lcb"]
    assert king["gate"]["suite"] == report["gate"]["suite"] and king["gate"]["outcome"] == "pass"
    assert king["license"] == "MIT"
    assert king_entry(report, licence="MIT", sentinel=sentinel) == king, "deterministic"


def test_an_armed_sentinel_needs_its_record_for_this_king(crowned_run):
    from cascade.lane_d.publish import PublishError, sentinel_entry

    report = json.loads(json.dumps(crowned_run["report"]))
    report["gate"]["sentinel"] = "boom"
    with pytest.raises(PublishError):
        sentinel_entry(report, None)
    rec = {"king_digest": report["crowned"], "suite": "boom", "boundary": report["boundary"],
           "status": "ok", "models": {"king": {"crps": 0.4, "mase": 0.9}}}
    assert sentinel_entry(report, rec) == {"suite": "boom", "status": "ok",
                                           "king": {"crps": 0.4, "mase": 0.9}}
    with pytest.raises(PublishError):
        sentinel_entry(report, {**rec, "king_digest": "f" * 64})


# ── king-publish -> king-check on a crowned testnet report ───────────────────


def test_king_check_is_red_before_and_green_after_king_publish(testnet, capsys):
    assert _check(testnet) == 1
    assert "index_missing" in json.dumps(_last(capsys)["problems"])
    assert _publish(testnet) == 0
    out = _last(capsys)
    assert out["action"] == "published" and out["planned"][0]["digest"] == testnet["digest"]
    assert _check(testnet) == 0
    out = _last(capsys)
    assert out["scope"] == "full" and out["problems"] == {} and out["crowned_reports"] == 1
    index = json.loads((testnet["tmp"] / "index.json").read_text())
    public = json.loads((testnet["tmp"] / "public" / "champions" / "lane_d" / "index.json")
                        .read_text())
    assert public == index and [k["digest"] for k in index["kings"]] == [testnet["digest"]]


def test_king_publish_is_idempotent_and_dry_run_writes_nothing(testnet, capsys):
    assert _publish(testnet, "--dry-run") == 0
    assert _last(capsys)["action"] == "planned"
    assert not (testnet["tmp"] / "index.json").exists()
    assert not (testnet["tmp"] / "mirrors").exists()
    assert _publish(testnet) == 0
    capsys.readouterr()
    before = (testnet["tmp"] / "index.json").read_bytes()
    assert _publish(testnet) == 0
    assert _last(capsys)["action"] == "current"
    assert (testnet["tmp"] / "index.json").read_bytes() == before


def test_king_publish_blocks_without_the_ready_licence(testnet, capsys):
    (testnet["intake"] / "lane_d_intake_queue.json").write_text(json.dumps({"entries": []}))
    assert _publish(testnet) == 3
    assert _last(capsys)["action"] == "blocked"
    assert not (testnet["tmp"] / "index.json").exists()


def test_king_publish_refuses_a_wallet_that_is_not_the_pinned_signer(testnet, monkeypatch,
                                                                     capsys):
    import cascade.shared.chain as chain

    monkeypatch.setattr(chain, "open_wallet", lambda *a, **k: fixture_key("stranger"))
    assert _publish(testnet) == 1
    assert not (testnet["tmp"] / "index.json").exists()


def test_takedown_unpublishes_hub_and_hf_and_king_check_stays_green(testnet, capsys):
    assert _publish(testnet) == 0
    argv = ["takedown", "--digest", testnet["digest"], "--reason", "licence", "--block",
            str(testnet["report"]["boundary"] + 1), "--index", str(testnet["tmp"] / "index.json"),
            "--signer-ss58", testnet["signer"].ss58_address, *_fs(testnet)]
    assert _main([*argv, "--dry-run"]) == 0
    capsys.readouterr()
    assert _main(argv) == 0
    index = json.loads((testnet["tmp"] / "index.json").read_text())
    assert index["kings"][0]["takedown"] == {"reason": "licence",
                                             "block": testnet["report"]["boundary"] + 1}
    for m in ("hub", "hf"):
        assert not list((testnet["tmp"] / "mirrors" / m).rglob("*.safetensors"))
    assert list((testnet["tmp"] / "mirrors" / "r2").rglob("*.safetensors")), "R2 copy stays"
    capsys.readouterr()
    assert _check(testnet) == 0


# ── the live mirror adapters (fakes of the API / client objects) ─────────────


class EntryNotFoundError(Exception):
    """Named like huggingface_hub's not-found error."""


class FakeRepoApi:
    """An HF-shaped API (``HippiusApi`` / ``HfApi``) over dicts."""

    def __init__(self):
        self.repos: dict[tuple[str, str], dict[str, bytes]] = {}

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        assert private is False and exist_ok is True
        self.repos.setdefault((repo_id, repo_type), {})

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.repos[(repo_id, repo_type)][path_in_repo] = bytes(path_or_fileobj)

    def hf_hub_download(self, *, repo_id, filename, repo_type, local_dir):
        files = self.repos.get((repo_id, repo_type), {})
        if filename not in files:
            raise EntryNotFoundError(filename)
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(files[filename])
        return str(p)

    def list_repo_files(self, repo_id, *, repo_type):
        return sorted(self.repos.get((repo_id, repo_type), {}))

    def get_paths_info(self, repo_id, paths, *, repo_type):
        files = self.repos.get((repo_id, repo_type), {})

        class Info:
            def __init__(self, path, size):
                self.path, self.size = path, size

        return [Info(p, len(files[p])) for p in paths if p in files]

    def delete_file(self, path, repo_id, *, repo_type, commit_message):
        self.repos.get((repo_id, repo_type), {}).pop(path, None)


class FakeBoto:
    def __init__(self, bucket):
        self.bucket, self.objects = bucket, {}

    def _missing(self):
        e = Exception("NoSuchKey")
        e.response = {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
        return e

    def put_object(self, *, Bucket, Key, Body, ContentType):
        assert Bucket == self.bucket
        self.objects[Key] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise self._missing()
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, *, Bucket, Prefix, **kw):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)],
                "IsTruncated": False}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise self._missing()
        return {"ContentLength": len(self.objects[Key])}

    def delete_object(self, *, Bucket, Key):
        self.objects.pop(Key, None)

    def create_bucket(self, **kw):
        raise AssertionError("the R2 mirror never creates a bucket")


def test_live_adapters_carry_publish_king_and_check_one_digest(tmp_path):
    from types import SimpleNamespace

    from cascade.lane_d.mirrors import open_live_mirrors
    from cascade.lane_d.publish import (
        HF_MIRROR_REPO,
        build_index,
        check_one_digest,
        hub_repo,
        publish_king,
        verify_mirrors,
    )

    hk = fixture_key("miner").ss58_address
    vault, digest, files = vault_with_king(tmp_path, hk)
    hub, hf, r2 = FakeRepoApi(), FakeRepoApi(), FakeBoto("cascade-manifests")
    cfg = SimpleNamespace(storage=SimpleNamespace(hub_namespace="cascade", backup_bucket="",
                                                  manifest_bucket="cascade-manifests",
                                                  backup_s3_endpoint="https://r2.example"))
    ms = open_live_mirrors(cfg, context=8192, hotkey=hk, digest=digest, environ={},
                           hub_api=hub, hf_api=hf, r2_client=r2)
    assert sorted(ms) == sorted(MIRRORS)
    publish_king(hotkey=hk, digest=digest, vault=vault, mirrors=ms, alert=lambda d: None)
    assert sorted(hub.repos[(hub_repo("cascade", 8192, hk), "model")]) == sorted(
        [*files, "manifest.json"])
    assert f"lane-d-kings/{digest}/weights.safetensors" in hf.repos[(HF_MIRROR_REPO, "dataset")]
    assert f"lane-d-kings/{digest}/manifest.json" in r2.objects
    assert verify_mirrors(digest, ms) == {m: "ok" for m in MIRRORS}
    assert ms["hub"].size(f"lane-d-kings/{digest}/weights.safetensors") == len(
        files["weights.safetensors"])
    signer = fixture_key("validator")
    from tests.accept.w13.helpers import king_record

    index = build_index([king_record(hk, digest, crown_block=7)], signer)
    report = {"crowned": digest, "boundary": 7}
    assert check_one_digest(index=index, report=report, vault=vault, mirrors=ms) == []
    with pytest.raises(KeyError):
        ms["hf"].get(f"lane-d-kings/{'e' * 64}/manifest.json")
    for key in ms["hf"].list(f"lane-d-kings/{digest}/"):
        ms["hf"].delete(key)
    assert verify_mirrors(digest, ms)["hf"] == "missing"


def test_live_clients_name_existing_credentials_only():
    from cascade.lane_d import mirrors

    src = Path(mirrors.__file__).read_text(encoding="utf-8")
    for name in ("BACKUP_S3_ACCESS_KEY", "BACKUP_S3_SECRET_KEY", "HF_TOKEN"):
        assert name in src
    assert "create_bucket" not in src, "the mirrors never create a bucket"
    with pytest.raises(mirrors.MirrorError):
        mirrors._hf_api({})
