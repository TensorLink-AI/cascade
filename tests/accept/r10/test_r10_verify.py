"""R10 verify: a bundle whose manifest doesn't verify is named and refused."""

from __future__ import annotations

import json
from pathlib import Path

from .helpers import build_source, manifest, ok, refused, run_cli, snapshot


def _bundle(tmp_path):
    bundle, _ = snapshot(build_source(tmp_path), tmp_path / "bundles")
    ok(run_cli("verify", "--bundle", bundle))
    return bundle


def _entry(bundle, source_name):
    return next(e for e in manifest(bundle)["files"] if e["source_name"] == source_name)


def test_verify_accepts_an_intact_bundle(tmp_path):
    _bundle(tmp_path)


def test_verify_refuses_a_tampered_file(tmp_path):
    bundle = _bundle(tmp_path)
    e = _entry(bundle, "cascade_state.json")
    path = bundle / e["path"]
    path.write_text(path.read_text().replace('"generation": 2', '"generation": 3'))
    refused(run_cli("verify", "--bundle", bundle), path.name)


def test_verify_refuses_a_missing_file(tmp_path):
    bundle = _bundle(tmp_path)
    e = _entry(bundle, "netconfig_state.json")
    (bundle / e["path"]).unlink()
    refused(run_cli("verify", "--bundle", bundle), Path(e["path"]).name)


def test_verify_refuses_an_unlisted_extra_file(tmp_path):
    bundle = _bundle(tmp_path)
    e = _entry(bundle, "validator_state.db")
    extra = (bundle / e["path"]).with_name("smuggled_state.json")
    extra.write_text("{}")
    refused(run_cli("verify", "--bundle", bundle), "smuggled_state.json")


def test_verify_refuses_a_tampered_manifest(tmp_path):
    bundle = _bundle(tmp_path)
    m = manifest(bundle)
    name = ""
    for e in m["files"]:
        if e["source_name"] == "validator_state.db":
            e["sha256"] = "0" * 63 + "1"
            name = Path(e["path"]).name
    (bundle / "manifest.json").write_text(json.dumps(m))
    refused(run_cli("verify", "--bundle", bundle), name)
