"""W13: a crowned lane D king is published to the Hippius Hub, the R2 public
mirror and the HF mirror, and every mirror's bytes hash to the one digest
(plan §11.1; rev6.4 §1.2: no Object Lock, so immutability is the
content-addressed path + the re-hash on every read). README.md has the
contract.

On main ``cascade.lane_d.publish`` does not exist: every test fails in its
body with ModuleNotFoundError for that PATHS module.
"""

from __future__ import annotations

import json

import pytest

from . import helpers as h
from .helpers import no_network  # noqa: F401  (autouse: no socket connects)


@pytest.fixture
def king(tmp_path):
    hotkey = h.fixture_key("king").ss58_address
    vault, digest, files = h.vault_with_king(tmp_path, hotkey)
    return hotkey, vault, digest, files


def _publish(king, mirrors, alerts=None):
    hotkey, vault, digest, _ = king
    sink = alerts if alerts is not None else []
    return h.publish().publish_king(hotkey=hotkey, digest=digest, vault=vault,
                                    mirrors=mirrors, alert=sink.append)


def test_publish_writes_every_mirror_and_each_hashes_to_the_digest(king):
    _, _, digest, files = king
    mirrors = h.mirrors()
    out = _publish(king, mirrors)
    assert out["digest"] == digest
    assert out["mirrors"] == {name: "ok" for name in h.MIRRORS}
    from cascade.lane_d.intake import manifest_digest

    for name, mirror in mirrors.items():
        manifest = json.loads(mirror.get(mirror.key_ending(digest, "manifest.json")))
        assert manifest_digest(manifest) == digest, name
        assert set(manifest["files"]) == set(h.CHECKPOINT_FILES), name
        for fname, data in files.items():
            got = mirror.get(mirror.key_ending(digest, fname))
            assert got == data, (name, fname)
            assert h.sha256(got) == manifest["files"][fname]["sha256"], (name, fname)


def test_mirror_paths_are_content_addressed(king):
    """Every object a publish writes carries the digest as a path component,
    and nothing else is written (the manifest files plus manifest.json)."""
    _, _, digest, _ = king
    mirrors = h.mirrors()
    _publish(king, mirrors)
    want = {*h.CHECKPOINT_FILES, "manifest.json"}
    for name, mirror in mirrors.items():
        written = {k for op, k in mirror.calls if op == "put"}
        assert written, name
        assert all(digest in k.split("/") for k in written), (name, sorted(written))
        assert {k.rsplit("/", 1)[-1] for k in written} == want, (name, sorted(written))
        assert set(mirror.objects) == written, name


def test_verify_mirrors_reports_ok_after_publish(king):
    _, _, digest, _ = king
    mirrors = h.mirrors()
    _publish(king, mirrors)
    assert h.publish().verify_mirrors(digest, mirrors) == {name: "ok" for name in h.MIRRORS}


@pytest.mark.parametrize("bad", h.MIRRORS)
def test_overwritten_weights_on_any_one_mirror_fail_the_rehash(king, bad):
    """No lock: an overwrite in place is possible, so every read re-hashes.
    A changed byte on one mirror is ``digest_mismatch`` there and only there."""
    _, _, digest, _ = king
    mirrors = h.mirrors()
    _publish(king, mirrors)
    m = mirrors[bad]
    key = m.key_ending(digest, "weights.safetensors")
    m.objects[key] = b"\x00" + m.objects[key][1:]
    got = h.publish().verify_mirrors(digest, mirrors)
    assert got[bad] == "digest_mismatch"
    assert {k: v for k, v in got.items() if k != bad} == {k: "ok" for k in h.MIRRORS if k != bad}


def test_a_consistently_rewritten_manifest_is_still_a_mismatch(king):
    """An attacker who rewrites weights AND manifest.json consistently still
    fails: the manifest no longer hashes to the digest in the path."""
    _, _, digest, _ = king
    mirrors = h.mirrors()
    _publish(king, mirrors)
    m = mirrors["hf"]
    wkey = m.key_ending(digest, "weights.safetensors")
    mkey = m.key_ending(digest, "manifest.json")
    forged = b"forged weights"
    m.objects[wkey] = forged
    manifest = json.loads(m.objects[mkey])
    manifest["files"]["weights.safetensors"] = {"sha256": h.sha256(forged), "size": len(forged)}
    m.objects[mkey] = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert h.publish().verify_mirrors(digest, mirrors)["hf"] == "digest_mismatch"


def test_a_missing_object_is_reported_missing(king):
    _, _, digest, _ = king
    mirrors = h.mirrors()
    _publish(king, mirrors)
    m = mirrors["r2"]
    del m.objects[m.key_ending(digest, "model.py")]
    got = h.publish().verify_mirrors(digest, mirrors)
    assert got["r2"] in ("missing", "digest_mismatch")
    assert got["hub"] == got["hf"] == "ok"
    empty = h.publish().verify_mirrors(digest, h.mirrors())
    assert empty == {name: "missing" for name in h.MIRRORS}


@pytest.mark.parametrize("bad", h.MIRRORS)
def test_publish_refuses_and_alerts_when_a_mirror_stores_other_bytes(king, bad):
    """The publish reads every mirror back; a mirror holding other bytes
    than the vault's is refused (``mirror_digest_mismatch``) and alerted."""
    _, _, digest, _ = king
    mirrors = h.mirrors(**{bad: h.CorruptingMirror(bad)})
    alerts = []
    with pytest.raises(h.publish().PublishError) as e:
        _publish(king, mirrors, alerts)
    assert e.value.code == "mirror_digest_mismatch"
    assert any(a.get("code") == "mirror_digest_mismatch" and a.get("mirror") == bad
               and a.get("digest") == digest for a in alerts), alerts


@pytest.mark.parametrize("missing", h.MIRRORS)
def test_publish_needs_all_three_mirrors(king, missing):
    """§11.1: Hub, R2 and HF. A publish with one mirror absent writes
    nothing anywhere and is refused ``mirror_missing``."""
    mirrors = h.mirrors()
    present = {k: v for k, v in mirrors.items() if k != missing}
    with pytest.raises(h.publish().PublishError) as e:
        _publish(king, present)
    assert e.value.code == "mirror_missing"
    assert all(not m.objects for m in present.values())


def test_publish_refuses_a_vault_entry_that_fails_its_rehash(king):
    """The source is re-hashed too: a vault object overwritten after ready is
    refused (``vault_digest_mismatch``), alerted, and nothing is mirrored."""
    hotkey, vault, digest, _ = king
    vault.put(h.vault_key(hotkey, digest, "weights.safetensors"), b"overwritten")
    mirrors, alerts = h.mirrors(), []
    with pytest.raises(h.publish().PublishError) as e:
        _publish(king, mirrors, alerts)
    assert e.value.code == "vault_digest_mismatch"
    assert any(a.get("code") == "vault_digest_mismatch" for a in alerts), alerts
    assert all(not m.objects for m in mirrors.values())


def test_publish_is_idempotent_and_deterministic(king, tmp_path):
    _, _, digest, _ = king
    first = h.mirrors()
    _publish(king, first)
    snapshot = {n: dict(m.objects) for n, m in first.items()}
    again = _publish(king, first)
    assert again["mirrors"] == {name: "ok" for name in h.MIRRORS}
    assert {n: dict(m.objects) for n, m in first.items()} == snapshot
    other = h.mirrors()
    _publish(king, other)
    assert {n: dict(m.objects) for n, m in other.items()} == snapshot


def test_publish_does_not_touch_the_vault(king, tmp_path):
    _, vault, _, _ = king
    before = h.tree(tmp_path / "vault")
    _publish(king, h.mirrors())
    assert h.tree(tmp_path / "vault") == before


def test_hub_repo_and_hf_mirror_names():
    """§11.1: ``{ns}/laned-king-c{context}-{hotkey8}`` and the HF dataset."""
    pub = h.publish()
    hotkey = h.fixture_key("king").ss58_address
    assert pub.hub_repo("tensorlink", 8192, hotkey) == f"tensorlink/laned-king-c8192-{hotkey[:8]}"
    assert pub.hub_repo("ns", 4096, hotkey) == f"ns/laned-king-c4096-{hotkey[:8]}"
    assert pub.HF_MIRROR_REPO == h.HF_REPO
    assert tuple(pub.MIRRORS) == h.MIRRORS
