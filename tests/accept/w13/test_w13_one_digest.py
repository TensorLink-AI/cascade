"""W13: the one-digest check (§11.1 "One digest, one identity: vault, public
repo, index and report must agree") and the (rev5, B13) takedown path for the
Hub and HF mirrors. README.md has the contract.

On main ``cascade.lane_d.publish`` does not exist: every test fails in its
body with ModuleNotFoundError for that PATHS module.
"""

from __future__ import annotations

import copy
import json

import pytest

from . import helpers as h
from .helpers import no_network  # noqa: F401  (autouse: no socket connects)

CROWN = 9_000_900


@pytest.fixture(scope="module")
def signer():
    return h.fixture_key("signer")


def build_world(tmp_path, signer):
    """A published king: vault entry, three mirrors, signed index, report.
    Called in each test body (never a fixture), so on main the missing
    module fails inside the test."""
    pub = h.publish()
    hotkey = h.fixture_key("king").ss58_address
    vault, digest, _ = h.vault_with_king(tmp_path, hotkey)
    mirrors = h.mirrors()
    pub.publish_king(hotkey=hotkey, digest=digest, vault=vault, mirrors=mirrors, alert=[].append)
    index = pub.build_index([h.king_record(hotkey, digest, crown_block=CROWN)], signer)
    report = {"crowned": digest, "boundary": CROWN}
    return {"pub": pub, "hotkey": hotkey, "vault": vault, "digest": digest,
            "mirrors": mirrors, "index": index, "report": report}


def _check(w, **override):
    args = {k: w[k] for k in ("index", "report", "vault", "mirrors")}
    args.update(override)
    return list(w["pub"].check_one_digest(**args))


def test_everything_agrees(tmp_path, signer):
    world = build_world(tmp_path, signer)
    assert _check(world) == []


def test_report_crown_not_in_the_index(tmp_path, signer):
    world = build_world(tmp_path, signer)
    report = dict(world["report"], crowned=h.sha256(b"some other king"))
    assert "index_missing" in _check(world, report=report)


def test_report_with_no_crown(tmp_path, signer):
    world = build_world(tmp_path, signer)
    assert "report_no_crown" in _check(world, report=dict(world["report"], crowned=None))


def test_crown_block_must_agree(tmp_path, signer):
    world = build_world(tmp_path, signer)
    assert "crown_block_mismatch" in _check(world, report=dict(world["report"], boundary=CROWN + 900))


def test_vault_rehash_failure(tmp_path, signer):
    world = build_world(tmp_path, signer)
    w = world
    w["vault"].put(h.vault_key(w["hotkey"], w["digest"], "weights.safetensors"), b"swapped")
    assert "vault_digest_mismatch" in _check(w)


@pytest.mark.parametrize("bad", h.MIRRORS)
def test_any_one_mirror_disagreeing_is_named(tmp_path, signer, bad):
    world = build_world(tmp_path, signer)
    m = world["mirrors"][bad]
    key = m.key_ending(world["digest"], "weights.safetensors")
    m.objects[key] = m.objects[key][::-1]
    problems = _check(world)
    assert f"{bad}_digest_mismatch" in problems
    assert not [p for p in problems if p.split("_", 1)[0] in h.MIRRORS and not p.startswith(bad)]


@pytest.mark.parametrize("gone", h.MIRRORS)
def test_any_one_mirror_missing_is_named(tmp_path, signer, gone):
    world = build_world(tmp_path, signer)
    world["mirrors"][gone].objects.clear()
    assert f"{gone}_missing" in _check(world)


def test_check_is_read_only(tmp_path, signer):
    world = build_world(tmp_path, signer)
    before_vault = h.tree(tmp_path / "vault")
    before = {n: dict(m.objects) for n, m in world["mirrors"].items()}
    index = copy.deepcopy(world["index"])
    marks = {n: len(m.calls) for n, m in world["mirrors"].items()}
    _check(world)
    assert h.tree(tmp_path / "vault") == before_vault
    assert {n: dict(m.objects) for n, m in world["mirrors"].items()} == before
    assert world["index"] == index
    for n, m in world["mirrors"].items():
        ops = {op for op, _ in m.calls[marks[n]:]}
        assert ops <= {"get", "list", "size"}, (n, ops)


# ── takedown (rev5, B13) ────────────────────────────────────────────────────


@pytest.mark.parametrize("reason", ["licence", "rights"])
def test_takedown_unpublishes_hub_and_hf_and_keeps_the_record(tmp_path, signer, reason):
    world = build_world(tmp_path, signer)
    w, pub = world, world["pub"]
    original = copy.deepcopy(w["index"])
    out = pub.takedown(w["index"], digest=w["digest"], reason=reason, block=CROWN + 1800,
                       mirrors=w["mirrors"], signer=signer)
    assert w["index"] == original, "takedown must not mutate its input"
    for name in ("hub", "hf"):
        assert w["mirrors"][name].keys_with(w["digest"]) == [], name
    got = pub.verify_mirrors(w["digest"], w["mirrors"])
    assert got["hub"] == got["hf"] == "missing"
    assert pub.verify_index(out, signer.ss58_address) is True
    (entry,) = [k for k in out["kings"] if k["digest"] == w["digest"]]
    (before,) = original["kings"]
    assert {f: entry[f] for f in h.KING_FIELDS} == {f: before[f] for f in h.KING_FIELDS}
    assert entry["takedown"] == {"reason": reason, "block": CROWN + 1800}


def test_takedown_keeps_the_vault(tmp_path, signer):
    world = build_world(tmp_path, signer)
    before = h.tree(tmp_path / "vault")
    world["pub"].takedown(world["index"], digest=world["digest"], reason="rights",
                          block=CROWN + 1, mirrors=world["mirrors"], signer=signer)
    assert h.tree(tmp_path / "vault") == before


def test_a_taken_down_king_still_passes_the_one_digest_check(tmp_path, signer):
    world = build_world(tmp_path, signer)
    """The digest and verdict record stay; only the Hub/HF copies are gone,
    which is not a disagreement for a king the index marks taken down."""
    w = world
    out = w["pub"].takedown(w["index"], digest=w["digest"], reason="licence", block=CROWN + 1,
                            mirrors=w["mirrors"], signer=signer)
    problems = _check(w, index=out)
    assert not [p for p in problems if p.startswith(("hub_", "hf_", "vault_", "index_", "report_"))]
    assert "crown_block_mismatch" not in problems


def test_takedown_of_an_unknown_digest_is_refused(tmp_path, signer):
    world = build_world(tmp_path, signer)
    pub = world["pub"]
    before = {n: dict(m.objects) for n, m in world["mirrors"].items()}
    with pytest.raises(pub.PublishError) as e:
        pub.takedown(world["index"], digest=h.sha256(b"never crowned"), reason="rights",
                     block=CROWN + 1, mirrors=world["mirrors"], signer=signer)
    assert e.value.code == "unknown_king"
    assert {n: dict(m.objects) for n, m in world["mirrors"].items()} == before


def test_takedown_needs_a_reason(tmp_path, signer):
    world = build_world(tmp_path, signer)
    pub = world["pub"]
    with pytest.raises(ValueError):
        pub.takedown(world["index"], digest=world["digest"], reason="", block=CROWN + 1,
                     mirrors=world["mirrors"], signer=signer)


def test_takedown_cards_keep_the_record(tmp_path, signer):
    world = build_world(tmp_path, signer)
    w, pub = world, world["pub"]
    out = pub.takedown(w["index"], digest=w["digest"], reason="licence", block=CROWN + 1,
                       mirrors=w["mirrors"], signer=signer)
    repo = tmp_path / "repo"
    pub.sync_cards(repo, out, signer.ss58_address)
    prov = json.loads((repo / "champions" / "lane_d" / w["digest"] / "PROVENANCE.json")
                      .read_text(encoding="utf-8"))
    assert prov["digest"] == w["digest"]
    assert prov["takedown"] == {"reason": "licence", "block": CROWN + 1}
