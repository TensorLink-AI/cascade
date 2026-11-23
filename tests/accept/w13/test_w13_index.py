"""W13: the signed ``champions/lane_d/index.json`` and the repo cards
``champions/lane_d/<digest>/{PROVENANCE.json, MODEL_CARD.md}`` (plan §11.1).
README.md has the contract.

On main ``cascade.lane_d.publish`` does not exist: every test fails in its
body with ModuleNotFoundError for that PATHS module.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from . import helpers as h
from .helpers import no_network  # noqa: F401  (autouse: no socket connects)


@pytest.fixture(scope="module")
def signer():
    return h.fixture_key("signer")


@pytest.fixture
def kings():
    a = h.fixture_key("king-a").ss58_address
    b = h.fixture_key("king-b").ss58_address
    return [h.king_record(a, h.sha256(b"w13 king a"), crown_block=9_000_900, context=8192),
            h.king_record(b, h.sha256(b"w13 king b"), crown_block=9_001_800, context=16384,
                          lcb=0.0311)]


# ── the signed index ────────────────────────────────────────────────────────


def test_index_carries_every_king_and_every_field(kings, signer):
    index = h.publish().build_index(kings, signer)
    assert index["schema"] == h.INDEX_SCHEMA
    assert index["signer"] == signer.ss58_address
    assert len(index["kings"]) == len(kings)
    by_digest = {k["digest"]: k for k in index["kings"]}
    for want in kings:
        got = by_digest[want["digest"]]
        assert {f: got[f] for f in h.KING_FIELDS} == {f: want[f] for f in h.KING_FIELDS}
        assert got["sentinel"]["suite"] == want["sentinel"]["suite"]


def test_index_verifies_under_the_pinned_signer(kings, signer):
    pub = h.publish()
    index = pub.build_index(kings, signer)
    assert pub.verify_index(index, signer.ss58_address) is True


@pytest.mark.parametrize("tamper", [
    lambda ix: ix["kings"][0].__setitem__("lcb", 0.5),
    lambda ix: ix["kings"][0].__setitem__("digest", h.sha256(b"another king")),
    lambda ix: ix["kings"][1]["sentinel"].__setitem__("suite", "boom"),
    lambda ix: ix["kings"][1].__setitem__("license", "proprietary"),
    lambda ix: ix["kings"].pop(),
    lambda ix: ix.pop("signature"),
    lambda ix: ix.__setitem__("signature", "00" * 64),
], ids=["lcb", "digest", "sentinel", "license", "dropped-king", "unsigned", "zero-sig"])
def test_a_tampered_or_unsigned_index_does_not_verify(kings, signer, tamper):
    pub = h.publish()
    index = copy.deepcopy(pub.build_index(kings, signer))
    tamper(index)
    assert pub.verify_index(index, signer.ss58_address) is False


def test_an_index_signed_by_another_key_does_not_verify(kings, signer):
    pub = h.publish()
    other = h.fixture_key("not-the-validator")
    forged = pub.build_index(kings, other)
    assert pub.verify_index(forged, other.ss58_address) is True
    assert pub.verify_index(forged, signer.ss58_address) is False
    relabelled = dict(forged, signer=signer.ss58_address)
    assert pub.verify_index(relabelled, signer.ss58_address) is False


@pytest.mark.parametrize("bad", [None, [], "index", 7])
def test_verify_index_is_false_on_non_index_input(signer, bad):
    assert h.publish().verify_index(bad, signer.ss58_address) is False


def test_index_body_is_deterministic_and_in_crown_order(kings, signer):
    """Two builds (input in either order) give the same canonical body; the
    kings are listed by crown block (digest breaks a tie)."""
    pub = h.publish()
    one = pub.build_index(kings, signer)
    two = pub.build_index(list(reversed(kings)), signer)
    assert h.body(one) == h.body(two)
    blocks = [(k["crown_block"], k["digest"]) for k in one["kings"]]
    assert blocks == sorted(blocks)


def test_build_index_does_not_mutate_its_input(kings, signer):
    before = copy.deepcopy(kings)
    h.publish().build_index(kings, signer)
    assert kings == before


@pytest.mark.parametrize("field", h.KING_FIELDS)
def test_a_king_without_a_required_field_is_refused(kings, signer, field):
    pub = h.publish()
    broken = copy.deepcopy(kings)
    del broken[0][field]
    with pytest.raises(pub.PublishError) as e:
        pub.build_index(broken, signer)
    assert e.value.code == "index_field_missing"


def test_a_sentinel_without_its_suite_is_refused(kings, signer):
    """§11.1 (rev5): sentinel numbers with the suite named."""
    pub = h.publish()
    broken = copy.deepcopy(kings)
    del broken[0]["sentinel"]["suite"]
    with pytest.raises(pub.PublishError) as e:
        pub.build_index(broken, signer)
    assert e.value.code == "index_field_missing"


def test_one_digest_appears_once(kings, signer):
    pub = h.publish()
    with pytest.raises(pub.PublishError) as e:
        pub.build_index([kings[0], dict(kings[1], digest=kings[0]["digest"])], signer)
    assert e.value.code == "index_duplicate_digest"


@pytest.mark.parametrize("digest", ["ABC", "0" * 63, "g" * 64, "0x" + "0" * 64])
def test_a_malformed_digest_is_refused(kings, signer, digest):
    pub = h.publish()
    with pytest.raises(pub.PublishError) as e:
        pub.build_index([dict(kings[0], digest=digest)], signer)
    assert e.value.code == "index_field_invalid"


def test_a_non_finite_lcb_is_refused(kings, signer):
    pub = h.publish()
    with pytest.raises(ValueError):
        pub.build_index([dict(kings[0], lcb=math.nan)], signer)


def test_the_index_claims_no_lock(kings, signer):
    """rev6.4: there is no Object Lock; the index says nothing of one."""
    text = json.dumps(h.publish().build_index(kings, signer), sort_keys=True)
    assert not h.LOCK_CLAIM.search(text), h.LOCK_CLAIM.search(text)


# ── cards and the repo copy (laned-king-sync) ──────────────────────────────


def test_sync_cards_unchanged_when_unarmed(tmp_path, signer):
    """Invariant 1: lane D unarmed means no published index; the sync writes
    nothing and the repo tree is byte-identical."""
    (tmp_path / "champions" / "king").mkdir(parents=True)
    (tmp_path / "champions" / "king" / "PROVENANCE.json").write_text("{}")
    before = h.tree(tmp_path)
    assert list(h.publish().sync_cards(tmp_path, None, signer.ss58_address)) == []
    assert h.tree(tmp_path) == before
    assert not (tmp_path / "champions" / "lane_d").exists()


def test_sync_cards_armed_nonempty_lane_d_effect(tmp_path, kings, signer):
    pub = h.publish()
    index = pub.build_index(kings, signer)
    written = pub.sync_cards(tmp_path, index, signer.ss58_address)
    assert written
    got = h.tree(tmp_path / "champions" / "lane_d")
    want = {"index.json"} | {f"{k['digest']}/{c}" for k in kings for c in h.CARD_FILES}
    assert set(got) == want
    assert json.loads(got["index.json"]) == index
    assert pub.verify_index(json.loads(got["index.json"]), signer.ss58_address) is True


def test_cards_carry_the_king_record(tmp_path, kings, signer):
    pub = h.publish()
    pub.sync_cards(tmp_path, pub.build_index(kings, signer), signer.ss58_address)
    for king in kings:
        d = tmp_path / "champions" / "lane_d" / king["digest"]
        prov = json.loads((d / "PROVENANCE.json").read_text(encoding="utf-8"))
        for f in ("hotkey", "context", "digest", "crown_block", "lcb", "license"):
            assert prov[f] == king[f], f
        assert prov["hf_repo"] == h.HF_REPO
        card = (d / "MODEL_CARD.md").read_text(encoding="utf-8")
        assert king["digest"] in card and king["license"] in card


def test_no_weights_in_git(tmp_path, kings, signer):
    """§11.1: cards only, no weights in git: every file the sync writes is
    index.json, PROVENANCE.json or MODEL_CARD.md under champions/lane_d/."""
    pub = h.publish()
    pub.sync_cards(tmp_path, pub.build_index(kings, signer), signer.ss58_address)
    for rel in h.tree(tmp_path):
        assert rel.startswith("champions/lane_d/"), rel
        assert rel.rsplit("/", 1)[-1] in ("index.json", *h.CARD_FILES), rel


def test_cards_claim_no_lock(tmp_path, kings, signer):
    pub = h.publish()
    pub.sync_cards(tmp_path, pub.build_index(kings, signer), signer.ss58_address)
    for rel, data in h.tree(tmp_path).items():
        hit = h.LOCK_CLAIM.search(data.decode("utf-8"))
        assert not hit, (rel, hit)


@pytest.mark.parametrize("tamper", ["lcb", "unsigned", "other-key"])
def test_sync_refuses_an_index_that_does_not_verify(tmp_path, kings, signer, tamper):
    pub = h.publish()
    index = copy.deepcopy(pub.build_index(kings, signer))
    if tamper == "lcb":
        index["kings"][0]["lcb"] = 9.9
    elif tamper == "unsigned":
        index.pop("signature")
    else:
        index = pub.build_index(kings, h.fixture_key("not-the-validator"))
    with pytest.raises(pub.PublishError) as e:
        pub.sync_cards(tmp_path, index, signer.ss58_address)
    assert e.value.code == "index_signature_invalid"
    assert h.tree(tmp_path) == {}


def test_sync_is_byte_deterministic(tmp_path, kings, signer):
    pub = h.publish()
    index = pub.build_index(kings, signer)
    pub.sync_cards(tmp_path / "one", index, signer.ss58_address)
    pub.sync_cards(tmp_path / "two", index, signer.ss58_address)
    pub.sync_cards(tmp_path / "two", index, signer.ss58_address)
    assert h.tree(tmp_path / "one") == h.tree(tmp_path / "two")
