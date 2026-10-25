"""DEC-CA-0056 lane-C manifest, the on-chain fine-tune commitment, and the
lane-A manifest's ``lane_c_digest`` binding (old manifests byte-identical)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.interface.validation import parse_commit
from cascade.shared.expert_manifest import parse_expert_commit
from cascade.shared.lane_c_manifest import (
    LaneCEntry,
    LaneCManifest,
    dump_lane_c_manifest,
    finetune_commit_as_of,
    format_finetune_commit,
    lane_c_manifest_key,
    lane_c_text_digest,
    load_lane_c_manifest,
    parse_finetune_commit,
    sign_lane_c_manifest,
    verify_lane_c_manifest,
)
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
)

REF = "alice/recipe@sha256:" + "a" * 64
SU = ("returns", "vol")         # [marketcast] suites admitted


def _ptr(tag: str) -> str:
    return format_trained_pointer(f"cascade/ckpt-{tag}@sha256:"
                                  + hashlib.sha256(tag.encode()).hexdigest())


def _m(**kw) -> LaneCManifest:
    e = LaneCEntry(miner_hotkey="alice", miner_uid=7, target="mkt:returns", recipe_ref=REF,
                   trained_pointer=_ptr("ft"), train_block=100)
    base = dict(round_id="123", era_index=4, created_block=200, base_pointer=_ptr("king"),
                base_round_id="122", pool_block=150, entries=(e,))
    base.update(kw)
    return LaneCManifest(**base)


def test_key_and_round_trip():
    assert lane_c_manifest_key("123") == "lane_c/round-123.json"
    m = _m()
    text = dump_lane_c_manifest(m)
    back = load_lane_c_manifest(text)
    assert back == m and back.digest() == m.digest()
    assert lane_c_text_digest(text) == hashlib.sha256(text.encode()).hexdigest()


def test_optional_entry_fields_drop_when_empty():
    body = json.loads(_m().canonical_body())
    assert "funded_by" not in body["entries"][0] and "public_rows" not in body["entries"][0]
    e = replace(_m().entries[0], funded_by="king", public_rows={"base": {}, "candidate": {}})
    m2 = _m(entries=(e,))
    assert json.loads(m2.canonical_body())["entries"][0]["funded_by"] == "king"
    assert load_lane_c_manifest(dump_lane_c_manifest(m2)) == m2


def test_entry_validation():
    with pytest.raises(ValueError):
        LaneCEntry("alice", 7, "mkt:Bad Suite", REF, _ptr("x"), 1)
    with pytest.raises(ValueError):
        LaneCEntry("alice", 7, "mkt:returns", "not-a-ref", _ptr("x"), 1)
    with pytest.raises(ValueError):
        LaneCEntry("alice", 7, "mkt:returns", REF, "garbage", 1)
    with pytest.raises(ValueError):
        _m(base_pointer="garbage")
    with pytest.raises(ValueError, match="version"):
        load_lane_c_manifest(json.dumps({**json.loads(_m().to_json()), "version": 9}))


def test_signature_covers_the_body():
    class _K:
        def sign(self, b):
            return hashlib.sha256(b).digest()

    s = sign_lane_c_manifest(_m(), _K())
    assert s.signature == hashlib.sha256(_m().canonical_body()).hexdigest()
    assert s.digest() == _m().digest()
    assert not verify_lane_c_manifest(_m(), "5Trainer")          # unsigned
    assert not verify_lane_c_manifest(s, "")                      # no pinned key


def test_real_sr25519_signature():
    sp = pytest.importorskip("bittensor.sp_core")
    kp = sp.Keypair.create_from_uri("//Trainer")
    s = sign_lane_c_manifest(_m(), SimpleNamespace(hotkey=kp))
    assert verify_lane_c_manifest(s, kp.ss58_address)
    tampered = replace(s, base_pointer=_ptr("other"))
    assert not verify_lane_c_manifest(tampered, kp.ss58_address)


def test_commit_format_is_distinct_from_lanes_a_and_b():
    p = format_finetune_commit("mkt:returns", REF, mkt_suites=SU)
    assert p == f"metro-v1:ft:mkt:returns:hippius:{REF}"
    assert parse_finetune_commit(p, mkt_suites=SU) == ("mkt:returns", REF)
    assert parse_commit(p) is None and parse_expert_commit(p) is None
    assert parse_finetune_commit(f"metro-v1:gen:hippius:{REF}", mkt_suites=SU) is None
    assert parse_finetune_commit(f"metro-v1:expert:energy:hippius:{REF}", mkt_suites=SU) is None
    for bad in (f"metro-v1:ft:mkt:Returns:hippius:{REF}", "metro-v1:ft:mkt:returns:hippius:x",
                f"metro-v1:ft:mkt:range:hippius:{REF}"):
        assert parse_finetune_commit(bad, mkt_suites=SU) is None
    with pytest.raises(ValueError):
        format_finetune_commit("mkt:returns", "bad", mkt_suites=SU)


def test_generic_commits_are_refused():
    """MarketCast-only lane C (owner 2026-10-20): the DEC-CA-0056 generic
    targets have no forward-only eval — never a lane-C commit or entry."""
    for t in ("domain:energy", "freq:daily", "horizon:720"):
        assert parse_finetune_commit(f"metro-v1:ft:{t}:hippius:{REF}", mkt_suites=SU) is None
        with pytest.raises(ValueError):
            format_finetune_commit(t, REF, mkt_suites=SU)
        with pytest.raises(ValueError, match="forward-only"):
            LaneCEntry("alice", 7, t, REF, _ptr("x"), 1)


def test_commit_as_of_takes_the_latest_reveal_by_then():
    C = SimpleNamespace
    hist = [C(hotkey="alice", commit_block=10,
              payload=format_finetune_commit("mkt:returns", REF, mkt_suites=SU)),
            C(hotkey="alice", commit_block=50, payload=f"metro-v1:gen:hippius:{REF}"),
            C(hotkey="bob", commit_block=5,
              payload=format_finetune_commit("mkt:vol", REF, mkt_suites=SU))]
    assert finetune_commit_as_of(hist, "alice", 9, mkt_suites=SU) is None
    assert finetune_commit_as_of(hist, "alice", 49, mkt_suites=SU) == ("mkt:returns", REF)
    assert finetune_commit_as_of(hist, "alice", 50, mkt_suites=SU) is None   # a lane-A commit
    assert finetune_commit_as_of(hist, "bob", 100, mkt_suites=SU) == ("mkt:vol", REF)
    assert finetune_commit_as_of(hist, "bob", 100) is None       # MarketCast off: refused


def _lane_a(**kw) -> TrainingManifest:
    return TrainingManifest(
        round_id="9", created_block=7300, contract_digest="1" * 64, base_arch_digest="2" * 64,
        eval_dataset="d",
        entries=[TrainedEntry("K", 0, "king", "k/g@sha256:" + "3" * 64, _ptr("king"),
                              "4" * 64, 7000)], **kw)


def test_lane_a_manifest_without_the_digest_is_byte_identical():
    m = _lane_a()
    assert b"lane_c_digest" not in m.canonical_body()
    assert "lane_c_digest" not in dump_manifest(m)
    assert load_manifest(dump_manifest(m)) == m
    # golden: the body is exactly the pre-field encoding
    want = json.dumps({"manifest_version": m.manifest_version, "round_id": "9",
                       "created_block": 7300, "contract_digest": "1" * 64,
                       "base_arch_digest": "2" * 64, "eval_dataset": "d",
                       "entries": json.loads(m.canonical_body())["entries"]},
                      sort_keys=True, separators=(",", ":")).encode()
    assert m.canonical_body() == want


def test_lane_a_manifest_binds_the_digest_when_set():
    m = _lane_a(lane_c_digest="f" * 64)
    assert json.loads(m.canonical_body())["lane_c_digest"] == "f" * 64
    back = load_manifest(dump_manifest(m))
    assert back.lane_c_digest == "f" * 64 and back.canonical_body() == m.canonical_body()
    assert m.canonical_body() != _lane_a().canonical_body()


def test_trainer_provenance_fields_round_trip():
    """The trainer stamps funded_by="miner", base_round_id = the binding
    lane-A round and pool_block = the boundary; none of them is judged."""
    e = replace(_m().entries[0], funded_by="miner")
    m = _m(entries=(e,), base_round_id="123", pool_block=200)
    back = load_lane_c_manifest(dump_lane_c_manifest(m))
    assert back == m and back.entries[0].funded_by == "miner"
    assert back.base_round_id == back.round_id and back.pool_block == 200
