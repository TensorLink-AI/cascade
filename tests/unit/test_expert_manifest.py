"""DEC-CA-0050: the lane-B expert manifest and the lane-B commitment."""

from __future__ import annotations

import pytest

from cascade.interface.validation import parse_commit
from cascade.shared.expert_manifest import (
    ExpertEntry,
    ExpertManifest,
    expert_manifest_key,
    format_expert_commit,
    load_expert_manifest,
    parse_expert_commit,
    sign_expert_manifest,
)
from cascade.shared.manifest import format_trained_pointer

GEN = "alice/metro-gen@sha256:" + "a" * 64
K0 = format_trained_pointer("cascade/ckpt-king0@sha256:" + "b" * 64)
K1 = format_trained_pointer("cascade/ckpt-king1@sha256:" + "c" * 64)
RB = format_trained_pointer("cascade/ckpt-rebased@sha256:" + "d" * 64)


def _m(**kw):
    e = ExpertEntry(miner_hotkey="hk", miner_uid=5, gen_ref=GEN, domain="energy",
                    selection={1: [9, 3], 0: [4]}, exclude={0: [7]}, base_king_pointer=K0,
                    rebased_pointer=RB, corpus_digest="x", train_block=10, commit_block=8)
    return ExpertManifest(round_id="r1", created_block=100, king_pointer=K1,
                          lineage_tokens=2 * 10**11, entries=(e,), **kw)


def test_round_trip_and_canonical_order():
    m = _m()
    back = load_expert_manifest(m.to_json())
    assert back.canonical_body() == m.canonical_body()
    assert back.entries[0].selection == {0: [4], 1: [3, 9]}
    assert b'"selection":{"0":[4],"1":[3,9]}' in m.canonical_body()
    assert expert_manifest_key("r1") == "lane_b/round-r1.json"


def test_signature_covers_body():
    class _K:
        def sign(self, b):
            return b"sig:" + b[:4]

    s = sign_expert_manifest(_m(), _K())
    assert s.signature and s.digest() == _m().digest()


def test_entry_guards():
    with pytest.raises(ValueError):
        ExpertEntry(miner_hotkey="h", miner_uid=1, gen_ref=GEN, domain="", selection={},
                    exclude={}, base_king_pointer=K0, rebased_pointer=RB, corpus_digest="",
                    train_block=0, commit_block=0)
    with pytest.raises(ValueError):
        ExpertEntry(miner_hotkey="h", miner_uid=1, gen_ref=GEN, domain="web", selection={},
                    exclude={}, base_king_pointer="nope", rebased_pointer=RB,
                    corpus_digest="", train_block=0, commit_block=0)


def test_expert_commit_is_invisible_to_lane_a():
    p = format_expert_commit("energy", GEN)
    assert parse_expert_commit(p) == ("energy", GEN)
    assert parse_commit(p) is None
    assert parse_expert_commit("metro-v1:gen:hippius:" + GEN) is None
    with pytest.raises(ValueError):
        format_expert_commit("Energy Domain", GEN)
