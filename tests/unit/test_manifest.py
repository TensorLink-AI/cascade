"""Training manifest: digests, trained pointers, and JSON round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from metronome.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    corpus_digest,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
    parse_trained_pointer,
)

SHA = "abc123def456abc123def456abc123def456abcd"
CID = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
CID_T = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"


def test_trained_pointer_round_trip():
    p = format_trained_pointer(CID_T)
    assert p == f"metro-v1:trained:hippius:{CID_T}"
    assert parse_trained_pointer(p) == CID_T
    assert parse_trained_pointer(f"metro-v1:gen:hippius:{CID_T}") is None
    assert parse_trained_pointer("metro-v1:trained:hippius:not-a-cid") is None


def test_corpus_digest_is_order_and_value_sensitive():
    a = [np.zeros(10), np.ones(20)]
    b = [np.ones(20), np.zeros(10)]
    assert corpus_digest(a) == corpus_digest([np.zeros(10), np.ones(20)])
    assert corpus_digest(a) != corpus_digest(b)
    assert corpus_digest(a) != corpus_digest([np.zeros(10), np.ones(20) + 1e-6])


def test_contract_digest_stable_for_dict():
    d1 = {"epochs": 3, "lr": 1e-4}
    d2 = {"lr": 1e-4, "epochs": 3}
    assert contract_digest(d1) == contract_digest(d2)
    assert contract_digest(d1) != contract_digest({"epochs": 4, "lr": 1e-4})


def _entry(role, uid):
    return TrainedEntry(
        miner_hotkey=f"hk{uid}",
        miner_uid=uid,
        role=role,
        gen_cid=CID,
        trained_pointer=format_trained_pointer(CID_T),
        corpus_digest="deadbeef",
        train_block=100,
        tar_digest="cafe",
    )


def test_entry_rejects_bad_role_and_pointer():
    with pytest.raises(ValueError):
        TrainedEntry("hk", 0, "emperor", CID, format_trained_pointer(CID_T), "d", 1)
    with pytest.raises(ValueError):
        TrainedEntry("hk", 0, "king", CID, "not-a-pointer", "d", 1)


def test_manifest_round_trip_and_role_lookup():
    m = TrainingManifest(
        round_id="42",
        created_block=1000,
        contract_digest=contract_digest({"epochs": 3}),
        base_arch_digest="a" * 64,
        eval_dataset="gift-eval",
        entries=[_entry("king", 0), _entry("challenger", 1)],
        signature="sig",
    )
    again = load_manifest(dump_manifest(m))
    assert again.round_id == "42"
    assert again.entry_for_role("king").miner_uid == 0
    assert again.entry_for_role("challenger").miner_uid == 1
    # canonical body excludes the signature and is stable.
    assert again.canonical_body() == m.canonical_body()
