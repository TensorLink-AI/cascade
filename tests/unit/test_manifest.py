"""Training manifest: digests, trained pointers, and JSON round-trip."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.shared.manifest import (
    HeatEntrant,
    HeatResult,
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    corpus_digest,
    dump_manifest,
    format_trained_pointer,
    load_manifest,
    parse_trained_pointer,
)

REF = "alice/metro-gen@sha256:" + "a" * 64
REF_T = "cascade/ckpt-r42-king@sha256:" + "b" * 64


def test_trained_pointer_round_trip():
    p = format_trained_pointer(REF_T)
    assert p == f"metro-v1:trained:hippius:{REF_T}"
    assert parse_trained_pointer(p) == REF_T
    assert parse_trained_pointer(f"metro-v1:gen:hippius:{REF_T}") is None
    assert parse_trained_pointer("metro-v1:trained:hippius:not-a-ref") is None


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
        gen_ref=REF,
        trained_pointer=format_trained_pointer(REF_T),
        corpus_digest="deadbeef",
        train_block=100,
    )


def test_entry_rejects_bad_role_and_pointer():
    with pytest.raises(ValueError):
        TrainedEntry("hk", 0, "emperor", REF, format_trained_pointer(REF_T), "d", 1)
    with pytest.raises(ValueError):
        TrainedEntry("hk", 0, "king", REF, "not-a-pointer", "d", 1)


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


def test_heat_is_unsigned_and_round_trips():
    base = TrainingManifest(
        round_id="42", created_block=1000,
        contract_digest=contract_digest({"epochs": 3}), base_arch_digest="a" * 64,
        eval_dataset="gift-eval", entries=[_entry("king", 0), _entry("challenger", 1)],
        signature="sig",
    )
    heat = HeatResult(
        screen_size="toto2-4m", finalists=1,
        entrants=(
            HeatEntrant(uid=1, hotkey="hk1", gen_ref=REF, status="advanced",
                        rank=1, rel_score=1.0),
            HeatEntrant(uid=2, hotkey="hk2", gen_ref=REF, status="screened",
                        rank=2, rel_score=1.08),
            HeatEntrant(uid=3, hotkey="hk3", gen_ref=REF, status="failed_train"),
        ),
    )
    m = replace(base, heat=heat)
    again = load_manifest(dump_manifest(m))
    assert again.heat == heat
    # heat is informational: it must NOT change what the trainer signs.
    assert m.canonical_body() == base.canonical_body()


def test_heat_rejects_unknown_status():
    with pytest.raises(ValueError):
        HeatEntrant(uid=1, hotkey="hk", gen_ref=REF, status="promoted")


def _sized_entry(role, uid, size):
    return TrainedEntry(
        miner_hotkey=f"hk{uid}", miner_uid=uid, role=role, gen_ref=REF,
        trained_pointer=format_trained_pointer(REF_T), corpus_digest="d",
        train_block=100, size=size,
    )


def test_multi_size_entries_round_trip_and_group_by_size():
    # A round with two sizes: a (king, challenger) pair per size, each size-tagged.
    m = TrainingManifest(
        round_id="7", created_block=1, contract_digest=contract_digest({"x": 1}),
        base_arch_digest="a" * 64, eval_dataset="cascade-private-v1",
        entries=[
            _sized_entry("king", 0, "toto2-4m"), _sized_entry("challenger", 1, "toto2-4m"),
            _sized_entry("king", 0, "toto2-22m"), _sized_entry("challenger", 1, "toto2-22m"),
        ],
    )
    again = load_manifest(dump_manifest(m))
    assert again.sizes() == ["toto2-4m", "toto2-22m"]
    assert [e.size for e in again.entries_for_role("king")] == ["toto2-4m", "toto2-22m"]
    assert [e.size for e in again.entries_for_role("challenger")] == ["toto2-4m", "toto2-22m"]
    # size is folded into the signed body (tampering with a size would break the sig)
    assert again.canonical_body() == m.canonical_body()
    assert b"toto2-22m" in m.canonical_body()


def test_legacy_single_size_entry_defaults_to_empty_size():
    e = _entry("king", 0)
    assert e.size == ""
    m = TrainingManifest(
        round_id="1", created_block=1, contract_digest="d", base_arch_digest="a" * 64,
        eval_dataset="x", entries=[e],
    )
    assert m.sizes() == [""]
