"""W3b2 (RV5): the owner's live verify of the pod evidence.

``python -P -m cascade.laned.bench_evidence verify --evidence evidence/w3b2.json``
(owner, read-only chain + registry access) re-reads ``draw.block``'s hash from
the chain, checks the block is finalized, re-derives the gate suite with
W3b1's ``gate_suite_for`` under the shipped ``chain.toml``, checks KING_D and
the house refs against the pointers in force at that block, resolves every
ref in the checkpoint registry, and only then writes a ``verified_live``
record bound to the evidence body by ``body_sha256``.

Pinned surface: ``bench_evidence.verify_live(evidence, *, chain, registry,
pointers, now=None)`` returns the record or raises
``bench_evidence.VerifyRefused`` (``.code``). The adapters are the external
services, faked here (``evidence_lib.FakeChain`` / ``FakeRegistry`` /
``FakePointers``); the in-process tests refuse every socket connect and make
bittensor/boto3 unimportable. Contract: README.md.
"""

from __future__ import annotations

import copy
import json

import pytest

from .evidence_lib import (
    MODULE,
    NETWORK,
    REFS,
    ROLES,
    VERIFY_SCHEMA,
    FakeChain,
    FakePointers,
    FakeRegistry,
    bench_evidence_module,
    body_sha256,
    evidence_problems,
    live_world,
    ref_for,
    run_cli,
    synthetic_body,
    verified_live_problems,
    write_json,
)


def _refused(be, ev, world, code):
    chain, registry, pointers = world
    with pytest.raises(be.VerifyRefused) as ei:
        be.verify_live(ev, chain=chain, registry=registry, pointers=pointers)
    assert getattr(ei.value, "code", None) == code, (getattr(ei.value, "code", None),
                                                     str(ei.value))
    for fake in world:
        assert fake.forbidden == [], f"verify must use reads only; it touched {fake.forbidden}"
    return ei.value


def _verify(be, ev):
    chain, registry, pointers = live_world(ev)
    rec = be.verify_live(ev, chain=chain, registry=registry, pointers=pointers)
    return rec, (chain, registry, pointers)


# ── a world that really holds the evidence's facts ───────────────────────────


def test_verify_against_the_real_facts_is_accepted(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    before = copy.deepcopy(ev)
    rec, world = _verify(be, ev)
    assert ev == before, "verify_live must not mutate the evidence"
    assert isinstance(rec, dict) and rec.get("verified") is True, rec
    assert rec.get("schema") == VERIFY_SCHEMA
    assert rec.get("body_sha256") == body_sha256(ev)
    assert json.loads(json.dumps(rec, allow_nan=False)) == rec, "the record is strict JSON"
    ev["verified_live"] = rec
    assert verified_live_problems(ev, check_git=False) == [], verified_live_problems(
        ev, check_git=False)
    assert evidence_problems(ev, check_git=False) == []
    assert rec["tool"]["module"] == MODULE
    for fake in world:
        assert fake.forbidden == [], fake.forbidden


def test_verify_reads_the_block_the_pointers_and_every_ref(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body(house_is_king=False)
    rec, (chain, registry, pointers) = _verify(be, ev)
    blk = ev["draw"]["block"]
    assert ("block_hash", blk) in chain.calls, chain.calls
    assert ("finalized_block",) in chain.calls, chain.calls
    assert ("at_block", blk) in pointers.calls, pointers.calls
    assert {c[1] for c in registry.calls if c[0] == "resolve"} >= set(ev["refs"].values())
    assert rec["pointers"] == {r: ev["refs"][r] for r in REFS}
    assert rec["resolved"] == {r: "sha256:" + ev["rows"][r]["digest"] for r in ROLES}
    assert rec["block_hash"] == ev["draw"]["block_hash"]
    assert rec["finalized_block"] >= blk


# ── refusals ────────────────────────────────────────────────────────────────


def test_fixture_source_is_refused_before_any_read(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    ev["source"] = "fixture"
    world = live_world(ev)
    _refused(be, ev, world, "not_pod")
    assert all(f.calls == [] for f in world), [f.calls for f in world]


@pytest.mark.parametrize("edit", [lambda d: d.update(network="test"),
                                  lambda d: d.update(netuid=1)],
                         ids=["network", "netuid"])
def test_another_network_is_refused_before_any_read(monkeypatch, edit):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    edit(ev["draw"])
    world = live_world(synthetic_body())
    _refused(be, ev, world, "wrong_network")
    assert all(f.calls == [] for f in world)


@pytest.mark.parametrize("edit", [lambda e: e["draw"].update(block_hash="nothex"),
                                  lambda e: e["draw"].update(block="9612000"),
                                  lambda e: e.pop("refs"),
                                  lambda e: e["refs"].update(house="no-digest-ref")],
                         ids=["block_hash", "block", "refs", "ref_format"])
def test_malformed_evidence_is_refused_before_any_read(monkeypatch, edit):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    world = live_world(copy.deepcopy(ev))
    edit(ev)
    _refused(be, ev, world, "malformed")
    assert all(f.calls == [] for f in world)


def test_an_invented_block_hash_is_refused(monkeypatch):
    """The forger's move: pick any hash whose draw is the suite they swept."""
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    chain, registry, pointers = live_world(ev)
    chain.hashes[ev["draw"]["block"]] = "0x" + "5" * 64
    _refused(be, ev, (chain, registry, pointers), "block_hash_mismatch")


def test_an_unreadable_block_is_refused(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    chain, registry, pointers = live_world(ev)
    chain.hashes.clear()
    _refused(be, ev, (chain, registry, pointers), "block_hash_mismatch")


def test_an_unfinalized_block_is_refused(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    chain, registry, pointers = live_world(ev)
    chain.finalized = ev["draw"]["block"] - 1
    _refused(be, ev, (chain, registry, pointers), "block_not_finalized")


def test_a_suite_that_is_not_the_draw_of_the_real_hash_is_refused(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body(suite="gift")
    other = synthetic_body(suite="boom")
    chain, registry, pointers = live_world(ev)
    chain.hashes[ev["draw"]["block"]] = other["draw"]["block_hash"]
    ev["draw"]["block_hash"] = other["draw"]["block_hash"]
    _refused(be, ev, (chain, registry, pointers), "draw_mismatch")


@pytest.mark.parametrize("ref", REFS)
def test_a_reference_that_is_not_the_pointer_at_the_block_is_refused(monkeypatch, ref):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body(house_is_king=False)
    chain, registry, pointers = live_world(ev)
    pointers.table[ev["draw"]["block"]][ref] = ref_for("f" * 64)
    _refused(be, ev, (chain, registry, pointers), "pointer_mismatch")


@pytest.mark.parametrize("role", ROLES)
def test_an_unpublished_checkpoint_is_refused(monkeypatch, role):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body(house_is_king=False)
    chain, _, pointers = live_world(ev)
    registry = FakeRegistry([v for r, v in ev["refs"].items() if r != role])
    _refused(be, ev, (chain, registry, pointers), "ref_unpublished")


def test_the_record_binds_the_body_so_a_later_edit_fails(monkeypatch):
    be = bench_evidence_module(monkeypatch)
    ev = synthetic_body()
    rec, _ = _verify(be, ev)
    ev["verified_live"] = rec
    ev["gate"]["lcb"]["house"]["crps"] -= 0.05
    assert any("body_sha256" in p for p in verified_live_problems(ev, check_git=False))


def test_fakes_are_chain_client_shaped():
    """The adapters the CLI builds must offer the same reads."""
    c = FakeChain({1: "0x" + "0" * 64}, 1)
    assert c.block_hash(1).startswith("0x") and c.finalized_block() == 1
    assert FakeRegistry([ref_for("a" * 64)]).resolve(ref_for("a" * 64)) == "sha256:" + "a" * 64
    assert FakePointers({1: {"house": "h", "king_d": "k"}}).at_block(1)["house"] == "h"
    assert NETWORK == "finney"


# ── the `verify` CLI refuses before any network use ──────────────────────────


def test_verify_cli_refuses_a_missing_or_non_json_evidence_file(tmp_path):
    proc = run_cli(tmp_path, "verify", "--evidence", str(tmp_path / "absent.json"))
    assert proc.returncode == 2 and "absent.json" in proc.stderr, proc.stderr[-2000:]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    proc = run_cli(tmp_path, "verify", "--evidence", str(bad))
    assert proc.returncode == 2 and "evidence" in proc.stderr.lower(), proc.stderr[-2000:]
    assert bad.read_text() == "{not json"


@pytest.mark.parametrize("edit,code", [
    (lambda e: e.update(source="fixture"), "not_pod"),
    (lambda e: e["draw"].update(network="test"), "wrong_network"),
], ids=["fixture", "testnet"])
def test_verify_cli_refuses_before_any_network_use(tmp_path, edit, code):
    ev = synthetic_body()
    edit(ev)
    f = write_json(tmp_path / "ev.json", ev)
    before = f.read_bytes()
    proc = run_cli(tmp_path, "verify", "--evidence", str(f))
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-2000:])
    assert code in proc.stderr, proc.stderr[-2000:]
    assert f.read_bytes() == before, "a refused verify leaves the evidence untouched"
