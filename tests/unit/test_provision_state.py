"""Rental ledger — crash-safe persistence and orphan reconciliation.

The ledger's job is 'never leak a billing pod': records are written atomically
before a pod is relied on, restarts resume teardown from disk, and reconcile
kills anything live-and-tagged that the ledger does not own."""

from __future__ import annotations

import json

import pytest

from cascade.provision.state import (
    PodInstance,
    RoundState,
    add_instance,
    drop_instances,
    instances_for_stage,
    load_state,
    owned_ids,
    reconcile,
    save_state,
)


def _inst(i="pod-0", stage="heat", provider="lium"):
    return PodInstance(provider=provider, instance_id=i, stage=stage,
                       rented_at_iso="2026-07-13T00:00:00+00:00")


# ── pure transforms ──────────────────────────────────────────────────────────


def test_add_and_drop_are_immutable():
    s0 = RoundState(round_id="900")
    s1 = add_instance(s0, _inst("a"))
    s2 = add_instance(s1, _inst("b", stage="final"))
    assert s0.instances == () and len(s2.instances) == 2     # originals untouched
    s3 = drop_instances(s2, {"a"})
    assert [i.instance_id for i in s3.instances] == ["b"]
    assert len(s2.instances) == 2


def test_instances_for_stage_and_owned_ids():
    s = RoundState(round_id="900", instances=(
        _inst("h0"), _inst("h1"), _inst("f0", stage="final")))
    assert [i.instance_id for i in instances_for_stage(s, "heat")] == ["h0", "h1"]
    assert [i.instance_id for i in instances_for_stage(s, "final")] == ["f0"]
    assert owned_ids(s) == {"h0", "h1", "f0"}


# ── reconcile (orphan detection) ─────────────────────────────────────────────


def test_reconcile_flags_tagged_pods_we_do_not_own():
    # A crash between the provider API call and the ledger save leaves a live,
    # tagged, unowned pod — the kill list.
    assert reconcile({"a"}, {"a", "b", "c"}) == ["b", "c"]   # sorted for stable logs


def test_reconcile_ignores_owned_but_dead_pods():
    # Owned-but-not-live needs no action: terminate is idempotent anyway.
    assert reconcile({"a", "b"}, {"a"}) == []


def test_reconcile_empty_ledger_kills_all_tagged():
    assert reconcile(set(), {"cascade-900-heat-0"}) == ["cascade-900-heat-0"]


# ── disk round trip ──────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "ledger" / "state.json"                # parent auto-created
    s = RoundState(round_id="7200", published=True, instances=(
        _inst("h0"), _inst("f0", stage="final", provider="shadeform")))
    save_state(path, s)
    assert load_state(path) == s
    assert not path.with_suffix(".json.tmp").exists()        # atomic: tmp renamed away


def test_load_missing_is_fresh_start(tmp_path):
    assert load_state(tmp_path / "nope.json") is None


def test_load_corrupt_raises_instead_of_silently_starting_fresh(tmp_path):
    # Starting fresh over an unreadable ledger is exactly how pods leak.
    p = tmp_path / "state.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_state(p)


def test_saved_json_is_stable_and_human_readable(tmp_path):
    p = tmp_path / "state.json"
    save_state(p, RoundState(round_id="900", instances=(_inst("h0"),)))
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["round_id"] == "900" and raw["published"] is False
    assert raw["instances"][0] == {
        "provider": "lium", "instance_id": "h0", "stage": "heat",
        "rented_at_iso": "2026-07-13T00:00:00+00:00",
    }
