"""T5/T6: generation VMs are ledgered and reaped, an uploaded corpus is reused,
and a sold-out CPU market is waited out before the leg fails as infra."""

from __future__ import annotations

import json

import pytest

from cascade.generation import stage as stage_mod
from cascade.generation.stage import GenerationInfraError, reap_generation_vms
from tests.unit.test_generation_stage import HK, FakeProvider, FakeStore, _go, _stage


class _FailTerminate(FakeProvider):
    def terminate(self, vm):
        raise RuntimeError("api down")


class _Lister(FakeProvider):
    name = "shadeform"

    def __init__(self, listed):
        super().__init__()
        self.listed = listed

    def list_tagged(self, prefix):
        return [v for n, v in self.listed if n.startswith(prefix)]


def _rows(path):
    return json.loads(path.read_text()) if path.is_file() else []


def test_vm_is_ledgered_during_the_run_and_dropped_after_terminate(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, _, _ = _stage(tmp_path, prov, store)
    st.ledger_path = tmp_path / "vms.json"
    seen = []
    orig = prov.wait_ready
    prov.wait_ready = lambda vm, *, timeout: seen.append(_rows(st.ledger_path)) or orig(
        vm, timeout=timeout)
    _go(st, tmp_path)
    assert seen[0][0]["vm_id"] == "vm-1" and seen[0][0]["provider"] == "targon"
    assert seen[0][0]["name"].startswith("cgen-")
    assert _rows(st.ledger_path) == [] and prov.terminated == ["vm-1"]


def test_failed_terminate_stays_on_the_ledger_and_the_reaper_stops_it(tmp_path):
    prov = _FailTerminate()
    st, _, _ = _stage(tmp_path, prov, FakeStore())
    st.ledger_path = tmp_path / "vms.json"
    _go(st, tmp_path)
    (row,) = _rows(st.ledger_path)
    assert row["vm_id"] == "vm-1"
    good = FakeProvider()
    assert reap_generation_vms(st.ledger_path, [good]) == ["vm-1"]
    assert good.terminated == ["vm-1"] and _rows(st.ledger_path) == []


def test_reaper_skips_live_stages_and_sweeps_unledgered_listed_vms(tmp_path):
    ledger = tmp_path / "vms.json"
    ledger.write_text(json.dumps([
        {"provider": "shadeform", "vm_id": "v-live", "name": "cgen-live", "created": 0},
        {"provider": "shadeform", "vm_id": "v-dead", "name": "cgen-dead", "created": 0},
    ]))
    prov = _Lister([("cgen-live", "v-live"), ("cgen-dead", "v-dead"),
                    ("cgen-orphan", "v-orphan"), ("gpu-pod", "v-gpu")])
    with stage_mod._LIVE_GUARD:
        stage_mod._LIVE.add("cgen-live")
    try:
        reaped = reap_generation_vms(ledger, [prov])
    finally:
        with stage_mod._LIVE_GUARD:
            stage_mod._LIVE.discard("cgen-live")
    assert sorted(reaped) == ["v-dead", "v-orphan"]      # never the live one, never non-cgen
    assert [r["name"] for r in _rows(ledger)] == ["cgen-live"]


def test_reaper_resolves_an_intent_row_by_name_listing(tmp_path):
    ledger = tmp_path / "vms.json"
    ledger.write_text(json.dumps([{"provider": "shadeform", "vm_id": "", "name": "cgen-x",
                                   "created": 0}]))
    prov = _Lister([("cgen-x", "v-x")])
    assert reap_generation_vms(ledger, [prov]) == ["v-x"]
    assert _rows(ledger) == []


def test_reaper_retains_an_unresolvable_targon_intent_row(tmp_path, caplog):
    ledger = tmp_path / "vms.json"
    ledger.write_text(json.dumps([{"provider": "targon", "vm_id": "", "name": "cgen-y",
                                   "created": 0}]))
    prov = FakeProvider()                      # targon: no list call
    assert reap_generation_vms(ledger, [prov]) == []
    assert prov.terminated == []
    assert [r["name"] for r in _rows(ledger)] == ["cgen-y"]
    assert "stop it by hand" in caplog.text


def test_an_uploaded_corpus_is_reused_without_a_vm(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, idx, _ = _stage(tmp_path, prov, store)
    first = _go(st, tmp_path)
    assert prov.launched                        # the first leg made it
    prov2 = FakeProvider()
    st.providers = [prov2]
    again = st.generate(round_id="77", hotkey=HK, uid=4, era=9,
                        gen_ref=idx["generator_ref"], repo_dir=tmp_path / "repo",
                        seed=idx["seed"])
    assert prov2.launched == []                 # no VM for the requeued leg
    assert again.index["corpus_digest"] == first.index["corpus_digest"]
    assert st.pod_files(again) == st.pod_files(first)


def test_a_mismatched_or_missing_corpus_is_not_reused(tmp_path):
    prov, store = FakeProvider(), FakeStore()
    st, idx, _ = _stage(tmp_path, prov, store)
    assert st.existing(round_id="77", hotkey=HK, gen_ref="g@sha256:1", seed=idx["seed"]) is None
    _go(st, tmp_path)
    assert st.existing(round_id="77", hotkey=HK, gen_ref="other@sha256:2",
                       seed=idx["seed"]) is None
    assert st.existing(round_id="77", hotkey=HK, gen_ref=idx["generator_ref"],
                       seed=idx["seed"] + 1) is None
    assert st.existing(round_id="77", hotkey=HK, gen_ref=idx["generator_ref"],
                       seed=idx["seed"]) is not None


class _SoldOutThenOk(FakeProvider):
    def __init__(self, empty_probes):
        super().__init__()
        self.left = empty_probes

    def offers(self, vcpus, price):
        if self.left > 0:
            self.left -= 1
            return []
        return super().offers(vcpus, price)


def test_cpu_drought_is_waited_out_with_backoff(tmp_path):
    prov = _SoldOutThenOk(2)
    st, _, _ = _stage(tmp_path, prov, FakeStore())
    t = [0.0]
    sleeps = []
    st.clock = lambda: t[0]
    st.sleep = lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s))
    st.capacity_wait_s, st.capacity_poll_s = 3600.0, 60.0
    _go(st, tmp_path)
    assert sleeps == [60.0, 120.0] and prov.launched


def test_cpu_drought_past_the_wait_fails_as_infra(tmp_path):
    prov = _SoldOutThenOk(10**6)
    st, _, _ = _stage(tmp_path, prov, FakeStore())
    t = [0.0]
    st.clock = lambda: t[0]
    st.sleep = lambda s: t.__setitem__(0, t[0] + s)
    st.capacity_wait_s, st.capacity_poll_s = 600.0, 60.0
    with pytest.raises(GenerationInfraError, match="no CPU capacity.*waited 600s"):
        _go(st, tmp_path)
    assert prov.launched == []
