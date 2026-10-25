"""Generation must not rent without durable intent or leak a known VM on a write fault."""

import pytest

from cascade.generation import stage

from .test_generation_stage import HK, FakeProvider, FakeStore, _go, _stage


@pytest.mark.parametrize("failed_write", [1, 2])
def test_ledger_disk_failure_prevents_rent_or_cleans_up_vm(tmp_path, monkeypatch, failed_write):
    provider = FakeProvider()
    st, _, _ = _stage(tmp_path, provider, FakeStore())
    st.ledger_path = tmp_path / "ledger.json"
    real, writes = stage.write_text_atomic, 0

    def failing_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == failed_write:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(stage, "write_text_atomic", failing_write)
    with pytest.raises(stage.GenerationInfraError, match="ledger"):
        _go(st, tmp_path)
    if failed_write == 1:
        assert provider.launched == []
    else:
        assert provider.terminated == ["vm-1"]


def test_lost_launch_response_cannot_create_a_second_vm(tmp_path):
    class LostResponse(FakeProvider):
        def launch(self, *args, **kwargs):
            super().launch(*args, **kwargs)
            raise TimeoutError("VM created but response lost")

    provider = LostResponse()
    st, _, _ = _stage(tmp_path, provider, FakeStore())
    st.ledger_path = tmp_path / "ledger.json"
    with pytest.raises(stage.GenerationInfraError, match="launch failed"):
        _go(st, tmp_path)
    intent = st.ledger_path.read_text()
    with pytest.raises(stage.GenerationInfraError, match="reconciliation"):
        _go(st, tmp_path)
    assert len(provider.launched) == 1
    assert st.ledger_path.read_text() == intent


def test_invisible_intent_stays_until_exact_lookup_can_resolve_it(tmp_path):
    class EventualProvider(FakeProvider):
        visible = False

        def find_by_name(self, name):
            return ["vm-1"] if self.visible else []

    provider = EventualProvider()
    path = tmp_path / "ledger.json"
    stage._ledger_put(path, {"provider": provider.name, "name": "cgen-x", "vm_id": ""})
    assert stage.reap_generation_vms(path, [provider]) == []
    assert stage._ledger_load(path)
    provider.visible = True
    assert stage.reap_generation_vms(path, [provider]) == ["vm-1"]
    assert stage._ledger_load(path) == []
    assert provider.terminated == ["vm-1"]


def test_stage_starting_during_lookup_is_not_reaped(tmp_path):
    class RacingProvider(FakeProvider):
        def find_by_name(self, name):
            with stage._LIVE_GUARD:
                stage._LIVE.add(name)
            return ["live-vm"]

    provider = RacingProvider()
    path = tmp_path / "ledger.json"
    stage._ledger_put(path, {"provider": provider.name, "name": "cgen-race", "vm_id": ""})
    try:
        assert stage.reap_generation_vms(path, [provider]) == []
        assert provider.terminated == []
        assert stage._ledger_load(path)
    finally:
        with stage._LIVE_GUARD:
            stage._LIVE.discard("cgen-race")


def test_unledgered_sweep_does_not_delete_a_new_live_launch(tmp_path):
    class RacingProvider(FakeProvider):
        def list_tagged(self, prefix):
            with stage._LIVE_GUARD:
                stage._LIVE.add("cgen-new")
                stage._LAUNCHING.add("cgen-new")
            return ["live-vm"]

    provider = RacingProvider()
    try:
        assert stage.reap_generation_vms(tmp_path / "ledger.json", [provider]) == []
        assert provider.terminated == []
    finally:
        with stage._LIVE_GUARD:
            stage._LIVE.discard("cgen-new")
            stage._LAUNCHING.discard("cgen-new")


def test_duplicate_stage_cannot_clear_an_existing_workers_live_marker(tmp_path):
    provider = FakeProvider()
    st, _, _ = _stage(tmp_path, provider, FakeStore())
    name = stage._vm_name(HK, "77")
    with stage._LIVE_GUARD:
        stage._LIVE.add(name)
    try:
        with pytest.raises(stage.GenerationInfraError, match="already running"):
            _go(st, tmp_path)
        assert name in stage._LIVE
        assert provider.launched == []
    finally:
        with stage._LIVE_GUARD:
            stage._LIVE.discard(name)
