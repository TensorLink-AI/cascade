"""A validator cannot judge under adopted policy that it failed to persist."""

import pytest

from cascade.shared import activation as activation
from cascade.shared import netconfig as nc

from .test_netconfig_core import E1, E2, MemStore, base_cfg, publish


def test_adoption_disk_failure_holds_config_until_durable(tmp_path, monkeypatch):
    cfg, store = base_cfg(), MemStore()
    publish(store, {"scoring": {"min_windows": 7}}, E2)
    resolver = nc.NetConfigResolver(cfg, state_path=tmp_path / "config.json", store=store)
    resolver.refresh(cfg, force=True)
    real = nc.write_text_atomic

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(nc, "write_text_atomic", disk_full)
    record = activation.ActivationRecord(lock_block=E1, activation_block=E2)
    with pytest.raises(nc.NetConfigError, match="persist"):
        resolver.adopt(cfg, resolver.state.candidate, record)
    with pytest.raises(nc.NetConfigError, match="persist"):
        resolver.config_at(cfg, E2)
    assert resolver.segment() is None
    monkeypatch.setattr(nc, "write_text_atomic", real)
    resolver.refresh(cfg, force=True)
    assert resolver.config_at(cfg, E2).scoring.min_windows == 7
    restarted = nc.NetConfigResolver(cfg, state_path=resolver.state_path, store=MemStore())
    assert restarted.config_at(cfg, E2).scoring.min_windows == 7
