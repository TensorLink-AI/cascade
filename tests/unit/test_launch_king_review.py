"""The carried king's CPU work and GPU funding, without provider calls."""
from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cascade.shared.config import load_chain_config
from cascade.trainer.runner_rolling import _KingKeyFailed, _RollingMixin


def king_runner(*, key_failure=True):
    cfg = load_chain_config()
    contract = cfg.training.v2_contract
    calls = []
    r = _RollingMixin()
    r.cfg = SimpleNamespace(round=SimpleNamespace(ops_king_backup_after_seconds=0),
                           throne_contracts_at=lambda _: [contract])
    r._rolling_seeds = lambda era: SimpleNamespace(base_seed=123)
    r._rolling_warm_start_ref = lambda *a: None
    r._king_self_funds = lambda era, gen=None: True
    def own(*a, **kw):
        calls.append("own_key")
        if key_failure:
            raise _KingKeyFailed("no key in the payer vault", key_digest="")
        return SimpleNamespace(role="king")
    r._rolling_train_king_own_key = own
    r._note_king_key_failed = lambda *a, **kw: calls.append("key_lapse")
    r._refuse_diverged_king = lambda *a: None
    r._persist_completed_leg = lambda *a, **kw: calls.append(("persist", kw.get("funded_by")))
    r._dispatcher = lambda: SimpleNamespace(dispatch=lambda *a, **kw: calls.append("operator_dispatch") or SimpleNamespace(role="king"))
    r._rent_king_host = lambda rid: calls.append("operator_gpu") or "host"
    r._stage_king_vault = lambda host, gen: host
    r._era_contract_block = lambda era: 9000
    r._era_router_slow = lambda era: False
    r._funded_king_lock = nullcontext()
    r._rolling_king_host_era = 0
    r._funded_wait_abort = threading.Event()
    r._final_role_hosts = {}
    return r, calls


def test_carried_king_without_key_uses_operator_and_keeps_cpu_priority(monkeypatch):
    from cascade.trainer import runner_rolling as M

    r, calls = king_runner()
    r._leg_local.funded_is_king = False
    def corpus(runner, *a, **kw):
        assert runner._leg_local.funded_is_king is True
        calls.append("operator_cpu")
        return {}
    monkeypatch.setattr(M, "leg_corpus", corpus)
    gen = SimpleNamespace(hotkey="inherited", uid=1, ref="ref")
    era = SimpleNamespace(start_block=9000, index=1, base_seed=123)
    r._rolling_train_king_single(gen, era, 9000, end_wall=10000)
    assert calls == ["own_key", "key_lapse", "operator_cpu", "operator_gpu", "operator_dispatch", ("persist", "operator")]
    assert r._leg_local.funded_is_king is False


def test_working_king_key_never_rents_an_operator_training_gpu():
    r, calls = king_runner(key_failure=False)
    r._rolling_train_king_single(SimpleNamespace(hotkey="king", uid=1, ref="ref"),
        SimpleNamespace(start_block=9000, index=1, base_seed=123), 9000, end_wall=10000)
    assert calls == ["own_key", ("persist", "king")]


def test_operator_failure_does_not_invent_a_key_lapse(monkeypatch):
    from cascade.trainer import runner_rolling as M

    r, calls = king_runner()
    def own(*a, **kw):
        raise _KingKeyFailed("provider capacity fallback", slash=False)
    r._rolling_train_king_own_key = own
    monkeypatch.setattr(M, "leg_corpus", lambda *a, **kw: {})
    r._rolling_train_king_single(SimpleNamespace(hotkey="king", uid=1, ref="ref"),
        SimpleNamespace(start_block=9000, index=1, base_seed=123), 9000, end_wall=10000)
    assert "key_lapse" not in calls and calls[-1] == ("persist", "operator_fault")


GENESIS = "vault/direct@sha256:" + "a" * 64


def _genesis_runner(monkeypatch):
    from cascade.trainer import runner_rolling as M

    r, calls = king_runner(key_failure=False)
    r.cfg.generator = SimpleNamespace(king_min_channels_refs=(GENESIS,))
    del r._king_self_funds                      # the real gate, not the stub
    monkeypatch.setattr(M, "leg_corpus", lambda *a, **kw: {})
    return r, calls


def test_genesis_king_is_always_operator_paid_never_on_its_own_key(monkeypatch):
    r, calls = _genesis_runner(monkeypatch)
    r._rolling_train_king_single(SimpleNamespace(hotkey="genesis", uid=1, ref=GENESIS),
        SimpleNamespace(start_block=9000, index=1, base_seed=123), 9000, end_wall=10000)
    assert "own_key" not in calls and "key_lapse" not in calls
    assert calls == ["operator_gpu", "operator_dispatch", ("persist", "operator_fault")]


def test_a_recommitted_king_ref_is_not_the_genesis_exemption(monkeypatch):
    r, _ = _genesis_runner(monkeypatch)
    era = SimpleNamespace(start_block=9000, index=1, base_seed=123)
    assert r._operator_funds_genesis_king(SimpleNamespace(ref=GENESIS)) is True
    assert r._operator_funds_genesis_king(SimpleNamespace(ref="bob/gen@sha256:" + "b" * 64)) is False
    assert r._operator_funds_genesis_king(None) is False
    assert r._king_self_funds(era, SimpleNamespace(ref=GENESIS)) is False


def test_failed_cpu_generation_restores_king_flag(monkeypatch):
    from cascade.trainer import runner_rolling as M

    r, calls = king_runner()
    r._leg_local.funded_is_king = False
    monkeypatch.setattr(M, "leg_corpus", lambda *a, **kw: (_ for _ in ()).throw(OSError("storage down")))
    with pytest.raises(OSError):
        r._rolling_train_king_single(SimpleNamespace(hotkey="king", uid=1, ref="ref"),
            SimpleNamespace(start_block=9000, index=1, base_seed=123), 9000, end_wall=10000)
    assert r._leg_local.funded_is_king is False and "operator_gpu" not in calls


def test_short_literal_source_is_checked_without_blanket_string_rejection():
    from cascade.interface.static_guard import scan_source

    payload = 'exec("import socket\\n")'
    assert not scan_source(payload, ("socket",), packed_sources="scan").ok
    assert not scan_source(payload, ("socket",), packed_sources="reject").ok
    assert scan_source('label = "import socket"', ("socket",), packed_sources="reject").ok
    assert scan_source('value = eval("2 + 3")', ("socket",), packed_sources="reject").ok
    assert scan_source(payload, ("socket",), unpack=False).ok
