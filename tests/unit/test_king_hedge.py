from __future__ import annotations

import threading
import time

import pytest

from cascade.trainer.king_hedge import KingHedge


def test_first_valid_king_completion_is_durable_and_late_completion_cannot_replace_it(tmp_path):
    path = tmp_path / "king.json"
    hedge = KingHedge(path, {"init": "pinned", "contract": "pinned", "seed": 7})
    release = threading.Event()
    cleaned = []

    def slow(cancel):
        release.wait(2)
        return {"entry": {"pointer": "late"}, "funded_by": "king"}

    result = hedge.run(
        slow,
        lambda cancel: {"entry": {"pointer": "first"}},
        delay=0,
        deadline=time.time() + 3,
        cleanup=cleaned.append,
    )
    assert result["replica"] == "backup" and result["funded_by"] == "operator_fault"
    release.set()
    deadline = time.time() + 1
    while "primary" not in cleaned and time.time() < deadline:
        time.sleep(0.005)
    assert cleaned == ["primary"]
    restarted = KingHedge(path, hedge.binding)

    def must_not_retrain(cancel):
        pytest.fail("durable winner must not be trained again")

    assert (
        restarted.run(must_not_retrain, must_not_retrain, delay=0, deadline=time.time() + 3)
        == result
    )


def test_primary_failure_launches_backup_and_preserves_proven_key_lapse(tmp_path):
    hedge = KingHedge(tmp_path / "king.json", {"init": "pinned"})

    def failed(cancel):
        hedge.note_key_failure()
        raise RuntimeError("fixture key rejected")

    result = hedge.run(
        failed,
        lambda cancel: {"entry": {"pointer": "backup"}},
        delay=3600,
        deadline=time.time() + 3,
    )
    assert result["funded_by"] == "operator"


def test_fast_primary_does_not_start_paid_backup(tmp_path):
    hedge = KingHedge(tmp_path / "king.json", {"init": "pinned"})

    def backup(cancel):
        pytest.fail("fast primary should never start a backup")

    assert (
        hedge.run(
            lambda cancel: {"entry": {"pointer": "primary"}, "funded_by": "king"},
            backup,
            delay=60,
            deadline=time.time() + 3,
        )["replica"]
        == "primary"
    )


def test_hedge_refuses_incompatible_job_after_restart(tmp_path):
    path = tmp_path / "king.json"
    KingHedge(path, {"init": "first"}).run(
        lambda cancel: {"entry": {}}, lambda cancel: {}, delay=60, deadline=time.time() + 3
    )
    with pytest.raises(ValueError, match="incompatible"):
        KingHedge(path, {"init": "changed"}).run(
            lambda cancel: {}, lambda cancel: {}, delay=60, deadline=time.time() + 3
        )


def test_runner_backup_keeps_pinned_experiment_and_persists_funding_classification(
    cfg, tmp_path, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    from cascade.shared.manifest import TrainedEntry
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
    from cascade.trainer.remote import RemoteHost

    from .test_rolling import _ptr

    cfg = replace(
        cfg,
        round=replace(
            cfg.round,
            ops_king_backup_after_seconds=1,
            ops_king_backup_cap_usd=5,
            ops_operator_leg_cap_usd=10,
            ops_operator_daily_cap_usd=25,
        ),
    )
    runner = TrainerRunner(
        cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="fixture:C"
    )
    runner.detached_dispatch = True
    era = SimpleNamespace(index=1, start_block=100, base_seed=77)
    gen = ResolvedGenerator("KING", 1, "fixture-generator-ref")
    contract = cfg.throne_contracts_at(era.start_block)[0]
    host = RemoteHost("backup", "127.0.0.1")
    observed = []

    def primary(*a, **kw):
        assert kw["persist"] is False
        runner._leg_local.king_hedge.note_key_failure()
        raise RuntimeError("fixture primary key unavailable")

    def rent(round_id, generator, *, name_prefix):
        assert round_id == "77" and generator is gen and "-backup-" in name_prefix
        return host, name_prefix + "-0"

    def dispatch(chosen, **kw):
        assert chosen is host
        observed.append(kw)
        return TrainedEntry(
            gen.hotkey, gen.uid, "king", gen.ref, _ptr("backup"), "fixture-data", 123
        )

    monkeypatch.setattr(runner, "_rolling_train_king_single", primary)
    monkeypatch.setattr(runner, "_rolling_warm_start_ref", lambda *a: None)
    monkeypatch.setattr(runner, "_era_router_slow", lambda *a: False)
    monkeypatch.setattr(runner, "_rent_operator_leg_host", rent)
    monkeypatch.setattr(runner, "_stage_king_vault", lambda chosen, gen: chosen)
    monkeypatch.setattr(runner, "_dispatcher", lambda: SimpleNamespace(dispatch=dispatch))
    monkeypatch.setattr(runner, "_refuse_diverged_king", lambda *a: None)
    monkeypatch.setattr("cascade.trainer.runner_generation.leg_corpus", lambda *a, **kw: {})
    result = runner._rolling_train_king(gen, era, 123, end_wall=time.time() + 3)
    assert result.trained_pointer == _ptr("backup")
    assert observed[0]["base_seed"] == 77 and observed[0]["arch_preset"] == contract.arch_preset
    assert observed[0]["gen_ref"] == gen.ref and observed[0]["warm_start_ref"] is None
    assert runner._rolling_king_funding(era, gen) == "operator"
    assert runner._rolling_king_hosts[era.index] is host


@pytest.mark.parametrize(
    "failure,status",
    [
        ("HTTP 401 unauthorized", "failed"),
        ("HTTP 402 insufficient balance", "failed"),
        ("HTTP 429 rate limit", "unknown"),
        ("API unavailable", "unknown"),
    ],
)
def test_backup_funding_probe_distinguishes_failed_key_from_unknown_provider(
    monkeypatch, failure, status
):
    from types import SimpleNamespace

    from cascade.trainer.king_hedge import king_account_status

    def unavailable(prefix):
        raise RuntimeError(failure)

    monkeypatch.setattr(
        "cascade.provision.funded.provider_for_key",
        lambda *a: SimpleNamespace(list_tagged=unavailable),
    )
    runner = SimpleNamespace(
        _payer_vault=lambda: SimpleNamespace(get_shared=lambda hk: "fixture-key")
    )
    assert king_account_status(runner, "hk")[0] == status
