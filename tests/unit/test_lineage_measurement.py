"""Checkpoint-bound actual counts, retry after disk/store faults and migration."""

import json
from types import SimpleNamespace

import pytest

from cascade.trainer.lane_b_driver import LaneBDriver
from cascade.trainer.lineage import MEASUREMENT_FILE, read_measurement, write_measurement


def binding():
    return dict(pointer="immutable", gen_ref="generator", corpus_digest="corpus",
                contract_digest="contract", warm_start_ref="base", token_budget=1000)


def measurement(path, n=317, **extra):
    b = binding()
    write_measurement(path, metrics={"tokens_seen": n, **extra},
                      **{k: v for k, v in b.items() if k != "pointer"})


def driver(tmp_path):
    return LaneBDriver(SimpleNamespace(), SimpleNamespace(fetch=lambda _: tmp_path),
                       tmp_path / "driver.json")


def test_actual_count_not_budget_and_epoch_replay_after_restart(tmp_path):
    measurement(tmp_path)
    d = driver(tmp_path)
    d.record_lineage_epoch("epoch1", binding())
    d.reconcile_lineage()
    assert d.st.lineage_tokens == 317
    # Crash after B save, before rolling scheduler retires the old epoch.
    d = driver(tmp_path)
    d.record_lineage_epoch("epoch1", binding())
    d.reconcile_lineage()
    assert d.st.lineage_tokens == 317
    with pytest.raises(ValueError, match="binding"):
        d.record_lineage_epoch("epoch1", {**binding(), "pointer": "other"})


def test_unavailable_evidence_is_pending_until_it_arrives(tmp_path):
    d = driver(tmp_path)
    d.record_lineage_epoch("epoch1", binding())
    d.reconcile_lineage()
    assert d.st.lineage_tokens == 0
    measurement(tmp_path)
    d = driver(tmp_path)
    d.reconcile_lineage()
    assert d.st.lineage_tokens == 317


def test_failed_counter_save_rolls_back_memory_and_can_retry(tmp_path, monkeypatch):
    measurement(tmp_path)
    d = driver(tmp_path)
    d.record_lineage_epoch("epoch1", binding())
    real = d.save
    monkeypatch.setattr(d, "save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        d.reconcile_lineage()
    assert d.st.lineage_tokens == 0
    assert "tokens_seen" not in d.st.lineage_epochs["epoch1"]
    monkeypatch.setattr(d, "save", real)
    d.reconcile_lineage()
    assert d.st.lineage_tokens == 317


def test_legacy_budget_is_retained_but_not_treated_as_measured(tmp_path):
    (tmp_path / "driver.json").write_text(json.dumps({"lineage_tokens": 200_000_000_000}))
    d = driver(tmp_path)
    assert d.st.lineage_tokens == 0
    assert d.st.legacy_budget_tokens == 200_000_000_000
    d.save()
    assert driver(tmp_path).st.legacy_budget_tokens == 200_000_000_000


def test_fork_anneal_tail_is_not_lineage_training(tmp_path):
    measurement(tmp_path, 1100, anneal_tokens_seen=100)
    assert read_measurement(tmp_path, binding()) == 1000


def test_backend_without_counts_cannot_publish_a_stale_measurement(tmp_path):
    measurement(tmp_path)
    b = binding()
    write_measurement(tmp_path, metrics={}, **{k: v for k, v in b.items() if k != "pointer"})
    assert not (tmp_path / MEASUREMENT_FILE).exists()


@pytest.mark.parametrize("value", [float("nan"), -1, True, "317", 1001])
def test_invalid_points_are_rejected(tmp_path, value):
    measurement(tmp_path)
    p = tmp_path / MEASUREMENT_FILE
    doc = json.loads(p.read_text())
    doc["tokens_seen"] = value
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        read_measurement(tmp_path, binding())


@pytest.mark.parametrize("field", ["gen_ref", "corpus_digest", "contract_digest", "warm_start_ref"])
def test_measurement_cannot_be_reused_for_a_different_leg(tmp_path, field):
    measurement(tmp_path)
    with pytest.raises(ValueError, match=field):
        read_measurement(tmp_path, {**binding(), field: "wrong"})


def test_worker_hook_keeps_actual_cpu_training_measurement_on_upload_retry(
        cfg, tmp_path, monkeypatch):
    from cascade.shared.manifest import contract_digest
    from cascade.trainer.contract import RoundSeeds
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
    from tests.unit.test_moe_trainer import _C, _train

    contract = _C(copula_loss_cap=0)
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path)
    calls = []

    def train(gen, seeds, contract, budget, out, **kw):
        calls.append(out)
        result = _train(out, contract, budget=budget)
        return result, "corpus", 48, 3072

    monkeypatch.setattr(r, "_train_checkpoint", train)
    gen = ResolvedGenerator("alice", 1, "generator")
    kw = dict(contract=contract, token_budget=600, repo_suffix="", heat=False,
              warm_start_ref=None)
    seeds = RoundSeeds.derive(3, cfg.training)
    out, _, _ = r._train_for_entry(gen, "king", seeds, **kw)
    b = dict(gen_ref=gen.ref, corpus_digest="corpus", contract_digest=contract_digest(contract),
             warm_start_ref="", token_budget=600)
    assert read_measurement(out, b) == 600
    original = (out / MEASUREMENT_FILE).read_bytes()
    r._train_for_entry(gen, "king", seeds, **kw)
    assert len(calls) == 1 and (out / MEASUREMENT_FILE).read_bytes() == original


def test_scheduler_crash_after_epoch_registration_replays_without_double_count(
        cfg, tmp_path, monkeypatch):
    from cascade.shared.era import era_for_block
    from cascade.shared.manifest import TrainedEntry, contract_digest
    from cascade.trainer.rolling import EraState, _entry_to_json
    from tests.unit.test_dec55_trainer_epochs import _v2
    from tests.unit.test_rolling import REF, Clock, FakeClient, _ptr, _sched

    cfg, block, length = _v2(cfg)
    sched, _ = _sched(cfg, tmp_path, Clock())
    first = era_for_block(cfg.round, block).index
    spec = era_for_block(cfg.round, block)
    entry = TrainedEntry("KING", 0, "king", REF["KING"], _ptr("trained"), "corpus", block)
    cur = EraState(first, block, spec.seed_block, 987, epoch_start=first,
                   king_hotkey="KING", king_entry=_entry_to_json(entry))
    sched.state.current = cur
    sched._save()
    d = driver(tmp_path)
    sched.lane_b = d
    c = cfg.throne_contracts_at(block)[0]
    write_measurement(tmp_path, metrics={"tokens_seen": 317}, gen_ref=entry.gen_ref,
                      corpus_digest="corpus", contract_digest=contract_digest(c),
                      warm_start_ref="", token_budget=c.train_tokens)
    nxt = era_for_block(cfg.round, block + 3 * length)
    monkeypatch.setattr(sched, "_save", lambda: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError, match="crash"):
        sched._roll_era(FakeClient(), nxt, nxt.start_block)
    d.reconcile_lineage()
    assert d.st.lineage_tokens == 317
    # Rolling state is still the previous epoch, but the B journal is durable.
    sched, _ = _sched(cfg, tmp_path, Clock())
    sched.lane_b = driver(tmp_path)
    sched._roll_era(FakeClient(), nxt, nxt.start_block)
    sched.lane_b.reconcile_lineage()
    assert sched.lane_b.st.lineage_tokens == 317
    assert len(sched.lane_b.st.lineage_epochs) == 1
