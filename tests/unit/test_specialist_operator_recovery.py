"""B/C use the same admission/pool and never enter payer rent on make-good."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue
from cascade.trainer import runner_funded_legs as legs
from cascade.trainer.lane_pool import _FinalLanePool
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
from cascade.trainer.remote import RemoteDispatchError, RemoteHost

GEN = "alice/generator@sha256:" + "a" * 64


@pytest.mark.parametrize("role", ["expert", "mkt_finetune"])
@pytest.mark.parametrize("outcome", ["success", "generator_fault", "operator_fault"])
def test_make_good_preserves_specialist_job_and_releases_resources(
        cfg, tmp_path, monkeypatch, role, outcome):
    cfg = replace(cfg, round=replace(cfg.round, funded_operator_fallback=True))
    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    from cascade.trainer.admission import remember_owners
    remember_owners(r, [SimpleNamespace(hotkey="alice", coldkey="fixture-owner", commit_block=1)])
    q = FundedQueue(tmp_path / "q.json", entry_ttl_seconds=3600)
    q.add("alice", GEN, reveal_block=1)
    q.mark_in_flight("alice", GEN, target_boundary=900, era_index=4, started_block=23)
    q.set_operator_billed("alice", True)
    host = RemoteHost(name="operator", host="10.1.1.1")
    pool = _FinalLanePool([host], lambda: [host])
    monkeypatch.setattr(TrainerRunner, "_operator_fallback_lanes", lambda self: [host])
    monkeypatch.setattr(TrainerRunner, "_rolling_lane_pool", lambda self: pool)
    # A flagged entry must exit rent BEFORE even loading the vault / payer key.
    monkeypatch.setattr(TrainerRunner, "_payer_vault",
                        lambda self: pytest.fail("payer vault accessed during make-good"))
    monkeypatch.setattr(legs, "leg_corpus", lambda *a, **kw: {"corpus_files": ["fixed"]})
    monkeypatch.setattr(legs, "_reupload_pending", lambda *a: None)
    seen = []
    harvested = []
    contract = cfg.training.primary_size

    def harvest(self, chosen, receipt, effective_contract, seeds, suffix):
        assert chosen.isolated and receipt == "local checkpoint"
        assert effective_contract is contract and seeds.base_seed == 99
        assert suffix == "-specialist-u7"
        assert self._funded_queue() is q
        assert q.get("alice").phase == "harvesting"
        harvested.append(receipt)
        return "checkpoint"

    monkeypatch.setattr(TrainerRunner, "_harvest_funded_checkpoint", harvest)

    class Dispatcher:
        def dispatch(self, chosen, **kw):
            assert chosen.name == host.name and chosen.isolated
            assert kw["local_checkpoint"] is True
            assert q.get("alice").phase == "training"
            assert r._funded_queue() is q
            assert r.__dict__["_leg_local_obj"].funded_contract is contract
            assert r.__dict__["_funded_job_slots"].active == 1
            seen.append(kw)
            if outcome != "success":
                raise RemoteDispatchError("worker failed", returncode=(
                    3 if outcome == "generator_fault" else 1))
            return "local checkpoint"

    kw = dict(warm_start_ref="pinned-base", role=role, bench=False, funded_queue=q,
              contract_block=17, era_index=4, domain="energy", expert_selection={0: [2]},
              finetune_recipe='{"marketcast":"recipe"}')
    args = (Dispatcher(), ResolvedGenerator("alice", 7, GEN), SimpleNamespace(base_seed=99),
            23, contract, "-specialist-u7")
    if outcome == "success":
        assert r._run_funded_leg(*args, **kw) == "checkpoint"
        assert harvested == ["local checkpoint"]
        assert q.get("alice").phase == "awaiting_settlement"
        assert "waiting_operator" in [e["phase"] for e in q.get("alice").phase_history]
    else:
        with pytest.raises(RemoteDispatchError):
            r._run_funded_leg(*args, **kw)
        verdict = r._funded_leg_failures["alice"]
        assert verdict[1:] == ((True, "generator", False) if outcome == "generator_fault"
                               else (False, "operator", False))
    assert seen
    for dispatch in seen:
        assert dispatch["role"] == role
        assert dispatch["warm_start_ref"] == "pinned-base"
        assert dispatch["contract_block"] == 17
        assert dispatch["repo_suffix"] == "-specialist-u7"
        assert dispatch["corpus_files"] == ["fixed"]
        if role == "expert":
            assert dispatch["domain"] == "energy" and dispatch["expert_selection"] == {0: [2]}
        else:
            assert dispatch["finetune_recipe"] == kw["finetune_recipe"]
    assert r.__dict__["_funded_job_slots"].active == 0
    assert pool.get_nowait() is host  # shared GPU lane returned even after both attempts fail
    assert q.get("alice").operator_billed
    assert not hasattr(r.__dict__["_leg_local_obj"], "funded_queue")
