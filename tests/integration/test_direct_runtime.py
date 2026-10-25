"""Real scheduler → torchrun → immutable checkpoint → forecasting evaluation."""

import io
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save

from cascade.direct.evaluation import PrivatePolicy, replay_decision
from cascade.direct.evaluator import CheckpointEvaluator
from cascade.direct.runtime import DirectRuntime, LocalCPUProvider, LocalTorchDispatcher
from cascade.direct.storage import publish
from cascade.eval.direct_guard import Case, GuardContract
from cascade.funding.direct_ledger import DirectLedger, replay_ledger
from cascade.shared.direct_moe import SUITES, BatchSchedule, Cohort, digest
from cascade.trainer.direct_model import build_direct_model
from cascade.trainer.direct_moe import DirectController
from tests.integration.test_direct_torch import training_spec


def test_scheduler_real_training_evaluation_restart(tmp_path):
    spec, objects = training_spec(tmp_path)
    torch.manual_seed(spec["seed"])
    model = build_direct_model(spec["architecture"], device="cpu")
    initial = {"ranks": {"0": {"weights": publish(objects, save(model.state_dict()))}}}
    anchor = digest(initial)
    guard = GuardContract(
        cases=tuple(Case(s, str(i), "mase", f"source:{i}") for s in SUITES for i in range(4)),
        suite_weights={s: 0.25 for s in SUITES},
        metric_weights={s: {"mase": 1.0} for s in SUITES},
        tau_overall=0.2,
        tau_suite={s: 0.2 for s in SUITES},
        min_clusters=2,
        draws=100,
        seed="fixture",
    )
    stream = io.BytesIO()
    rng = np.random.default_rng(99)
    np.savez(stream, history=rng.normal(size=(4, 2, 8)), target=rng.normal(size=(4, 2, 2)))
    panel = {
        "object": publish(objects, stream.getvalue()),
        "fixture": True,
        "cases": [
            {"id": str(i), "source": f"source:{i}", "domain": "test", "seasonal_period": 1}
            for i in range(4)
        ],
    }
    evaluator = CheckpointEvaluator(
        objects,
        tmp_path / "eval",
        guard=guard,
        private_policy=PrivatePolicy(0.05, 100, "fixture-only-permissive-gain", -1.0, 2, 0.0, 2),
        private_panel=panel,
        public_panels={s: panel for s in SUITES},
        initial_artifacts={anchor: initial},
        samples=8,
        seed=99,
        block_source=lambda: 1000,
    )
    policy = dict(
        max_jobs=4,
        spend_pool=100,
        reward_pool=100,
        founding_count=2,
        founding_deadline=1000,
        specialist_seconds=20,
        retry_base=1,
        retry_cap=10,
        capacity_deadline=100,
        funding_cure_seconds=20,
        circuit_failures=4,
    )
    path = tmp_path / "journal.sqlite"
    ledger = DirectLedger(path, policy)
    controller = DirectController(ledger, {"local": LocalCPUProvider()}, objects)
    schedule = BatchSchedule(**spec["schedule"])
    cohort = Cohort(
        anchor,
        0,
        spec["budget"],
        "fixture-generator",
        digest(spec["optimizer"]),
        schedule,
        spec["seed"],
        digest(spec["architecture"]),
        "local-cpu",
    )
    controller.cohort(cohort, now=0)
    terms = dict(
        cohort=cohort.key,
        role="incumbent",
        payer="fixture-payer",
        generator="fixture-generator",
        funding_ref="existing-cpu",
        anchor=anchor,
        budget=spec["budget"],
        spend_cap=0,
        reward=10,
        slot=cohort.key,
        profiles=["local-cpu"],
        providers=["local"],
        world_size=2,
        reference=anchor,
        guard_contract=digest(guard.to_dict()),
    )
    controller.admit(spec["job"], terms, now=1)
    spec["terms_digest"] = digest(terms)
    controller.apply("bind", "bind_training", 1, job=spec["job"], spec=spec)
    dispatcher = LocalTorchDispatcher(tmp_path / "workers")
    runtime = DirectRuntime(controller, dispatcher, evaluator)
    actual_evaluate = evaluator.evaluate
    outage = {"pending": True}

    def evaluate_with_outage(*args, **kwargs):
        if outage["pending"]:
            outage["pending"] = False
            raise OSError("injected evaluation storage outage")
        return actual_evaluate(*args, **kwargs)

    evaluator.evaluate = evaluate_with_outage
    runtime.tick(2)
    ledger.close()  # supervisor and workers continue across controller restart
    ledger = DirectLedger(path, policy)
    controller = DirectController(ledger, {"local": LocalCPUProvider()}, objects)
    runtime = DirectRuntime(controller, dispatcher, evaluator)
    deadline = time.monotonic() + 90
    now = 3
    while time.monotonic() < deadline:
        runtime.tick(now)
        now += 1
        if ledger.state["jobs"][spec["job"]]["state"] in ("accepted", "rejected"):
            break
        time.sleep(0.2)
    job = ledger.state["jobs"][spec["job"]]
    assert job["state"] in ("accepted", "rejected"), job
    assert job["consumed"] == 135
    assert job["failures"] == 1 and job["reason"] == "evaluation"
    assert len(job["decisions"]) == 1
    for decision in job["decisions"].values():
        replay_decision(decision)
    assert job["state"] == "accepted"  # fixture policy; not approved production thresholds
    runtime.settle(spec["job"], now=now)
    # A second funded founder uses the identical random initialization and
    # scientific contract. It cannot dethrone an identical trained incumbent.
    challenger_terms = {
        **terms,
        "role": "challenger",
        "payer": "challenger",
        "reward": 0,
        "generator": "second-generator",
        "funding_ref": "second-existing-cpu",
    }
    controller.admit("challenger", challenger_terms, now=now)
    challenger = {**spec, "job": "challenger", "terms_digest": digest(challenger_terms)}
    controller.apply("bind:challenger", "bind_training", now, job="challenger", spec=challenger)

    def finish(identity):
        nonlocal now
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            now += 1
            runtime.tick(now)
            if ledger.state["jobs"][identity]["state"] in ("accepted", "rejected"):
                return
            time.sleep(0.2)
        raise AssertionError(ledger.state["jobs"][identity])

    finish("challenger")
    with pytest.raises(ValueError):
        runtime.settle("challenger", now=now)
    assert ledger.state["king"] == terms["payer"]
    controller.advance(spec["job"], routing_stable=False, now=now)
    controller.advance(spec["job"], routing_stable=False, now=now + 1)
    now += 1
    assert ledger.state["exposure"] == spec["budget"]
    # The completed incumbent block holds a finite entitlement, settled later
    # through emissions at a payout boundary (validator weights, no transfer).
    assert ledger.state["rewards"][spec["job"]]["amount"] == 10
    promoted = ledger.state["anchor"]
    next_cohort = replace(cohort, anchor=promoted, exposure=spec["budget"])
    controller.cohort(next_cohort, now=now)
    next_terms = {
        **terms,
        "cohort": next_cohort.key,
        "slot": next_cohort.key,
        "anchor": promoted,
        "reference": promoted,
        "funding_ref": "next-existing-cpu",
    }
    controller.admit("next", next_terms, now=now)
    next_spec = {
        **spec,
        "job": "next",
        "terms_digest": digest(next_terms),
        "exposure": spec["budget"],
        "initial": job["checkpoint"]["ranks"]["0"],
    }
    controller.apply("bind:next", "bind_training", now, job="next", spec=next_spec)
    finish("next")
    assert ledger.state["jobs"]["next"]["consumed"] == spec["budget"]
    assert ledger.state["exposure"] == spec["budget"]  # training itself never advances lineage
    assert replay_ledger(ledger.export()) == ledger.state
    reproduce_by_audit(tmp_path, objects, panel, replay_ledger(ledger.export()))
    owner_config_admits_only_listed_rules(ledger.export())
    evaluator.close()
    ledger.close()


def owner_config_admits_only_listed_rules(journal):
    """Validators enforce the owner's signed network config (DEC-CA-0052 §8, §9)."""
    from copy import deepcopy

    from cascade.direct.network import (
        audit_obligations,
        check_conformance,
        decision_times,
        journal_decisions,
    )

    state = replay_ledger(journal)
    decisions = journal_decisions(state).values()
    assert decisions
    panels = sorted({d["scoring"]["private_panel"] for d in decisions})
    assert len(panels) == 1 and {d["scoring"]["block"] for d in decisions} == {1000}
    schedule = [{"panel": panels[0], "from_block": 0}]
    config = {"ledger_policy": journal["policy"], "reward_policy": state["reward_policy"],
              "guards": sorted({j["terms"]["guard_contract"] for j in state["jobs"].values()}),
              "private_policies": sorted({digest(d["private"]["policy"]) for d in decisions}),
              "private_panels": schedule, "reveal_after": 100}
    check_conformance(journal, state, config, now_block=1000)
    with pytest.raises(ValueError, match="future block"):
        check_conformance(journal, state, config, now_block=999)
    # One panel at a time: a newer panel active at the scoring block refuses.
    later = schedule + [{"panel": "e" * 64, "from_block": 500}]
    with pytest.raises(ValueError, match="active at block 1000"):
        check_conformance(journal, state, {**config, "private_panels": later})
    for change, error in (({"private_policies": []}, "unlisted private policy"),
                          ({"guards": []}, "guard"),
                          ({"private_panels": []}, "active at block")):
        with pytest.raises(ValueError, match=error):
            check_conformance(journal, state, {**config, **change})
    unbound = deepcopy(state)
    next(d for j in unbound["jobs"].values() for d in j.get("decisions", {}).values()).pop("scoring")
    with pytest.raises(ValueError, match="not auditable"):
        check_conformance(journal, unbound, config)

    # Auditability: public checkpoints, and private panels revealed in time.
    times = decision_times(journal)
    fetched = []

    def obligations(boundary, *, available=lambda uri: fetched.append(uri) or True,
                    revealed=lambda panel: False):
        return audit_obligations(journal, state, config, available=available,
                                 revealed=revealed, boundary=boundary)

    assert obligations(min(times.values()) + 99) == []
    assert len(set(fetched)) >= 2
    assert obligations(max(times.values()) + 100) == panels
    assert obligations(max(times.values()) + 100, revealed=lambda panel: True) == []
    with pytest.raises(ValueError, match="not publicly fetchable"):
        obligations(0, available=lambda uri: False)


def reproduce_by_audit(tmp_path, objects, panel, state):
    """The trust model: anyone re-scores the published checkpoints (DEC-CA-0052)."""
    from copy import deepcopy

    from cascade.audit.direct_reproduce import (
        AuditObjects,
        journal_decisions,
        reproduce_decision,
        reproduce_journal,
    )

    revealed = {digest(panel): (panel, objects.get(panel["object"]["uri"]))}
    audit = dict(objects=AuditObjects(tmp_path / "audit", remote=lambda uri: objects),
                 workdir=tmp_path / "audit-work", reveal=revealed.get)
    # While the panel is private nothing is claimed either way.
    assert {r.status for r in reproduce_journal(state, **{**audit, "reveal": lambda _: None})} == {"WARN"}
    results = reproduce_journal(state, **audit)
    assert len(results) == 2 * len(journal_decisions(state)) >= 6
    assert all(r.status == "PASS" for r in results), results
    # A trainer that recorded a better candidate score than its checkpoint earns is caught.
    decision = deepcopy(next(iter(journal_decisions(state).values())))
    decision["candidate_private"][0]["mase"] *= 0.5
    first = next(iter(decision["public"]["candidate_rows"]))
    decision["public"]["candidate_rows"][first] *= 0.5
    private, public = reproduce_decision("lie", decision, **audit)
    assert private.status == public.status == "FAIL"
    assert "mase" in private.detail and first in public.detail
    # A binding that names a different checkpoint than the verdict judged is refused.
    decision = deepcopy(next(iter(journal_decisions(state).values())))
    decision["scoring"]["manifests"] = {k: {**m, "swapped": True} for k, m in decision["scoring"]["manifests"].items()}
    assert reproduce_decision("swap", decision, **audit)[0].status == "FAIL"
