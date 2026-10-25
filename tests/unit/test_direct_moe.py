"""Offline protocol acceptance and crash/fault tests. No rented GPUs or keys."""
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from cascade.eval.direct_guard import (
    Case,
    GuardContract,
    judge_public,
    make_receipt,
    replay_receipt,
)
from cascade.funding.direct_ledger import DirectLedger, replay_ledger
from cascade.provision.direct import FakeProvider, MemoryObjects
from cascade.shared.direct_moe import SUITES, BatchSchedule, Cohort, DirectConfig, digest
from cascade.trainer.direct_moe import DirectController


@pytest.fixture
def guard():
    return GuardContract(
        cases=tuple(Case(s, str(i), "mase", f"source:{i}") for s in SUITES for i in range(12)),
        suite_weights={s: .25 for s in SUITES}, metric_weights={s: {"mase": 1.} for s in SUITES},
        tau_overall=.2, tau_suite={s: .2 for s in SUITES}, min_clusters=5, draws=300, seed="pinned")


def rows(guard, ratio=.9):
    return ({c.key: 1. for c in guard.cases}, {c.key: ratio for c in guard.cases})


def verdict(guard, k, c):
    return judge_public(guard, k, c, reference_id="anchor", candidate_id="candidate", private_pass=True)


def test_joint_guard_and_exact_replay(guard):
    k, c = rows(guard)
    # Slight point gain, wide enough confidence interval to include regression.
    c = {case.key: (.6 if int(case.case) < 6 else 1.3) for case in guard.cases}
    v = verdict(guard, k, c)
    assert v["passed"] and v["point"] > 0 and -.2 <= v["lcb"] < 0
    receipt = make_receipt(guard, k, c, reference_id="a", candidate_id="b", private_pass=True)
    assert replay_receipt(json.loads(json.dumps(receipt))) == receipt["verdict"]
    receipt["verdict"]["point"] += .01
    with pytest.raises(ValueError, match="replay mismatch"):
        replay_receipt(receipt)


@pytest.mark.parametrize("ratio", [1., 1.1])
def test_nonpositive_point_fails(guard, ratio):
    assert not verdict(guard, *rows(guard, ratio))["passed"]


def test_suite_regression_not_hidden(guard):
    k, c = rows(guard, .5)
    c.update({case.key: 1.3 for case in guard.cases if case.suite == "FEV"})
    v = verdict(guard, k, c)
    assert v["point"] > 0 and not v["passed"]


@pytest.mark.parametrize("fault", ["missing", "nan", "mismatch", "zero", "coverage"])
def test_incomplete_cannot_pass(guard, fault):
    k, c = rows(guard)
    key = next(iter(c))
    if fault == "missing":
        c.pop(key)
    if fault == "nan":
        c[key] = np.nan
    if fault == "mismatch":
        c[key + "other"] = c.pop(key)
    if fault == "zero":
        k[key] = 0
    if fault == "coverage":
        guard = replace(guard, min_clusters=13)
    v = verdict(guard, k, c)
    assert not v["passed"] and v["status"] == "inconclusive"


def test_horizons_do_not_multiply_evidence(guard):
    k, c = rows(guard)
    c = {case.key: (.6 if int(case.case) < 6 else 1.3) for case in guard.cases}
    old = verdict(guard, k, c)
    duplicate = replace(guard, cases=tuple(replace(case, case=f"{case.case}:h{h}") for case in guard.cases for h in range(4)))
    kd = {case.key: 1. for case in duplicate.cases}
    cd = {case.key: (.6 if int(case.case.split(":")[0]) < 6 else 1.3) for case in duplicate.cases}
    new = verdict(duplicate, kd, cd)
    assert new["lcb"] == pytest.approx(old["lcb"], abs=1e-14)
    assert new["coverage"]["GIFT"]["clusters"] == 12


def test_lifetime_ramp_partial_budget_and_warmup():
    schedule = BatchSchedule()
    assert schedule.batch(150_000_000_000 - 1) == 64
    assert schedule.batch(150_000_000_000) == 128
    assert schedule.batch(400_000_000_000) == 256
    steps = list(schedule.steps(149_999_999_999, 50_000_000_000))
    assert steps[0]["valid_values"] == 1
    assert steps[0]["accumulation"] == 8
    assert steps[1]["global_batch"] == 128
    assert sum(x["valid_values"] for x in steps) == 50_000_000_000
    assert all(x["lr_scale"] == 1 for x in steps)
    assert list(schedule.steps(399_999_999_999, 10))[1]["global_batch"] == 256
    assert list(schedule.steps(5_000_000_000, 10))[0]["lr_scale"] == .5


def test_cache_contract_and_activation():
    c = Cohort("a", 0, 25_000_000_000, "gen", "opt", BatchSchedule(), 3, "unresolved-shape", "8xh200")
    assert c.cache_key("g") == c.cache_key("g")
    for change in ({"anchor": "b"}, {"budget": 50_000_000_000}, {"schedule": replace(c.schedule, warmup=0)}):
        assert replace(c, **change).cache_key("g") != c.cache_key("g")
    assert "architecture" in DirectConfig().unresolved()
    with pytest.raises(ValueError, match="unresolved"):
        DirectConfig(from_block=1).validate()


@pytest.fixture
def environment(tmp_path, guard):
    policy = dict(max_jobs=12, spend_pool=10000, reward_pool=100, founding_count=2,
                  founding_deadline=1000, specialist_seconds=20, retry_base=1, retry_cap=10,
                  capacity_deadline=100, funding_cure_seconds=20, circuit_failures=4)
    path = tmp_path / "jobs.sqlite"
    ledger = DirectLedger(path, policy)
    provider, objects = FakeProvider(), MemoryObjects()
    controller = DirectController(ledger, {"fake": provider}, objects)
    cohort = Cohort("random-init", 0, 100, "gen", "opt", BatchSchedule(), 3, "fixture-only", "fake-8")
    controller.cohort(cohort, now=0)
    return controller, cohort, provider, objects, policy, path


def admit(env, guard, job="A", role="incumbent", reference="random-init", reward=10, now=1):
    ctl, cohort, *_ = env
    terms = dict(cohort=cohort.key, role=role, payer=job, generator=f"gen-{job}", funding_ref=f"vault:{job}",
                 anchor=cohort.anchor, budget=cohort.budget, spend_cap=100, reward=reward, slot=cohort.key,
                 profiles=["fake-8"], providers=["fake", "a", "b"], world_size=8, reference=reference, guard_contract=digest(guard.to_dict()))
    ctl.admit(job, terms, now=now)
    return terms


def checkpoint(env, job, consumed=None):
    ctl, _, _, objects, *_ = env
    j = ctl.ledger.state["jobs"][job]
    consumed = j["terms"]["budget"] if consumed is None else consumed
    ranks = {}
    for rank in range(8):
        ranks[str(rank)] = {}
        for kind in ("weights", "optimizer", "scheduler", "router", "rng", "sampler", "generator"):
            body = f"{job}:{rank}:{kind}:{consumed}".encode()
            checksum = hashlib.sha256(body).hexdigest()
            uri = f"memory://{checksum}"
            objects.put(uri, body)
            ranks[str(rank)][kind] = {"uri": uri, "sha256": checksum}
    return dict(job=job, fence=j["fence"], contract=digest(j["terms"]), kind="continuation",
                accumulation_boundary=True, consumed=consumed, ranks=ranks)


def complete(env, guard, job="A", now=2):
    ctl = env[0]
    ctl.reserve(job, now=now)
    ctl.launch(job, ready_ranks=list(range(8)), now=now)
    cp = checkpoint(env, job)
    ctl.checkpoint(job, cp, now=now)
    ctl.release(job, now=now, reason="completed")
    ctl.evaluate(job, guard, *rows(guard), private_pass=True, now=now)
    return digest(cp)


def test_founders_cache_and_single_advancement(environment, guard):
    ctl, c, *_ = environment
    admit(environment, guard)
    endpoint = complete(environment, guard)
    ctl.settle("A", now=3)
    with pytest.raises(ValueError, match="founding window"):
        ctl.advance("A", routing_stable=False, now=3)
    admit(environment, guard, "B", role="challenger", reward=0, reference=endpoint, now=3)
    b = complete(environment, guard, "B", now=4)
    ctl.settle("B", now=4)
    s = ctl.ledger.state
    assert s["exposure"] == 0 and s["cohorts"][c.key]["cache"]["gen-A"] == endpoint
    assert s["cohorts"][c.key]["cache"]["gen-B"] == b
    ctl.advance("B", routing_stable=False, now=5)
    ctl.advance("B", routing_stable=False, now=5)
    assert ctl.ledger.state["exposure"] == 100
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_dethroned_payer_keeps_completion_right(environment, guard):
    ctl, _, provider, objects, policy, path = environment
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.launch("A", ready_ranks=list(range(8)), now=2)
    ctl.checkpoint("A", checkpoint(environment, "A", 40), now=3)
    ctl.fault("A", "worker", now=4)
    admit(environment, guard, "B", role="challenger", reward=0, now=5)
    complete(environment, guard, "B", now=6)
    ctl.settle("B", now=6)
    # Restart controller while dethroned A is waiting. Its pinned job survives.
    ctl.ledger.close()
    ledger = DirectLedger(path, policy)
    ctl = DirectController(ledger, {"fake": provider}, objects)
    env = (ctl, *environment[1:])
    assert ctl.resume_manifest("A")["consumed"] == 40
    complete(env, guard, "A", now=7)
    ledger.apply("pay-A", "pay", job="A", now=8)
    ledger.apply("pay-A", "pay", job="A", now=8)
    ledger.apply("pay-A-again", "pay", job="A", now=9)
    s = ledger.state
    assert s["king"] == "B"
    assert s["rewards"]["A"] == {"payer": "A", "amount": 10}
    assert s["paid"] == {"A": 10} and s["reserved_reward"] == 0
    assert replay_ledger(ledger.export()) == s


@pytest.mark.parametrize("mode,expected", [("scarcity", "waiting_capacity"), ("partial", "waiting_capacity"), ("revoked", "funding_cure")])
def test_provider_failure_attribution(environment, guard, mode, expected):
    ctl, _, provider, *_ = environment
    admit(environment, guard)
    provider.mode = mode
    ctl.reserve("A", now=2)
    j = ctl.ledger.state["jobs"]["A"]
    assert j["state"] == expected and j["spent"] == 0 and j["consumed"] == 0
    assert j["terms"]["payer"] == "A"
    if mode == "partial":
        assert provider.released and j["reservation"] is None


def test_timeout_after_reserve_restart_reconciles(environment, guard):
    ctl, _, provider, objects, policy, path = environment
    admit(environment, guard)
    provider.mode = "timeout_after_reserve"
    with pytest.raises(TimeoutError):
        ctl.reserve("A", now=2)
    ctl.ledger.close()
    ctl = DirectController(DirectLedger(path, policy), {"fake": provider}, objects)
    ctl.reserve("A", now=3)
    assert provider.reservations == 1
    assert ctl.ledger.state["jobs"]["A"]["state"] == "reserved"


def test_one_rank_missing_and_old_worker_fenced(environment, guard):
    ctl = environment[0]
    admit(environment, guard)
    ctl.reserve("A", now=2)
    with pytest.raises(ValueError, match="rank missing"):
        ctl.launch("A", ready_ranks=list(range(7)), now=2)
    ctl.launch("A", ready_ranks=list(range(8)), now=2)
    cp = checkpoint(environment, "A", 40)
    ctl.checkpoint("A", cp, now=3)
    stale = checkpoint(environment, "A", 60)
    ctl.fault("A", "worker", now=4)
    ctl.reserve("A", now=6)
    ctl.launch("A", ready_ranks=list(range(8)), now=6)
    with pytest.raises(ValueError, match="stale worker"):
        ctl.checkpoint("A", stale, now=7)
    assert ctl.resume_manifest("A")["consumed"] == 40


@pytest.mark.parametrize("fault", ["corrupt", "partial", "outage", "evaluation_artifact", "expired"])
def test_checkpoint_failures_preserve_pointer(environment, guard, fault):
    ctl, _, _, objects, *_ = environment
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.launch("A", ready_ranks=list(range(8)), now=2)
    good = checkpoint(environment, "A", 40)
    ctl.checkpoint("A", good, now=3)
    bad = checkpoint(environment, "A", 60)
    if fault == "corrupt":
        objects.objects[bad["ranks"]["0"]["weights"]["uri"]] = b"corrupt"
    if fault == "partial":
        bad["ranks"].pop("7")
    if fault == "outage":
        objects.outage = True
    if fault == "evaluation_artifact":
        bad["kind"] = "cooled"
    with pytest.raises((ValueError, OSError)):
        ctl.checkpoint("A", bad, now=100 if fault == "expired" else 4)
    assert ctl.ledger.state["jobs"]["A"]["checkpoint"] == good


def test_evaluator_outage_retains_training_and_entitlement(environment, guard):
    ctl = environment[0]
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.launch("A", ready_ranks=list(range(8)), now=2)
    ctl.checkpoint("A", checkpoint(environment, "A"), now=2)
    ctl.release("A", now=3, reason="completed")
    ctl.fault("A", "evaluation", now=4)
    ctl.evaluate("A", guard, {}, {}, private_pass=True, now=5)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "inconclusive"
    ctl.evaluate("A", guard, *rows(guard), private_pass=True, now=6)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "accepted"
    assert ctl.ledger.state["rewards"]["A"]["amount"] == 10


def test_backpressure_unfunded_caps_and_idempotent_charges(environment, guard):
    ctl = environment[0]
    terms = admit(environment, guard)
    ctl.ledger.apply("charge1", "charge", job="A", charge_key="provider-invoice", amount=40, now=2)
    ctl.ledger.apply("charge2", "charge", job="A", charge_key="provider-invoice", amount=40, now=3)
    assert ctl.ledger.state["spent"] == 40
    with pytest.raises(ValueError, match="cap exhausted"):
        ctl.ledger.apply("charge3", "charge", job="A", charge_key="extra", amount=61, now=4)
    for key, value, match in (("funding_ref", "", "unfunded"), ("reward", 200, "overcommitted"), ("spend_cap", 20000, "cap exhausted")):
        invalid = {**terms, key: value}
        with pytest.raises(ValueError, match=match):
            ctl.admit(f"bad-{key}", invalid, now=4)
    with pytest.raises(ValueError, match="already committed"):
        ctl.admit("B", {**terms, "payer": "B"}, now=4)


def test_circuit_breaker_and_durable_generator_progress(environment, guard):
    ctl, _, provider, *_ = environment
    admit(environment, guard)
    provider.mode = "scarcity"
    for now in (2, 12, 22, 32):
        ctl.reserve("A", now=now)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "paused_capacity"
    assert ctl.ledger.state["jobs"]["A"]["cure_until"] is None
    ctl.ledger.apply("recover", "resume_queue", job="A", evidence="capacity restored", now=40)
    provider.mode = "ok"
    ctl.reserve("A", now=40)
    ctl.launch("A", ready_ranks=list(range(8)), now=40)
    fence = ctl.ledger.state["jobs"]["A"]["fence"]
    ctl.ledger.apply("progress1", "progress", job="A", fence=fence, stage="generation", position=1, now=41)
    ctl.ledger.apply("heartbeat", "progress", job="A", fence=fence, stage="generation", position=1, now=50)
    assert ctl.ledger.state["jobs"]["A"]["progress"]["generation"]["at"] == 41
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_audit_cli(environment, guard, tmp_path):
    from cascade.audit.main import main
    ctl = environment[0]
    admit(environment, guard)
    complete(environment, guard)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(ctl.ledger.export()))
    assert main(["direct-moe", str(receipt)]) == 0


@pytest.mark.parametrize("combined_ratio,accepted", [(1.1, False), (.8, True)])
def test_combined_specialists_gate_next_anchor(environment, guard, combined_ratio, accepted):
    ctl, original, provider, objects, policy, path = environment
    # Configure a large synthetic budget; objects remain tiny fault fixtures.
    cohort = replace(original, budget=200_000_000_000)
    ctl.cohort(cohort, now=1)
    env = (ctl, cohort, provider, objects, policy, path)
    admit(env, guard, now=2)
    a = complete(env, guard, now=3)
    ctl.settle("A", now=3)
    admit(env, guard, "B", role="challenger", reward=0, reference=a, now=4)
    complete(env, guard, "B", now=5)
    ctl.settle("B", now=5)
    ctl.advance("B", routing_stable=True, now=6)
    before = ctl.ledger.state["anchor"]
    assert ctl.ledger.state["phase"] == "specialist"
    specialist = replace(original, anchor=before, exposure=200_000_000_000)
    ctl.cohort(specialist, now=7)
    env = (ctl, specialist, provider, objects, policy, path)
    for name in ("expert-1", "expert-2"):
        admit(env, guard, name, role="expert", reference=before, now=8)
        complete(env, guard, name, now=8)
        assert name not in ctl.ledger.state["rewards"]
    receipt = make_receipt(guard, *rows(guard, combined_ratio), reference_id=before,
                           candidate_id="actual-combined-artifact", private_pass=True)
    ctl.merge(["expert-1", "expert-2"], "actual-combined-artifact", receipt, now=9)
    state = ctl.ledger.state
    assert state["phase"] == "broad" and state["exposure"] == 200_000_000_000
    assert state["anchor"] == ("actual-combined-artifact" if accepted else before)
    assert ("expert-1" in state["rewards"]) == accepted
    next_cohort = replace(specialist, anchor=state["anchor"])
    if next_cohort.key != specialist.key:
        ctl.cohort(next_cohort, now=10)
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_evaluation_cache_retries_failed_cases_only(environment, guard):
    ctl = environment[0]
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.launch("A", ready_ranks=list(range(8)), now=2)
    ctl.checkpoint("A", checkpoint(environment, "A"), now=2)
    ctl.release("A", reason="completed", now=3)
    seen = []

    def outage(case):
        seen.append(case.key)
        if len(seen) == 4:
            raise TimeoutError("evaluator outage")
        return 1., .9

    with pytest.raises(TimeoutError):
        ctl.evaluate_cases("A", guard, outage, private_pass=True, now=4)
    cached = set(ctl.ledger.state["jobs"]["A"]["evaluations"])
    assert len(cached) == 3
    retried = []

    def healthy(case):
        retried.append(case.key)
        return 1., .9

    ctl.evaluate_cases("A", guard, healthy, private_pass=True, now=5)
    assert not cached.intersection(retried)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "accepted"


def test_fallback_only_after_reconciled_release(environment, guard):
    ctl, _, first, objects, *_ = environment
    second = FakeProvider()
    ctl.providers = {"a": first, "b": second}
    admit(environment, guard)
    first.mode = "partial"
    ctl.reserve("A", now=2)
    ctl.reserve("A", now=4)
    assert first.released and second.reservations == 1
    assert ctl.ledger.state["jobs"]["A"]["reservation"]["provider"] == "b"


def test_release_timeout_blocks_replacement(environment, guard):
    ctl, _, provider, *_ = environment
    admit(environment, guard)
    ctl.reserve("A", now=2)
    provider.mode = "release_timeout"
    with pytest.raises(TimeoutError):
        ctl.release("A", now=3, reason="worker")
    with pytest.raises(TimeoutError):
        ctl.reserve("A", now=4)
    assert provider.reservations == 1
    provider.mode = "ok"
    ctl.reserve("A", now=5)
    ctl.reserve("A", now=6)
    assert provider.reservations == 2


def test_finite_royalty_options_and_deduplication():
    from cascade.shared.direct_rewards import RewardPolicy, allocate_boundary
    history = [{"payer": "prior", "contribution": "one", "boundary": 1},
               {"payer": "prior", "contribution": "one", "boundary": 3}]
    for fraction, current, _prior in (("1/2", 25, 25), ("1/4", 37, 13)):
        p = RewardPolicy("0", "1/2", fraction, 5, 10, "burn", "burn")
        v = allocate_boundary(p, emission=100, boundary=4, king="king", history=history)
        assert v["throne_royalties"]["king"] in (current, current + 1)
        assert v["continuation_pool"] == 50
        assert sum(v["throne_royalties"].values()) == 50
        expired = allocate_boundary(p, emission=100, boundary=6, king="king", history=history)
        assert "prior" not in expired["throne_royalties"]
        paused = allocate_boundary(p, emission=100, boundary=6, king="king", history=history, paused=True)
        assert not paused["throne_royalties"] and paused["burn"] == 50


def test_stale_king_burns_ninety_percent_of_its_throne():
    from cascade.shared.direct_rewards import RewardPolicy, allocate_boundary
    history = [{"payer": "king", "contribution": "first", "boundary": 10},
               {"payer": "prior", "contribution": "old", "boundary": 5}]
    p = RewardPolicy("0", "0", "0", 50, 10, "burn", "burn", stale_after=100, stale_burn="9/10")

    def throne(boundary, rows=history):
        return allocate_boundary(p, emission=1000, boundary=boundary, king="king", history=rows)

    # Within the window the king keeps the whole throne.
    assert throne(109)["throne_royalties"] == {"king": 1000} and throne(109)["burn"] == 0
    stale = throne(110)
    assert stale["throne_royalties"] == {"king": 100}
    assert stale["burn"] == stale["stale_burn"] == 900
    # A new promotion by the king's generator restores the full throne.
    improved = history + [{"payer": "king", "contribution": "second", "boundary": 150}]
    assert throne(200, improved)["throne_royalties"] == {"king": 1000}
    # A predecessor's promotions do not count for the current king.
    assert throne(200, [{"payer": "prior", "contribution": "x", "boundary": 199}])["throne_royalties"]["king"] == 100
    # Disabled by default: old reward policies replay unchanged.
    assert RewardPolicy("0", "0", "0", 50, 10, "burn", "burn").stale_after == 0
    with pytest.raises(ValueError):
        RewardPolicy("0", "0", "0", 50, 10, "burn", "burn", stale_after=1, stale_burn="3/2")


def test_rounds_on_the_era_clock_gate_progression_and_burn_a_stale_king(tmp_path, guard):
    from cascade.shared.direct_rewards import RewardPolicy

    policy = dict(max_jobs=12, spend_pool=10000, reward_pool=100, founding_count=1,
                  founding_deadline=0, specialist_seconds=20, retry_base=1, retry_cap=10,
                  capacity_deadline=100, funding_cure_seconds=20, circuit_failures=4,
                  round_origin=1000, round_blocks=600, era_blocks=100)  # a round = 6 eras
    block = [1000]
    objects = MemoryObjects()
    ctl = DirectController(DirectLedger(tmp_path / "rounds.sqlite", policy), {"fake": FakeProvider()},
                           objects, block=lambda: block[0])
    ctl.apply("reward-policy", "reward_policy", 0, policy=asdict(
        RewardPolicy("0", "0", "0", 50, 10, "burn", "burn", stale_burn="9/10")))
    first = Cohort("random-init", 0, 100, "gen", "opt", BatchSchedule(), 3, "fixture-only", "fake-8")
    ctl.cohort(first, now=0)
    env = (ctl, first, None, objects, policy, None)
    admit(env, guard)
    complete(env, guard)
    block[0] = 1100
    ctl.settle("A", now=3)
    retry = replace(first, seed=4)
    with pytest.raises(ValueError, match="previous round's winner"):
        block[0] = 1600
        ctl.cohort(retry, now=4)
    block[0] = 1599
    with pytest.raises(ValueError, match="after the round closes"):
        ctl.advance("A", routing_stable=False, now=4)
    block[0] = 1600
    ctl.advance("A", routing_stable=False, now=4)
    promoted = ctl.ledger.state["anchor"]
    ctl.cohort(replace(first, anchor=promoted, exposure=100), now=5)
    with pytest.raises(ValueError, match="one cohort per round"):
        ctl.cohort(replace(first, anchor=promoted, exposure=100, seed=4), now=5)

    def payout(at, now):
        block[0] = at
        ctl.apply(f"payout:{now}", "payout_boundary", now, boundary=now, emission=1000, paused=False)
        return ctl.ledger.state["payout_boundaries"][str(now)]["throne_royalties"]

    # Crowned in round 0: paid in full through round 1.
    assert payout(1650, 10) == {"A": 1000}
    with pytest.raises(ValueError, match="one payout per era"):
        payout(1699, 11)
    # A whole round (1) without a promotion of its own: 90% burns in round 2.
    assert payout(2200, 12) == {"A": 100}
    with pytest.raises(ValueError, match="non-decreasing"):
        payout(2100, 13)
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state
    # Validators refuse round blocks the chain has not reached.
    from cascade.direct.network import check_conformance
    config = {"ledger_policy": policy, "reward_policy": None, "guards": [digest(guard.to_dict())],
              "private_policies": [], "private_panels": []}
    check_conformance(ctl.ledger.export(), ctl.ledger.state, config, now_block=2200)
    with pytest.raises(ValueError, match="future round block"):
        check_conformance(ctl.ledger.export(), ctl.ledger.state, config, now_block=2199)
    with pytest.raises(ValueError, match="whole number of eras"):
        DirectLedger(tmp_path / "bad.sqlite", {**policy, "round_blocks": 650})


def test_worker_resume_full_state_and_routing_counts(environment, guard):
    from cascade.trainer.direct_worker import COMPONENTS, run_worker
    ctl, original, provider, objects, policy, path = environment
    schedule = BatchSchedule(context=1, channels=1, thresholds=(150, 400), batches=(8, 16, 32), horizon=1000, warmup=10)
    cohort = replace(original, schedule=schedule)
    ctl.cohort(cohort, now=1)
    env = (ctl, cohort, provider, objects, policy, path)
    admit(env, guard, now=2)
    ctl.reserve("A", now=3)
    ctl.launch("A", ready_ranks=list(range(8)), now=3)

    class Backend:
        world_size = 8

        def __init__(self):
            self.position = 0
            self.loads = None

        def restore(self, states):
            positions = {int(parts["generator"]) for parts in states.values()}
            assert len(positions) == 1
            self.position = positions.pop()

        def step(self, plan):
            self.position += plan["valid_values"]
            return {rank: [[plan["accumulation"], 2]] for rank in range(8)}

        def balance(self, loads):
            self.loads = loads

        def snapshot(self):
            return {rank: {kind: str(self.position).encode() for kind in COMPONENTS} for rank in range(8)}

    first = Backend()
    cp = run_worker(ctl, "A", schedule, first, now=lambda: 4, max_steps=3)
    assert cp["consumed"] == 24 and first.loads == [[8, 16]]
    ctl.fault("A", "worker", now=5)
    ctl.reserve("A", now=7)
    ctl.launch("A", ready_ranks=list(range(8)), now=7)
    second = Backend()
    cp = run_worker(ctl, "A", schedule, second, now=lambda: 8)
    assert cp["consumed"] == 100 and second.position == 100
    ctl.release("A", now=9, reason="completed")
    ctl.evaluate("A", guard, *rows(guard), private_pass=True, now=10)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "accepted"
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_attributable_cure_and_spending_refusal(environment, guard):
    ctl, _, provider, *_ = environment
    admit(environment, guard)
    provider.price = 101
    ctl.reserve("A", now=2)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "funding_cure"
    with pytest.raises(ValueError, match="deadline"):
        ctl.ledger.apply("early", "expire_funding", job="A", evidence="cap refusal", now=3)
    ctl.ledger.apply("expired", "expire_funding", job="A", evidence="cap refusal uncured", now=23)
    state = ctl.ledger.state
    assert state["jobs"]["A"]["state"] == "contract_failure"
    assert state["reserved_reward"] == state["reserved_spend"] == 0
    assert "A" not in state["paid"]


def test_restart_after_every_payment_and_promotion_transition(environment, guard):
    ctl, _, _, _, policy, path = environment
    original_apply = ctl.apply

    def restarting_apply(*args, **kw):
        result = original_apply(*args, **kw)
        export = ctl.ledger.export()
        ctl.ledger.close()
        ctl.ledger = DirectLedger(path, policy)
        assert replay_ledger(export) == ctl.ledger.state
        return result

    ctl.apply = restarting_apply
    admit(environment, guard)
    a = complete(environment, guard)
    ctl.settle("A", now=3)
    admit(environment, guard, "B", role="challenger", reward=0, reference=a, now=4)
    complete(environment, guard, "B", now=5)
    ctl.settle("B", now=5)
    ctl.advance("B", routing_stable=False, now=6)
    ctl.apply("payout", "pay", 7, job="A")
    assert ctl.ledger.state["paid"] == {"A": 10}
    assert ctl.ledger.state["exposure"] == 100


def test_partial_training_cannot_enter_evaluation(environment, guard):
    ctl = environment[0]
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.launch("A", ready_ranks=list(range(8)), now=2)
    ctl.checkpoint("A", checkpoint(environment, "A", 40), now=3)
    with pytest.raises(ValueError, match="complete training"):
        ctl.fault("A", "evaluation", now=4)
    assert ctl.ledger.state["jobs"]["A"]["consumed"] == 40
    assert ctl.ledger.state["jobs"]["A"]["state"] == "retry_wait"


def test_edited_rows_combined_against_same_anchor():
    torch = pytest.importorskip("torch")
    from cascade.trainer.direct_experts import combine_experts
    base = {"in": torch.zeros(3, 2), "out": torch.zeros(3, 2), "router": torch.zeros(2)}
    one = {k: v.clone() for k, v in base.items()}
    two = {k: v.clone() for k, v in base.items()}
    one["in"][0] = 1
    two["out"][1] = 2
    merged, reset = combine_experts(base, [(one, {0: [0]}), (two, {0: [1]})], [("in", "out")], optimizer_policy="reset_edited_rows")
    assert torch.equal(merged["in"][0], one["in"][0])
    assert torch.equal(merged["out"][1], two["out"][1])
    assert reset == {"in": [0, 1], "out": [0, 1]}
    two["router"][0] = 1
    with pytest.raises(ValueError, match="outside"):
        combine_experts(base, [(two, {0: [1]})], [("in", "out")], optimizer_policy="reset_edited_rows")


def test_model_shape_and_actual_parameter_count():
    from pathlib import Path
    torch = pytest.importorskip("torch")
    from cascade.trainer import moe_model
    from cascade.trainer.direct_model import build_direct_model
    config = moe_model.CascadeMoEConfig()
    with torch.device("meta"):
        original = moe_model.CascadeMoEModel(config)
    count = moe_model.param_counts(original)
    architecture = {"config": config.to_dict(),
                    "source_sha256": hashlib.sha256(Path(moe_model.__file__).read_bytes()).hexdigest(),
                    "expected_active": count["active"], "expected_total": count["total"]}
    model = build_direct_model(architecture)
    assert moe_model.param_counts(model) == count
    assert count["active"] < 30_000_000  # Existing preset is NOT the requested ~220M.
    architecture["expected_active"] = 220_000_000
    with pytest.raises(ValueError, match="counts differ"):
        build_direct_model(architecture)


def test_payout_boundary_keeps_throne_and_completion_separate(environment, guard):
    from dataclasses import asdict

    from cascade.shared.direct_rewards import RewardPolicy
    ctl = environment[0]
    policy = RewardPolicy("1/2", "1/2", "1/4", 20, 5, "burn", "burn")
    ctl.apply("reward-policy", "reward_policy", 1, policy=asdict(policy))
    admit(environment, guard)
    a = complete(environment, guard)
    ctl.settle("A", now=3)
    admit(environment, guard, "B", role="challenger", reward=0, reference=a, now=4)
    complete(environment, guard, "B", now=5)
    ctl.settle("B", now=5)
    ctl.apply("payout6", "payout_boundary", 6, boundary=6, emission=800, paused=False)
    ctl.apply("payout6", "payout_boundary", 6, boundary=6, emission=800, paused=False)
    allocation = ctl.ledger.state["payout_boundaries"]["6"]
    assert allocation["throne_royalties"] == {"A": 50, "B": 150}
    assert allocation["continuation_pool"] == 200 and allocation["lane_b_pool"] == 400
    assert ctl.ledger.state["rewards"]["A"]["amount"] == 10
    ctl.apply("settle6", "settle_boundary", 7, boundary=6)
    allocation = ctl.ledger.state["payout_boundaries"]["6"]
    # Emissions settle the finite entitlement; the unowed rest of the pool burns.
    assert allocation["continuation_paid"] == {"A": {"payer": "A", "amount": 10}}
    assert allocation["continuation_burn"] == 190
    assert ctl.ledger.state["paid"] == {"A": 10}
    with pytest.raises(ValueError, match="already settled"):
        ctl.apply("settle6-again", "settle_boundary", 8, boundary=6)
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state

    from cascade.direct.weights import boundary_units, latest_settled_boundary, weight_vector
    state = ctl.ledger.state
    units = boundary_units(state, latest_settled_boundary(state))
    assert units == {"hotkeys": {"A": 60, "B": 150}, "burn": 190, "lane_b_pool": 400}
    uids = {"A": 1, "B": 2}
    vec = weight_vector(units, uid_of=uids.get, n_uids=4, burn_uid=0)
    assert vec == [590 / 800, 60 / 800, 150 / 800, 0.0]
    # An unregistered payer's slice burns; it is never redistributed.
    vec = weight_vector(units, uid_of={"B": 2}.get, n_uids=4, burn_uid=0)
    assert vec == [650 / 800, 0.0, 150 / 800, 0.0]


def test_incomplete_private_evaluation_is_inconclusive(guard):
    k, c = rows(guard)
    v = judge_public(guard, k, c, reference_id="a", candidate_id="b", private_pass=None)
    assert v["status"] == "inconclusive" and not v["passed"]


def test_direct_activation_config_rejects_unresolved_values(tmp_path):
    from pathlib import Path

    from cascade.shared.config import load_chain_config
    source = Path(__file__).resolve().parents[2] / "chain.toml"
    path = tmp_path / "chain.toml"
    path.write_text(source.read_text() + "\n[direct_moe]\nversion = 1\nfrom_block = 100\n")
    with pytest.raises(ValueError, match="unresolved direct-MoE policies"):
        load_chain_config(path)


def test_executable_simulation_trainer_validator_audit(tmp_path):
    from cascade.audit.main import main as audit
    from cascade.trainer.main import main as train
    from cascade.validator.main import main as validate
    directory = tmp_path / "simulation"
    assert train(["--direct-simulation", str(directory)]) == 0
    receipt = directory / "receipt.json"
    assert validate(["--direct-receipt", str(receipt)]) == 0
    assert audit(["direct-moe", str(receipt)]) == 0
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["king"] == "B" and summary["paid"] == {"A": 10}
    assert summary["exposure"] == 100 and summary["replayed"]
    with pytest.raises(ValueError, match="existing evidence"):
        train(["--direct-simulation", str(directory)])


def test_round_clock_must_sit_on_the_live_era_grid():
    from cascade.direct.network import check_round_clock
    from cascade.shared.config import load_chain_config
    from cascade.shared.era import era_length_blocks

    cfg = load_chain_config().round
    era = era_length_blocks(cfg, 6 * 10**7)
    origin = (6 * 10**7 // era + 1) * era
    check_round_clock({"round_origin": origin, "round_blocks": 6 * era, "era_blocks": era}, cfg)
    check_round_clock({}, cfg)
    for bad in ({"round_origin": origin + 1, "round_blocks": 6 * era, "era_blocks": era},
                {"round_origin": origin, "round_blocks": 6 * era, "era_blocks": era + 1}):
        with pytest.raises(ValueError, match="era grid"):
            check_round_clock(bad, cfg)
