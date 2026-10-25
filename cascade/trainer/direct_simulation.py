"""Credential-free executable simulation of funded direct-MoE recovery.

Uses counters, NOT a neural model or an H200 benchmark. All policy values here
are explicitly synthetic fixtures and are never loaded by live contracts.
"""
from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

from ..eval.direct_guard import Case, GuardContract
from ..funding.direct_ledger import DirectLedger
from ..provision.direct import FakeProvider, MemoryObjects
from ..shared.direct_moe import SUITES, BatchSchedule, Cohort, DirectConfig, digest
from .direct_moe import DirectController
from .direct_worker import COMPONENTS, run_worker


class CounterGang:
    world_size = 8

    def __init__(self):
        self.position = 0

    def restore(self, states):
        positions = {int(rank["generator"]) for rank in states.values()}
        if len(positions) != 1:
            raise ValueError("ranks disagree on data position")
        self.position = positions.pop()

    def step(self, plan):
        self.position += plan["valid_values"]
        return {rank: [[plan["accumulation"], 0]] for rank in range(self.world_size)}

    def balance(self, global_loads):
        if global_loads[0][0] < self.world_size:
            raise ValueError("incomplete accumulated routing counts")

    def snapshot(self):
        return {rank: {kind: str(self.position).encode() for kind in COMPONENTS}
                for rank in range(self.world_size)}


def simulate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "direct.sqlite"
    if path.exists():
        raise ValueError("simulation requires a new directory; existing evidence is preserved")
    policy = dict(max_jobs=4, spend_pool=1000, reward_pool=100, founding_count=2,
                  founding_deadline=100, specialist_seconds=10, retry_base=1, retry_cap=10,
                  capacity_deadline=100, funding_cure_seconds=10, circuit_failures=4)
    schedule = BatchSchedule(context=1, channels=1, thresholds=(150, 400), batches=(8, 16, 32),
                             horizon=1000, warmup=10)
    cohort = Cohort("fixture-random", 0, 100, "fixture-gen", "fixture-opt", schedule, 7,
                    "counter-NOT-220M", "memory-8-NOT-H200")
    guard = GuardContract(tuple(Case(s, str(i), "mase", str(i)) for s in SUITES for i in range(8)),
                          {s: .25 for s in SUITES}, {s: {"mase": 1.} for s in SUITES},
                          .1, {s: .1 for s in SUITES}, 4, draws=300, seed="simulation-only")
    provider, objects = FakeProvider(mode="timeout_after_reserve"), MemoryObjects()
    ctl = DirectController(DirectLedger(path, policy), {"fake": provider}, objects)
    ctl.cohort(cohort, now=0)
    terms = dict(cohort=cohort.key, role="incumbent", payer="A", generator="gen-A", funding_ref="fixture-funded-A",
                 anchor=cohort.anchor, budget=100, spend_cap=100, reward=10, slot=cohort.key,
                 profiles=["fake"], providers=["fake"], world_size=8, reference=cohort.anchor,
                 guard_contract=digest(guard.to_dict()))
    ctl.admit("A", terms, now=1)
    with suppress(TimeoutError):
        ctl.reserve("A", now=2)
    ctl.ledger.close()
    ctl = DirectController(DirectLedger(path, policy), {"fake": provider}, objects)
    ctl.reserve("A", now=3)
    ctl.launch("A", ready_ranks=list(range(8)), now=3)
    run_worker(ctl, "A", schedule, CounterGang(), now=lambda: 4, max_steps=3)
    ctl.fault("A", "worker", now=5)
    ctl.admit("B", {**terms, "role": "challenger", "payer": "B", "generator": "gen-B", "reward": 0,
                    "funding_ref": "fixture-funded-B"}, now=6)
    ctl.reserve("B", now=7)
    ctl.launch("B", ready_ranks=list(range(8)), now=7)
    run_worker(ctl, "B", schedule, CounterGang(), now=lambda: 8)
    ctl.release("B", reason="completed", now=9)
    ctl.evaluate_cases("B", guard, lambda case: (1., .9), private_pass=True, now=10)
    ctl.settle("B", now=11)
    ctl.reserve("A", now=12)
    ctl.launch("A", ready_ranks=list(range(8)), now=12)
    backend = CounterGang()
    run_worker(ctl, "A", schedule, backend, now=lambda: 13)
    ctl.release("A", reason="completed", now=14)
    ctl.evaluate_cases("A", guard, lambda case: (1., .9), private_pass=True, now=15)
    ctl.apply("pay-A", "pay", 16, job="A")
    ctl.advance("B", routing_stable=False, now=17)
    journal = ctl.ledger.export()
    ctl.ledger.close()
    from ..validator.direct_moe import verify_direct_receipt
    verify_direct_receipt(journal)
    receipt = directory / "receipt.json"
    receipt.write_text(json.dumps(journal, indent=2) + "\n")
    summary = {"mode": "synthetic offline simulation", "receipt": str(receipt),
               "king": journal["state"]["king"], "paid": journal["state"]["paid"],
               "exposure": journal["state"]["exposure"], "replayed": True,
               "unresolved_activation_settings": DirectConfig().unresolved()}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
