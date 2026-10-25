"""Offline direct-MoE orchestration: effects follow committed intents.

This controller is deliberately separate from the historical rolling scheduler.
No production provider is selected implicitly. All operations accept recorded
clock values and restart from the journal, including uncertain reservations.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict

from ..eval.direct_guard import GuardContract, make_receipt
from ..funding.direct_ledger import ROUND_EVENTS
from ..provision.direct import CapacityUnavailable, CredentialsRevoked, SpendingRefused
from ..shared.direct_moe import digest


class DirectController:
    def __init__(self, ledger, providers, objects, *, clock=None, block=None):
        self.ledger, self.providers, self.objects = ledger, providers, objects
        self.clock = clock
        # Chain block source; a round journal (DEC-CA-0052 §11) stamps it on
        # every round event.
        self.block = block

    def apply(self, command, kind, now, **data):
        if "rounds" in self.ledger.state and kind in ROUND_EVENTS and "block" not in data:
            if self.block is None:
                raise ValueError("round journal requires a chain block source")
            data["block"] = int(self.block())
        return self.ledger.apply(command, kind, now=self.clock() if self.clock is not None else now, **data)

    def cohort(self, cohort, *, now):
        self.apply(f"cohort:{cohort.key}", "cohort", now, cohort=asdict(cohort))
        return cohort.key

    def admit(self, job, terms, *, now):
        # No credentials are persisted: funding_ref identifies a scoped vault entry.
        required = {"cohort", "role", "payer", "generator", "funding_ref", "anchor",
                    "budget", "spend_cap", "reward", "slot", "profiles", "providers", "world_size",
                    "reference", "guard_contract"}
        if set(terms) != required or terms["world_size"] < 1 or not terms["profiles"]:
            raise ValueError("incomplete immutable job terms")
        return self.apply(f"admit:{job}", "admit", now, job=job, terms=terms)

    def reserve(self, job, *, now):
        j = self.ledger.state["jobs"][job]
        if j["state"] == "releasing":
            reason = j["reason"]
            result = self.release(job, now=now, reason=reason)
            return result if reason == "completed" else self.fault(job, reason, now=now)
        if j["reservation"] is None:
            approved = [(name, profile) for name in sorted(self.providers) if name in j["terms"]["providers"] for profile in j["terms"]["profiles"]]
            if not approved:
                raise ValueError("no configured providers")
            provider, profile = approved[j["attempt"] % len(approved)]
            key = digest({"job": job, "attempt": j["attempt"] + 1})
            self.apply(f"reserve:{key}", "reserve_intent", now, job=job, key=key, provider=provider, profile=profile)
            j = self.ledger.state["jobs"][job]
        r = j["reservation"]
        if r["status"] != "intent":
            return j
        provider = self.providers[r["provider"]]
        conservative = getattr(provider, "conservative_accounting", False)
        cap = j["cap"] - j["spent"]
        try:
            if conservative:
                if r["key"] not in j["charges"]:
                    amount = provider.estimate(r["profile"], j["terms"]["world_size"])
                    if type(amount) is not int or amount <= 0 or amount > cap:
                        raise SpendingRefused("allocation estimate exceeds remaining cap")
                    # This debit precedes ANY potentially billable provider effect.
                    self.apply(f"allocation-budget:{r['key']}", "charge", now,
                               job=job, charge_key=r["key"], amount=amount)
                cap = self.ledger.state["jobs"][job]["charges"][r["key"]]
            # A timeout never creates a second key or switches provider. Query/retry
            # the same reservation until its outcome is known.
            a = provider.lookup(r["key"])
            if a is None:
                a = provider.reserve(r["key"], profile=r["profile"], world_size=j["terms"]["world_size"],
                                     cap=cap, now=now)
        except (CapacityUnavailable, CredentialsRevoked, SpendingRefused) as exc:
            reason = ("auth" if isinstance(exc, CredentialsRevoked) else
                      "spend" if isinstance(exc, SpendingRefused) else "capacity")
            self.release(job, now=now, reason=reason)
            return self.fault(job, reason, now=now)
        # Other exceptions (particularly timeout) retain the durable intent and debit.
        now = self.clock() if self.clock is not None else now
        if (a.ranks != tuple(range(j["terms"]["world_size"])) or a.lease_until <= now
                or a.profile != r["profile"] or a.max_charge > cap):
            self.release(job, now=now, reason="capacity")
            return self.fault(job, "capacity", now=now)
        return self.apply(f"held:{r['key']}", "reserved", now, job=job, key=r["key"],
                          ranks=list(a.ranks), lease_until=a.lease_until, allocation=a.identity)

    def release(self, job, *, now, reason):
        j = self.ledger.state["jobs"][job]
        r = j["reservation"]
        if r is None:
            return j
        if r["status"] != "releasing":
            self.apply(f"release:{r['key']}", "release_intent", now, job=job, reason=reason)
        provider = self.providers[r["provider"]]
        provider.release(r["key"])
        now = self.clock() if self.clock is not None else now
        if getattr(provider, "conservative_accounting", False):
            evidence = provider.no_charge_evidence(r["key"]) if hasattr(provider, "no_charge_evidence") else None
            if evidence:
                self.apply(f"allocation-no-charge:{r['key']}", "allocation_no_charge", now,
                           job=job, key=r["key"], evidence=evidence)
        return self.apply(f"released:{r['key']}", "released", now, job=job)

    def fault(self, job, reason, *, now):
        j = self.ledger.state["jobs"][job]
        if j["reservation"] is not None:
            self.release(job, now=now, reason=reason)
        j = self.ledger.state["jobs"][job]
        return self.apply(f"fault:{job}:{j['failures'] + 1}", "fault", now, job=job, reason=reason)

    def launch(self, job, *, ready_ranks, now):
        j = self.ledger.state["jobs"][job]
        return self.apply(f"launch:{job}:{j['fence']}", "launch", now, job=job, fence=j["fence"], ready_ranks=ready_ranks)

    def checkpoint(self, job, manifest, *, now):
        # Remote checksum verification precedes the atomic authoritative pointer.
        # Objects may be orphaned on crash; an unverified pointer is never published.
        for rank in manifest["ranks"].values():
            for obj in rank.values():
                if hasattr(self.objects, "verify"):
                    self.objects.verify(obj["uri"], obj["sha256"])
                elif hashlib.sha256(self.objects.get(obj["uri"])).hexdigest() != obj["sha256"]:
                    raise ValueError("corrupt checkpoint object")
        return self.apply(f"checkpoint:{job}:{digest(manifest)}", "checkpoint", now,
                          job=job, fence=manifest["fence"], manifest=manifest)

    def resume_manifest(self, job):
        cp = self.ledger.state["jobs"][job]["checkpoint"]
        if cp is not None:
            for rank in cp["ranks"].values():
                for obj in rank.values():
                    if hashlib.sha256(self.objects.get(obj["uri"])).hexdigest() != obj["sha256"]:
                        raise ValueError("resume object checksum mismatch")
        return cp

    def evaluate(self, job, guard: GuardContract, reference, candidate, *, private_pass, now):
        j = self.ledger.state["jobs"][job]
        receipt = make_receipt(guard, reference, candidate,
                              reference_id=j["terms"]["reference"], candidate_id=digest(j["checkpoint"]),
                              private_pass=private_pass)
        return self.apply(f"evaluate:{job}:{digest(receipt)}", "evaluation", now, job=job, receipt=receipt)

    def settle(self, job, *, now):
        return self.apply(f"settle:{job}", "settle", now, job=job)

    def advance(self, job, *, routing_stable, now):
        return self.apply(f"advance:{job}", "advance", now, job=job, routing_stable=routing_stable)

    def merge(self, jobs, combined, receipt, *, now):
        return self.apply(f"merge:{digest([jobs, combined, receipt])}", "merge", now,
                          jobs=jobs, combined=combined, receipt=receipt)

    def evaluate_cases(self, job, guard, scorer, *, private_pass, now):
        """Cache successful fixed cases and retry only missing evaluator work."""
        j = self.ledger.state["jobs"][job]
        for case in sorted(guard.cases, key=lambda c: c.key):
            if case.key in j["evaluations"]:
                continue
            king_score, candidate_score = scorer(case)
            self.apply(f"eval-case:{job}:{digest(guard.to_dict())}:{case.key}", "eval_case", now,
                       job=job, guard=digest(guard.to_dict()), reference=j["terms"]["reference"],
                       candidate=digest(j["checkpoint"]), case=case.key,
                       king_score=king_score, candidate_score=candidate_score)
        cache = self.ledger.state["jobs"][job]["evaluations"]
        return self.evaluate(job, guard, {k: v["king"] for k, v in cache.items()},
                             {k: v["candidate"] for k, v in cache.items()},
                             private_pass=private_pass, now=now)

    def compare(self, job, guard, reference, candidate, *, reference_id, private_pass, now):
        """Rescore a fixed trained endpoint against a new king without rewriting admission."""
        j = self.ledger.state["jobs"][job]
        receipt = make_receipt(guard, reference, candidate, reference_id=reference_id,
                               candidate_id=digest(j["checkpoint"]), private_pass=private_pass)
        return self.apply(f"compare:{job}:{digest(receipt)}", "comparison", now, job=job, receipt=receipt)
