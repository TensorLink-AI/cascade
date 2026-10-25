"""Transactional direct-MoE obligations and event journal (schema 1).

All timestamps are supplied and recorded. SQLite holds metadata only; checkpoint
objects stay in remote storage. A command and its idempotency result commit with
its state/event in one transaction. Historical funding stores are untouched.
"""
from __future__ import annotations

import json
import sqlite3
from copy import deepcopy

from ..eval.direct_guard import replay_receipt
from ..shared.direct_moe import BatchSchedule, Cohort, canonical, digest, finite

TERMINAL = {"accepted", "rejected", "contract_failure"}
# Rounds on the era clock (DEC-CA-0052 §11): optional, all three or none.
ROUND_FIELDS = {"round_origin", "round_blocks", "era_blocks"}
ROUND_EVENTS = {"cohort", "settle", "advance", "payout_boundary"}


def initial_state(policy):
    required = {"max_jobs", "spend_pool", "reward_pool", "founding_count",
                "founding_deadline", "specialist_seconds", "retry_base", "retry_cap",
                "capacity_deadline", "funding_cure_seconds", "circuit_failures"}
    if set(policy) not in (required, required | ROUND_FIELDS):
        raise ValueError(f"pin ledger policy fields: {sorted(required)} (+ {sorted(ROUND_FIELDS)})")
    for value in policy.values():
        finite(value)
    for key in ("max_jobs", "founding_count", "circuit_failures"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f"invalid {key}")
    if policy["retry_base"] <= 0 or policy["retry_cap"] < policy["retry_base"]:
        raise ValueError("invalid retry policy")
    rounds = {}
    if set(policy) >= ROUND_FIELDS:
        if (any(type(policy[k]) is not int for k in ROUND_FIELDS) or policy["era_blocks"] < 1
                or policy["round_blocks"] < 1 or policy["round_blocks"] % policy["era_blocks"]):
            raise ValueError("rounds must be a whole number of eras (integer blocks)")
        # Present only for round journals, so older journals replay byte-identically.
        rounds = {"rounds": {"block": policy["round_origin"], "opened": {}, "paid_eras": []}}
    return {**rounds, "version": 1, "policy": policy, "jobs": {}, "cohorts": {}, "slots": {},
            "king": None, "anchor": None, "exposure": 0, "phase": "broad",
            "specialist_until": None, "merged": [], "rewards": {}, "paid": {},
            "reserved_spend": 0, "spent": 0, "reserved_reward": 0,
            "founding_completed": [], "advanced": [], "clock": 0,
            "reward_policy": None, "payout_boundaries": {}, "king_history": []}


def reduce_event(previous, kind, data):
    s = deepcopy(previous)
    d = deepcopy(data)
    now = d.pop("now")
    finite(now)
    # Wall time controls leases; commit time orders concurrent journal writers.
    committed = d.pop("commit_time", now)
    finite(committed)
    if committed < max(now, s["clock"]):
        raise ValueError("event time moved backwards")
    s["clock"] = committed
    p = s["policy"]
    jobs = s["jobs"]
    j = jobs.get(d.get("job"))

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    block = None
    if "rounds" in s and kind in ROUND_EVENTS:
        block = d.pop("block", None)
        require(type(block) is int and block >= s["rounds"]["block"], "round events need a non-decreasing chain block")
        s["rounds"]["block"] = block

    def round_of(at):
        return (at - p["round_origin"]) // p["round_blocks"]

    def fenced():
        require(j is not None and j["state"] not in TERMINAL, "inactive job")
        require(d["fence"] == j["fence"] and now < j["lease_until"], "stale worker lease")

    def finish_charge():
        if not j["budget_released"]:
            s["reserved_spend"] -= j["cap"] - j["spent"]
            j["budget_released"] = True

    if kind == "cohort":
        c = d["cohort"]
        schedule = {**c["schedule"]}
        schedule["thresholds"] = tuple(schedule["thresholds"])
        schedule["batches"] = tuple(schedule["batches"])
        Cohort(**{**c, "schedule": BatchSchedule(**schedule)})
        key = digest(c)
        require(c["exposure"] == s["exposure"], "cohort exposure mismatch")
        require(s["anchor"] in (None, c["anchor"]), "cohort anchor mismatch")
        require(key not in s["cohorts"], "cohort already exists")
        if block is not None:
            r = round_of(block)
            require(str(r) not in s["rounds"]["opened"], "one cohort per round")
            require(all(old["winner"] is None or k in s["advanced"] for k, old in s["cohorts"].items()
                        if old["contract"]["anchor"] == s["anchor"]), "advance the previous round's winner first")
            s["rounds"]["opened"][str(r)] = key
        s["anchor"] = c["anchor"]
        s["cohorts"][key] = {"contract": c, "cache": {}, "winner": None,
                             **({"round": round_of(block)} if block is not None else {})}
    elif kind == "admit":
        terms = d["terms"]
        if "authorization" in d:
            from ..direct.signing import verify
            authorization = verify(d["authorization"], expected_signer=terms["payer"], scope="fund-job")
            require(authorization["terms"] == terms and authorization["job"] == d["job"] and authorization["nonce"] == d["job"], "funding signature does not bind admitted terms")
            require(all(old["terms"]["funding_ref"] != terms["funding_ref"] for old in jobs.values()), "funding grant already bound to another job")
            if terms["role"] == "incumbent":
                require(bool(s["king"] or d.get("bootstrap_payer")) and terms["payer"] == (s["king"] or d.get("bootstrap_payer")), "incumbent requires king or configured bootstrap payer")
        required = {"cohort", "role", "payer", "generator", "funding_ref", "anchor",
                    "budget", "spend_cap", "reward", "slot", "profiles", "providers",
                    "world_size", "reference", "guard_contract"}
        require(set(terms) == required, "incomplete job terms")
        require(type(terms["world_size"]) is int and terms["world_size"] > 0, "invalid gang size")
        require(bool(terms["profiles"] and terms["providers"]), "pin qualified resources/providers")
        jid = d["job"]
        require(jid not in jobs, "job identity already admitted")
        require(len([x for x in jobs.values() if x["state"] not in TERMINAL]) < p["max_jobs"], "admission backpressure")
        require(terms["cohort"] in s["cohorts"], "unknown cohort")
        c = s["cohorts"][terms["cohort"]]["contract"]
        require(terms["role"] in ("challenger", "incumbent", "expert"), "invalid role")
        require(bool(terms["payer"] and terms["generator"] and terms["funding_ref"]), "unfunded admission")
        require(terms["anchor"] == c["anchor"] and terms["budget"] == c["budget"], "unmatched anchor/budget")
        require(terms["anchor"] == s["anchor"], "stale anchor admission")
        require((terms["role"] == "expert") == (s["phase"] == "specialist"), "wrong phase")
        if terms["role"] == "expert":
            require(now < s["specialist_until"], "specialist window expired")
        cap, reward = finite(terms["spend_cap"]), finite(terms["reward"])
        require(type(cap) is int and type(reward) is int, "money must use integer smallest units")
        require(terms["role"] != "challenger" or reward == 0, "challenger completion is not an incumbent obligation")
        require(s["spent"] + s["reserved_spend"] + cap <= p["spend_pool"], "spending cap exhausted")
        require(sum(s["paid"].values()) + s["reserved_reward"] + reward <= p["reward_pool"], "reward pool overcommitted")
        if terms["role"] == "incumbent":
            slot = terms["slot"]
            require(slot == terms["cohort"], "incumbent obligation must identify its cohort")
            require(slot not in s["slots"], "incumbent obligation already committed")
            s["slots"][slot] = jid
        jobs[jid] = {"terms": terms, "state": "admitted", "cap": cap, "spent": 0,
                     "budget_released": False, "fence": 0, "lease_until": 0,
                     "checkpoint": None, "consumed": 0, "reservation": None,
                     "attempt": 0, "next_retry": now, "admitted_at": now,
                     "capacity_since": None, "capacity_wait": 0, "failures": 0,
                     "cure_until": None, "evaluations": {}, "receipt": None,
                     "progress": {}, "reason": "", "charges": {}}
        s["reserved_spend"] += cap
        s["reserved_reward"] += reward
    elif kind == "bind_training":
        require(j is not None and j["state"] == "admitted", "bind training before external effects")
        spec = d["spec"]
        require(isinstance(spec.get("implementation"), str) and len(spec["implementation"]) == 64, "pin worker implementation fingerprint")
        require(spec["job"] == d["job"] and spec["terms_digest"] == digest(j["terms"]), "training admission mismatch")
        c = s["cohorts"][j["terms"]["cohort"]]["contract"]
        require(spec["budget"] == c["budget"] and spec["exposure"] == c["exposure"] and digest(spec["schedule"]) == digest(c["schedule"]), "training scientific contract mismatch")
        require(digest(spec["architecture"]) == c["architecture"] and spec["world_size"] == j["terms"]["world_size"], "architecture/topology mismatch")
        require(spec["seed"] == c["seed"] and digest(spec["optimizer"]) == c["optimizer_contract"], "seed/optimizer contract mismatch")
        if c["exposure"] > 0:
            anchors = [old["checkpoint"] for old in jobs.values() if old["checkpoint"] is not None and digest(old["checkpoint"]) == c["anchor"]]
            anchors += [old["manifest"] for old in s["merged"] if "manifest" in old and digest(old["manifest"]) == c["anchor"]]
            require(bool(anchors) and spec.get("initial") == anchors[0]["ranks"]["0"], "warm start must use the promoted continuation checkpoint")
        else:
            require(not spec.get("initial"), "founders must share the contracted random initialization")
        require("work" not in j, "training already bound")
        require(spec["device"] != "cuda" or spec["storage"]["kind"] in ("hippius", "s3"), "CUDA recovery requires remote storage")
        if j["terms"]["role"] == "expert":
            from ..direct.experts import verify_selection
            require(bool(spec.get("selection")) and bool(spec.get("expert_domain")), "specialist requires pinned selection/domain")
            cfg = spec["architecture"]["config"]
            verify_selection(spec["selection_evidence"], anchor=c["anchor"], domain=spec["expert_domain"], selection=spec["selection"], layers=len(cfg["moe_blocks"]), experts=cfg["num_experts"])
        else:
            require(not spec.get("selection"), "broad treatment cannot restrict expert rows")
        j["work"] = spec
    elif kind == "reserve_intent":
        require(j is not None and j["state"] in ("admitted", "waiting_capacity", "retry_wait"), "job not reservable")
        require(now >= j["next_retry"], "retry not due")
        require(j["reservation"] is None, "reconcile previous reservation first")
        require(d["profile"] in j["terms"]["profiles"] and d["provider"] in j["terms"]["providers"], "unapproved hardware profile/provider")
        j["attempt"] += 1
        j["reservation"] = {"key": d["key"], "provider": d["provider"], "profile": d["profile"], "status": "intent"}
        j["state"] = "reserving"
    elif kind == "reserved":
        require(j is not None and j["state"] == "reserving", "no reservation intent")
        require(d["key"] == j["reservation"]["key"], "reservation mismatch")
        require(d["ranks"] == list(range(j["terms"]["world_size"])), "incomplete gang")
        require(d["lease_until"] > now, "expired allocation")
        if "committed_charge" in d:
            amount = d["committed_charge"]
            require(type(amount) is int and 0 <= amount <= j["cap"] - j["spent"], "allocation exceeds remaining spending cap")
            j["charges"][d["key"]] = amount
            j["spent"] += amount
            s["spent"] += amount
            s["reserved_spend"] -= amount
        j["reservation"].update(status="held", allocation=d["allocation"])
        j["fence"] += 1
        j["lease_until"] = d["lease_until"]
        j["state"] = "reserved"
        if j["capacity_since"] is not None:
            j["capacity_wait"] += now - j["capacity_since"]
            j["capacity_since"] = None
    elif kind == "release_intent":
        require(j is not None and j["reservation"] is not None, "no reservation to release")
        j["reservation"]["status"] = "releasing"
        j["fence"] += 1
        j["lease_until"] = 0
        j["state"] = "releasing"
        j["reason"] = d["reason"]
    elif kind == "released":
        require(j is not None and j["state"] == "releasing", "no release intent")
        j["reservation"] = None
        j["state"] = "evaluating" if j["consumed"] == j["terms"]["budget"] else "retry_wait"
    elif kind == "fault":
        require(j is not None and j["state"] not in TERMINAL, "inactive job")
        require(j["reservation"] is None, "release/reconcile allocation before fault transition")
        reason = d["reason"]
        require(reason in ("capacity", "provider", "auth", "spend", "worker", "storage", "generator", "evaluation"), "unknown fault attribution")
        j["reason"] = reason
        j["failures"] += 1
        if reason in ("auth", "spend"):
            j["state"] = "funding_cure"
            j["cure_until"] = now + p["funding_cure_seconds"]
        elif reason == "evaluation":
            require(j["consumed"] == j["terms"]["budget"] and j["checkpoint"] is not None, "evaluation requires complete training")
            j["state"] = "inconclusive"
        else:
            if j["capacity_since"] is None:
                j["capacity_since"] = now
            elapsed = j["capacity_wait"] + now - j["capacity_since"]
            j["state"] = "paused_capacity" if (elapsed >= p["capacity_deadline"] or j["failures"] >= p["circuit_failures"]) else "waiting_capacity"
        # Deterministic jitter is receipt-visible, independent of process RNG.
        jitter = int(digest({"job": d["job"], "failure": j["failures"]})[:8], 16) / 2**32
        j["next_retry"] = now + min(p["retry_cap"], p["retry_base"] * 2**min(20, j["failures"] - 1)) * (.75 + .25 * jitter)
    elif kind == "expire_funding":
        require(j is not None and j["state"] == "funding_cure", "not attributable funding failure")
        require(now >= j["cure_until"], "funding cure deadline not elapsed")
        require(bool(d["evidence"]), "attribution evidence required")
        j["state"] = "contract_failure"
        finish_charge()
        s["reserved_reward"] -= j["terms"]["reward"]
    elif kind == "release_slot":
        require(j is not None and j["state"] == "contract_failure" and j["terms"]["role"] == "incumbent", "only a failed unfunded incumbent slot can be replaced")
        require(s["slots"].get(j["terms"]["slot"]) == d["job"], "slot already replaced")
        require(d["authorised_by"] == (s["king"] or j["terms"]["payer"]) and bool(d["evidence"]), "replacement authority/evidence required")
        del s["slots"][j["terms"]["slot"]]
    elif kind == "resume_queue":
        require(j is not None and j["state"] in ("paused_capacity", "funding_cure", "inconclusive"), "job not paused")
        require(bool(d["evidence"]), "explicit recovery evidence required")
        j["state"] = "evaluating" if j["consumed"] == j["terms"]["budget"] else "retry_wait"
        j["next_retry"] = now
    elif kind == "launch":
        fenced()
        require(j["state"] == "reserved", "gang not reserved")
        require(d["ready_ranks"] == list(range(j["terms"]["world_size"])), "rank missing at launch")
        if "work" in j:
            j["launch_resume"] = j["checkpoint"]
        j["state"] = "running"
    elif kind == "progress":
        fenced()
        require(j["state"] in ("running", "checkpointing"), "worker not running")
        require(d["stage"] in ("training", "checkpointing", "generation"), "unknown stage")
        old = j["progress"].get(d["stage"], {"position": -1})
        require(d["position"] >= old["position"], "progress moved backwards")
        if d["position"] > old["position"]:
            j["progress"][d["stage"]] = {"position": d["position"], "at": now}
    elif kind == "checkpoint":
        fenced()
        require(j["state"] in ("running", "checkpointing"), "worker not running")
        cp = d["manifest"]
        require(cp["job"] == d["job"] and cp["fence"] == d["fence"], "checkpoint job/fence mismatch")
        require(cp["contract"] == digest(j["terms"]), "checkpoint contract mismatch")
        require(cp["kind"] == "continuation" and cp["accumulation_boundary"] is True, "evaluation/partial checkpoint cannot resume")
        require(type(cp["consumed"]) is int and j["consumed"] <= cp["consumed"] <= j["terms"]["budget"], "checkpoint budget mismatch")
        require(sorted(int(r) for r in cp["ranks"]) == list(range(j["terms"]["world_size"])), "checkpoint missing rank")
        components = {"weights", "optimizer", "scheduler", "router", "rng", "sampler", "generator"}
        for rank in cp["ranks"].values():
            require(set(rank) == components, "incomplete resume state")
            for obj in rank.values():
                require(obj["uri"].startswith(("hippius://", "s3://", "memory://", "file://")) and len(obj["sha256"]) == 64, "nonremote checkpoint")
        j["checkpoint"] = cp
        j["consumed"] = cp["consumed"]
        if cp["consumed"] == j["terms"]["budget"]:
            j["state"] = "checkpointing"
    elif kind == "allocation_no_charge":
        require(j is not None and j["state"] == "releasing", "allocation must be releasing")
        require(d["key"] == j["reservation"]["key"], "reservation mismatch")
        require(d["evidence"] in ("never_submitted", "provider_rejected"), "missing no-charge evidence")
        amount = j["charges"].get(d["key"], 0)
        j["charges"][d["key"]] = 0
        j["spent"] -= amount
        s["spent"] -= amount
        s["reserved_spend"] += amount
    elif kind == "charge":
        require(j is not None, "unknown job")
        charge_key, amount = d["charge_key"], finite(d["amount"])
        require(type(amount) is int, "charge requires integer units")
        if charge_key in j["charges"]:
            require(j["charges"][charge_key] == amount, "conflicting charge replay")
        else:
            require(not j["budget_released"] and j["spent"] + amount <= j["cap"], "spending cap exhausted")
            j["charges"][charge_key] = amount
            j["spent"] += amount
            s["spent"] += amount
            s["reserved_spend"] -= amount
    elif kind == "eval_case":
        require(j is not None and j["state"] in ("evaluating", "inconclusive") and j["checkpoint"] is not None and j["consumed"] == j["terms"]["budget"], "training incomplete")
        require(d["guard"] == j["terms"]["guard_contract"], "evaluation fingerprint mismatch")
        require(d["candidate"] == digest(j["checkpoint"]) and d["reference"] == j["terms"]["reference"], "evaluation model mismatch")
        finite(d["king_score"])
        finite(d["candidate_score"])
        value = {"king": d["king_score"], "candidate": d["candidate_score"]}
        old = j["evaluations"].get(d["case"])
        require(old is None or old == value, "conflicting cached evaluation")
        j["evaluations"][d["case"]] = value
    elif kind == "evaluation":
        require(j is not None and j["state"] in ("evaluating", "inconclusive") and j["checkpoint"] is not None and j["consumed"] == j["terms"]["budget"], "training incomplete")
        require(j["reservation"] is None, "release GPU gang before evaluating")
        if "work" in j:
            require(any(digest(e["public"]) == digest(d["receipt"]) for e in j.get("decisions", {}).values()), "production evaluation requires replayable private evidence")
        v = replay_receipt(d["receipt"])
        require(v["candidate"] == digest(j["checkpoint"]), "candidate checkpoint mismatch")
        require(v["reference"] == j["terms"]["reference"], "evaluation reference mismatch")
        require(v["contract"] == j["terms"]["guard_contract"], "evaluation contract mismatch")
        j["receipt"] = d["receipt"]
        j["state"] = v["status"]
        if v["status"] in TERMINAL:
            finish_charge()
            if j["terms"]["role"] != "expert" and d["job"] not in s["founding_completed"]:
                s["founding_completed"].append(d["job"])
            if v["passed"]:
                c = s["cohorts"][j["terms"]["cohort"]]
                c["cache"][j["terms"]["generator"]] = digest(j["checkpoint"])
                if j["terms"]["role"] != "expert":
                    s["rewards"][d["job"]] = {"payer": j["terms"]["payer"], "amount": j["terms"]["reward"]}
            else:
                s["reserved_reward"] -= j["terms"]["reward"]
    elif kind == "decision_evidence":
        from ..direct.evaluation import replay_decision
        require(j is not None, "unknown job")
        replay_decision(d["decision"])
        require(d["decision"]["public"]["verdict"]["candidate"] == digest(j["checkpoint"]), "decision checkpoint mismatch")
        j.setdefault("decisions", {})[digest(d["decision"])] = d["decision"]
    elif kind == "comparison":
        require(j is not None and j["checkpoint"] is not None and j["consumed"] == j["terms"]["budget"] and j["state"] in ("accepted", "rejected"), "comparison requires completed evaluated training")
        require(j["terms"]["role"] != "expert", "broad comparison required")
        c = s["cohorts"][j["terms"]["cohort"]]
        require(c["contract"]["anchor"] == s["anchor"] and s["phase"] == "broad", "stale comparison")
        v = replay_receipt(d["receipt"])
        ref = digest(jobs[c["winner"]]["checkpoint"]) if c["winner"] else j["terms"]["reference"]
        require(v["reference"] == ref and v["candidate"] == digest(j["checkpoint"]), "comparison model mismatch")
        require(v["contract"] == j["terms"]["guard_contract"], "comparison guard mismatch")
        j.setdefault("comparisons", {})[ref] = d["receipt"]
    elif kind == "settle":
        require(j is not None and j["state"] in ("accepted", "rejected") and j["terms"]["role"] != "expert", "not an evaluated broad contender")
        c = s["cohorts"][j["terms"]["cohort"]]
        require(c["contract"]["anchor"] == s["anchor"] and s["phase"] == "broad", "stale settlement")
        previous_winner = c["winner"]
        if previous_winner is not None and previous_winner != d["job"]:
            ref = digest(jobs[previous_winner]["checkpoint"])
            comparison = j.get("comparisons", {}).get(ref, j["receipt"])
            require(comparison["verdict"]["reference"] == ref and comparison["verdict"]["passed"], "duel did not compare current cached king")
        else:
            require(j["receipt"]["verdict"]["passed"], "initial comparison failed")
        c["winner"] = d["job"]
        s["king"] = j["terms"]["payer"]
        if not any(row["contribution"] == d["job"] for row in s["king_history"]):
            s["king_history"].append({"payer": s["king"], "contribution": d["job"], "boundary": now,
                                      **({"round": round_of(block)} if block is not None else {})})
    elif kind == "advance":
        require(j is not None and j["state"] in ("accepted", "rejected"), "unevaluated endpoint")
        ckey = j["terms"]["cohort"]
        c = s["cohorts"][ckey]
        require(c["winner"] == d["job"] and s["phase"] == "broad", "only settled winner advances")
        require(ckey not in s["advanced"], "cohort already advanced")
        require(s["anchor"] == c["contract"]["anchor"], "stale advancement")
        if block is not None:
            require(round_of(block) > c["round"], "a round's winner advances only after the round closes")
        require(len(s["founding_completed"]) >= p["founding_count"] or now >= p["founding_deadline"], "founding window open")
        s["anchor"] = digest(j["checkpoint"])
        s["exposure"] += c["contract"]["budget"]
        s["advanced"].append(ckey)
        if s["exposure"] >= 200_000_000_000 and d["routing_stable"] is True:
            s["phase"] = "specialist"
            s["specialist_until"] = now + p["specialist_seconds"]
    elif kind == "merge":
        require(s["phase"] == "specialist", "specialist phase not open")
        ids = d["jobs"]
        require(len(set(ids)) == len(ids), "duplicate expert contribution")
        require(all(jobs[x]["state"] == "accepted" and jobs[x]["terms"]["role"] == "expert" and jobs[x]["terms"]["anchor"] == s["anchor"] for x in ids), "invalid individual expert results")
        if ids:
            if "decision" in d:
                from ..direct.evaluation import replay_decision
                replay_decision(d["decision"])
                require(d["decision"]["public"] == d["receipt"] and digest(d["manifest"]) == d["combined"], "combined evidence mismatch")
            v = replay_receipt(d["receipt"])
            require(v["reference"] == s["anchor"], "combined reference mismatch")
            require(v["candidate"] == d["combined"], "combined artifact mismatch")
            require(all(v["contract"] == jobs[x]["terms"]["guard_contract"] for x in ids), "combined guard mismatch")
            if v["status"] == "inconclusive":
                raise ValueError("combined evaluation incomplete; retain window/results")
            if v["passed"]:
                s["anchor"] = d["combined"]
                s["merged"].append({"jobs": ids, "receipt": d["receipt"], **({"manifest": d["manifest"], "decision": d["decision"]} if "manifest" in d else {})})
                for x in ids:
                    s["rewards"][x] = {"payer": jobs[x]["terms"]["payer"], "amount": jobs[x]["terms"]["reward"]}
            else:
                for x in ids:
                    s["reserved_reward"] -= jobs[x]["terms"]["reward"]
        else:
            require(now >= s["specialist_until"], "empty specialist window not expired")
        # Refuse to abandon funded in-flight specialist jobs at the boundary.
        require(not any(x["terms"]["role"] == "expert" and x["terms"]["anchor"] == (d.get("receipt", {}).get("verdict", {}).get("reference", s["anchor"])) and x["state"] not in TERMINAL for x in jobs.values()), "funded specialists still in flight")
        require(all(x in ids or jobs[x]["state"] != "accepted" for x in jobs if jobs[x]["terms"]["role"] == "expert" and jobs[x]["terms"]["anchor"] == d.get("receipt", {}).get("verdict", {}).get("reference", s["anchor"])), "omit no accepted specialist")
        s["phase"] = "broad"
        s["specialist_until"] = None
    elif kind == "reward_policy":
        from ..shared.direct_rewards import RewardPolicy
        require(s["reward_policy"] is None and not jobs, "reward policy must be fixed before admission")
        RewardPolicy(**d["policy"])
        s["reward_policy"] = d["policy"]
    elif kind == "payout_boundary":
        from ..shared.direct_rewards import RewardPolicy, allocate_boundary
        require(s["reward_policy"] is not None, "unresolved royalty/pause payout policy")
        key = str(d["boundary"])
        require(d["boundary"] == now, "payout boundary must match recorded protocol clock")
        require(key not in s["payout_boundaries"], "payout boundary already accounted")
        stale = None
        if block is not None:
            era = (block - p["round_origin"]) // p["era_blocks"]
            require(era not in s["rounds"]["paid_eras"], "one payout per era")
            s["rounds"]["paid_eras"].append(era)
            # A king without a promotion of its own in the last full round is stale.
            last = max((r["round"] for r in s["king_history"] if r["payer"] == s["king"] and "round" in r), default=None)
            stale = last is None or last < round_of(block) - 1
        # Pool allocations are kept separate from previously escrowed completion
        # entitlements. They do not silently enlarge the admitted reward pool.
        s["payout_boundaries"][key] = allocate_boundary(
            RewardPolicy(**s["reward_policy"]), emission=d["emission"], boundary=now,
            king=s["king"], history=s["king_history"], paused=d["paused"], stale=stale)
    elif kind == "settle_boundary":
        # Emissions settle entitlements (validator weights), never transfers.
        # The split is derived here from journal state, so every validator
        # replaying the journal reaches the same weights without trusting it.
        from ..shared.direct_rewards import continuation_shares
        key = str(d["boundary"])
        require(key in s["payout_boundaries"], "unknown payout boundary")
        alloc = s["payout_boundaries"][key]
        require("continuation_paid" not in alloc, "payout boundary already settled")
        shares = continuation_shares(s["rewards"], s["paid"], alloc["continuation_pool"])
        for job, amount in shares.items():
            s["paid"][job] = s["paid"].get(job, 0) + amount
            s["reserved_reward"] -= amount
        alloc["continuation_paid"] = {job: {"payer": s["rewards"][job]["payer"], "amount": amount}
                                      for job, amount in shares.items()}
        alloc["continuation_burn"] = alloc["continuation_pool"] - sum(shares.values())
    elif kind == "pay":
        require(d["job"] in s["rewards"], "conditional entitlement not earned")
        entitlement = s["rewards"][d["job"]]
        if d["job"] not in s["paid"]:
            s["paid"][d["job"]] = entitlement["amount"]
            s["reserved_reward"] -= entitlement["amount"]
    else:
        raise ValueError(f"unknown direct-MoE event {kind}")
    return s


class DirectLedger:
    def __init__(self, path, policy):
        self.db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError("unsupported direct ledger schema")
        self.db.executescript('''CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, command TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL, prev TEXT NOT NULL, hash TEXT NOT NULL);
        PRAGMA user_version=1;''')
        state = initial_state(policy)
        self.db.execute("INSERT OR IGNORE INTO state VALUES (1, ?)", (canonical(state),))
        if self.state["policy"] != policy:
            raise ValueError("policy changed for existing ledger")

    @property
    def state(self):
        return json.loads(self.db.execute("SELECT body FROM state WHERE id=1").fetchone()[0])

    def close(self):
        self.db.close()

    def apply(self, command, kind, **data):
        if "commit_time" in data:
            raise ValueError("commit ordering is assigned by the ledger")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT kind, body FROM events WHERE command=?", (command,)).fetchone()
            if old:
                # Arrival time is metadata, not the semantic identity of a command.
                old_payload = {k: v for k, v in json.loads(old[1]).items() if k not in ("now", "commit_time")}
                new_payload = {k: v for k, v in data.items() if k not in ("now", "commit_time")}
                if old[0] != kind or canonical(old_payload) != canonical(new_payload):
                    raise ValueError("idempotency key reused with different event")
                self.db.execute("COMMIT")
                return self.state
            previous = self.state
            if data["now"] < previous["clock"]:
                data["commit_time"] = previous["clock"]
            body = canonical(data)
            state = reduce_event(previous, kind, data)
            last = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last[0] if last else digest(initial_state(state["policy"]))
            event_hash = digest({"command": command, "kind": kind, "data": data, "prev": prev})
            self.db.execute("INSERT INTO events(command,kind,body,prev,hash) VALUES (?,?,?,?,?)", (command, kind, body, prev, event_hash))
            self.db.execute("UPDATE state SET body=? WHERE id=1", (canonical(state),))
            self.db.execute("COMMIT")
            return state
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def export(self):
        self.db.execute("BEGIN")
        try:
            result = self._export_snapshot()
            self.db.execute("COMMIT")
            return result
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _export_snapshot(self):
        return {"protocol": "direct-moe-ledger/1", "policy": self.state["policy"],
                "events": [{"command": r[0], "kind": r[1], "data": json.loads(r[2]), "prev": r[3], "hash": r[4]}
                           for r in self.db.execute("SELECT command,kind,body,prev,hash FROM events ORDER BY seq")],
                "state": self.state}


def replay_ledger(receipt):
    if receipt["protocol"] != "direct-moe-ledger/1":
        raise ValueError("unsupported journal protocol")
    state = initial_state(receipt["policy"])
    prev = digest(state)
    seen = set()
    for event in receipt["events"]:
        body = {k: event[k] for k in ("command", "kind", "data", "prev")}
        if event["prev"] != prev or digest(body) != event["hash"] or event["command"] in seen:
            raise ValueError("invalid event chain")
        seen.add(event["command"])
        state = reduce_event(state, event["kind"], event["data"])
        prev = event["hash"]
    if canonical(state) != canonical(receipt["state"]):
        raise ValueError("ledger state replay mismatch")
    return state
