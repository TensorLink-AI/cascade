"""Owner-signed direct-MoE network config (DEC-CA-0052 §8).

The owner's main validator signs and publishes the rules every other validator
enforces: the trainer hotkey the journal must be signed by, where the journal
lives, the ledger and reward policies, the evaluation guards and private
policies a decision may use, the burn uid and an all-burn kill switch.
Validators pin only the owner hotkey and the config bucket; every other change
is a republish here, never a validator update or a ``chain.toml`` edit.

Versions are hash-chained (``seq`` + ``prev``) and a validator never accepts a
lower ``seq`` than one it has seen, so an old signed config cannot be replayed.
Guards and private policies are allow-lists: a journal that already used one
stays valid only while it remains listed, so retire one by rotating the cohort
first, then dropping it.

Chain anchor (§9): the owner hotkey commits ``cascade-direct:1:<digest(body)>``
as its plain commitment. Validators accept only the config whose digest is
committed, and the chain's block for that commitment is its timestamp: a panel
newly added to ``private_panels`` must start (``from_block``) at or after it,
so a panel cannot be listed after the scores it produced were known.

Auditability (§9): every decision must carry its scoring binding, including the
chain block it was scored at, and use the panel active at that block (one at a
time: the entry with the greatest ``from_block`` <= it); its bound checkpoints
must be publicly fetchable; and a private panel first used more than
``reveal_after`` seconds before the boundary being paid must have been revealed,
or validators burn that boundary until it is.
"""

from __future__ import annotations

import json

from ..shared.direct_moe import canonical, digest

PROTOCOL = "direct-network/1"
SCOPE = "network-config"
LATEST = "direct/network/latest.json"
COMMITMENT = "cascade-direct:1:"
FIELDS = {"protocol", "seq", "prev", "trainer", "journal", "burn_uid", "burn_all",
          "ledger_policy", "reward_policy", "guards", "private_policies", "private_panels",
          "reveal_after", "poll_seconds"}


def check_network_config(body):
    """Validate a config body's shape; returns it unchanged."""
    if set(body) != FIELDS or body["protocol"] != PROTOCOL:
        raise ValueError(f"network config fields must be exactly {sorted(FIELDS)}")
    if type(body["seq"]) is not int or body["seq"] < 1:
        raise ValueError("network config seq must be a positive integer")
    if (body["seq"] == 1) != (body["prev"] is None):
        raise ValueError("network config prev must be null exactly at seq 1")
    if not isinstance(body["trainer"], str) or not body["trainer"]:
        raise ValueError("network config must name the trainer hotkey")
    if not isinstance(body["journal"], dict) or not body["journal"].get("bucket"):
        raise ValueError("network config must name the journal bucket")
    if type(body["burn_uid"]) is not int or body["burn_uid"] < 0 or type(body["burn_all"]) is not bool:
        raise ValueError("invalid burn_uid/burn_all")
    if not isinstance(body["ledger_policy"], dict):
        raise ValueError("network config must pin the ledger policy")
    for key in ("guards", "private_policies"):
        if not isinstance(body[key], list) or not all(
                isinstance(x, str) and len(x) == 64 for x in body[key]):
            raise ValueError(f"{key} must be a list of sha256 digests")
    panels = body["private_panels"]
    if not isinstance(panels, list) or not all(
            isinstance(e, dict) and set(e) == {"panel", "from_block"}
            and isinstance(e["panel"], str) and len(e["panel"]) == 64
            and type(e["from_block"]) is int and e["from_block"] >= 0 for e in panels):
        raise ValueError("private_panels must be [{panel: sha256, from_block: int}]")
    starts = [e["from_block"] for e in panels]
    if starts != sorted(set(starts)) or len({e["panel"] for e in panels}) != len(panels):
        raise ValueError("private_panels must have distinct panels and strictly increasing from_block")
    for key in ("reveal_after", "poll_seconds"):
        if not isinstance(body[key], (int, float)) or isinstance(body[key], bool) or body[key] <= 0:
            raise ValueError(f"invalid {key}")
    return body


def _store(location):
    from .journal import _store

    return _store(location)


def config_key(signed):
    return f"direct/network/{digest(signed)}.json"


def publish_network_config(location, signed, *, store=None):
    """Write the immutable config object, then advance the ``latest`` pointer."""
    check_network_config(signed["body"])
    store = store or _store(location)
    key = config_key(signed)
    store.put_text(key, canonical(signed), content_type="application/json", acl="public-read")
    store.put_text(LATEST, canonical({"key": key, "sha256": digest(signed)}),
                   content_type="application/json", acl="public-read")
    return key


def latest_network_config(location, *, owner, store=None):
    """The current verified config body, or None before the first publish."""
    from ..shared.hippius import ObjectNotFound
    from .signing import verify

    store = store or _store(location)
    try:
        pointer = json.loads(store.get_text(LATEST))
    except ObjectNotFound:
        return None
    signed = json.loads(store.get_text(pointer["key"]))
    if digest(signed) != pointer["sha256"]:
        raise ValueError("network config object does not match its pointer")
    return check_network_config(verify(signed, expected_signer=owner, scope=SCOPE))


def check_round_clock(ledger_policy, round_cfg):
    """A round journal's clock must sit on the live era grid (DEC-CA-0052 §11)."""
    if "round_blocks" not in ledger_policy:
        return
    from ..shared.era import era_length_blocks

    origin, era = ledger_policy["round_origin"], ledger_policy["era_blocks"]
    if era != era_length_blocks(round_cfg, origin) or origin % era:
        raise ValueError(f"round_origin {origin}/era_blocks {era} are not on the live era grid "
                         f"(era length {era_length_blocks(round_cfg, origin)} blocks)")


def commitment_payload(body):
    return COMMITMENT + digest(body)


def fetch_network_config(location, *, owner, last=None, committed=None, store=None):
    """Verify the latest config against the owner hotkey; refuse rollbacks.

    ``last`` is the config this validator accepted before, if any.
    ``committed`` is the owner's on-chain ``(payload, block)``; with it the
    config must be the committed one and its new panels may not start before
    that block. Without it (offline) the chain anchor is not checked.
    """
    body = latest_network_config(location, owner=owner, store=store)
    if body is None:
        raise ValueError("no network config published yet")
    if committed is not None:
        payload, block = committed
        if payload != commitment_payload(body):
            raise ValueError("network config is not the one the owner committed on chain")
        # First sight (no ``last``) trusts the schedule; a running validator
        # checks every panel the owner adds or re-times after that.
        known = ({e["panel"]: e["from_block"] for e in last["private_panels"]}
                 if last is not None else {})
        for entry in body["private_panels"] if last is not None else ():
            if known.get(entry["panel"]) == entry["from_block"]:
                continue
            if entry["from_block"] < block:
                raise ValueError(f"panel {entry['panel'][:12]} starts at block {entry['from_block']}, "
                                 f"before its config was committed at {block}")
    if last is not None:
        if body["seq"] < last["seq"]:
            raise ValueError(f"network config rolled back: seq {body['seq']} < {last['seq']}")
        if body["seq"] == last["seq"] and digest(body) != digest(last):
            raise ValueError("two different network configs share one seq")
        if body["seq"] == last["seq"] + 1 and body["prev"] != digest(last):
            raise ValueError("network config does not chain from the last accepted one")
    return body


def next_network_config(rules, previous):
    """Fill ``protocol``/``seq``/``prev`` for a new version of ``rules``."""
    return check_network_config({
        **rules, "protocol": PROTOCOL,
        "seq": 1 if previous is None else previous["seq"] + 1,
        "prev": None if previous is None else digest(previous)})


def active_panel(config, block):
    """The private panel scoring at ``block`` (None before the first starts)."""
    live = [e for e in config["private_panels"] if e["from_block"] <= block]
    return live[-1]["panel"] if live else None


def journal_decisions(state):
    """Every decision recorded in a replayed journal state, keyed by digest."""
    out = {}
    for job in state["jobs"].values():
        for decision in job.get("decisions", {}).values():
            out[digest(decision)] = decision
    for merge in state["merged"]:
        if "decision" in merge:
            out[digest(merge["decision"])] = merge["decision"]
    return dict(sorted(out.items()))


def check_conformance(journal, state, config, *, now_block=None):
    """Refuse a replayed journal that used rules the owner config does not list."""
    if canonical(journal["policy"]) != canonical(config["ledger_policy"]):
        raise ValueError("journal ledger policy differs from the network config")
    if (config["reward_policy"] is not None and state["reward_policy"] is not None
            and canonical(state["reward_policy"]) != canonical(config["reward_policy"])):
        raise ValueError("journal reward policy differs from the network config")
    if "rounds" in state and now_block is not None and state["rounds"]["block"] > now_block:
        raise ValueError(f"journal claims future round block {state['rounds']['block']}")
    guards, policies = set(config["guards"]), set(config["private_policies"])
    for job_id, job in state["jobs"].items():
        if job["terms"]["guard_contract"] not in guards:
            raise ValueError(f"job {job_id} admitted under a guard the network config does not list")
    for key, decision in journal_decisions(state).items():
        if digest(decision["public"]["guard"]) not in guards:
            raise ValueError(f"decision {key[:12]} used an unlisted guard")
        if digest(decision["private"]["policy"]) not in policies:
            raise ValueError(f"decision {key[:12]} used an unlisted private policy")
        if "scoring" not in decision:
            raise ValueError(f"decision {key[:12]} carries no scoring binding; not auditable")
        block = decision["scoring"].get("block")
        if type(block) is not int:
            raise ValueError(f"decision {key[:12]} records no scoring block; not auditable")
        if now_block is not None and block > now_block:
            raise ValueError(f"decision {key[:12]} claims future block {block}")
        if decision["scoring"]["private_panel"] != active_panel(config, block):
            raise ValueError(f"decision {key[:12]} was not scored on the private panel "
                             f"active at block {block}")


def decision_times(journal):
    """When each decision entered the journal (its event's commit time)."""
    out = {}
    for event in journal["events"]:
        data = event["data"]
        if event["kind"] in ("decision_evidence", "merge") and "decision" in data:
            out.setdefault(digest(data["decision"]), data.get("commit_time", data["now"]))
    return out


def audit_obligations(journal, state, config, *, available, revealed, boundary):
    """Check a conforming journal stays reproducible (DEC-CA-0052 §9).

    Raises when a decision's bound checkpoint is not publicly fetchable
    (``available(uri)``); returns the private panels overdue for reveal at
    ``boundary`` (first used ``reveal_after`` or more before it and not yet
    ``revealed(panel_digest)``).
    """
    times = decision_times(journal)
    first_use = {}
    for key, decision in journal_decisions(state).items():
        scoring = decision["scoring"]
        for pointer, manifest in sorted(scoring["manifests"].items()):
            if not available(manifest["ranks"]["0"]["weights"]["uri"]):
                raise ValueError(f"decision {key[:12]}: checkpoint {pointer[:12]} is not publicly fetchable")
        panel = scoring["private_panel"]
        first_use[panel] = min(first_use.get(panel, times[key]), times[key])
    return sorted(panel for panel, used in first_use.items()
                  if boundary - used >= config["reveal_after"] and not revealed(panel))
