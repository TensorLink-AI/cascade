"""Pin the effective contract and init together for a whole local comparison."""
from __future__ import annotations

from dataclasses import fields, replace

from ..shared.manifest import contract_digest
from .live_status import LiveStateUnavailable


class UnresolvedLaunch(ValueError):
    """``--at-block`` cannot bind a contract while the dated v2 launch is
    unresolved offline: every block would silently fall back to the pre-v2
    policy and pass a generator the launched trainer rejects."""


def configured_context(cfg, block=None):
    """Explicit offline contract; no network and no claim that it is live."""
    if block is not None:
        from ..shared.calendar_launch import dated_launch_pending

        if dated_launch_pending(cfg):
            raise UnresolvedLaunch(
                f"--at-block {int(block)} cannot resolve the dated v2 launch "
                f"({cfg.v2.launch_at_utc}) offline: [v2] from_block is 0 in this chain.toml, so "
                "every block would bind the pre-v2 policy. Omit --at-block to verify the "
                "intersection of both policies, or use --live to bind the deployed contract.")
    contract = cfg.throne_contracts_at(block)[0] if block is not None else cfg.screen_contract()
    return with_contract(cfg, contract)


def with_contract(cfg, contract):
    return replace(cfg, training=contract,
                   round=replace(cfg.round, screen_size=contract.arch_preset),
                   eval=replace(cfg.eval, context_length=contract.context_length, horizon=contract.horizon),
                   generator=replace(cfg.generator_for_contract(contract), accepted_fields=contract.accepted_fields,
                                     allow_future_known=contract.allow_future_known,
                                     real_corpus_ref=contract.real_corpus_ref))


def contract_from_snapshot(snapshot):
    from ..shared.config.training import parse_training_contract

    if not snapshot.get("ready_for_training"):
        raise LiveStateUnavailable("Current era init is stale or unconfirmed; refresh cascade status.")
    doc = snapshot.get("training_contract") or {}
    if doc.get("schema") != 1 or doc.get("era_start_block") != snapshot["warm_start"].get("era_start_block"):
        raise LiveStateUnavailable("Current era training contract is unavailable; updated trainer discovery is required.")
    try:
        body = doc["body"]
        contract = parse_training_contract(body)
        if contract_digest(body) != doc["digest"] or contract_digest(contract) != doc["digest"]:
            raise ValueError("contract digest differs or this CLI cannot represent the contract")
        if contract.base_arch not in ("toto2", "cascade_moe", "cascade_pyramid"):
            raise ValueError("unsupported architecture")
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveStateUnavailable(f"Unusable live contract: {exc}") from exc
    return contract


def live_context(cfg):
    from .live_status import fetch_snapshot

    snapshot = fetch_snapshot(cfg)
    contract = contract_from_snapshot(snapshot)
    cfg = with_contract(published_policy(cfg, snapshot), contract)
    return cfg, snapshot["warm_start"]["pointer"] or None, snapshot


def published_policy(cfg, snapshot):
    for section in ("lane_b", "marketcast", "eval", "generator", "generation", "scoring"):
        obj = getattr(cfg, section)
        published = (snapshot.get("policy") or {}).get(section)
        if published:
            known = {f.name for f in fields(obj)}
            if set(published) - known:
                raise LiveStateUnavailable(f"Update the CLI: unknown {section} policy fields.")

            def freeze(value):
                return tuple(freeze(v) for v in value) if isinstance(value, list) else value

            cfg = replace(cfg, **{section: replace(obj, **{k: freeze(v) for k, v in published.items()})})
    return cfg
