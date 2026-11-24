"""``cascade rules <lane>`` — print the competition rules a lane is judged and
paid under, read from ``chain.toml`` (the same loader the trainer, intake and
validators use), so the per-lane miner guides never have to restate numbers
that can drift.

* ``rules a|b|c|d`` — the headline rules (what is scored, the bar, the pay
  schedule, the caps), each labelled with its ``[section] key``.
* ``--all`` — every key of the lane's sections, not only the headline.
* ``--key K`` — one value (``K`` or ``section.K``), for scripts.
* ``--chain-toml PATH`` — read a deployed copy instead of the checkout's
  template. The checked-in template is not proof of what is live: a signed
  overlay can change a key at an era boundary; ``cascade status --json``
  shows what the network currently runs.

Read only: no wallet, no network, no torch.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .output import OK, USAGE, reporter

# The chain.toml sections each lane's rules live in.
LANE_SECTIONS: dict[str, tuple[str, ...]] = {
    "a": ("v2", "scoring"),
    "b": ("lane_b",),
    "c": ("lane_c", "marketcast"),
    "d": ("lane_d",),
}

LANE_TITLES = {
    "a": "Lane A: synthetic-data generators (king of the hill)",
    "b": "Lane B: domain and suite experts",
    "c": "Lane C: MarketCast fine-tuning recipes",
    "d": "Lane D: open weights (cascade_pyramid checkpoints)",
}

LANE_GUIDES = {
    "a": "docs/MINER_LANE_A.md",
    "b": "docs/MINER_LANE_B.md",
    "c": "docs/MINER_LANE_C_MARKETCAST.md",
    "d": "docs/MINER_LANE_D.md",
}

# Headline rules: (section, key, what it means). Every pair must name a real
# config field (tests/accept/w10 pins it), so a renamed key fails CI instead
# of silently vanishing from the printout.
HEADLINES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "a": (
        ("v2", "from_block", "v2 lane A opened at this block"),
        ("v2", "win_margin_start", "dethrone margin against a fresh king (increment units)"),
        ("v2", "win_margin_end", "dethrone margin floor after the king's tenure"),
        ("v2", "margin_warmup_blocks", "blocks over which the margin decays start -> end"),
        ("v2", "bench_gate_mode", "public bench gate (GIFT-Eval/BOOM/TIME/FEV) can block a win"),
        ("scoring", "bench_gate_tolerance", "per-suite ratio bound allowed above 1"),
        ("scoring", "bench_gate_min_win_rate", "suite-weighted share of bench tasks to win"),
        ("v2", "ladder_start_tiers", "(crowning LCB, starting share of A in basis points)"),
        ("v2", "ladder_step_bp", "share of A gained per passed king check (bp)"),
        ("v2", "ladder_cap_bp", "ladder cap (bp of A)"),
        ("v2", "court_bp", "immediate former king's share of A (bp)"),
        ("v2", "court_blocks", "how long the former king is paid (blocks)"),
        ("v2", "random_init_eras", "eras trained from random init before warm starts"),
        ("scoring", "burn_uid", "UID that receives every unpaid share"),
    ),
    "b": (
        ("lane_b", "from_block", "lane B active from this block (0 = template default)"),
        ("lane_b", "share", "lane B's pool, as a fraction of all emissions"),
        ("lane_b", "domains", "domain targets"),
        ("lane_b", "horizons", "horizon targets (when horizon_targets is on)"),
        ("lane_b", "max_domains_per_leg", "declared targets one leg may name"),
        ("lane_b", "suite_targets", "suite:<s> targets accepted"),
        ("lane_b", "gain_margin", "gain LCB a declared target must clear"),
        ("lane_b", "tau_domain", "allowed regression on non-target domains"),
        ("lane_b", "tau_public", "allowed regression on each public suite"),
        ("lane_b", "require_public", "public suites checked for no regression"),
        ("lane_b", "min_experts", "fewest experts a leg trains"),
        ("lane_b", "max_experts", "most experts a leg trains"),
        ("lane_b", "leg_hours", "default leg length (hours)"),
        ("lane_b", "max_leg_hours", "longest leg a miner may request (hours)"),
        ("lane_b", "funded_pod_skus", "GPU layouts a lane B leg may rent"),
        ("lane_b", "contamination_min_hits", "benchmark sub-window hits that refuse a tree"),
    ),
    "c": (
        ("lane_c", "from_block", "lane C active from this block"),
        ("lane_c", "share", "lane C's pool, as a fraction of all emissions"),
        ("lane_c", "takes_from", "lane whose pool lane C is carved from"),
        ("lane_c", "targets", "the competition targets"),
        ("marketcast", "base_ref", "pinned base every recipe fine-tunes"),
        ("marketcast", "suites", "scored suites"),
        ("marketcast", "composite_suite_weights", "suite weights in the composite"),
        ("marketcast", "assets", "asset universe by class"),
        ("marketcast", "frequencies", "bar timeframes per suite"),
        ("marketcast", "horizons", "forecast horizons per suite (bar x steps)"),
        ("marketcast", "gain_margin", "composite gain LCB to pass"),
        ("marketcast", "min_windows", "fewest scored windows"),
        ("marketcast", "min_clusters", "fewest independent clusters"),
        ("marketcast", "embargo_days", "training data must end this many days before submission"),
        ("marketcast", "verdict_wait_blocks", "longest wait for an inconclusive entry (blocks)"),
        ("marketcast", "max_leg_hours", "longest leg a recipe may request (hours)"),
        ("marketcast", "max_lr", "largest learning rate"),
        ("marketcast", "max_steps", "most steps"),
        ("marketcast", "max_batch", "largest batch"),
        ("marketcast", "max_lora_rank", "largest LoRA rank"),
        ("marketcast", "max_context_length", "longest training context"),
        ("marketcast", "max_open_legs_per_coldkey", "open legs per coldkey"),
    ),
    "d": (
        ("lane_d", "from_block", "lane D active from this block (0 = unarmed template)"),
        ("lane_d", "share", "lane D's pool, as a fraction of all emissions"),
        ("lane_d", "takes_from", "lane whose pool lane D is carved from"),
        ("lane_d", "contexts", "contexts an entry may declare"),
        ("lane_d", "dtypes", "accepted weight dtypes"),
        ("lane_d", "horizon", "forecast horizon (steps)"),
        ("lane_d", "d_margin_start", "crown bar against a fresh KING_D (LCB of the gain)"),
        ("lane_d", "d_margin", "crown bar floor, reached after the warm-up"),
        ("lane_d", "d_margin_warmup_settlements", "settlements of KING_D tenure over which the bar decays"),
        ("lane_d", "alpha", "one-sided test level"),
        ("lane_d", "embargo_hours", "live windows are this much newer than the entry"),
        ("lane_d", "min_windows", "fewest windows before a verdict"),
        ("lane_d", "min_clusters", "fewest source clusters before a verdict"),
        ("lane_d", "max_windows_per_settlement", "windows drawn per settlement"),
        ("lane_d", "min_fresh_window_share", "least share of live windows that are fresh (not embargo)"),
        ("lane_d", "gate_suites", "bench suites the regression gate draws from"),
        ("lane_d", "tau_public", "allowed bench regression per metric"),
        ("lane_d", "reward_prior_kings", "prior kings kept in the paid court"),
        ("lane_d", "court_weights", "court pay split: king, newest prior king, next"),
        ("lane_d", "max_entries_per_settlement", "entries judged per settlement"),
        ("lane_d", "max_open_per_coldkey", "open entries per coldkey"),
        ("lane_d", "max_judged_per_coldkey_per_settlement", "entries judged per coldkey per settlement"),
        ("lane_d", "max_coldkey_unresolved_settlements", "settlements an unresolved coldkey may wait"),
        ("lane_d", "max_reveal_delay_days", "days an entry may stay pending"),
        ("lane_d", "ready_ttl_blocks", "blocks between ready and the on-chain commit"),
        ("lane_d", "max_mirror_lag_hours", "largest data mirror lag before skipping"),
    ),
}


def _plain(value):
    """A config value as JSON-friendly data (tuples -> lists, dataclasses -> dicts)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def section_values(cfg, section: str) -> dict:
    """Every field of one loaded ``[section]`` (dataclass order)."""
    obj = getattr(cfg, section)
    return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


def headline_rows(cfg, lane: str) -> list[dict]:
    """The lane's headline rules as ``{section, key, value, meaning}`` rows."""
    return [{"section": s, "key": k, "value": _plain(getattr(getattr(cfg, s), k)), "meaning": m}
            for s, k, m in HEADLINES[lane]]


def lookup(cfg, lane: str, key: str):
    """``key`` or ``section.key`` within the lane's sections; raises KeyError."""
    section, _, name = key.rpartition(".")
    sections = (section,) if section else LANE_SECTIONS[lane]
    for s in sections:
        if s in LANE_SECTIONS[lane] and name in section_values(cfg, s):
            return s, name, section_values(cfg, s)[name]
    raise KeyError(key)


def add_rules(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "rules",
        help="Print a lane's live competition rules from chain.toml (read only).",
        description="Print the rules one lane is judged and paid under, read from chain.toml "
                    "with the validators' own loader. No wallet, network or GPU needed.",
        epilog="examples:\n"
               "  cascade rules d                     headline lane D rules\n"
               "  cascade rules b --all               every [lane_b] key\n"
               "  cascade rules c --key gain_margin   one value\n"
               "  cascade rules a --chain-toml /path/to/chain.toml --json\n"
               "Per-lane guides: " + ", ".join(LANE_GUIDES.values()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lane", choices=sorted(LANE_SECTIONS), type=str.lower,
                   help="The lane: a (generators), b (experts), c (MarketCast), d (open weights).")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--all", action="store_true",
                       help="Print every key of the lane's chain.toml sections.")
    which.add_argument("--key", default=None, metavar="KEY",
                       help="Print one key (KEY or section.KEY).")
    p.add_argument("--chain-toml", type=Path, default=None,
                   help="Read this chain.toml (e.g. a deployed copy) instead of the checkout's.")
    p.set_defaults(func=cmd_rules, command="rules")


def cmd_rules(args: argparse.Namespace) -> int:
    from ..shared.config import load_chain_config

    out = reporter(args)
    lane = args.lane
    cfg = load_chain_config(args.chain_toml)
    source = str(args.chain_toml) if args.chain_toml is not None else "chain.toml (checkout template)"
    out.result(lane=lane, title=LANE_TITLES[lane], guide=LANE_GUIDES[lane], source=source,
               sections=list(LANE_SECTIONS[lane]))
    if args.key is not None:
        try:
            section, name, value = lookup(cfg, lane, args.key)
        except KeyError:
            return out.fail("unknown_key", f"--key {args.key!r} is not a key of "
                            f"{', '.join(f'[{s}]' for s in LANE_SECTIONS[lane])}", USAGE)
        out.result(section=section, key=name, value=value)
        out.info(f"[{section}] {name} = {value!r}")
        return OK
    out.info(f"{LANE_TITLES[lane]}  (source: {source}; guide: {LANE_GUIDES[lane]})")
    if args.all:
        values = {s: section_values(cfg, s) for s in LANE_SECTIONS[lane]}
        out.result(values=values)
        for s, table in values.items():
            out.info(f"[{s}]")
            for k, v in table.items():
                out.info(f"  {k} = {v!r}")
        return OK
    rows = headline_rows(cfg, lane)
    out.result(rules=rows)
    for row in rows:
        out.info(f"  [{row['section']}] {row['key']} = {row['value']!r}   # {row['meaning']}")
    out.info("A signed overlay can change these at an era boundary; "
             "`cascade status --json` shows what is live.")
    return OK
