"""Check guides against CLI definitions, links, configuration and reward calculations.

Run offline from the checkout: .venv/bin/python scripts/check_docs.py
Historical decisions, archived competitors and third-party generator docs are
outside the current-guide scope. Checks cover interfaces and marked training,
configuration and reward references, not arbitrary prose, remote availability
or live workflow execution.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG_START = "<!-- cascade:config-reference:start -->"
CONFIG_END = "<!-- cascade:config-reference:end -->"


def guide_paths(root=ROOT):
    return sorted({*(root / "docs").glob("*.md"),
                   *(root / name for name in ("README.md", "AGENTS.md", "CLAUDE.md",
                     "agent.md", "llms.txt", "scripts/README.md", "benchmarks/README.md",
                     "cascade/website/llms.txt"))})


def config_reference(root=ROOT):
    """Render checked-in values, without implying that a deployment uses them."""
    from cascade.shared.config import load_chain_config

    rows = ["| Template | V2 block | Loaded B gate | C / MarketCast blocks | Random-init eras | B pool | CPU generation |",
            "|---|---:|---:|---|---:|---|---|"]
    for name in ("chain.toml", "chain.testnet.toml"):
        cfg = load_chain_config(root / name)
        generation = cfg.generation
        mode = "enabled for shards" if generation.enabled else "off"
        rows.append(f"| `{name}` | {cfg.v2.from_block} | {cfg.lane_b.from_block} | "
                    f"{cfg.lane_c.from_block} / {cfg.marketcast.from_block} | "
                    f"{cfg.v2.random_init_eras} | {cfg.lane_b.share:.0%} | "
                    f"{mode}; {generation.window_seconds // 60} min configured |")
    return "\n".join(rows)


def training_reference(root=ROOT):
    """Show template contracts separately from live deployment state."""
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(root / "chain.toml")
    before, after = cfg.training, cfg.training.v2_contract
    rows = ["| Setting | Mainnet before v2 | Mainnet v2 A |",
            "|---|---|---|"]
    for label, field in (("Model preset", "arch_preset"), ("Corpus feed", "corpus_mode"),
                         ("Budget accounting", "budget_denomination"),
                         ("Batch accounting", "batch_denomination"),
                         ("Batch size", "batch_size"), ("Model context length", "context_length"),
                         ("Training wall limit (seconds)", "max_train_seconds")):
        rows.append(f"| {label} | `{getattr(before, field)}` | `{getattr(after, field)}` |")
    rows.extend([
        f"| Generator channels per yield | {cfg.generator_for_contract(before).min_channels}–{cfg.generator.max_channels} | "
        + (f"{cfg.generator.max_channels} |" if cfg.generator_for_contract(after).min_channels == cfg.generator.max_channels
         else f"{cfg.generator_for_contract(after).min_channels}–{cfg.generator.max_channels} |"),
        f"| Generator length limit | {cfg.generator.min_length}–{cfg.generator.max_length} | "
        f"{cfg.generator.min_length}–{cfg.generation.max_length} in the CPU shard tier |",
        f"| CPU shard generation window | Not used by this contract | {cfg.generation.window_seconds // 60} minutes |",
    ])
    return "\n".join(rows)


def reward_reference(root=ROOT):
    """Mainnet tables computed from configuration and the actual payout helpers."""
    from dataclasses import replace

    from cascade.eval.expert_lane import LaneBWin, full_pay, reward_multiplier
    from cascade.shared.config import load_chain_config
    from cascade.shared.emission import (
        EmissionPolicy,
        crown_start,
        emission_vector,
        ladder_step,
        lane_parts,
    )
    from cascade.shared.lane_b import lane_b_slots, win_expired
    from cascade.shared.lane_c import LaneCWin, lane_c_split, lane_pool_shares

    cfg = load_chain_config(root / "chain.toml")

    def percent(value):
        return f"{value * 100:.6g}%"

    allocations = ["| Launched lanes | A pool | B pool | C pool |", "|---|---:|---:|---:|"]
    for label, bcfg, ccfg in (("A", None, None), ("A + C", None, cfg.lane_c),
                              ("A + B", cfg.lane_b, None),
                              ("A + B + C", cfg.lane_b, cfg.lane_c)):
        b, c = lane_pool_shares(bcfg, ccfg)
        allocations.append(f"| {label} | {percent(1-b-c)} | {percent(b)} | {percent(c)} |")
    starts = ["| Crowning LCB in increment units | Starting share of A |", "|---|---:|"]
    for threshold, _ in cfg.v2.ladder_start_tiers:
        starts.append(f"| ≥ {threshold:g} | {percent(crown_start(threshold, 'increment', cfg).start_bp / 10000)} |")
    tiers = ["| Lane | Target gain LCB | Full-pay eras |", "|---|---:|---:|"]
    for label, policy in (("B: accepted marginal gain", cfg.lane_b), ("C: confirmed MarketCast gain", cfg.marketcast)):
        for threshold, _ in policy.full_pay_tiers:
            tiers.append(f"| {label} | ≥ {threshold:g} | {full_pay(threshold, policy.full_pay_tiers).full_eras} |")
    b_share, c_share = lane_pool_shares(cfg.lane_b, cfg.lane_c)
    b_slot = b_share / len(lane_b_slots(cfg.lane_b))
    full = full_pay(.02, cfg.lane_b.full_pay_tiers).full_eras
    groups = []
    for age in range(cfg.lane_b.win_ttl_eras + 1):
        multiplier = (0.0 if win_expired(age, cfg.lane_b) else reward_multiplier(
            age, full_eras=full, half_life_rounds=cfg.lane_b.half_life_rounds, floor=cfg.lane_b.floor))
        if groups and multiplier == groups[-1][2]:
            groups[-1][1] = age
        else:
            groups.append([age, age, multiplier])
    decay = [f"B example: accepted marginal gain LCB 0.02, recorded full-pay period {full} eras.",
             "", "| Age since acceptance (eras) | Slot multiplier | Share of total emissions |",
             "|---|---:|---:|"]
    for start, end, multiplier in groups:
        label = f"{start}+" if start == cfg.lane_b.win_ttl_eras else str(start) if start == end else f"{start}–{end}"
        decay.append(f"| {label} | {percent(multiplier)} | {percent(b_slot * multiplier)} |")
    # Explicit hypothetical launched-lane state, with eligible holders, no I/O.
    _, slots = lane_c_split(cfg.lane_c, 1, 1, mc=replace(cfg.marketcast, from_block=1))
    target = cfg.lane_c.targets[0]
    a, other = lane_parts([1, 2], 5, king_decay=cfg.scoring.king_decay, burn_uid=0,
        lane_b_cfg=cfg.lane_b, wins=[LaneBWin(3, lane_b_slots(cfg.lane_b)[0], 1.0)],
        lane_c_cfg=cfg.lane_c, c_wins=[LaneCWin(4, target, 1.0)], c_slot_frac=slots)
    king_share = ladder_step(cfg.v2.ladder_start_level_bp, crowned=False, check="pass", cfg=cfg) / 10000
    vector = emission_vector(a, other, EmissionPolicy(king_share=king_share,
        former_share=cfg.v2.court_bp / 10000), burn_uid=0, lane_a_uids=[1, 2])
    example = ["| Recipient in the worked example | Share of total emissions |", "|---|---:|"]
    for label, uid in (("A king", 1), ("Eligible immediate former king", 2),
                       ("One full-pay B holder", 3), (f"One full-pay C holder ({target})", 4), ("Burn", 0)):
        example.append(f"| {label} | {percent(vector[uid])} |")
    tier_descriptions = []
    for label, policy in (("B", cfg.lane_b), ("MarketCast C", cfg.marketcast)):
        tier_descriptions.append(label + ": " + ", ".join(
            f"LCB ≥ {threshold:g} → {full_pay(threshold, policy.full_pay_tiers).full_eras} eras"
            for threshold, _ in policy.full_pay_tiers))
    decay_steps = " → ".join(percent(reward_multiplier(full + offset, full_eras=full,
        half_life_rounds=cfg.lane_b.half_life_rounds, floor=cfg.lane_b.floor)) for offset in range(3))
    starts_text = "; ".join(f"LCB ≥ {threshold:g} → {percent(bp / 10000)}"
                            for threshold, bp in cfg.v2.ladder_start_tiers)
    summary = [
        "These are the checked-in mainnet v2 rules. Check live activation and signed policy before submitting.",
        "Percentages are shares of the competition reward vector. A lane's pool is its budget; miners receive only the shares they earn.", "",
        *allocations, "",
        f"- **At launch:** a separately activated pre-v2 policy burns {percent(cfg.v2.prelaunch_burn)}. Until the first v2 winner is crowned, the inherited king/court receives at most {percent(cfg.v2.carryover_king_share)} of total emissions; the rest of A burns.",
        f"- **A winner:** a random-init or other level-unit win starts at {percent(cfg.v2.ladder_start_level_bp / 10000)} of A. Increment-unit wins use the duel's lower confidence bound (LCB): {starts_text}. These are the duel's increment units, not percentages of absolute forecast-error reduction.",
        f"- **A progression:** each passed scheduled king check adds {cfg.v2.ladder_step_bp / 100:g} percentage points of A, up to {percent(cfg.v2.ladder_cap_bp / 10000)}. Failure sets {percent(cfg.v2.ladder_fail_bp / 10000)}; a deferred check holds the share. Time alone does not increase it.",
        f"- **Former king:** only the immediate former king can receive {percent(cfg.v2.court_bp / 10000)} of A for {cfg.v2.court_blocks:,} blocks, while eligible. A qualifying compute-key lapse reduces the current king's payout to {percent(cfg.v2.key_slash_bp / 10000)} of A, without reducing separately earned B/C rewards.",
        f"- **B winner:** with merge acceptance enabled, qualification is unpaid. Acceptance into a scheduled shared starting checkpoint starts payment. The {len(lane_b_slots(cfg.lane_b))} targets each reserve {percent(b_slot)} of total emissions at full pay.",
        f"- **C winner:** an initial test pass is unpaid. A later confirmation must win the target slot. C's pool is divided by competition weight, then target count. The current sole competition, MarketCast, has {len(cfg.lane_c.targets)} targets: each full-pay slot is {percent(c_share * slots[target])} of total emissions.",
        "- **Full-pay duration:** " + "; ".join(tier_descriptions) + ". B uses accepted marginal gain; C uses confirmed gain. Each win keeps its recorded tier.",
        f"- **B/C progression:** after full pay, B's per-era slot multipliers are {decay_steps}, then a {percent(cfg.lane_b.floor)} floor. Payment ends at age {cfg.lane_b.win_ttl_eras} eras, including the full-pay period. C has its own policy (currently a {percent(cfg.lane_c.floor)} floor and age {cfg.lane_c.win_ttl_eras} expiry). Age starts at B acceptance or C confirmation. A new holder stops the old holder's pay.",
        "- **Burn:** unpaid A, empty/decayed/expired slots in launched lanes and inactive competition allocations go to `burn_uid`; they are not shared among other miners. An unlaunched lane has no reserved pool.",
        "- **Check payment:** read signed reward records and on-chain weights. Queue completion, qualification and pending confirmation are not paid wins. Public status reports `reward_status=not_verified` because it does not verify payment.",
    ]
    return {"allocation": "\n".join(allocations), "king-start": "\n".join(starts),
            "slot-tiers": "\n".join(tiers), "slot-decay": "\n".join(decay),
            "worked-example": "\n".join(example), "summary": "\n".join(summary)}


def reward_issues(paths=None, root=ROOT):
    expected = reward_reference(root)
    issues = []
    for path in guide_paths(root) if paths is None else paths:
        text = path.read_text()
        for key, body in expected.items():
            start, end = f"<!-- cascade:rewards:{key}:start -->", f"<!-- cascade:rewards:{key}:end -->"
            summaries = {root / name for name in ("README.md", "AGENTS.md", "agent.md", "llms.txt",
                          "docs/AGENT_MINING.md", "docs/VALIDATOR.md", "cascade/website/llms.txt")}
            required = ((path == root / "docs/MINER.md" and key != "summary")
                        or (path in summaries and key == "summary"))
            if start not in text and not required:
                continue
            if start not in text or end not in text:
                issues.append(f"{path.relative_to(root)}: missing reward reference {key}")
            elif text.split(start, 1)[1].split(end, 1)[0].strip() != body:
                issues.append(f"{path.relative_to(root)}: reward reference differs from code/config: {key}")
    return issues


def parser_tree(parser, path=("cascade",), inherited=frozenset()):
    flags = inherited | {f for a in parser._actions for f in a.option_strings}
    yield path, flags
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                yield from parser_tree(child, (*path, name), flags)


def entrypoint_flags(root=ROOT):
    """Other console scripts: check option spellings against their source modules."""
    result = {}
    for project, module_root in ((root, root), (root / "benchmarks", root / "benchmarks")):
        spec = tomllib.loads((project / "pyproject.toml").read_text())
        for command, target in spec["project"]["scripts"].items():
            module, _ = target.split(":", 1)
            path = module_root / (module.replace(".", "/") + ".py")
            flags = {"--help"}
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                    flags.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                                 and isinstance(arg.value, str) and arg.value.startswith("--"))
            result[command] = flags
    return result


def command_examples(text):
    """CLI snippets, including partial invocations; never execute their contents."""
    fenced = re.findall(r"^```[^\n]*\n(.*?)^```", text, re.M | re.S)
    outside = re.sub(r"^```[^\n]*\n.*?^```", "", text, flags=re.M | re.S)
    inline = re.findall(r"(?<!`)`([^`]+)`(?!`)", outside)
    for snippet in [*(line for block in fenced for line in block.replace("\\\n", " ").splitlines()), *inline]:
        snippet = snippet.strip()
        if snippet.startswith("#"):
            continue
        match = re.search(r"(?<![\w/.-])(cascade(?:-[a-z][a-z-]*)?)(?=\s|$)", snippet)
        if match:
            yield snippet[match.start():].split(" #", 1)[0]


def cli_issues(paths, root=ROOT):
    from cascade.miner.cli import build_parser

    tree = dict(parser_tree(build_parser()))
    leaf_paths = {path for path in tree if not any(other[:len(path)] == path and len(other) > len(path)
                                                  for other in tree)}
    flags_by_entry = entrypoint_flags()
    covered, mentioned, issues, count = set(), set(), [], 0
    for path in paths:
        for example in command_examples(path.read_text()):
            words = example.split()
            command = words[0]
            if command not in flags_by_entry:
                continue
            count += 1
            mentioned.add(command)
            allowed = flags_by_entry[command]
            if command == "cascade":
                current = (command,)
                index = 1
                while index < len(words) and (*current, words[index]) in tree:
                    current = (*current, words[index])
                    index += 1
                if (current not in leaf_paths and index < len(words)
                        and re.fullmatch(r"[a-z][a-z-]*", words[index])):
                    issues.append(f"{path.relative_to(root)}: unknown command in {example!r}")
                # A group-only snippet can discuss flags of several children.
                allowed = tree[current] if current in leaf_paths else set().union(
                    *(flags for child, flags in tree.items() if child[:len(current)] == current))
                if current in leaf_paths:
                    covered.add(current)
            unknown = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", example)) - set(allowed)
            if unknown:
                issues.append(f"{path.relative_to(root)}: unknown options {sorted(unknown)} in {example!r}")
    issues.extend("Undocumented miner command: " + " ".join(path) for path in sorted(leaf_paths - covered))
    issues.extend("Undocumented console entrypoint: " + name for name in sorted(set(flags_by_entry) - mentioned))
    return issues, count, len(leaf_paths), len(flags_by_entry)


def anchors(text):
    result, used = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.M):
        heading = re.sub(r"<[^>]+>", "", heading).strip().lower()
        slug = "".join(c for c in heading if c.isalnum() or c in " _-").replace(" ", "-")
        number = used.get(slug, 0)
        used[slug] = number + 1
        result.add(slug + (f"-{number}" if number else ""))
    return result


def link_issues(paths, root=ROOT):
    issues = []
    # These site-relative links target published objects, not checkout files.
    # Bind each to its writer's actual string constants instead of ignoring JSON links.
    published = {"receipts/index.json": "cascade/shared/hippius.py",
                 "status/v2.json": "cascade/shared/v2_status.py",
                 "status/chain.json": "cascade/shared/chain_status.py"}
    for path in paths:
        text = re.sub(r"^```[^\n]*\n.*?^```", "", path.read_text(), flags=re.M | re.S)
        for target in re.findall(r"\]\(([^)\s]+)(?:\s+[^)]*)?\)", text):
            if re.match(r"\w+://|mailto:|data:", target):
                continue
            destination, _, anchor = unquote(target.strip("<>")).partition("#")
            route = destination.removeprefix("./")
            if path == root / "cascade/website/llms.txt" and route in published:
                constants = {n.value for n in ast.walk(ast.parse((root / published[route]).read_text()))
                             if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                if route not in constants:
                    issues.append(f"{path.relative_to(root)}: no publisher for {target}")
                continue
            resolved = path.parent / destination if destination else path
            if not resolved.exists():
                issues.append(f"{path.relative_to(root)}: missing local link {target}")
            elif anchor and resolved.suffix in (".md", ".txt") and anchor not in anchors(resolved.read_text()):
                issues.append(f"{path.relative_to(root)}: missing heading {target}")
    return issues


def config_issues(root=ROOT):
    text = (root / "docs/AGENT_MINING.md").read_text()
    if CONFIG_START not in text or CONFIG_END not in text:
        return ["docs/AGENT_MINING.md: missing checked-in config reference markers"]
    issues = []
    if text.split(CONFIG_START, 1)[1].split(CONFIG_END, 1)[0].strip() != config_reference(root):
        issues.append("docs/AGENT_MINING.md: config reference differs from checked-in TOML")
    interface = root / "docs/INTERFACE.md"
    if interface.exists():
        start, end = "<!-- cascade:training-reference:start -->", "<!-- cascade:training-reference:end -->"
        text = interface.read_text()
        if start not in text or end not in text:
            issues.append("docs/INTERFACE.md: missing training reference markers")
        elif text.split(start, 1)[1].split(end, 1)[0].strip() != training_reference(root):
            issues.append("docs/INTERFACE.md: training reference differs from checked-in TOML")
    return issues


def main():
    paths = guide_paths()
    issues, examples, commands, entries = cli_issues(paths)
    issues.extend(link_issues(paths))
    issues.extend(config_issues())
    issues.extend(reward_issues())
    for issue in issues:
        print(issue)
    print(f"Checked {len(paths)} current guides, {examples} CLI snippets, {commands} miner commands "
          f"and {entries} console entrypoints; {len(issues)} issue(s).")
    return bool(issues)


if __name__ == "__main__":
    raise SystemExit(main())
