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


# W-LAUNCH (owner 2026-11-25): the mainnet launch overlays. Step 1 (published,
# cfg-961291bc) arms lanes B and C and freezes lane A; step 2 adds lane D.
LAUNCH_BLOCK = 9586800
LAUNCH_OVERLAYS = ("deploy/overlays/launch-20261125.pre.mainnet.toml",
                   "deploy/overlays/launch-20261125.mainnet.toml")


def launch_configs(root=ROOT):
    """(template, step 1, step 2): ``chain.toml`` and the configs the two
    mainnet launch overlays put in force on it, through the real overlay path."""
    from cascade.shared import netconfig as nc
    from cascade.shared.config import load_chain_config

    base = load_chain_config(root / "chain.toml")
    ident, prev, out = nc.network_identity(base), None, []
    for rel in LAUNCH_OVERLAYS:
        src = root / rel if (root / rel).is_file() else ROOT / rel   # tmp-root tests copy chain.toml only
        prev = nc.next_overlay(tomllib.loads(src.read_text()), LAUNCH_BLOCK, prev,
                               network=ident)
        out.append((nc.build_config(base, prev), nc.cfg_feature_name(prev)))
    return base, out[0], out[1]


def reward_reference(root=ROOT):
    """Mainnet tables computed from configuration and the actual payout helpers.

    The ``summary``, ``slot-decay`` and ``worked-example`` blocks state the LIVE
    launch rules (``chain.toml`` + the launch overlays); ``allocation``,
    ``king-start`` and ``slot-tiers`` are the checked-in template."""
    from dataclasses import replace

    from cascade.eval.expert_lane import LaneBWin
    from cascade.shared.config import load_chain_config
    from cascade.shared.emission import (
        EmissionPolicy,
        crown_start,
        emission_vector,
        ladder_step,
        lane_parts,
    )
    from cascade.shared.era import era_length_blocks
    from cascade.shared.lane_b import lane_b_slots
    from cascade.shared.lane_c import (
        LaneCWin,
        lane_c_split,
        lane_pool_shares,
        target_full_pay_record,
    )

    cfg, (live1, cfg1_name), (live2, _) = launch_configs(root)
    assert cfg == load_chain_config(root / "chain.toml")

    def percent(value):
        return f"{value * 100:.6g}%"

    allocations = ["| Launched lanes | A pool | B pool | C pool |", "|---|---:|---:|---:|"]
    for label, bcfg, ccfg in (("A", None, None), ("A + C", None, cfg.lane_c),
                              ("A + B", cfg.lane_b, None),
                              ("A + B + C", cfg.lane_b, cfg.lane_c)):
        b, c, _ = lane_pool_shares(bcfg, ccfg)
        allocations.append(f"| {label} | {percent(1-b-c)} | {percent(b)} | {percent(c)} |")
    starts = ["| Crowning LCB in increment units | Starting share of A |", "|---|---:|"]
    for threshold, _ in cfg.v2.ladder_start_tiers:
        starts.append(f"| ≥ {threshold:g} | {percent(crown_start(threshold, 'increment', cfg).start_bp / 10000)} |")
    # LANE_D_PLAN rev6.22 (owner 2026-11-24): lane B is untiered — a held slot
    # pays in full until replaced. Lane C (DEC-CA-0073): a gain-scaled
    # duration on the era grid in force at lane C's start, then the slot ends.
    tiers = ["| Lane | Target gain LCB | Full-pay eras |", "|---|---:|---:|"]
    c_eb = era_length_blocks(cfg.round, max(1, int(cfg.lane_c.from_block)))
    c_margin, c_top = float(cfg.marketcast.gain_margin), float(cfg.lane_c.full_pay_max_gain)
    c_target = cfg.lane_c.targets[0]

    def c_eras(g):
        return target_full_pay_record(c_target, g, cfg.lane_c, cfg.marketcast,
                                      era_blocks=c_eb)["full_eras"]
    c_points = [c_margin + (c_top - c_margin) * f for f in (0.0, 0.25, 0.5, 1.0)]
    for g in c_points:
        mark = "≥ " if g == c_points[-1] else ""
        tiers.append(f"| C: confirmed MarketCast gain | {mark}{g:g} | {c_eras(g)} |")
    b_share, c_share, _ = lane_pool_shares(live1.lane_b, live1.lane_c)
    n_slots = len(lane_b_slots(live1.lane_b))
    b_slot = b_share / n_slots
    decay = [f"B (live launch split): a held slot pays 100% of its slot ({percent(b_slot)} of total "
             "emissions) every settlement, with no decay and no expiry, until a newer leg's merge "
             "passes acceptance on that slot. C has no decay: its slot pays 100% for its recorded "
             "full-pay duration, then ends."]
    # Explicit hypothetical launched-lane state, with eligible holders, no I/O.
    # Step 1 of the launch (A + B + C, lane D not yet armed).
    lb, lc = live1.lane_b, live1.lane_c
    _, slots = lane_c_split(lc, 1, 1, mc=replace(live1.marketcast, from_block=1))
    target = lc.targets[0]
    a, other = lane_parts([1, 2], 5, king_decay=cfg.scoring.king_decay, burn_uid=0,
        lane_b_cfg=lb, wins=[LaneBWin(3, lane_b_slots(lb)[0], 1.0)],
        lane_c_cfg=lc, c_wins=[LaneCWin(4, target, 1.0)], c_slot_frac=slots)
    king_share = ladder_step(cfg.v2.ladder_start_level_bp, crowned=False, check="pass", cfg=cfg) / 10000
    vector = emission_vector(a, other, EmissionPolicy(king_share=king_share,
        former_share=cfg.v2.court_bp / 10000), burn_uid=0, lane_a_uids=[1, 2])
    example = ["| Recipient in the worked example | Share of total emissions |", "|---|---:|"]
    for label, uid in (("A king", 1), ("Eligible immediate former king", 2),
                       ("One full-pay B holder", 3), (f"One full-pay C holder ({target})", 4), ("Burn", 0)):
        example.append(f"| {label} | {percent(vector[uid])} |")
    c_duration = (f"MarketCast C: {cfg.lane_c.full_pay_min_days:g} days ({c_eras(c_margin)} eras) "
                  f"at LCB {c_margin:g}, rising linearly to {cfg.lane_c.full_pay_max_days:g} days "
                  f"({c_eras(c_top)} eras) at LCB ≥ {c_top:g}")
    split = ["| From block 9,586,800 | A pool | B pool | C pool | D pool |", "|---|---:|---:|---:|---:|"]
    for label, live in ((f"Step 1 (overlay {cfg1_name}, lanes B and C)", live1),
                        ("Step 2 (second overlay, lane D armed)", live2)):
        sh = lane_pool_shares(live.lane_b, live.lane_c, live.lane_d)
        split.append(f"| {label} | {percent(1 - sh.b - sh.c - sh.d)} | {percent(sh.b)} "
                     f"| {percent(sh.c)} | {percent(sh.d)} |")
    d = live2.lane_d
    court = "/".join(f"{w * 100:g}" for w in d.court_weights)
    pw = float(d.public_weight)
    public = ("currently 0, live-only: no public index is scored" if pw == 0
              else f"currently {pw:g}")
    summary = [
        "These are the live mainnet rules from the launch at block 9,586,800 (about 2026-11-25 22:27 UTC): the checked-in `chain.toml` plus the launch overlays `deploy/overlays/launch-20261125.pre.mainnet.toml` (step 1) and `launch-20261125.mainnet.toml` (step 2). A later signed overlay can change them; check `cascade status --json` and the signed reward records before submitting.",
        "Percentages are shares of total emissions. A lane's pool is its budget; miners receive only the shares they earn, and the rest burns.", "",
        *split, "",
        f"- **Lane A is closed to new challengers from block 9,586,800.** The intake refuses a new lane A fund or funded submit (`409 lane_ended`) and the trainer seats no new challenger. King checks and promotions are off (`[v2] reign_warm_start_blocks = {live1.v2.reign_warm_start_blocks}`), so the lane A king is frozen as the reference model for lanes B and D. In step 1, A's pool pays that king its ladder share, which no longer changes; the rest of A burns.",
        f"- **Lane D:** step 2 arms lane D at 9,586,800 if its overlay locks in by then, otherwise at a later era start. D takes its {percent(d.share)} from lane A. At lane D genesis the house entry (the frozen lane A king) holds the lane D throne and D's share stays with the lane A king until the first miner is crowned lane D king. Then D's pool pays its court {court} (king, then the two previous kings). The crown bar is a {percent(d.d_margin_start)} LCB falling linearly to {percent(d.d_margin)} over {d.d_margin_warmup_settlements} settlements. The embargo is {d.embargo_hours:g} h at launch and the public-half weight is set by overlay ({public}).",
        f"- **B winner:** the {n_slots} slots each pay {percent(b_slot)} of total emissions to their holder every settlement, with no decay or expiry, until a newer leg's accepted merge takes the slot. Qualification is unpaid: pay starts only when a qualified expert is accepted into a shared era init, and the code opens a new init only at a lane A promotion. With lane A promotions off, **no lane B slot is paid under the launch config**; qualified experts wait as pending until the owner changes this. Every leg is tested on every slot: up to {live1.lane_b.max_domains_per_leg} declared targets clear at a {percent(live1.lane_b.gain_margin)} gain LCB, any other slot at {percent(live1.lane_b.undeclared_gain_margin)}; one miner may hold every slot.",
        "- **B timing:** a finished leg typically settles about one settlement (3 h) after it finishes: its public bench is queued at the first settlement and the leg is judged once the bench rows exist.",
        f"- **C winner:** an initial test pass is unpaid. A later confirmation must win the target slot. C's pool is divided by competition weight, then target count. The current sole competition, MarketCast, has {len(lc.targets)} target{'' if len(lc.targets) == 1 else 's'}: each full-pay slot is {percent(c_share * slots[target])} of total emissions.",
        "- **C full-pay duration:** " + c_duration + ". C uses confirmed gain; each win keeps its recorded duration. B has no tiers.",
        "- **C progression:** C has no decay: its slot ends when the full-pay duration ends. Age starts at C confirmation. A new holder stops the old holder's pay.",
        "- **Burn:** unpaid A, empty B slots, empty/ended C slots and inactive competition allocations go to `burn_uid`; they are not shared among other miners. An unlaunched lane has no reserved pool.",
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
