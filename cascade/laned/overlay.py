"""``cascade-laned overlay``: the S0c overlay bring-up kit (lane D plan R7).

Four commands, none of which touches a chain, a store or a wallet:

- ``benign``: a seq-1 overlay body that restates current ``[lane_d]`` values of
  the given base (no parsed value changes; the fleet's first overlay drill).
- ``refused``: a seq-1 body that ``netconfig`` refuses WHOLE: valid
  restatements plus one release-only entry, so the refusal visibly throws
  away keys that would otherwise apply.
- ``kill``: the lane D kill switch, seq N+1 after ``--previous``: every
  table restated, only ``[lane_d] until_block`` set to the effective block.
- ``publish``: the Approve-tier runbook wrapper. It prints the plan and the
  exact ``cascade-validator --netconfig-overlay ... --netconfig-publish``
  command (dry run, the default). ``--execute`` runs that command only with
  the ``--confirm`` token the dry run printed for the same inputs; signing
  and publishing are done by ``cascade-validator`` with the owner's wallet.
  #911: ``--in-force`` names the body nodes run when ``--previous`` (the
  latest published, seq/prev source) was refused; CHANGE / LANE lines and
  the implicit lane-off guard (``--allow-lane-off``) are judged against it.

The generators take ``effective_block`` from R1's block arithmetic (the next
era start after ``--after`` on the live grid, cross-checked, with drift
between the base's ``[round]`` and the grid refused) and validate every body
with the real ``cascade.shared.netconfig`` checks. On success they write
``DIR/body.json`` and ``DIR/overrides.toml`` and print one JSON summary.

Exit codes: 0 done; 1 refused by a netconfig check (or a wrong confirm token);
2 usage error or unreadable input; 3 block cross-check / drift refusal.
Contract: ``tests/accept/r7/README.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import fields, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cascade.laned import blocks
from cascade.shared import netconfig as nc
from cascade.shared.config import ChainConfig, effective_epoch_blocks, load_chain_config
from cascade.shared.era import era_length_blocks

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_INVALID = 2
EXIT_CROSS_CHECK = 3

LANE_D = "lane_d"
# The lane D switch keys a benign body restates (whichever the base types).
BENIGN_KEYS = ("from_block", "until_block", "share")
# Release-only burn destinations, in the order the refused body picks one
# the base types (check_overrides refuses each even when restated).
RELEASE_ONLY_KEYS = (("lane_d", "remainder"), ("lane_b", "remainder"),
                     ("lane_c", "remainder"), ("scoring", "burn_uid"))
VALIDATOR = "cascade-validator"
# Owner 2026-11-24 (lane D step 2 fallback): a block the publish tool fills —
# the first era start at least one era past --head-block.
NEXT_ERA = "next-era"
# ... and the stake share of the release that must signal before lane D arms
UPGRADE_THRESHOLD = Fraction(2, 3)


class Refused(Exception):
    """A body or plan the netconfig checks refuse (exit 1)."""


class Invalid(Exception):
    """Unreadable input or a usage error (exit 2)."""


# ── inputs ───────────────────────────────────────────────────────────────────


def read_base(path: str) -> ChainConfig:
    try:
        return load_chain_config(Path(path))
    except FileNotFoundError as e:
        raise Invalid(f"cannot read config {path}: {e}") from e
    except (ValueError, KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        raise Invalid(f"config {path} does not load: {e}") from e


def read_body(path: str) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise Invalid(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise Invalid(f"{path} is not JSON: {e}") from e
    try:
        return nc.check_body(obj)
    except nc.NetConfigError as e:
        raise Invalid(f"{path} is not an overlay body: {e}") from e


def read_overrides(path: str) -> dict:
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise Invalid(f"cannot read overrides {path}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise Invalid(f"overrides {path} is not TOML: {e}") from e


def effective_block_after(grid_path: str, config_path: str, after: int) -> int:
    """R1's ``blocks next-era-start <after> --grid G --config C``, in process."""
    grid = blocks.load_grid(Path(grid_path))
    blocks.check_config_drift(grid, Path(config_path))
    out = blocks.answer(SimpleNamespace(query="next-era-start", block=int(after)), grid)
    return int(out["next_era_start"])


# ── TOML for an overrides table ──────────────────────────────────────────────

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}"
                                for k, v in value.items()) + " }"
    raise Invalid(f"cannot write {value!r} as TOML")


def overrides_toml(overrides: dict) -> str:
    """The ``--netconfig-overlay`` file for ``overrides``; round-trips through
    ``tomllib`` to exactly ``overrides`` or raises."""
    out: list[str] = []
    for table, kv in overrides.items():
        out.append(f"[{_toml_key(str(table))}]")
        if not isinstance(kv, dict):
            raise Invalid(f"override [{table}] is not a table")
        for key, value in kv.items():
            out.append(f"{_toml_key(str(key))} = {_toml_value(value)}")
        out.append("")
    text = "\n".join(out)
    if tomllib.loads(text) != json.loads(json.dumps(overrides)):
        raise Invalid("overrides do not round-trip through TOML")
    return text


# ── comparing built configs ──────────────────────────────────────────────────


def _parsed(cfg: ChainConfig) -> ChainConfig:
    return replace(cfg, raw={}, netconfig=replace(cfg.netconfig, seq=0, digest="",
                                                  applied_block=0))


def changed_fields(a: ChainConfig, b: ChainConfig) -> list[str]:
    """Top-level parsed fields that differ, ignoring ``raw`` and the overlay stamp."""
    pa, pb = _parsed(a), _parsed(b)
    return [f.name for f in fields(pa) if getattr(pa, f.name) != getattr(pb, f.name)]


def _validate(base: ChainConfig, body: dict, what: str) -> ChainConfig:
    try:
        return nc.validate_overlay(base, body)
    except nc.NetConfigError as e:
        raise Refused(f"{what}: {e}") from e


def _plan_ok(base: ChainConfig, overrides: dict, eff: int, previous: dict | None,
             what: str) -> None:
    plan = nc.plan_overlay(base, overrides, eff, previous)
    if not plan.ok:
        raise Refused(f"{what}: " + "; ".join(plan.errors))


# ── the three bodies ─────────────────────────────────────────────────────────


def benign_overrides(base: ChainConfig) -> dict:
    """Restatements of keys the base types, with their typed values."""
    raw = base.raw or {}
    lane_d = raw.get(LANE_D)
    if isinstance(lane_d, dict):
        kv = {k: copy.deepcopy(lane_d[k]) for k in BENIGN_KEYS if k in lane_d}
        if kv:
            return {LANE_D: kv}
    for table in sorted(nc.ALLOWED):
        tbl = raw.get(table)
        if not isinstance(tbl, dict):
            continue
        allowed = nc.ALLOWED[table]
        for key in sorted(tbl):
            value = tbl[key]
            if isinstance(value, dict) or (allowed is not None and key not in allowed):
                continue
            candidate = {table: {key: copy.deepcopy(value)}}
            try:
                nc.check_overrides(candidate)
            except nc.NetConfigError:
                continue
            return candidate
    raise Refused("the base types no overlay-settable key to restate")


def benign_body(base: ChainConfig, eff: int) -> dict:
    overrides = benign_overrides(base)
    body = nc.next_overlay(overrides, eff, None, network=nc.network_identity(base))
    built = _validate(base, body, "benign body")
    moved = changed_fields(base, built)
    if moved:
        raise Refused(f"benign body changes parsed values: {moved}")
    plan = nc.plan_overlay(base, overrides, eff, None)
    if not plan.ok or plan.diff:
        raise Refused("benign body is not a no-op plan: "
                      + "; ".join(plan.errors + plan.diff))
    return body


def refused_body(base: ChainConfig, eff: int) -> tuple[dict, str]:
    """A body refused WHOLE: the benign restatements plus one release-only
    burn destination the base types. Returns (body, the refusal text)."""
    raw = base.raw or {}
    valid = benign_body(base, eff)
    overrides = copy.deepcopy(valid["overrides"])
    for table, key in RELEASE_ONLY_KEYS:
        tbl = raw.get(table)
        if isinstance(tbl, dict) and key in tbl:
            overrides.setdefault(table, {})[key] = copy.deepcopy(tbl[key])
            break
    else:
        raise Invalid("the base types no release-only burn destination to restate")
    body = dict(valid, overrides=overrides)
    try:
        nc.validate_overlay(base, body)
    except nc.NetConfigError as e:
        why = str(e)
    else:
        raise Invalid("netconfig accepted the release-only restatement")
    if "release-only" not in why:
        raise Invalid(f"refused for another reason than release-only: {why}")
    return body, why


def kill_body(base: ChainConfig, eff: int, previous: dict) -> dict:
    """seq N+1: the previous overrides restated, ``[lane_d] until_block = eff``."""
    try:
        prev_cfg = nc.build_config(base, previous)
    except nc.NetConfigError as e:
        raise Refused(f"the previous overlay does not build on this base: {e}") from e
    ld = prev_cfg.lane_d
    if int(ld.from_block) <= 0:
        raise Refused("lane D is not armed in the config the previous overlay builds "
                      "(from_block = 0): nothing to kill")
    if 0 < int(ld.until_block) <= eff:
        raise Refused(f"lane D already ends at {ld.until_block}, at or before {eff}")
    if eff <= int(ld.from_block):
        raise Refused(f"effective block {eff} is not after lane D's from_block "
                      f"{ld.from_block}")
    overrides = copy.deepcopy(previous["overrides"])
    overrides.setdefault(LANE_D, {})["until_block"] = int(eff)
    body = nc.next_overlay(overrides, eff, previous, network=nc.network_identity(base))
    killed = _validate(base, body, "kill body")
    _plan_ok(base, overrides, eff, previous, "kill body")
    moved = changed_fields(prev_cfg, killed)
    if moved != [LANE_D]:
        raise Refused(f"kill body changes {moved}, not only lane_d")
    first = eff + int(effective_epoch_blocks(base.round, eff))
    if not (prev_cfg.lane_d.active(first) and prev_cfg.lane_d.share > 0):
        raise Refused(f"lane D is not paid at settlement {first} under the previous overlay")
    if killed.lane_d.active(first) and killed.lane_d.share > 0:
        raise Refused(f"lane D would still be paid at settlement {first}")
    return body


# ── generator commands ───────────────────────────────────────────────────────


def _emit(out_dir: str, body: dict, extra: dict | None = None) -> None:
    toml_text = overrides_toml(body["overrides"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "overrides.toml").write_text(toml_text, encoding="utf-8")
    (out / "body.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    summary = {"seq": body["seq"], "prev": body["prev"],
               "effective_block": body["effective_block"],
               "digest": nc.body_digest(body), "note": nc.cfg_feature_name(body),
               "body": str(out / "body.json"), "overrides": str(out / "overrides.toml")}
    summary.update(extra or {})
    print(json.dumps(summary, sort_keys=True))


def _generate(args: argparse.Namespace) -> int:
    try:
        base = read_base(args.config)
        eff = effective_block_after(args.grid, args.config, args.after)
        extra: dict = {}
        if args.kind == "benign":
            body = benign_body(base, eff)
        elif args.kind == "refused":
            body, why = refused_body(base, eff)
            extra["refusal"] = why
        else:
            body = kill_body(base, eff, read_body(args.previous))
        _emit(args.out, body, extra)
    except blocks.InvalidInput as e:
        print(f"cascade-laned overlay {args.kind}: invalid input: {e}", file=sys.stderr)
        return EXIT_INVALID
    except blocks.CrossCheckRefused as e:
        print(f"cascade-laned overlay {args.kind}: refused: {e}", file=sys.stderr)
        return EXIT_CROSS_CHECK
    except Invalid as e:
        print(f"cascade-laned overlay {args.kind}: invalid input: {e}", file=sys.stderr)
        return EXIT_INVALID
    except Refused as e:
        print(f"cascade-laned overlay {args.kind}: refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


# ── the runbook wrapper ──────────────────────────────────────────────────────


def validator_argv(args: argparse.Namespace) -> list[str]:
    """The exact ``cascade-validator`` publish command; names forwarded verbatim."""
    argv = [VALIDATOR, "--chain-toml", args.config, "--netconfig-overlay",
            getattr(args, "command_overrides", None) or args.overrides,
            "--netconfig-effective-block",
            str(getattr(args, "resolved_effective_block", None) or args.effective_block),
            "--netconfig-publish"]
    if args.network is not None:
        argv += ["--network", args.network]
    if args.comment:
        argv += ["--netconfig-comment", args.comment]
    if args.wallet_name is not None:
        argv += ["--wallet-name", args.wallet_name]
    if args.wallet_hotkey is not None:
        argv += ["--wallet-hotkey", args.wallet_hotkey]
    seq = getattr(args, "in_force_seq", None)
    if seq is not None:
        argv += ["--netconfig-in-force-seq", str(seq)]
    if getattr(args, "allow_lane_off", False):
        argv.append("--netconfig-allow-lane-off")
    return argv


def _file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirm_token(argv: list[str], digest_hex: str, args: argparse.Namespace) -> str:
    """Binds the command, the planned body and the bytes of every input file."""
    material = {"argv": argv, "digest": digest_hex,
                "config": _file_sha(args.config), "overrides": _file_sha(args.overrides),
                "previous": _file_sha(args.previous) if args.previous else None}
    if getattr(args, "in_force", None):
        material["in_force"] = _file_sha(args.in_force)
    raw = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return "ok-" + hashlib.sha256(raw).hexdigest()[:20]


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def next_era_start(base: ChainConfig, head: int) -> int:
    """The first era start at least one era (at the head) past ``head``."""
    b = int(head) + int(era_length_blocks(base.round, int(head)))
    for _ in range(4 * int(era_length_blocks(base.round, b)) + 1):
        if nc.is_era_start(base, b):
            return b
        b += 1
    raise Invalid(f"no era start found after block {head}")


def resolve_next_era(base: ChainConfig, overrides: dict, head: int | None,
                     effective: int | str) -> tuple[dict, int, list[str]]:
    """(overrides with every ``"next-era"`` value resolved, the effective
    block, ``RESOLVED`` lines). Only ``[lane_d] from_block`` may carry it."""
    found = [(t, k) for t, kv in overrides.items() if isinstance(kv, dict)
             for k, v in kv.items() if v == NEXT_ERA]
    if not found and effective != NEXT_ERA:
        return overrides, int(effective), []
    bad = [f"[{t}] {k}" for t, k in found if (t, k) != (LANE_D, "from_block")]
    if bad:
        raise Invalid(f"{NEXT_ERA!r} is supported only for [lane_d] from_block, not "
                      + ", ".join(bad))
    if head is None:
        raise Invalid(f"{NEXT_ERA!r} needs --head-block (the chain head, read by the owner)")
    block = next_era_start(base, head)
    out = copy.deepcopy(overrides)
    lines = []
    for t, k in found:
        out[t][k] = block
        lines.append(f"[{t}] {k} = {block} (the first era start >= head {head} + one era)")
    eff = block if effective == NEXT_ERA else int(effective)
    if effective == NEXT_ERA:
        lines.append(f"effective_block = {block}")
    return out, eff, lines


def upgrade_share(args: argparse.Namespace) -> tuple[dict | None, str]:
    """(the upgrade tracker's report for the release, or None, why not)."""
    from cascade.laned import upgrade_tracker as ut

    if not (args.upgrade_metagraph and args.upgrade_notes and args.upgrade_feature):
        return None, ("--upgrade-metagraph, --upgrade-notes and --upgrade-feature are "
                      "needed to measure the release's stake share")
    feat = str(args.upgrade_feature)
    try:
        if feat.startswith(nc.FEATURE_PREFIX) or re.fullmatch(r"[0-9a-f]{8}|[0-9a-f]{64}", feat):
            feat = ut.parse_target(feat)
        rows = ut.parse_metagraph(ut._load_json(args.upgrade_metagraph, "metagraph"))
        notes = ut.parse_notes(ut._load_json(args.upgrade_notes, "notes"))
        return ut.track(rows, notes, feat, UPGRADE_THRESHOLD), ""
    except ut.InputError as e:
        return None, f"upgrade tracker: {e}"


def _publish(args: argparse.Namespace) -> int:
    try:
        base = read_base(args.config)
        overrides = read_overrides(args.overrides)
        previous = read_body(args.previous) if args.previous else None
        in_force_body = read_body(args.in_force) if args.in_force else None
        overrides, eff, resolved = resolve_next_era(base, overrides, args.head_block,
                                                    args.effective_block)
    except Invalid as e:
        print(f"cascade-laned overlay publish: invalid input: {e}", file=sys.stderr)
        return EXIT_INVALID
    args.command_overrides = None
    args.resolved_effective_block = eff if args.effective_block == NEXT_ERA else None
    if resolved:
        out_path = Path(args.resolved_out or f"{args.overrides}.resolved.toml")
        try:
            out_path.write_text(overrides_toml(overrides), encoding="utf-8")
        except (OSError, Invalid) as e:
            print(f"cascade-laned overlay publish: cannot write {out_path}: {e}",
                  file=sys.stderr)
            return EXIT_INVALID
        args.command_overrides = str(out_path)
    pre_errors: list[str] = []
    args.in_force_seq = None
    if in_force_body is not None:
        # #911: seq / prev follow --previous; the checks follow --in-force
        if previous is None:
            print("cascade-laned overlay publish: --in-force needs --previous (the latest "
                  "published body)", file=sys.stderr)
            return EXIT_INVALID
        if int(in_force_body["seq"]) > int(previous["seq"]) or (
                int(in_force_body["seq"]) == int(previous["seq"])
                and nc.body_digest(in_force_body) != nc.body_digest(previous)):
            print("cascade-laned overlay publish: --in-force is not an overlay at or before "
                  "--previous in the published chain", file=sys.stderr)
            return EXIT_INVALID
        in_force = in_force_body
        if nc.body_digest(in_force_body) != nc.body_digest(previous):
            args.in_force_seq = int(in_force_body["seq"])
    else:
        in_force = previous
        if previous is not None:
            _, skipped = nc.newest_valid(base, [previous])
            if skipped:
                pre_errors.append(
                    f"{skipped[0]}. Nodes keep the newest overlay that validates in force: "
                    "pass it as --in-force (seq/prev still follow --previous); if this seq "
                    "was adopted under an older release (#891), pass --in-force with the "
                    "--previous file itself")
    plan = nc.plan_overlay(base, overrides, eff, previous, comment=args.comment,
                           in_force=in_force, allow_lane_off=args.allow_lane_off,
                           head_block=args.head_block)
    plan.errors[:0] = pre_errors
    # owner 2026-11-24: lane D arms (step 2) only once >= 2/3 of stake runs
    # the release that carries the satellite fix (#913)
    upgrade_lines: list[str] = []
    if any(x.startswith("LANE D: OFF -> ON") for x in plan.lanes):
        report, why = upgrade_share(args)
        if report is not None:
            upgrade_lines.append(
                f"UPGRADE {report['pct']}% of eligible stake ({report['signed_stake']} / "
                f"{report['total_stake']}) signals {report['target']} (needs >= 66.67%)")
            ok = bool(report["meets_threshold"])
            why = "" if ok else (f"only {report['pct']}% of eligible stake signals "
                                 f"{report['target']}")
        else:
            upgrade_lines.append(f"UPGRADE unknown: {why}")
            ok = False
        if not ok:
            msg = (f"arming lane D needs >= 2/3 of stake on the release: {why}")
            if args.force_upgrade_threshold:
                plan.warnings.append(msg + " — overridden (--force-upgrade-threshold)")
            else:
                plan.errors.append(msg + " (override: --force-upgrade-threshold)")
    try:
        body = nc.next_overlay(overrides, eff, previous, network=nc.network_identity(base),
                               comment=args.comment)
        digest_hex, note = nc.body_digest(body), nc.cfg_feature_name(body)
    except nc.NetConfigError:
        digest_hex, note = "-", "-"
    if all(isinstance(kv, dict) for kv in overrides.values()):
        diff = nc.diff_tables(base.raw or {}, nc.merge_raw(base.raw or {}, overrides))
    else:
        diff = list(plan.diff)

    out = ["DRY RUN: nothing is signed or published by this step",
           f"SEQ {plan.body['seq']}",
           f"PREV {plan.body['prev'] or '-'}",
           f"EFFECTIVE_BLOCK {eff}",
           f"DIGEST {digest_hex}",
           f"NOTE {note}",
           f"IN_FORCE {plan.in_force}"]
    out += [f"RESOLVED {r}" for r in resolved]
    out += upgrade_lines
    # #911: CHANGE = effective values vs the config in force (dropped
    # restatements marked); LANE = lane on/off and share moves; DIFF = vs
    # chain.toml as typed
    out += [f"CHANGE {_one_line(c)}" for c in plan.changes] or ["CHANGE (none)"]
    out += [f"LANE {_one_line(x)}" for x in plan.lanes]
    out += [f"DIFF {d}" for d in diff] or ["DIFF (none)"]
    out += [f"WARNING {_one_line(w)}" for w in plan.warnings]
    argv: list[str] = []
    token = ""
    if plan.ok:
        out.append("CHECK VALID")
        argv = validator_argv(args)
        token = confirm_token(argv, digest_hex, args)
        out.append(f"COMMAND {shlex.join(argv)}")
        out.append(f"CONFIRM {token}")
        out.append("DRY RUN: Approve-tier. After the owner approves this plan, rerun the "
                   f"same arguments with --execute --confirm {token}")
    else:
        out.append("CHECK REFUSED " + _one_line("; ".join(plan.errors)))
        out.append("DRY RUN: refused, there is nothing to publish")
    print("\n".join(out), flush=True)

    if not plan.ok:
        return EXIT_REFUSED
    if not args.execute:
        if args.confirm is not None:
            print("cascade-laned overlay publish: --confirm is used only with --execute",
                  file=sys.stderr)
        return EXIT_OK
    if args.confirm is None:
        print("cascade-laned overlay publish: --execute needs --confirm <token> from the "
              "dry run", file=sys.stderr)
        return EXIT_INVALID
    if args.confirm != token:
        print("cascade-laned overlay publish: the --confirm token does not match this plan; "
              "rerun the dry run and review it", file=sys.stderr)
        return EXIT_REFUSED
    exe = shutil.which(VALIDATOR)
    if exe is None:
        print(f"cascade-laned overlay publish: {VALIDATOR} is not on PATH", file=sys.stderr)
        return EXIT_INVALID
    return subprocess.run([exe, *argv[1:]], check=False).returncode


# ── argument parsing ─────────────────────────────────────────────────────────


def _block(text: str) -> int:
    try:
        value = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not an integer block: {text!r}") from e
    if value < 0:
        raise argparse.ArgumentTypeError(f"block must be >= 0, got {value}")
    return value


def _block_or_next(text: str) -> int | str:
    return NEXT_ERA if text == NEXT_ERA else _block(text)


def _generator(cmds: Any, name: str, help_text: str) -> argparse.ArgumentParser:
    p = cmds.add_parser(name, help=help_text)
    p.add_argument("--config", required=True, help="the base chain.toml (the DEPLOYED file)")
    p.add_argument("--grid", required=True, help="live grid JSON (schema laned_grid/1)")
    p.add_argument("--after", type=_block, required=True,
                   help="effective_block = the first era start after this block")
    p.add_argument("--out", required=True, help="directory for body.json and overrides.toml")
    p.set_defaults(func=_generate, kind=name)
    return p


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``overlay`` and its commands on the ``cascade-laned`` parser."""
    overlay = subparsers.add_parser(
        "overlay", help="S0c overlay bring-up: benign / refused / kill bodies, publish wrapper")
    cmds = overlay.add_subparsers(dest="overlay_command", required=True)
    _generator(cmds, "benign", "seq-1 body restating current base values (no-op)")
    _generator(cmds, "refused", "seq-1 body refused whole (one release-only entry)")
    kill = _generator(cmds, "kill", "kill switch: seq N+1 ending lane D at the era start")
    kill.add_argument("--previous", required=True, help="the overlay body in force (seq N)")

    p = cmds.add_parser("publish", help="dry run (default) of the cascade-validator publish; "
                                        "--execute --confirm TOKEN runs it")
    p.add_argument("--config", required=True, help="the base chain.toml")
    p.add_argument("--overrides", required=True, help="overrides TOML for --netconfig-overlay")
    p.add_argument("--effective-block", type=_block_or_next, required=True,
                   help=f"an era start, or {NEXT_ERA!r} (needs --head-block)")
    p.add_argument("--head-block", type=_block,
                   help=f"the chain head, read by the owner: resolves {NEXT_ERA!r} (the "
                        "first era start at least one era past it) and runs the "
                        "one-era-past-head checks")
    p.add_argument("--resolved-out",
                   help=f"where the overrides with {NEXT_ERA!r} resolved are written "
                        "(default: <overrides>.resolved.toml); the printed command uses it")
    p.add_argument("--upgrade-metagraph",
                   help="upgrade_tracker metagraph JSON (needed to arm lane D)")
    p.add_argument("--upgrade-notes", help="upgrade_tracker readiness notes JSON")
    p.add_argument("--upgrade-feature",
                   help="the release's readiness segment (or cfg-<8 hex>) whose stake share "
                        "gates arming lane D")
    p.add_argument("--force-upgrade-threshold", action="store_true",
                   help="arm lane D although under 2/3 of stake signals --upgrade-feature")
    p.add_argument("--previous", help="the latest published body (seq N), if any")
    p.add_argument("--in-force", help="the body nodes run, when --previous was refused "
                                      "(default: --previous); checks and CHANGE/LANE lines "
                                      "are judged against it, seq/prev follow --previous")
    p.add_argument("--allow-lane-off", action="store_true",
                   help="allow turning a lane the config in force arms off without setting "
                        "its switch keys (a dropped restatement)")
    p.add_argument("--comment", default="", help="free text carried in the body")
    p.add_argument("--network", help="forwarded to cascade-validator --network")
    p.add_argument("--wallet-name", help="forwarded to cascade-validator --wallet-name")
    p.add_argument("--wallet-hotkey", help="forwarded to cascade-validator --wallet-hotkey")
    p.add_argument("--execute", action="store_true",
                   help="run the printed command (needs the matching --confirm)")
    p.add_argument("--confirm", help="the CONFIRM token the dry run printed")
    p.set_defaults(func=_publish)
    return overlay
