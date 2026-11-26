"""Overlay lint for ``[lane_d]`` netconfig overlays (lane D R3, plan §5 S6 and
the kill switch, §14.2, §14.5; CI parts of T21 and T24).

``python -P -m cascade.laned.overlay_lint --overlay F [--previous G] --base B
[--policies P]`` checks the unsigned DEC-CA-0057 body of seq N (``F``) against
seq N-1 (``G``), the ``chain.toml`` it is validated against (``B``) and,
optionally, the overlay policies of the target fleet (``P``).

A node whose release lacks ``lane_d`` refuses a ``[lane_d]`` overlay WHOLE, so
any non-``lane_d`` change it carries would split the fleet. The rules:

- ``non-lane-d``: in an overlay with a ``lane_d`` table, every other
  ``(table, key)`` must be a byte-identical (canonical JSON) restatement of
  seq N-1: no changed value, no key seq N-1 lacks, no key seq N-1 sets and
  seq N drops.
- ``release-only``: any overlay setting a burn destination that
  ``netconfig.check_overrides`` keeps release-only, even restated.
- ``chain``: ``seq``/``prev`` against seq N-1 and ``body_digest``.
- ``invalid``: whatever ``netconfig.validate_overlay`` refuses.
- ``unsafe-fleet``: a node whose ``overlay_keys`` lacks a table (or key) the
  overlay sets.

Stdout: sorted ``REFUSED <rule> ...`` lines, then ``RESULT OK`` or ``RESULT
REFUSED <n>``. Exit 0 = OK, 1 = refused, 2 = usage error or unreadable input.
Inputs are only read. Contract: ``tests/accept/r3/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from cascade.shared import netconfig as N
from cascade.shared.config import ChainConfig, parse_chain_config
from cascade.shared.direct_moe import canonical

LANE_D = "lane_d"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2

# The marker ``check_overrides`` prints for a burn destination.
_BURN_MARKER = "burn destination is release-only"


class InputError(Exception):
    """An input file that is missing, unreadable or not a JSON object."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise InputError(f"usage: {message}")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -P -m cascade.laned.overlay_lint",
                description="Lint a [lane_d] netconfig overlay body before it is signed.")
    p.add_argument("--overlay", required=True, help="unsigned body of seq N (JSON)")
    p.add_argument("--previous", help="body of seq N-1 (JSON); required when seq > 1")
    p.add_argument("--base", required=True, help="chain.toml the overlay is validated against")
    p.add_argument("--policies", help='fleet policies: {"nodes": [{"name", "policy"}]} (JSON)')
    return p


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name} is not strict JSON")


def read_json_object(path: str, what: str) -> dict:
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise InputError(f"{what}: cannot read {path}: {e}") from e
    try:
        obj = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:
        raise InputError(f"{what}: {path} is not JSON: {e}") from e
    if not isinstance(obj, dict):
        raise InputError(f"{what}: {path} is not a JSON object")
    return obj


def read_base(path: str) -> ChainConfig:
    try:
        raw = tomllib.loads(Path(path).read_text())
    except OSError as e:
        raise InputError(f"base: cannot read {path}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise InputError(f"base: {path} is not TOML: {e}") from e
    try:
        return parse_chain_config(raw)
    except (ValueError, KeyError, TypeError) as e:
        raise InputError(f"base: {path} is refused by the loader: {e}") from e


def read_policies(path: str) -> list[tuple[str, dict]]:
    obj = read_json_object(path, "policies")
    nodes = obj.get("nodes")
    if not isinstance(nodes, list):
        raise InputError(f"policies: {path} has no 'nodes' list")
    out: list[tuple[str, dict]] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            raise InputError(f"policies: node {i} has no string 'name'")
        pol = node.get("policy")
        if not isinstance(pol, dict) or not isinstance(pol.get("overlay_keys"), dict):
            raise InputError(f"policies: node {node['name']!r} has no policy.overlay_keys table")
        out.append((node["name"], pol["overlay_keys"]))
    return out


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _tables(overrides: object) -> dict:
    return overrides if isinstance(overrides, dict) else {}


def _keyed(overrides: object) -> dict[tuple[str, str], object]:
    """``(table, key) -> value`` for every key of every table; a table that is
    not a table maps under the key ``""`` (``validate_overlay`` refuses it)."""
    out: dict[tuple[str, str], object] = {}
    for table, kv in _tables(overrides).items():
        if isinstance(kv, dict):
            for key, value in kv.items():
                out[(str(table), str(key))] = value
        else:
            out[(str(table), "")] = kv
    return out


def _label(table: str, key: str) -> str:
    return f"[{table}] {key}" if key else f"[{table}]"


def is_release_only(table: str, key: str, value: object) -> bool:
    """Whether ``netconfig.check_overrides`` refuses this key as a burn
    destination (the release-only set lives there, not here)."""
    if not key:
        return False
    try:
        N.check_overrides({table: {key: value}})
    except N.NetConfigError as e:
        return _BURN_MARKER in str(e)
    return False


def rule_release_only(overlay: dict) -> list[str]:
    return [f"REFUSED release-only {_label(t, k)}: burn destination is release-only "
            f"(set to {canonical(v)}); ship it with a release, never an overlay"
            for (t, k), v in _keyed(overlay.get("overrides")).items()
            if is_release_only(t, k, v)]


def rule_non_lane_d(overlay: dict, previous: dict | None) -> list[str]:
    if LANE_D not in _tables(overlay.get("overrides")):
        return []
    now = {tk: v for tk, v in _keyed(overlay.get("overrides")).items() if tk[0] != LANE_D}
    before = ({} if previous is None else
              {tk: v for tk, v in _keyed(previous.get("overrides")).items() if tk[0] != LANE_D})
    out: list[str] = []
    for tk in sorted(set(now) | set(before)):
        label = _label(*tk)
        if tk not in before:
            what = ("seq 1 has no seq N-1 to restate" if previous is None
                    else "not set by seq N-1")
            out.append(f"REFUSED non-lane-d {label}: {what} (seq N sets {canonical(now[tk])}); "
                       "a [lane_d] overlay may only restate seq N-1 byte-identically")
        elif tk not in now:
            out.append(f"REFUSED non-lane-d {label}: set by seq N-1 ({canonical(before[tk])}) "
                       "but dropped by seq N, which reverts it to each node's base")
        elif canonical(now[tk]) != canonical(before[tk]):
            out.append(f"REFUSED non-lane-d {label}: changed from {canonical(before[tk])} to "
                       f"{canonical(now[tk])}; not a byte-identical restatement of seq N-1")
    return out


def rule_chain(overlay: dict, previous: dict | None) -> list[str]:
    seq = overlay.get("seq")
    if type(seq) is not int:
        return [f"REFUSED chain: overlay seq {canonical(seq)} is not an integer"]
    if previous is None:
        return ([f"REFUSED chain: seq {seq} > 1 needs --previous (the body of seq {seq - 1})"]
                if seq > 1 else [])
    if seq == 1:
        return ["REFUSED chain: seq 1 has no predecessor, but --previous was given"]
    out: list[str] = []
    pseq = previous.get("seq")
    if type(pseq) is not int or seq != pseq + 1:
        out.append(f"REFUSED chain: seq {seq} does not follow previous seq {canonical(pseq)}")
    want = N.body_digest(previous)
    if overlay.get("prev") != want:
        out.append(f"REFUSED chain: prev {canonical(overlay.get('prev'))} != "
                   f"body_digest(previous) {want}")
    return out


def rule_invalid(base: ChainConfig, overlay: dict) -> list[str]:
    try:
        N.validate_overlay(base, overlay)
    except (ValueError, KeyError, TypeError) as e:
        return [f"REFUSED invalid {_one_line(e)}"]
    return []


def rule_unsafe_fleet(overlay: dict, nodes: list[tuple[str, dict]]) -> list[str]:
    tables = _tables(overlay.get("overrides"))
    out: list[str] = []
    for name, keys in nodes:
        for table in sorted(map(str, tables)):
            accepted = keys.get(table)
            if not isinstance(accepted, list):
                out.append(f"REFUSED unsafe-fleet {name} [{table}]: this node's release does "
                           "not accept the table and refuses the whole overlay")
                continue
            if "*" in accepted or not isinstance(tables[table], dict):
                continue
            for key in sorted(map(str, tables[table])):
                if key not in accepted:
                    out.append(f"REFUSED unsafe-fleet {name} [{table}] {key}: this node's "
                               "release does not accept the key and refuses the whole overlay")
    return out


def lint(overlay: dict, previous: dict | None, base: ChainConfig,
         nodes: list[tuple[str, dict]] | None) -> list[str]:
    found = (rule_chain(overlay, previous) + rule_invalid(base, overlay)
             + rule_release_only(overlay) + rule_non_lane_d(overlay, previous))
    if nodes is not None:
        found += rule_unsafe_fleet(overlay, nodes)
    return sorted(found)


def main(argv: list[str]) -> int:
    try:
        args = _parser().parse_args(list(argv))
        overlay = read_json_object(args.overlay, "overlay")
        previous = (None if args.previous is None
                    else read_json_object(args.previous, "previous"))
        base = read_base(args.base)
        nodes = None if args.policies is None else read_policies(args.policies)
    except InputError as e:
        print(f"overlay_lint: {e}", file=sys.stderr)
        return EXIT_USAGE
    found = lint(overlay, previous, base, nodes)
    for line in found:
        print(line)
    print(f"RESULT REFUSED {len(found)}" if found else "RESULT OK")
    return EXIT_REFUSED if found else EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
