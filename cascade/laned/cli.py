"""``cascade-laned <command> ...``: dispatcher for the lane D rollout tools.

Each command is a module under :mod:`cascade.laned` exposing
``add_parser(subparsers)``, which registers its sub-parser and sets
``func(args) -> int`` as the handler. To add a command, add its module name to
:data:`COMMANDS`; modules are imported only when their command runs, so one
tool's imports never load for another.
"""

from __future__ import annotations

import argparse
import importlib
import sys

# command name -> module under cascade.laned
COMMANDS: dict[str, str] = {
    "blocks": "cascade.laned.blocks",
}


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="cascade-laned",
        description="Lane D rollout tooling.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="tool to run")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for the tool")
    if not args_in or args_in[0] in ("-h", "--help"):
        parser.print_help()
        return 0 if args_in else 2
    if args_in[0] not in COMMANDS:
        parser.parse_args(args_in[:1])
    name = args_in[0]
    module = importlib.import_module(COMMANDS[name])
    sub = argparse.ArgumentParser(prog="cascade-laned")
    subparsers = sub.add_subparsers(dest="command", required=True)
    module.add_parser(subparsers)
    args = sub.parse_args(args_in)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
