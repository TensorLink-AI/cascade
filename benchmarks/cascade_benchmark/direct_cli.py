"""Compatibility entry point for the hash-pinned FEV benchmark CLI."""

import argparse

from .provenance import canonical, tree_hash, verify_fev_manifest  # noqa: F401


def main(argv=None):
    from .cli import main as benchmark

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--fev-manifest", required=True)
    args, remaining = p.parse_known_args(argv)
    return benchmark(remaining + ["--fev-manifest", str(args.fev_manifest)])


if __name__ == "__main__":
    raise SystemExit(main())
