"""W10: the `cascade lane-d` command group and the W3a-chosen upload path
(plan §3.0 W10 row; decisions/STORAGE-LANE-D.md). Contract: README.md
"The command group" and "Upload path"."""

from __future__ import annotations

import argparse

from .helpers import PATH_IDS, SUBCOMMANDS, chosen_upload_path, decision_path_ids


def _lane_d_parser():
    import cascade.miner.lane_d  # noqa: F401  (PATHS)
    # isort: split
    from cascade.miner.cli import build_parser

    root = build_parser()
    (sub,) = [a for a in root._actions if isinstance(a, argparse._SubParsersAction)]
    assert "lane-d" in sub.choices, f"no `cascade lane-d` group: {sorted(sub.choices)}"
    return sub.choices["lane-d"]


def _children(parser) -> dict:
    acts = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert acts, "`cascade lane-d` has no subcommands"
    return dict(acts[0].choices)


def _flags(parser) -> set[str]:
    return {s for a in parser._actions for s in a.option_strings}


def test_the_lane_d_group_has_exactly_the_five_commands():
    kids = _children(_lane_d_parser())
    assert set(kids) >= set(SUBCOMMANDS), sorted(kids)
    extra = set(kids) - set(SUBCOMMANDS) - {"test"}
    assert not extra, f"unexpected lane-d commands: {sorted(extra)}"


def test_every_command_takes_json():
    for name, p in _children(_lane_d_parser()).items():
        assert "--json" in _flags(p), f"`cascade lane-d {name}` has no --json"


def test_the_signing_commands_take_the_wallet_flags():
    kids = _children(_lane_d_parser())
    for name in ("submit", "ready"):
        flags = _flags(kids[name])
        for f in ("--wallet-name", "--wallet-hotkey", "--wallet-path", "--network",
                  "--chain-toml"):
            assert f in flags, f"`cascade lane-d {name}` lacks {f}"


def test_status_needs_no_wallet():
    flags = _flags(_children(_lane_d_parser())["status"])
    assert "--wallet-name" not in flags and "--wallet-hotkey" not in flags


def test_no_credential_ever_comes_from_the_command_line():
    """§1.2: credentials and URLs are never put on a command line. No lane-d
    flag takes a key, secret, token, password or presigned URL."""
    for name, p in _children(_lane_d_parser()).items():
        for f in _flags(p):
            low = f.lower()
            assert not any(w in low for w in ("secret", "token", "password", "api-key",
                                              "apikey", "presign", "grant")), \
                f"`cascade lane-d {name} {f}` takes a credential on the command line"


# W9 invariant 12 (tests/accept/w9/test_w9_independence.py): the audit's
# static import closure reaches cascade.miner.cli (cascade.audit.checks ->
# validator.marketcast -> ... -> miner.live_status -> miner.cli), so no module
# cli.py imports may name a lane D producer module statically.
W9_FORBIDDEN_PACKAGES = ("cascade.validator.lane_d", "cascade.lane_d", "cascade.laned",
                         "cascade.satellite", "torch")
W9_FORBIDDEN_STEMS = ("cascade.validator.lane_d_", "cascade.eval.lane_d_", "cascade.lane_d_")


def _static_imports(path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ["cascade", "miner"][:max(0, 2 - (node.level - 1))]
                mod = ".".join([*base, node.module] if node.module else base)
            else:
                mod = node.module or ""
            out.add(mod)
            out |= {f"{mod}.{a.name}" for a in node.names}
    return out


def test_the_lane_d_cli_imports_no_lane_d_producer_module_statically():
    from .helpers import LANE_D_PY

    assert LANE_D_PY.is_file(), "cascade/miner/lane_d.py does not exist"
    bad = sorted(m for m in _static_imports(LANE_D_PY)
                 if any(m == p or m.startswith(p + ".") for p in W9_FORBIDDEN_PACKAGES)
                 or m.startswith(W9_FORBIDDEN_STEMS))
    assert not bad, (f"cascade/miner/lane_d.py statically imports {bad}: W9's invariant-12 "
                     "closure reaches it through cascade.miner.cli (load them at call time)")


def test_the_decision_record_lists_the_three_path_ids():
    assert decision_path_ids() == PATH_IDS


def test_the_cli_offers_only_the_decision_records_choice():
    from cascade.miner import lane_d

    assert tuple(lane_d.UPLOAD_PATH_IDS) == decision_path_ids()
    chosen = chosen_upload_path()
    assert chosen is not None, (
        "decisions/STORAGE-LANE-D.md has no filled 'Chosen path:' (W3a owner run): W10 "
        "builds only the chosen path, so it is blocked until the record names one")
    assert chosen == lane_d.UPLOAD_PATH
