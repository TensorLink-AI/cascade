"""`cascade rules <lane>` and the lane D help examples (owner 2026-11-24: "make
the miner CLI up to date"; per-lane miner guides docs/MINER_LANE_*.md).

`cascade rules a|b|c|d` prints the live rule values from chain.toml with the
validators' own loader, so the guides never restate drifting numbers. The
`lane-d` group's help carries runnable examples and points at it.

Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import contextlib
import io
import json
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHAIN_TOML = REPO / "chain.toml"
LANES = ("a", "b", "c", "d")


def _parser():
    from cascade.miner.cli import build_parser

    return build_parser()


def _parse(argv: list[str]):
    try:
        return _parser().parse_args(argv)
    except SystemExit as e:  # argparse refusal -> an assertion, never a bare exit
        raise AssertionError(f"argparse refused {argv!r} (exit {e.code})") from None


def _refused(argv: list[str]) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            _parser().parse_args(argv)
    except SystemExit as e:
        return e.code == 2
    return False


def _run_json(argv: list[str]) -> tuple[int, dict]:
    from cascade.miner.cli import main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = main([*argv, "--json"])
    return rc, json.loads(buf.getvalue())


def _cfg(path: Path = CHAIN_TOML):
    from cascade.shared.config import load_chain_config

    return load_chain_config(path)


def _subparser(*names: str):
    import argparse

    p = _parser()
    for name in names:
        action = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        assert name in action.choices, f"no `{name}` subcommand"
        p = action.choices[name]
    return p


def test_rules_parses_every_lane():
    for lane in LANES:
        args = _parse(["rules", lane])
        assert args.lane == lane and args.command == "rules"
    assert _parse(["rules", "D"]).lane == "d", "the lane letter is case-insensitive"


def test_rules_all_and_key_are_exclusive_and_lane_is_checked():
    assert _parse(["rules", "b", "--all"]).all is True
    assert _parse(["rules", "b", "--key", "gain_margin"]).key == "gain_margin"
    assert _refused(["rules", "b", "--all", "--key", "gain_margin"])
    assert _refused(["rules", "e"])
    assert _refused(["rules"])


def test_every_headline_names_a_real_config_field_of_its_lane():
    import dataclasses

    from cascade.miner import rules

    cfg = _cfg()
    assert set(rules.HEADLINES) == set(LANES) == set(rules.LANE_SECTIONS)
    for lane, rows in rules.HEADLINES.items():
        assert rows, f"lane {lane} has no headline rules"
        for section, key, meaning in rows:
            assert section in rules.LANE_SECTIONS[lane], (lane, section)
            names = {f.name for f in dataclasses.fields(getattr(cfg, section))}
            assert key in names, f"[{section}] {key} is not a config field"
            assert meaning.strip(), (section, key)


def test_rules_json_values_equal_the_loaded_config():
    from cascade.miner.rules import _plain

    cfg = _cfg()
    for lane in LANES:
        rc, doc = _run_json(["rules", lane])
        assert rc == 0 and doc["ok"] is True, doc
        data = doc["data"]
        assert data["lane"] == lane
        assert (REPO / data["guide"]).is_file(), f"guide {data['guide']} missing"
        assert data["rules"], lane
        for row in data["rules"]:
            want = _plain(getattr(getattr(cfg, row["section"]), row["key"]))
            assert row["value"] == json.loads(json.dumps(want)), row


def test_rules_all_prints_every_field_of_the_lane_sections():
    import dataclasses

    cfg = _cfg()
    rc, doc = _run_json(["rules", "d", "--all"])
    assert rc == 0, doc
    values = doc["data"]["values"]
    assert set(values) == {"lane_d"}
    assert set(values["lane_d"]) == {f.name for f in dataclasses.fields(cfg.lane_d)}
    rc, doc = _run_json(["rules", "c", "--all"])
    assert rc == 0 and set(doc["data"]["values"]) == {"lane_c", "marketcast"}


def test_rules_key_lookup_and_unknown_key():
    cfg = _cfg()
    rc, doc = _run_json(["rules", "d", "--key", "contexts"])
    assert rc == 0, doc
    assert doc["data"]["section"] == "lane_d"
    assert doc["data"]["value"] == [int(c) for c in cfg.lane_d.contexts]
    rc, doc = _run_json(["rules", "c", "--key", "marketcast.gain_margin"])
    assert rc == 0 and doc["data"]["value"] == cfg.marketcast.gain_margin, doc
    rc, doc = _run_json(["rules", "a", "--key", "no_such_rule"])
    assert rc == 2 and doc["error"]["code"] == "unknown_key", doc
    # a key of another lane's section is not this lane's rule
    rc, doc = _run_json(["rules", "a", "--key", "lane_d.contexts"])
    assert rc == 2 and doc["error"]["code"] == "unknown_key", doc


def test_rules_reads_the_chain_toml_it_is_given(tmp_path):
    text = CHAIN_TOML.read_text()
    old = "max_open_per_coldkey = 2"
    assert text.count(old) == 1, "fixture anchor moved: update this test"
    deployed = tmp_path / "chain.toml"
    deployed.write_text(text.replace(old, "max_open_per_coldkey = 3"))
    rc, doc = _run_json(["rules", "d", "--key", "max_open_per_coldkey",
                         "--chain-toml", str(deployed)])
    assert rc == 0 and doc["data"]["value"] == 3, doc
    assert doc["data"]["source"] == str(deployed)


def test_lane_d_help_has_runnable_examples_and_points_at_rules():
    epilog = _subparser("lane-d").epilog or ""
    assert "cascade rules d" in epilog
    assert "MINER_LANE_D.md" in epilog
    examples = [line.strip() for line in epilog.splitlines()
                if line.strip().startswith("cascade lane-d ")]
    seen = set()
    for line in examples:
        args = _parse(shlex.split(line)[1:])
        seen.add(args.lane_d_cmd)
    assert seen == {"export", "verify", "submit", "ready", "status"}, seen
    assert any("--licence MIT" in line for line in examples)


def test_lane_d_flag_help_states_the_licence_and_context_rules():
    ready = _subparser("lane-d", "ready")
    licence = next(a for a in ready._actions if "--licence" in a.option_strings)
    assert "MIT" in licence.help and "only" in licence.help
    submit = _subparser("lane-d", "submit")
    context = next(a for a in submit._actions if "--context" in a.option_strings)
    assert "cascade rules d" in context.help and "never above" in context.help
