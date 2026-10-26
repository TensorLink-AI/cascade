"""Offline documentation checks must catch drift in both directions."""

import shutil

import pytest

from scripts.check_docs import (
    CONFIG_END,
    CONFIG_START,
    ROOT,
    cli_issues,
    config_issues,
    config_reference,
    guide_paths,
    link_issues,
    reward_issues,
    reward_reference,
    training_reference,
)


def test_current_guides_match_interfaces_and_configuration():
    paths = guide_paths()
    issues, examples, commands, entries = cli_issues(paths)
    assert not issues
    assert examples > 0 and commands > 0 and entries > 0
    assert not link_issues(paths)
    assert not config_issues()
    assert not reward_issues()


def test_bad_documented_option_is_rejected_without_executing_command(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("```sh\ncascade fund https://fixture.invalid --gpou H100\n```\n")
    issues, *_ = cli_issues([guide], root=tmp_path)
    assert any("unknown options ['--gpou']" in issue for issue in issues)


def test_code_command_missing_from_guides_is_reported(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("`cascade verify DIR --json`\n")
    # CLI sources are in the real checkout; only the guide corpus is incomplete.
    issues, *_ = cli_issues([guide], root=tmp_path)
    assert "Undocumented miner command: cascade status" in issues


def test_bad_nested_subcommand_is_reported(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("`cascade direct nonexistent --json`\n")
    issues, *_ = cli_issues([guide], root=tmp_path)
    assert any("unknown command" in issue for issue in issues)


def test_missing_link_and_heading_are_reported(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("# Exists\n[missing](absent.md) [wrong heading](#absent)\n")
    issues = link_issues([guide], root=tmp_path)
    assert len(issues) == 2


def test_published_site_route_must_exist_in_writer(tmp_path):
    guide = tmp_path / "cascade/website/llms.txt"
    writer = tmp_path / "cascade/shared/v2_status.py"
    guide.parent.mkdir(parents=True)
    writer.parent.mkdir(parents=True)
    guide.write_text("[status](./status/v2.json)\n")
    writer.write_text('KEY = "status/v3.json"\n')
    assert link_issues([guide], root=tmp_path)
    writer.write_text('KEY = "status/v2.json"\n')
    assert not link_issues([guide], root=tmp_path)


def test_changed_template_cannot_leave_old_documented_values(tmp_path):
    (tmp_path / "docs").mkdir()
    for name in ("chain.toml", "chain.testnet.toml"):
        shutil.copy2(ROOT / name, tmp_path / name)
    guide = tmp_path / "docs/AGENT_MINING.md"
    guide.write_text(CONFIG_START + "\n" + config_reference(tmp_path) + "\n" + CONFIG_END)
    assert not config_issues(tmp_path)
    config = tmp_path / "chain.toml"
    text = config.read_text()
    assert "window_seconds     = 5400" in text
    config.write_text(text.replace("window_seconds     = 5400", "window_seconds     = 3600"))
    assert config_issues(tmp_path)


def test_reward_examples_include_all_pools_decay_expiry_and_burn():
    tables = reward_reference()
    assert "| A | 100% | 0% | 0% |" in tables["allocation"]
    assert "| A + B | 75% | 25% | 0% |" in tables["allocation"]
    assert "| A + B + C | 50% | 25% | 25% |" in tables["allocation"]
    for row in ("| 0–4 | 100% | 2.5% |", "| 5 | 50% | 1.25% |",
                "| 8–27 | 10% | 0.25% |", "| 28+ | 0% | 0% |"):
        assert row in tables["slot-decay"]
    example = tables["worked-example"]
    assert "| A king | 30% |" in example
    assert "| Eligible immediate former king | 2.5% |" in example
    assert "| One full-pay B holder | 2.5% |" in example
    assert "| One full-pay C holder (mkt:returns) | 12.5% |" in example
    assert "| Burn | 52.5% |" in example


def test_wrong_reward_arithmetic_in_document_is_rejected(tmp_path):
    shutil.copy2(ROOT / "chain.toml", tmp_path / "chain.toml")
    guide = tmp_path / "guide.md"
    guide.write_text("<!-- cascade:rewards:worked-example:start -->\n"
                     + reward_reference()["worked-example"].replace("52.5%", "0%")
                     + "\n<!-- cascade:rewards:worked-example:end -->\n")
    issues = reward_issues([guide], root=tmp_path)
    assert len(issues) == 1 and "worked-example" in issues[0]


@pytest.mark.parametrize("entrypoint", ["AGENTS.md", "README.md", "docs/VALIDATOR.md"])
def test_entrypoint_cannot_drop_reward_instructions(tmp_path, entrypoint):
    shutil.copy2(ROOT / "chain.toml", tmp_path / "chain.toml")
    guide = tmp_path / entrypoint
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("# Agent instructions\n")
    issues = reward_issues([guide], root=tmp_path)
    assert len(issues) == 1 and "missing reward reference summary" in issues[0]


@pytest.mark.parametrize("old,new", [
    ('budget_denomination         = "series_points"', 'budget_denomination         = "points"'),
    ('batch_size                  = 64', 'batch_size                  = 32'),
    ('max_length         = 32768', 'max_length         = 16384'),
])
def test_training_limits_cannot_drift_from_interface_guide(tmp_path, old, new):
    (tmp_path / "docs").mkdir()
    for name in ("chain.toml", "chain.testnet.toml", "docs/AGENT_MINING.md", "docs/INTERFACE.md"):
        shutil.copy2(ROOT / name, tmp_path / name)
    assert not config_issues(tmp_path)
    config = tmp_path / "chain.toml"
    text = config.read_text()
    assert old in text
    config.write_text(text.replace(old, new, 1))
    assert "docs/INTERFACE.md: training reference differs from checked-in TOML" in config_issues(tmp_path)


def test_training_reference_distinguishes_streaming_and_cpu_shard_limits():
    table = training_reference()
    assert "| Budget accounting | `series_points` | `points` |" in table
    assert "64–4096 | 64–32768 in the CPU shard tier" in table
    assert "| Corpus feed | `stream_cpu` | `shards` |" in table
    assert "90 minutes" in table
