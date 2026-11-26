"""Scheduled pool commands must not silently override the reviewed caps."""
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest
import yaml

from cascade.pool.cli import _add_build_args

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/publish-pool.yml').read_text())
COMMANDS = [step['run'] for step in WORKFLOW['jobs']['publish']['steps']
            if re.search(r'^\s*TSFORGE_DIR=.*\bcascade-pool publish\b', step.get('run', ''), re.M)]


@pytest.mark.parametrize('command', COMMANDS, ids=('primary-pool',))
def test_scheduled_publish_uses_the_reviewed_cli_capacity(command):
    parser = argparse.ArgumentParser()
    _add_build_args(parser)
    defaults = parser.parse_args([])
    match = re.search(r'^\s*TSFORGE_DIR=.*?cascade-pool publish\b(.*)', command, re.M | re.S)
    parsed, _ = parser.parse_known_args(shlex.split(match[1].replace('\\\n', ' ')))
    assert parsed.max_series_per_domain == defaults.max_series_per_domain
    assert parsed.max_series_per_domain_freq == defaults.max_series_per_domain_freq
    # Reviewed values (owner 2026-11-23). The panel cap has no CLI default
    # (None = the source's built-in 200), so it is pinned on the command itself.
    assert (parsed.max_series_per_domain, parsed.max_series_per_domain_freq) == (1500, 750)
    assert parsed.max_panel_series_per_feed == 20
    assert len(COMMANDS) == 1
    assert parsed.lane == 'a'


def test_no_retired_lane_c_forge_command_is_scheduled():
    commands = [step.get('run', '') for step in WORKFLOW['jobs']['publish']['steps']]
    assert not any(re.search(r'cascade-pool\s+publish\s+--lane\s+c\b', command) for command in commands)


def _flag(command: str, name: str) -> str:
    match = re.search(rf'{name}\s+(\S+)', command)
    assert match, f'{name} missing'
    return match[1]


def test_hourly_publish_keeps_the_pending_block_an_epoch_past_the_build():
    on = WORKFLOW.get('on', WORKFLOW.get(True))
    assert [c['cron'] for c in on['schedule']] == ['0 * * * *']
    (command,) = COMMANDS
    # Buffer 2: a ~55 min build can never land on an epoch that began during it.
    assert int(_flag(command, '--round-buffer')) >= 2


def test_reveal_runs_once_a_day_and_never_reveals_the_active_snapshot():
    (step,) = [s for s in WORKFLOW['jobs']['publish']['steps']
               if s.get('name', '').startswith('Reveal retired snapshots')]
    run = step['run']
    # scheduled runs skip the reveal outside one UTC hour, before any reveal
    assert 'github.event_name }}" = "schedule"' in run and 'date -u +%H' in run
    assert run.index('date -u +%H') < run.index('cascade-pool reveal')
    (command,) = COMMANDS
    buffer = int(_flag(command, '--round-buffer'))
    primary = re.search(r'cascade-pool reveal --chain-toml.*?(?=\n\s*#|\n\s*if |\Z)', run, re.S)[0]
    # buffer pending entries sit above the active one: withhold all of them + it
    assert int(_flag(primary, '--keep-newest')) >= buffer + 1
    # owner 2026-11-24 option (b): ONE snapshot per day — the newest retiree;
    # the rest stay in the pool bucket (no backfill)
    assert '--newest-only' in primary.split()
    assert int(_flag(primary, '--max-reveal')) == 1


def test_reveal_cli_parses_the_scheduled_flags():
    from cascade.pool.cli import _add_reveal

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    _add_reveal(sub)
    args = parser.parse_args(['reveal', '--hf-repo', 'o/r', '--keep-newest', '3',
                              '--newest-only', '--max-reveal', '1'])
    assert (args.keep_newest, args.newest_only, args.max_reveal) == (3, True, 1)
    assert parser.parse_args(['reveal', '--hf-repo', 'o/r']).newest_only is False


def test_reveal_delay_covers_the_lane_d_embargo():
    """The HF reveal must never expose lane A bytes lane D still scores as
    embargoed: a revealed tar's effective block E <= N - K*epoch (one entry per
    epoch boundary at most), its bytes were built at head <= E, and the reveal
    runs at head >= N - round_buffer*epoch, so the delay is at least
    (K - round_buffer) epochs. #880 (hourly, K=3) cut it to ~6-9 h."""
    from cascade.shared.config import BLOCK_SECONDS, load_chain_config

    cfg = load_chain_config(ROOT / 'chain.toml')
    (step,) = [s for s in WORKFLOW['jobs']['publish']['steps']
               if s.get('name', '').startswith('Reveal retired snapshots')]
    primary = re.search(r'cascade-pool reveal --chain-toml.*?(?=\n\s*#|\n\s*if |\Z)',
                        step['run'], re.S)[0]
    (command,) = COMMANDS
    keep = int(_flag(primary, '--keep-newest'))
    buffer = int(_flag(command, '--round-buffer'))
    epoch = int(cfg.round.epoch_blocks)
    delay_h = (keep - buffer) * epoch * BLOCK_SECONDS / 3600.0
    assert delay_h >= float(cfg.lane_d.embargo_hours), (keep, buffer, epoch, delay_h)
    # the pool index keeps enough entries for anything to be revealed at all
    from inspect import signature

    from cascade.shared.hippius import publish_pool_snapshot
    assert signature(publish_pool_snapshot).parameters['max_keep'].default > keep
