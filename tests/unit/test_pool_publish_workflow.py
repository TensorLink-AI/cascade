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
    assert len(COMMANDS) == 1
    assert parsed.lane == 'a'


def test_no_retired_lane_c_forge_command_is_scheduled():
    commands = [step.get('run', '') for step in WORKFLOW['jobs']['publish']['steps']]
    assert not any(re.search(r'cascade-pool\s+publish\s+--lane\s+c\b', command) for command in commands)
