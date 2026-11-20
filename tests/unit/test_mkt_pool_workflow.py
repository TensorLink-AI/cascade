"""The MarketCast publish workflow: real flags, scoped keys, an always-removed signer."""
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import yaml

from cascade.pool.cli import _add_publish

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/publish-mkt-pool.yml').read_text())
STEPS = WORKFLOW['jobs']['publish']['steps']


def _step(name_part: str) -> dict:
    (step,) = [s for s in STEPS if name_part in s.get('name', '')]
    return step


def test_publish_command_parses_as_a_signed_mkt_publish_from_the_recording():
    run = _step('Build + sign + publish')['run']
    match = re.search(r'cascade-pool publish\b(.*)', run, re.S)
    parser = argparse.ArgumentParser()
    _add_publish(parser.add_subparsers())
    parsed = parser.parse_args(['publish', *shlex.split(match[1].replace('\\\n', ' '))])
    assert parsed.lane == 'mkt'
    assert str(parsed.bar_store) == 'mkt/bars'     # never a cold (--allow-cold) build
    assert not parsed.allow_cold
    assert parsed.wallet_name and parsed.wallet_hotkey and parsed.wallet_path


def test_forge_read_key_is_only_on_the_sync_step():
    forge = [s for s in STEPS if 'TSFORGE_S3_ACCESS_KEY' in str(s.get('env', {}))]
    assert [s['name'] for s in forge] == [_step('Sync the bar recording')['name']]
    assert 'TSFORGE_S3' not in str(WORKFLOW['jobs']['publish']['env'])


def test_signer_hotkey_is_scoped_and_always_removed():
    holders = [s for s in STEPS if 'MKT_SIGNER_HOTKEY' in str(s.get('env', {}))]
    assert [s['name'] for s in holders] == [_step('Install the signer hotkey')['name']]
    cleanup = _step('Remove the signer hotkey')
    assert cleanup['if'] == 'always()'
    assert STEPS[-1] is cleanup


def test_scheduled_runs_are_gated_on_marketcast_being_armed():
    gate = _step('Skip while MarketCast is off')['run']
    assert 'marketcast.from_block' in gate and 'schedule' in gate
    for name in ('Sync the bar recording', 'Install the signer hotkey', 'Build + sign + publish'):
        assert _step(name)['if'] == 'steps.gate.outputs.run'


def test_dispatch_defaults_to_the_inert_sentinel_block():
    on = WORKFLOW.get('on', WORKFLOW.get(True))
    assert on['workflow_dispatch']['inputs']['effective_block']['default'] == '999999999999'
