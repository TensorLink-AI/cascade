"""`cascade round` — the epoch-grid countdown math, frame rendering, and the
CLI wiring, over a fake chain client (no network)."""

from __future__ import annotations

import io
import types
from dataclasses import replace

import pytest

from cascade.miner import cli
from cascade.miner.dashboard import (
    DEFAULT_SECONDS_PER_BLOCK,
    format_duration,
    render,
    round_status,
    run_dashboard,
    seconds_per_block,
)
from cascade.shared.config import RoundConfig


class _FakeClient:
    def __init__(self, block):
        self.block = block

    def current_block(self):
        return self.block


def test_round_status_epoch_grid():
    # mirrors trainer run_forever: epoch = block // epoch_blocks
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200))
    assert st.epoch == 600
    assert st.epoch_start == 4_320_000
    assert st.next_epoch_start == 4_327_200
    assert st.blocks_elapsed == 1_004
    assert st.blocks_remaining == 6_196
    assert st.blocks_elapsed + st.blocks_remaining == st.epoch_blocks
    assert 0.0 <= st.progress < 1.0


def test_round_status_at_boundary():
    # the boundary block belongs to the NEW epoch: a full window remains
    st = round_status(7200, RoundConfig(epoch_blocks=7200))
    assert st.epoch == 1
    assert st.blocks_elapsed == 0
    assert st.blocks_remaining == 7200


def test_seconds_per_block_derived_from_config():
    # 24h over 7200 blocks = Bittensor's 12s cadence
    assert seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=24.0)) == 12.0
    # halve the round length, same grid → 6s blocks
    assert seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=12.0)) == 6.0
    # placeholder config falls back to the ~12s default
    assert (seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=0.0))
            == DEFAULT_SECONDS_PER_BLOCK)


def test_seconds_remaining_uses_cadence():
    st = round_status(7100, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    assert st.seconds_remaining == 100 * 12.0


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (59, "59s"), (61, "1m 1s"), (3600, "1h 0m 0s"),
     (93_784, "1d 2h 3m 4s"), (-5, "0s")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_render_frame_contents():
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    frame = render(st, "finney")
    assert "network: finney" in frame
    assert "current block   4,321,004" in frame
    assert "round (epoch)   600" in frame
    assert "epoch 601 at block 4,327,200" in frame
    assert "commit strictly before block 4,327,200" in frame
    assert "13.9%" in frame  # 1004/7200
    assert "(~12.0s/block)" in frame


def test_render_drift_ticks_countdown_down():
    st = round_status(100, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    base = render(st, "test")
    drifted = render(st, "test", drift_seconds=60.0)
    assert base != drifted  # a minute of wall clock moved the countdown


def test_run_dashboard_once_snapshot():
    out = io.StringIO()  # no isatty → snapshot mode even without --once
    rc = run_dashboard(_FakeClient(14_500), RoundConfig(epoch_blocks=7200),
                       "test", out=out)
    assert rc == 0
    text = out.getvalue()
    assert "current block   14,500" in text
    assert "\x1b[" not in text  # piped output stays escape-code free


def test_cmd_round_wiring(monkeypatch, cfg, capsys):
    client = _FakeClient(4_321_004)
    monkeypatch.setattr(cli, "load_chain_config",
                        lambda *_a, **_k: replace(cfg, round=RoundConfig(epoch_blocks=7200)))
    from cascade.shared.chain import ChainClient
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: client))
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True, refresh=30.0)
    rc = cli._cmd_round(args)
    assert rc == 0
    assert "until next round" in capsys.readouterr().out


def test_cmd_round_chain_error_exits_3(monkeypatch, cfg, capsys):
    from cascade.shared.chain import ChainClient, ChainError

    class _Dead:
        def current_block(self):
            raise ChainError("get_current_block_failed: boom")

    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: _Dead()))
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True, refresh=30.0)
    rc = cli._cmd_round(args)
    assert rc == 3
    assert "chain error" in capsys.readouterr().err


def test_round_registered_in_parser():
    with pytest.raises(SystemExit) as e:
        cli.main(["round", "--help"])
    assert e.value.code == 0
