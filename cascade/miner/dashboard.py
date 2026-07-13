"""``cascade round`` — a terminal dashboard counting down to the next round.

A round is one ``[round] epoch_blocks`` window on the chain block grid (the
same math the trainer uses in :meth:`~cascade.trainer.loop.TrainerRunner.
run_forever`): the current epoch is ``block // epoch_blocks`` and the next
round starts at the next epoch boundary. Only commitments revealed STRICTLY
BEFORE that boundary enter the next round, so the boundary is also the
submission deadline — the number a miner actually watches.

Everything on-chain-exact here is in *blocks*; the wall-clock countdown is an
estimate derived from the configured cadence (``round_hours`` over
``epoch_blocks``, ~12s/block on Bittensor). In watch mode the display ticks
every second by interpolating between chain polls, and re-syncs to the real
block height every ``refresh`` seconds.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from ..shared.config import RoundConfig

DEFAULT_SECONDS_PER_BLOCK = 12.0
BAR_WIDTH = 28


def seconds_per_block(round_cfg: RoundConfig) -> float:
    """The configured wall-clock cadence: ``round_hours`` spread over
    ``epoch_blocks``. Falls back to Bittensor's ~12s when the config carries a
    placeholder (non-positive) value."""
    if round_cfg.round_hours > 0 and round_cfg.epoch_blocks > 0:
        return round_cfg.round_hours * 3600.0 / round_cfg.epoch_blocks
    return DEFAULT_SECONDS_PER_BLOCK


@dataclass(frozen=True)
class RoundStatus:
    """A snapshot of where the current block sits on the epoch grid."""

    block: int              # chain block the snapshot was taken at
    epoch_blocks: int       # blocks per round ([round] epoch_blocks)
    spb: float              # estimated seconds per block

    @property
    def epoch(self) -> int:
        return self.block // self.epoch_blocks

    @property
    def epoch_start(self) -> int:
        return self.epoch * self.epoch_blocks

    @property
    def next_epoch_start(self) -> int:
        return self.epoch_start + self.epoch_blocks

    @property
    def blocks_elapsed(self) -> int:
        return self.block - self.epoch_start

    @property
    def blocks_remaining(self) -> int:
        return self.next_epoch_start - self.block

    @property
    def seconds_remaining(self) -> float:
        return self.blocks_remaining * self.spb

    @property
    def progress(self) -> float:
        return self.blocks_elapsed / self.epoch_blocks


def round_status(block: int, round_cfg: RoundConfig) -> RoundStatus:
    return RoundStatus(
        block=int(block),
        epoch_blocks=max(1, round_cfg.epoch_blocks),
        spb=seconds_per_block(round_cfg),
    )


def format_duration(seconds: float) -> str:
    """``93784.0`` → ``"1d 2h 3m 4s"`` (leading zero units dropped)."""
    s = max(0, int(seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts: list[str] = []
    if d:
        parts.append(f"{d}d")
    if h or parts:
        parts.append(f"{h}h")
    if m or parts:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _bar(progress: float, width: int = BAR_WIDTH) -> str:
    filled = min(width, max(0, round(progress * width)))
    return "█" * filled + "░" * (width - filled)


def render(st: RoundStatus, network: str, *, drift_seconds: float = 0.0) -> str:
    """The dashboard frame. ``drift_seconds`` is the wall-clock time elapsed
    since ``st.block`` was fetched, so watch mode can tick the countdown every
    second between chain polls."""
    remaining = max(0.0, st.seconds_remaining - drift_seconds)
    eta = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(time.time() + remaining))
    pct = st.progress * 100.0
    return (
        f"cascade round — network: {network}\n"
        f"  current block   {st.block:,}\n"
        f"  round (epoch)   {st.epoch:,}  ·  started at block {st.epoch_start:,}\n"
        f"  next round      epoch {st.epoch + 1:,} at block {st.next_epoch_start:,}\n"
        f"  progress        [{_bar(st.progress)}]  {pct:.1f}%"
        f"  ({st.blocks_elapsed:,} / {st.epoch_blocks:,} blocks)\n"
        f"  countdown       {format_duration(remaining)} until next round"
        f"  (~{st.spb:.1f}s/block)\n"
        f"  deadline        commit strictly before block {st.next_epoch_start:,}"
        f" to enter epoch {st.epoch + 1:,}\n"
        f"  eta             {eta} (estimated)"
    )


def run_dashboard(
    client,
    round_cfg: RoundConfig,
    network: str,
    *,
    once: bool = False,
    refresh: float = 30.0,
    out=None,
) -> int:
    """Print the round dashboard; in watch mode, keep it live until Ctrl+C.

    Watch mode redraws in place (ANSI cursor-up), ticking the countdown every
    second and re-fetching the real block height every ``refresh`` seconds.
    A non-TTY ``out`` degrades to a single snapshot so piped/scripted runs
    never emit escape codes.
    """
    out = out if out is not None else sys.stdout
    st = round_status(client.current_block(), round_cfg)
    frame = render(st, network)
    print(frame, file=out)
    if once or not getattr(out, "isatty", lambda: False)():
        return 0

    lines = frame.count("\n") + 1
    fetched_at = time.monotonic()
    try:  # pragma: no cover — interactive loop; the frame logic is tested above
        while True:
            time.sleep(1.0)
            drift = time.monotonic() - fetched_at
            if drift >= refresh:
                st = round_status(client.current_block(), round_cfg)
                fetched_at, drift = time.monotonic(), 0.0
            # move to the top of the previous frame, clear below, redraw
            print(f"\x1b[{lines}F\x1b[J" + render(st, network, drift_seconds=drift),
                  file=out)
    except KeyboardInterrupt:  # pragma: no cover
        return 0
