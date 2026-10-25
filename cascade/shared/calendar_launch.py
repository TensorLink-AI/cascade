"""Resolve a UTC launch onto the era grid using finalized chain timestamps."""
from __future__ import annotations

import math
from datetime import datetime, timedelta


def launch_timestamp(value: str) -> float | None:
    if not value:
        return None
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ValueError("[v2] launch_at_utc must be a UTC ISO timestamp with Z or +00:00")
    return instant.timestamp()


def dated_launch_pending(cfg) -> bool:
    """A launch date is set and no block has been resolved or typed for it."""
    return launch_timestamp(cfg.v2.launch_at_utc) is not None and not cfg.v2.from_block


def check_launch_loads(cfg, head_block: int) -> None:
    """Trial-parse the config with v2 armed at the era boundary after
    ``head_block``: the v2 load checks run now, not on every poll after the
    date passes. Raises ``ValueError`` naming the launch when it cannot load."""
    if not dated_launch_pending(cfg):
        return
    from .config import parse_chain_config
    from .era import next_era_start
    from .netconfig import effective_raw

    raw = effective_raw(cfg)
    raw.setdefault("v2", {})["from_block"] = next_era_start(cfg.round, int(head_block))
    try:
        parse_chain_config(raw)
    except ValueError as exc:
        raise ValueError(
            f"[v2] launch_at_utc = {cfg.v2.launch_at_utc!r} schedules a v2 config that does not "
            f"load: {exc}. Complete the v2 configuration or clear launch_at_utc before "
            "starting this service") from exc


def check_calendar_launch(cfg, client) -> None:
    """Refuse a service start whose scheduled launch could not be parsed.

    Resolution re-parses the config with v2 active, which runs the v2 load
    checks. Fail now rather than halting every poll once the date passes.
    The finalized head places the trial launch on the grid the fleet would use.
    """
    if not dated_launch_pending(cfg):
        return
    check_launch_loads(cfg, int(client.finalized_block()))


def resolve_calendar_launch(cfg, client):
    """No wall-clock estimate, readiness vote or write to live state.

    The first finalized block at/after the target selects the next era start.
    Every late joiner derives that same block; the lane never opens early.
    A typed from_block remains an explicit operator override.
    """
    target = launch_timestamp(cfg.v2.launch_at_utc)
    if target is None or cfg.v2.from_block:
        return cfg
    from .config import parse_chain_config
    from .era import next_era_start
    from .netconfig import effective_raw

    head = int(client.finalized_block())

    def stamp(block):
        try:
            value = float(client.block_timestamp(block))
        except Exception as exc:
            # A pruned node cannot read state far behind its head; every node must
            # derive the same launch block, so guessing is not an option.
            raise RuntimeError(
                f"calendar launch: cannot read the timestamp of block {block} ({exc}); use an "
                "archive endpoint or type [v2] from_block to the resolved launch block") from exc
        if not math.isfinite(value):
            raise ValueError("chain timestamp is non-finite; calendar launch is unresolved")
        return value

    if head < 1 or stamp(head) < target:
        return cfg
    # Most nodes observe the crossing immediately. Bracket it from the recent
    # head first rather than requiring an archive read halfway to genesis.
    hi, step = head, 1
    lo = max(1, head - step)
    while lo > 1 and stamp(lo) >= target:
        hi = lo
        step *= 2
        lo = max(1, head - step)
    while lo < hi:
        mid = (lo + hi) // 2
        if stamp(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    block = next_era_start(cfg.round, lo - 1)
    raw = effective_raw(cfg)
    raw.setdefault("v2", {})["from_block"] = block
    resolved = parse_chain_config(raw)
    from dataclasses import replace

    return replace(resolved, activation=cfg.activation)


def receipt_calendar_config(cfg, receipt, client=None):
    """Resolve a receipt's rules without trusting its claimed activation block."""
    if not cfg.v2.launch_at_utc or cfg.v2.from_block:
        return cfg
    if client is None:
        raise ValueError("dated launch requires a chain connection or resolved [v2] from_block")
    if int(receipt.epoch_start_block) > int(client.finalized_block()):
        raise ValueError("receipt boundary is not finalized")
    resolved = resolve_calendar_launch(cfg, client)
    if resolved.v2.from_block:
        return resolved
    # The finalized head is before launch; this historical receipt is pre-v2.
    # Clear only the replay copy, so nested replay cannot mistake it for unknown.
    from dataclasses import replace

    from .config import parse_chain_config
    from .netconfig import effective_raw

    raw = effective_raw(cfg)
    raw["v2"]["launch_at_utc"] = ""
    return replace(parse_chain_config(raw), activation=cfg.activation)
