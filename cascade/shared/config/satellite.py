"""``[satellite]`` — satellite validator settings (DEC-CA-0058).

Read straight from the loaded config's ``raw`` table by the satellite, so the
main loader and every other service are untouched: a node that is not a
satellite never reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_STALE_AFTER_BLOCKS = 3600     # one mainnet era (DEC-CA-0058 §5)
DEFAULT_POLL_SECONDS = 60
DEFAULT_MAX_WALK = 64                 # receipts walked back per poll to reach the anchor


@dataclass(frozen=True)
class SatelliteConfig:
    """``stale_after_blocks``: blocks past the last verified settlement's
    boundary after which a satellite stops re-asserting its last verified
    vector and burns (the main validator is silent). ``poll_seconds``: how
    often the main validator's receipts are re-read. ``max_walk``: how many
    receipts one poll walks back from the main validator's latest to reach
    the last one this satellite replayed (a longer gap fails continuity)."""

    stale_after_blocks: int = DEFAULT_STALE_AFTER_BLOCKS
    poll_seconds: int = DEFAULT_POLL_SECONDS
    max_walk: int = DEFAULT_MAX_WALK


def parse_satellite(table: dict | None) -> SatelliteConfig:
    """Parse a ``[satellite]`` table (None/absent = the defaults)."""
    t = dict(table or {})
    known = {"stale_after_blocks", "poll_seconds", "max_walk"}
    unknown = sorted(set(t) - known)
    if unknown:
        raise ValueError(f"[satellite] unknown key(s): {unknown}")
    out = SatelliteConfig(
        stale_after_blocks=int(t.get("stale_after_blocks", DEFAULT_STALE_AFTER_BLOCKS)),
        poll_seconds=int(t.get("poll_seconds", DEFAULT_POLL_SECONDS)),
        max_walk=int(t.get("max_walk", DEFAULT_MAX_WALK)),
    )
    if out.stale_after_blocks <= 0:
        raise ValueError("[satellite] stale_after_blocks must be > 0")
    if out.poll_seconds <= 0 or out.max_walk <= 0:
        raise ValueError("[satellite] poll_seconds and max_walk must be > 0")
    return out


def satellite_config(cfg: Any) -> SatelliteConfig:
    """The ``[satellite]`` table of a loaded :class:`ChainConfig`."""
    return parse_satellite((getattr(cfg, "raw", None) or {}).get("satellite"))


__all__ = ["SatelliteConfig", "parse_satellite", "satellite_config"]
