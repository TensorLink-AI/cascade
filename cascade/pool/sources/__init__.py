"""Concrete data sources and a name registry.

``cascade-pool build --sources openmeteo,wikimedia`` resolves names here.
Register a new source by adding it to :data:`_REGISTRY`; it then works in the
CLI and through :func:`get_sources` with no other wiring.
"""

from __future__ import annotations

from collections.abc import Callable

from ..source import DataSource
from .openmeteo import OpenMeteoSource
from .synthetic import SyntheticSource
from .wikimedia import WikimediaSource

# Name → zero-arg factory (so each `build` gets a fresh instance with defaults).
_REGISTRY: dict[str, Callable[[], DataSource]] = {
    "openmeteo": OpenMeteoSource,
    "wikimedia": WikimediaSource,
    "synthetic": SyntheticSource,
}

# Real-world sources used when the operator doesn't name any explicitly.
DEFAULT_SOURCES = ("openmeteo", "wikimedia")


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_source(name: str) -> DataSource:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown source {name!r}; available: {', '.join(available())}") from None


def get_sources(names: list[str]) -> list[DataSource]:
    return [get_source(n) for n in names]


__all__ = [
    "OpenMeteoSource",
    "WikimediaSource",
    "SyntheticSource",
    "DEFAULT_SOURCES",
    "available",
    "get_source",
    "get_sources",
]
