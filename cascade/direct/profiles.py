"""Direct-MoE gang layouts a job may name, and which ones it gets by default.

A funded job's terms pin the hardware profiles its gang may rent
(``terms["profiles"]``), signed by the miner before any operator config is
seen, so the catalog miners choose from ships with the code. A profile name
here must be the key of the operator's qualified :class:`~.compute.PodProfile`
of the same SKU and gang size (the runtime refuses a mismatch at start).

Reference job: 27,500 updates, batch 64, eight variates, context 4096, at the
advertised Lium rates (USD, 2026-10-29). ``measured`` marks wall times that
were timed; the rest are estimates. A layout whose estimated compute is at or
under :data:`DEFAULT_MAX_USD` is a DEFAULT: ``cascade direct fund`` without
``--profile`` offers every default of the job's ``--world-size``. The others
are OPT-IN: a job rents one only when ``--profile`` names it.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_USD = 75.0


@dataclass(frozen=True)
class Layout:
    name: str
    sku: str
    world_size: int
    hours: float            # reference-job wall time
    measured: bool
    hourly_usd: float       # whole-gang rental, advertised

    @property
    def estimated_usd(self) -> float:
        return round(self.hours * self.hourly_usd, 2)

    @property
    def default(self) -> bool:
        return self.estimated_usd <= DEFAULT_MAX_USD


LAYOUTS: tuple[Layout, ...] = (
    Layout("8xB200", "B200", 8, 2.2, False, 33.60),
    Layout("8xB300", "B300", 8, 2.2, False, 56.00),
    Layout("8xH200", "H200", 8, 2.67, True, 18.00),
    Layout("8xH100", "H100", 8, 2.82, True, 24.00),
    Layout("4xB200", "B200", 4, 3.26, True, 16.80),
    Layout("4xB300", "B300", 4, 3.3, False, 32.00),
    Layout("4xH200", "H200", 4, 4.0, False, 18.00),
    Layout("4xH100", "H100", 4, 4.23, True, 24.00),
)
BY_NAME = {lay.name: lay for lay in LAYOUTS}


def default_profiles(world_size: int) -> list[str]:
    """The default layouts for a gang of ``world_size`` GPUs, cheapest first."""
    rows = [lay for lay in LAYOUTS if lay.world_size == int(world_size) and lay.default]
    return [lay.name for lay in sorted(rows, key=lambda lay: (lay.estimated_usd, lay.name))]


def resolve_profiles(named: list[str] | None, world_size: int) -> list[str]:
    """The profiles a job's terms pin: ``named`` as given (opt-ins included),
    or the defaults for ``world_size`` when none is named. A catalog layout
    of a different gang size is refused: a job has one world size."""
    if named:
        for name in named:
            lay = BY_NAME.get(name)
            if lay is not None and lay.world_size != int(world_size):
                raise ValueError(f"profile {name} is a {lay.world_size}-GPU gang, "
                                 f"not --world-size {world_size}")
        return sorted(set(named))
    out = default_profiles(world_size)
    if not out:
        raise ValueError(f"no default profile for --world-size {world_size}; name one with "
                         "--profile (" + ", ".join(lay.name for lay in LAYOUTS) + ")")
    return out


def check_operator_profiles(profiles: dict) -> None:
    """Refuse an operator PodProfile keyed by a catalog name whose SKU or
    gang size differs from the catalog's: miners chose it by that name."""
    for name, p in profiles.items():
        lay = BY_NAME.get(name)
        if lay is not None and (p.world_size != lay.world_size
                                or p.sku.upper() != lay.sku.upper()):
            raise ValueError(f"operator profile {name} is {p.world_size}x{p.sku}; the "
                             f"catalog layout is {lay.world_size}x{lay.sku}")
