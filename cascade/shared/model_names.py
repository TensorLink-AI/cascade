"""Public MoE names use total parameters; contract IDs remain machine identifiers."""
from __future__ import annotations

MOE_FAMILY_NAME = "Cascade MoE Base"


def parameter_count_label(total: int) -> str:
    scale, suffix = (1_000_000_000, "B") if total >= 1_000_000_000 else (1_000_000, "M")
    return f"{total / scale:.2f}".rstrip("0").rstrip(".") + suffix


def moe_display_name(total: int) -> str:
    return f"{MOE_FAMILY_NAME} - {parameter_count_label(total)}"
