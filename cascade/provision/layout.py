"""GPU layout SKUs: ``"8xH200"`` = one pod with eight H200s.

``[round] funded_pod_skus`` may list layouts beside plain types. A layout
rents a pod of exactly that many GPUs of the base type, and every per-SKU
table (walls, bench floors, qualification) keys on the layout name, so a
4×H100 and an 8×H100 are priced and timed separately. A plain type is a
one-GPU layout.
"""

from __future__ import annotations

import re

_LAYOUT = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\S+)\s*$")


def split_layout(sku: str) -> tuple[str, int]:
    """``"8xH200"`` → ``("H200", 8)``; ``"RTX4090"`` → ``("RTX4090", 1)``."""
    m = _LAYOUT.match(str(sku or ""))
    if m and int(m.group(1)) >= 1:
        return m.group(2), int(m.group(1))
    return str(sku or "").strip(), 1


def layout_gpus(sku: str) -> int:
    return split_layout(sku)[1]


def visible_devices(sku: str) -> str:
    """``CUDA_VISIBLE_DEVICES`` for a leg on a pod rented as ``sku``: every
    GPU of a layout (``"0,1,2,3"`` for ``4xH100``), ``"0"`` for one GPU."""
    return ",".join(str(i) for i in range(layout_gpus(sku)))
