"""hosts.toml publication — the provisioner's half of the trainer contract.

The trainer re-reads its ``--remote-hosts`` file at every round start
(``TrainerRunner._reload_remote_hosts``), so this file IS the interface: write
it and the next round trains on the fleet; empty it and the next round falls
back to local training (``load_hosts`` raises on zero ``[[host]]`` entries,
which the trainer treats as "no fleet this round" — the round is never lost).

Rendering itself stays in :func:`cascade.provision.core.render_hosts_toml`
(stage-aware, per-GPU fan-out); this module owns only the two write
disciplines: **atomic** (tmp + ``os.replace``, so the trainer's tomllib parse
never sees a torn file — it may read at any moment) and **clear** (the
all-providers-down escape hatch).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["clear_hosts", "write_hosts"]


def write_hosts(path: Path | str, content: str) -> None:
    """Atomically publish ``content`` as the trainer's hosts file.

    tmp + ``os.replace`` in the same directory: the trainer polls this path on
    its own schedule, so a plain ``write_text`` could hand it half a fleet
    (parse error → treated as no fleet → a round trains locally for nothing).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, p)


def clear_hosts(path: Path | str) -> None:
    """Publish an EMPTY hosts file — the trainer then trains locally.

    Used when no provider has capacity (or every health check failed): an
    empty file is the contract's explicit "no fleet" signal (``load_hosts``
    raises on no ``[[host]]`` entries and the trainer falls back local), which
    is strictly better than leaving a stale previous round's pods listed —
    those boxes are torn down and every dispatch to them would burn a retry.
    """
    write_hosts(path, "# cascade-provisioner: no fleet this round (trainer falls back local)\n")
