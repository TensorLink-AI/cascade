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

import json
import os
import tomllib
from pathlib import Path

__all__ = ["clear_hosts", "parse_host_entries", "render_host_entries", "write_hosts"]

# Field order for re-rendered entries (the trainer's load_hosts schema order);
# unknown keys follow, sorted, so nothing an operator wrote is dropped.
_ENTRY_KEY_ORDER = ("name", "host", "port", "user", "key_path", "remote_python",
                    "workdir", "cuda_device", "stage", "chain_toml", "forward_env",
                    "ssh_options")


def parse_host_entries(text: str) -> list[dict]:
    """The ``[[host]]`` tables of a hosts.toml body, as dicts (``[]`` when none).

    Raises ``tomllib.TOMLDecodeError`` on a malformed file — the caller decides
    whether a torn file means "preserve nothing" (it does: never re-emit
    garbage into the trainer's contract file).
    """
    return [dict(h) for h in tomllib.loads(text).get("host", []) if isinstance(h, dict)]


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, list | tuple):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    # json.dumps yields a valid TOML basic string for any str (same escapes).
    return json.dumps(str(v))


def render_host_entries(entries: list[dict]) -> str:
    """Re-render parsed ``[[host]]`` entries as a hosts.toml fragment.

    Byte-for-byte fidelity is not the goal (tomllib drops comments and
    layout); semantic fidelity is: the fragment parses back to the same dicts
    through ``tomllib`` and loads through the trainer's ``load_hosts``.
    """
    out = []
    for e in entries:
        keys = [k for k in _ENTRY_KEY_ORDER if k in e] + sorted(
            k for k in e if k not in _ENTRY_KEY_ORDER)
        lines = ["[[host]]"] + [f"{k} = {_toml_value(e[k])}" for k in keys]
        out.append("\n".join(lines) + "\n\n")
    return "".join(out)


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
