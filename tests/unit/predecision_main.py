"""The release chain.toml as it shipped BEFORE 2026-11-13 typed the decided
rollover and the resolved v2 launch: every DEC-CA-0043 rollover key back to
its vote-open value and ``[v2] from_block`` / ``commit_from_block`` back to 0.

Tests that exercise the on-chain rollover vote or the dated-launch resolution
from a fresh node's point of view load this; tests about the shipped release
load ``chain.toml`` itself. ``test_predecision_main_is_the_exact_inverse``
pins that this is the exact inverse of the typed lines — and that applying the
decided rollover + resolving the launch lands back on the shipped config.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (section, key, pre-decision value) — exactly the lines typed on 2026-11-13.
UNTYPED = (
    ("round", "epoch_blocks", "3600"),
    ("round", "epoch_blocks_prev", "0"),
    ("round", "epoch_activation_block", "0"),
    ("round", "rolling_from_block", "0"),
    ("scoring", "cohort_maxt_increment_from_block", "0"),
    ("scoring", "era_king_from_block", "0"),
    ("scoring", "tenure_blocks_from_block", "0"),
    ("v2", "commit_from_block", "0"),
    ("v2", "from_block", "0"),
)


def predecision_text(text: str | None = None) -> str:
    """``chain.toml`` text with the decided blocks untyped (section-scoped:
    only the key inside its own table is rewritten)."""
    text = (REPO / "chain.toml").read_text() if text is None else text
    for section, key, value in UNTYPED:
        head = re.search(rf"(?m)^\[{re.escape(section)}\]\s*$", text)
        assert head, section
        nxt = re.search(r"(?m)^\[", text[head.end():])
        end = head.end() + (nxt.start() if nxt else len(text) - head.end())
        body, n = re.subn(rf"(?m)^{re.escape(key)}(\s*)=\s*[^\n#]*?(\s*#.*)?$",
                          rf"{key}\g<1>= {value}", text[head.end():end], count=1)
        assert n == 1, (section, key)
        text = text[:head.end()] + body + text[end:]
    return text


def predecision_path(tmp_path) -> Path:
    p = Path(tmp_path) / "chain.predecision.toml"
    p.write_text(predecision_text())
    return p


def predecision_main(tmp_path):
    from cascade.shared.config import load_chain_config

    return load_chain_config(predecision_path(tmp_path))
