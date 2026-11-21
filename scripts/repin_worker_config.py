#!/usr/bin/env python3
"""Atomically align all worker pins with a published, source-matching image."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascade.provision.codeprint import (  # noqa: E402
    canonical_fingerprint,
    registry_code_fingerprint,
)
from cascade.shared.atomic import write_text_atomic  # noqa: E402
from cascade.shared.config import parse_chain_config  # noqa: E402


def _sub_in_section(text: str, section: str, field: str, value: str) -> tuple[str, int]:
    """``text`` with ``field = "<value>"`` set inside ``[section]`` only."""
    head = re.search(rf"(?m)^\[{re.escape(section)}\]\s*(#.*)?$", text)
    if head is None:
        return text, 0
    nxt = re.search(r"(?m)^\[", text[head.end():])
    end = head.end() + (nxt.start() if nxt else len(text) - head.end())
    body, count = re.subn(rf'(?m)^{field}\s*=\s*"[^"\n]*"[^\n]*$',
                          f'{field} = "{value}"', text[head.end():end])
    return text[:head.end()] + body + text[end:], count


def repinned_config(text: str, image: str, fingerprint: str) -> str:
    if not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("worker image must be an immutable registry reference")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("invalid worker source fingerprint")
    root = Path(__file__).resolve().parents[1]
    local = canonical_fingerprint((str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted((root / "cascade").rglob("*.py")))
    if fingerprint != local:
        raise ValueError("published worker source differs from this release; rebuild before repinning")
    digest = image.rsplit("@", 1)[1]
    # section-scoped: the fleet pins only ([marketcast] pod_image /
    # worker_code_fingerprint name MarketCast's own release and are untouched)
    for section, field, value in (
        ("training", "train_image_digest", digest),
        ("v2.training", "train_image_digest", digest),
        ("round", "funded_pod_image", image),
        ("round", "worker_code_fingerprint", fingerprint),
    ):
        text, count = _sub_in_section(text, section, field, value)
        if count != 1:
            raise ValueError(f"expected 1 [{section}] {field} pin, found {count}")
    cfg = parse_chain_config(tomllib.loads(text))
    if (cfg.training.train_image_digest != digest
            or cfg.training.v2_contract is None
            or cfg.training.v2_contract.train_image_digest != digest):
        raise ValueError("training image pins are not in the expected contract sections")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("image", help="public registry image pinned by sha256 digest")
    args = parser.parse_args(argv)
    # Fetch and validate before touching the file. No rental or deployment occurs.
    fingerprint, count = registry_code_fingerprint(args.image)
    if not count:
        parser.error("published worker image contains no Cascade source")
    text = repinned_config(args.config.read_text(), args.image, fingerprint)
    write_text_atomic(args.config, text)
    print(f"Aligned all four worker pins in {args.config} ({count} source files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
