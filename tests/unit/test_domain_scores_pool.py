"""DEC-CA-0050 P2: the v2 primary pool — every series domain-labeled, and a
DEC-CA-0032 capacity floor per domain enforced at build time."""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pytest

from cascade.pool.builder import (
    DomainFloorError,
    build_pool,
    check_domain_floor,
    prepare_series,
)
from cascade.pool.cli import _add_build_args

from .test_pool_builder import CFG, CTX, _ListSource, _series


def _items(counts: dict[str, int]):
    out, k = [], 0
    for dom, n in counts.items():
        for _ in range(n):
            out.append(_series(f"{dom}{k}", domain=dom,
                               base=10 + k + np.sin(np.arange(600) / (5.0 + k))))
            k += 1
    return out


def test_require_domain_drops_unlabeled():
    cfg = dataclasses.replace(CFG, require_domain=True)
    for label in ("unknown", "", "None"):
        rec, reason = prepare_series(_series("u", domain=label), cfg)
        assert rec is None and reason == "missing_domain"
    rec, reason = prepare_series(_series("e", domain="energy"), cfg)
    assert rec is not None and reason is None
    # off by default: legacy pools keep unlabeled series
    assert prepare_series(_series("u", domain="unknown"), CFG)[0] is not None


def test_domain_floor_refuses_thin_domains(tmp_path):
    cfg = dataclasses.replace(CFG, require_domain=True, min_series_per_domain=3)
    with pytest.raises(DomainFloorError, match="web=2"):
        build_pool([_ListSource(_items({"energy": 3, "web": 2}))], tmp_path / "a",
                   CTX, cfg, fetch=None)
    assert not (tmp_path / "a").exists()
    s = build_pool([_ListSource(_items({"energy": 3, "web": 3, "unknown": 1}))],
                   tmp_path / "b", CTX, cfg, fetch=None)
    assert s.per_domain == {"energy": 3, "web": 3}
    assert s.dropped.get("missing_domain") == 1


def test_floor_ignores_unlabeled_and_can_be_off():
    check_domain_floor({"unknown": 1, "energy": 5}, 5)
    check_domain_floor({"energy": 1}, None)
    with pytest.raises(DomainFloorError):
        check_domain_floor({"energy": 4}, 5)


def test_cli_flags_parse():
    p = argparse.ArgumentParser()
    _add_build_args(p)
    args = p.parse_args(
        ["--sources", "tsbench_forge", "--require-domain",
         "--require-source", "--min-series-per-domain", "300"])
    assert args.require_domain and args.require_source
    assert args.min_series_per_domain == 300
