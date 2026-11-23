"""WB22: the shipped lane B GPU list (owner 2026-11-23) and lane A's list
left alone."""

from __future__ import annotations

from .helpers import (
    CHAIN_TOML,
    MAIN_ROUND_SKUS,
    OWNER_LANE_B_SKUS,
    TESTNET_TOML,
    _raw,
    chain_cfg,
    norm,
    raw_toml,
)


def test_shipped_lane_b_list_is_the_owner_list():
    lb = raw_toml().get("lane_b") or {}
    assert norm(lb.get("funded_pod_skus")) == OWNER_LANE_B_SKUS, (
        f"chain.toml [lane_b] funded_pod_skus = {lb.get('funded_pod_skus')!r}, "
        f"owner list {OWNER_LANE_B_SKUS!r}")
    assert norm(chain_cfg().lane_b.funded_pod_skus) == OWNER_LANE_B_SKUS


def test_no_single_gpu_layout_and_8x4090_is_a_layout():
    from cascade.provision.layout import split_layout

    skus = norm(chain_cfg().lane_b.funded_pod_skus)
    assert "8xRTX4090" in skus, "owner 2026-11-23: lane B tests 8x RTX 4090"
    singles = [s for s in skus if split_layout(s)[1] < 2]
    assert not singles, f"single-GPU layouts in lane B's list: {singles} (owner: too slow)"
    assert split_layout("8xRTX4090") == ("RTX4090", 8)


def test_lane_a_round_list_unchanged():
    assert norm(raw_toml()["round"]["funded_pod_skus"]) == MAIN_ROUND_SKUS
    cfg = chain_cfg()
    assert norm(cfg.round.funded_pod_skus) == MAIN_ROUND_SKUS
    assert "8xRTX4090" not in norm(cfg.round.funded_pod_skus)


def test_testnet_lane_b_list_stays_its_round_list():
    t = raw_toml(TESTNET_TOML)
    assert norm(t["lane_b"]["funded_pod_skus"]) == norm(t["round"]["funded_pod_skus"])


def test_shipped_comment_names_the_owner_decision():
    lines = _raw(CHAIN_TOML).splitlines()
    i = next((n for n, ln in enumerate(lines)
              if ln.strip().startswith("funded_pod_skus")
              and "8xRTX4090" in ln), None)
    assert i is not None, "no [lane_b] funded_pod_skus line with 8xRTX4090 in chain.toml"
    above = "\n".join(lines[max(0, i - 12):i])
    assert "owner 2026-11-23" in above and "8xRTX4090" in above, (
        "the comment above [lane_b] funded_pod_skus must name the owner decision "
        "(owner 2026-11-23) and 8xRTX4090")


def test_shipped_list_moves_no_lane_a_contract_digest():
    from cascade.shared.config import parse_chain_config
    from cascade.shared.manifest import contract_digest

    raw = raw_toml()
    dropped = dict(raw)
    dropped["lane_b"] = {k: v for k, v in raw["lane_b"].items() if k != "funded_pod_skus"}
    a, b = parse_chain_config(raw), parse_chain_config(dropped)
    for block in (0, int(a.training.v2_from_block)):
        assert ([contract_digest(c) for c in a.throne_contracts_at(block)]
                == [contract_digest(c) for c in b.throne_contracts_at(block)])
    assert a.round == b.round and a.training == b.training
