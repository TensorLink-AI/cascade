"""Size-conditional GPU provisioning + per-size budget overrides (DEC-CA-0023
Stage 6 plumbing — inert for every deployed config)."""

from __future__ import annotations

from dataclasses import replace

from cascade.provision.main import build_policy
from cascade.provision.policy import (
    ProvisionPolicy,
    StagePolicy,
    select_final_stage,
)
from cascade.shared.config import SizeSpec, TrainingContractConfig
from cascade.shared.manifest import contract_digest

H100 = "NVIDIA H100 80GB HBM3"


def _contract(**kw) -> TrainingContractConfig:
    base = dict(
        base_arch="toto2", arch_preset="toto2-4m", base_arch_digest="a" * 64,
        d_model=256, num_layers=4, num_heads=4, head_dim=64, patch_size=32,
        mlp_expansion=2, num_quantiles=9, masking="contiguous_patch",
        cpm_c_max=16, cpm_p_max=0.4, input_transform="arcsinh_causal",
        context_length=4096, horizon=64, target_train_hours=3.0,
        ref_throughput_tokens_per_s=3_700_000, warmup_fraction=0.05,
        batch_size=64, optimizer="normuon_adamw", base_lr=4e-3,
        weight_decay=0.05, lr_schedule="warmup_cosine", umup_base_d_model=256,
        train_seed_salt=1337, max_train_seconds=10800,
        expected_gpu="NVIDIA L40S",
    )
    base.update(kw)
    return TrainingContractConfig(**base)


def _size(**kw) -> SizeSpec:
    base = dict(
        arch_preset="toto2-313m", base_arch_digest="b" * 64, d_model=1024,
        num_layers=16, num_heads=16, mlp_expansion=2,
        ref_throughput_tokens_per_s=100_000,
    )
    base.update(kw)
    return SizeSpec(**base)


# ── SizeSpec overrides through for_size ──────────────────────────────────────


def test_unset_overrides_inherit_the_base():
    c = _contract(extra_sizes=(_size(),))
    per = c.contract_for("toto2-313m")
    assert per.expected_gpu == "NVIDIA L40S"
    assert per.target_train_hours == 3.0


def test_size_overrides_swap_gpu_and_hours():
    c = _contract(extra_sizes=(_size(expected_gpu=H100, target_train_hours=6.0),))
    per = c.contract_for("toto2-313m")
    assert per.expected_gpu == H100
    assert per.target_train_hours == 6.0
    # The token budget follows the size's own hours × its own throughput.
    assert per.train_tokens == round(6.0 * 3600 * 100_000)
    # The primary size is untouched.
    assert c.primary_size.expected_gpu == "NVIDIA L40S"
    assert c.primary_size.target_train_hours == 3.0


def test_digest_unmoved_without_sizes_and_moved_with():
    # No [[training.sizes]] blocks (every deployed config): adding the SizeSpec
    # fields changes nothing — the digest has no sizes payload at all.
    assert contract_digest(_contract()) == contract_digest(_contract())
    # A sizes block's silicon/budget are part of that size's identity.
    a = _contract(extra_sizes=(_size(),))
    b = _contract(extra_sizes=(_size(expected_gpu=H100),))
    assert contract_digest(a) != contract_digest(b)


# ── select_final_stage ───────────────────────────────────────────────────────


def _stage(sku: str, **kw) -> StagePolicy:
    base = dict(gpus_per_pod=2, max_pods=2, providers=("shadeform",),
                max_price_hr=5.0)
    base.update(kw)
    return StagePolicy(sku=sku, **base)


def _policy(**kw) -> ProvisionPolicy:
    base = dict(
        heat=_stage("NVIDIA GeForce RTX 4090"),
        final=_stage("NVIDIA L40S"),
        trigger_margin_blocks=45,
        max_spend_per_round=120.0,
    )
    base.update(kw)
    return ProvisionPolicy(**base)


def test_no_alternates_always_base_final():
    p = _policy()
    assert select_final_stage(p, "") is p.final
    assert select_final_stage(p, "NVIDIA L40S") is p.final
    assert select_final_stage(p, H100) is p.final   # unserved pin → base (caller warns)


def test_matching_alternate_selected_by_exact_device_string():
    large = _stage(H100, max_price_hr=6.0)
    p = _policy(final_alternates=(large,))
    assert select_final_stage(p, H100) is large
    # Byte-exact match only — the PCIe variant is a different pin.
    assert select_final_stage(p, "NVIDIA H100 PCIe") is p.final
    assert select_final_stage(p, "") is p.final


# ── config parsing: [provisioner.final.large] ────────────────────────────────

_RAW = {
    "provisioner": {
        "trigger_margin_blocks": 45,
        "max_spend_per_round": 120.0,
        "heat": {"sku": "NVIDIA GeForce RTX 4090", "gpus_per_pod": 4,
                 "max_pods": 8, "max_price_hr": 2.6},
        "final": {
            "sku": "NVIDIA L40S", "gpus_per_pod": 2, "max_pods": 2,
            "max_price_hr": 2.6,
            "candidate": [{"sku": "NVIDIA L40S", "gpus_per_pod": 1,
                           "max_price_hr": 1.3}],
            "large": {
                "sku": H100, "market_sku": "H100", "gpus_per_pod": 2,
                "max_pods": 2, "max_price_hr": 6.0,
                "candidate": [{"sku": H100, "gpus_per_pod": 1,
                               "max_price_hr": 3.0}],
            },
        },
    }
}


def test_final_sub_table_parses_as_alternate():
    p = build_policy(_RAW, epoch_blocks=3600)
    assert p.final.sku == "NVIDIA L40S"
    assert len(p.final_alternates) == 1
    large = p.final_alternates[0]
    assert large.sku == H100 and large.max_price_hr == 6.0
    assert len(large.sku_candidates) == 2   # primary + its own ladder
    # The candidate ARRAY under final never parses as an alternate.
    assert all(a.sku != "NVIDIA L40S" for a in p.final_alternates)
    assert select_final_stage(p, H100) is large


def test_config_without_alternates_unchanged():
    raw = {"provisioner": dict(_RAW["provisioner"])}
    raw["provisioner"]["final"] = {
        k: v for k, v in _RAW["provisioner"]["final"].items() if k != "large"
    }
    p = build_policy(raw, epoch_blocks=3600)
    assert p.final_alternates == ()


def test_for_hours_screen_contract_keeps_size_overrides_out():
    # A heat/screen derivation from a per-size contract keeps the size's pin
    # (screen runs never gate on it) but must not explode: sanity only.
    c = _contract(extra_sizes=(_size(expected_gpu=H100, target_train_hours=6.0),))
    heat = c.contract_for("toto2-313m").for_hours(1.0)
    assert heat.target_train_hours == 1.0
    assert heat.expected_gpu == H100


def test_policy_replace_final_keeps_alternates():
    # main._run swaps policy.final for the selected stage; the dataclass copy
    # must carry the alternates through untouched.
    large = _stage(H100)
    p = _policy(final_alternates=(large,))
    swapped = replace(p, final=large)
    assert swapped.final is large and swapped.final_alternates == (large,)
