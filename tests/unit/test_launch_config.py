"""Launch-readiness guards + computable base_arch_digest."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cascade.shared.config import LaunchConfigError, assert_launch_ready
from cascade.trainer.contract import compute_base_arch_digest

REF = "cascade/eval-pool@sha256:" + "a" * 64


def test_arch_digest_deterministic_and_arch_sensitive(cfg):
    d1 = compute_base_arch_digest(cfg.training)
    d2 = compute_base_arch_digest(cfg.training)
    assert d1 == d2 and len(d1) == 64
    # changing an architecture field changes the digest
    other = replace(cfg.training, d_model=cfg.training.d_model + 8)
    assert compute_base_arch_digest(other) != d1


def _launch_ready(cfg):
    digest = compute_base_arch_digest(cfg.training)
    training = replace(cfg.training, base_arch_digest=digest)
    subnet = replace(cfg.subnet, netuid=42)
    manifest = replace(cfg.manifest, trainer_hotkey="5Fhotkeyaddress")
    eval_ = replace(cfg.eval, window_pool=REF)
    return replace(cfg, subnet=subnet, training=training, manifest=manifest, eval=eval_)


def test_assert_launch_ready_flags_default_placeholders(cfg):
    # The shipped template is now fully launch-valued (netuid 91, trainer
    # hotkey set, digests pinned) — flagging is tested by explicitly blanking,
    # and the shipped file itself must PASS for the trainer role.
    assert_launch_ready(cfg, role="trainer")
    blanked = replace(cfg, subnet=replace(cfg.subnet, netuid=0),
                      manifest=replace(cfg.manifest, trainer_hotkey=""))
    with pytest.raises(LaunchConfigError) as ei:
        assert_launch_ready(blanked, role="trainer")
    msg = str(ei.value)
    assert "netuid" in msg and "trainer_hotkey" in msg
    assert "base_arch_digest" not in msg


def test_assert_launch_ready_flags_zero_digest(cfg):
    zeroed = replace(cfg, training=replace(cfg.training, base_arch_digest="0" * 64))
    with pytest.raises(LaunchConfigError) as ei:
        assert_launch_ready(zeroed, role="trainer")
    assert "base_arch_digest" in str(ei.value)


def test_assert_launch_ready_passes_when_set(cfg):
    ready = _launch_ready(cfg)
    assert_launch_ready(ready, role="trainer")        # no raise
    assert_launch_ready(ready, role="validator")      # window_pool ref set too


def test_validator_requires_some_eval_pool(cfg):
    """The validator needs A pool source: the daily bucket (recommended) OR a
    static window_pool ref — only both blank is unlaunchable."""
    ready = _launch_ready(cfg)
    bucket_only = replace(ready, eval=replace(ready.eval, window_pool=""))
    assert_launch_ready(bucket_only, role="validator")   # template bucket suffices
    no_pool = replace(bucket_only,
                      storage=replace(bucket_only.storage, pool_bucket=""))
    with pytest.raises(LaunchConfigError) as ei:
        assert_launch_ready(no_pool, role="validator")
    assert "pool" in str(ei.value)
    # trainer doesn't need the pool, so it still passes
    assert_launch_ready(no_pool, role="trainer")


# ── funded_sku_wall_seconds vs the epoch: loud, never a silent narrowing ─────


def _rent_cfg(cfg, *, epoch_blocks, skus, walls):
    ready = _launch_ready(cfg)
    training = replace(ready.training, expected_gpu="")
    rnd = replace(ready.round, funded_pods="rent", funded_pod_skus=tuple(skus),
                  funded_pod_sku=skus[0], funded_sku_wall_seconds=tuple(sorted(walls.items())),
                  epoch_blocks=epoch_blocks, epoch_blocks_prev=0, epoch_activation_block=0)
    return replace(ready, training=training, round=rnd)


def test_wall_table_excluding_every_sku_refuses_launch(cfg):
    from cascade.shared.config import funded_sku_wall_fit

    # a 30-min testnet grid inheriting mainnet's measured walls
    c = _rent_cfg(cfg, epoch_blocks=150, skus=("RTX4090", "L40S"),
                  walls={"RTX4090": 13500, "L40S": 10200})
    assert funded_sku_wall_fit(c) == ((), ("RTX4090", "L40S"))
    with pytest.raises(LaunchConfigError, match="excludes EVERY configured SKU"):
        assert_launch_ready(c, role="trainer")
    # the validator role never rents — no problem raised
    assert_launch_ready(c, role="validator")


def test_wall_table_excluding_every_sku_only_warns_under_rolling(cfg, caplog):
    import logging

    # Rolling intake armed: a wall longer than one grid step settles a
    # boundary later, it does not "never start" — launch proceeds, loudly.
    c = _rent_cfg(cfg, epoch_blocks=150, skus=("RTX4090", "L40S"),
                  walls={"RTX4090": 13500, "L40S": 10200})
    c = replace(c, round=replace(c.round, rolling_from_block=600))
    with caplog.at_level(logging.WARNING, logger="cascade.config"):
        assert_launch_ready(c, role="trainer")
    assert any("rolling intake is armed" in r.getMessage() and "RTX4090, L40S" in r.getMessage()
               for r in caplog.records)


def test_wall_table_excluding_some_skus_warns(cfg, caplog):
    import logging

    from cascade.shared.config import funded_sku_wall_fit

    # RTX3090 has no entry ⇒ the contract's max_train_seconds (fits a 12 h
    # epoch); RTX4090's measured wall is longer than the epoch ⇒ excluded.
    c = _rent_cfg(cfg, epoch_blocks=3600, skus=("RTX4090", "RTX3090"),
                  walls={"RTX4090": 50000})
    assert funded_sku_wall_fit(c) == (("RTX3090",), ("RTX4090",))
    with caplog.at_level(logging.WARNING, logger="cascade.config"):
        assert_launch_ready(c, role="trainer")
    assert any("excludes RTX4090" in r.getMessage() and "only RTX3090" in r.getMessage()
               for r in caplog.records)


def test_wall_table_that_fits_is_silent(cfg, caplog):
    import logging

    from cascade.shared.config import funded_sku_wall_fit

    c = _rent_cfg(cfg, epoch_blocks=3600, skus=("RTX4090", "L40S"),
                  walls={"RTX4090": 13500, "L40S": 10200})
    assert funded_sku_wall_fit(c) == (("RTX4090", "L40S"), ())
    with caplog.at_level(logging.WARNING, logger="cascade.config"):
        assert_launch_ready(c, role="trainer")
    assert not [r for r in caplog.records if "funded_sku_wall_seconds" in r.getMessage()]
