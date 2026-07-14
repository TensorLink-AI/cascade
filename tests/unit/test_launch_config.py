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
    # netuid is now the real mainnet value (91, decided 2026-07-14) so it must
    # NOT be flagged; trainer_hotkey remains an operator secret placeholder and
    # base_arch_digest is pinned — only the former should trip the check.
    with pytest.raises(LaunchConfigError) as ei:
        assert_launch_ready(cfg, role="trainer")
    msg = str(ei.value)
    assert "trainer_hotkey" in msg
    assert "netuid" not in msg
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


def test_validator_requires_window_pool_ref(cfg):
    ready = _launch_ready(cfg)
    no_pool = replace(ready, eval=replace(ready.eval, window_pool=""))
    with pytest.raises(LaunchConfigError) as ei:
        assert_launch_ready(no_pool, role="validator")
    assert "window_pool" in str(ei.value)
    # trainer doesn't need the pool, so it still passes
    assert_launch_ready(no_pool, role="trainer")
