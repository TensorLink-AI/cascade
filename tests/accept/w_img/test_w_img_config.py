"""W-IMG config: the typed activation key (release-then-activate) and the
trainer's bound epoch block load, round-trip and refuse bad values; the
shipped files leave both off (plan rev6.24 (e))."""

from __future__ import annotations

import pytest

from .helpers import REPO, with_keys

# era starts on the shipped grid (3600-block eras), past the typed v2b switch
DECOUPLED = 9493200


def test_shipped_configs_leave_w_img_off():
    from cascade.shared.config import load_chain_config

    for name in ("chain.toml", "chain.testnet.toml"):
        cfg = load_chain_config(REPO / name)
        assert cfg.v2.image_epoch_from_block == 0, name
        assert cfg.v2.image_after_epoch_block == 0, name


def test_activation_key_round_trips(tmp_path):
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(with_keys(tmp_path, image_epoch_from_block=DECOUPLED + 3600))
    assert cfg.v2.image_epoch_from_block == DECOUPLED + 3600


@pytest.mark.parametrize("value,match", [
    (DECOUPLED + 1800, "not an era start"),
    (DECOUPLED - 3600, "precedes"),
    (-1, ">= 0"),
    ("true", "integer"),
])
def test_activation_key_refuses_bad_values(tmp_path, value, match):
    from cascade.shared.config import load_chain_config

    with pytest.raises(ValueError, match=match):
        load_chain_config(with_keys(tmp_path, image_epoch_from_block=value))


def test_bound_block_needs_the_rule_and_an_after_release(tmp_path):
    from cascade.shared.config import load_chain_config

    with pytest.raises(ValueError, match="image_after_epoch_block"):
        load_chain_config(with_keys(tmp_path, image_after_epoch_block=DECOUPLED + 7200))
    with pytest.raises(ValueError, match="image_after_epoch_block needs the"):
        load_chain_config(with_keys(tmp_path, image_epoch_from_block=DECOUPLED + 3600,
                                    image_after_epoch_block=DECOUPLED + 7200))


def test_activation_key_is_release_only_not_overlay_settable():
    """A typed activation (every validator deploys it) — never a signed
    overlay value, which would let one publish move a consensus rule."""
    from cascade.shared import netconfig
    from cascade.shared.config.v2 import V2Config

    assert {"image_epoch_from_block", "image_after_epoch_block"} <= set(
        V2Config.__dataclass_fields__)
    allowed = netconfig.ALLOWED.get("v2", ())
    assert "image_epoch_from_block" not in allowed
    assert "image_after_epoch_block" not in allowed
