"""W7b: the lane D report signer is the pinned main validator hotkey,
``[manifest] validator_hotkey`` (plan rev6 §1.5 "the main validator signs the
lane D report"; README "Pinned signer").

A lane D report is never trusted against its self-declared ``signer``:
unlike the receipt audit, an empty pin is a configuration error, not a
fallback. On main these tests fail at the import of
``pinned_report_signer`` (an ``ImportError`` for a path in PATHS).
"""

from __future__ import annotations

import pytest

from .helpers import chain_raw, make_key, signed_report


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7b_pinned")


def _cfg(validator_hotkey: str):
    from cascade.shared.config.loader import load_chain_config

    return load_chain_config(raw=chain_raw(validator_hotkey))


def test_pinned_signer_is_the_manifest_validator_hotkey(scratch):
    from cascade.validator.lane_d_report import pinned_report_signer  # noqa: I001

    key = make_key(scratch / "keys", "pin")
    cfg = _cfg(key.ss58_address)
    assert pinned_report_signer(cfg) == key.ss58_address


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_pin_is_refused_never_the_self_declared_signer(scratch, value):
    from cascade.validator.lane_d_report import pinned_report_signer  # noqa: I001

    with pytest.raises(ValueError):
        pinned_report_signer(_cfg(value))


def test_report_by_the_pinned_validator_verifies(scratch):
    from cascade.validator.lane_d_report import pinned_report_signer, verify_report  # noqa: I001

    built = signed_report(scratch / "chain")
    cfg = _cfg(built["signer"].ss58_address)
    assert verify_report(built["report"], pinned_report_signer(cfg)) is True


def test_report_validly_signed_by_another_key_is_refused(scratch):
    """The other key's signature is genuine (the sr25519 layer accepts it),
    but it is not the pinned validator's."""
    from cascade.validator.lane_d_report import (  # noqa: I001
        canonical_body, pinned_report_signer, sign_report, verify_report)
    from cascade.shared.sr25519 import verify_hotkey

    built = signed_report(scratch / "chain")
    cfg = _cfg(built["signer"].ss58_address)
    other = built["other"]
    by_other = sign_report(built["report"], other)
    assert verify_hotkey(other.ss58_address, canonical_body(by_other), by_other["signature"])
    assert verify_report(by_other, pinned_report_signer(cfg)) is False
