"""Shared spec and helpers for the W-IMG acceptance tests (plan rev6.24 (e)).

Owner decision, 2026-11-24 (verbatim): "can we relax need for image to be
pinned"; "we need a system so that we roll all these lanes out, validators
only need to update for major upgrades, not if we tweak training image or
comp rules"; "we just log what image pin was used, it's auditable".

The scenarios run the REAL validator gate (``ValidatorRunner.check_manifest``
over real ``TrainingManifest`` objects, the fixtures of
``tests/unit/test_v2_golive_validator.py`` / ``test_era_validator.py``), the
real audit Tier-0 checks, the real loader and the real trainer binding. No
torch anywhere (no checkpoint is loaded, no model is built).

Nothing in W-IMG's PATHS is imported at module level, so a missing name is
reported inside the test that needs it.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
IMG_X = "sha256:" + "a" * 64
IMG_Y = "sha256:" + "b" * 64
FP_Y = "c" * 64


def require_w_img():
    """Fail first (on a build without W-IMG) on the missing W-IMG names
    themselves, never on a later stand-in mismatch."""
    from cascade.shared.era import image_epoch_from, image_record_at  # noqa: F401
    from cascade.trainer.cost_guards import unmeasured_wall_cap  # noqa: F401


def base_cfg(tmp_path):
    """The mainnet template with its enforcing pins neutralised — the same
    neutralisation as ``tests/conftest.py``'s ``cfg`` fixture (repeated here
    so the tests also run under ``--noconftest``)."""
    from cascade.shared.config import load_chain_config
    from tests.unit.predecision_main import predecision_path

    require_w_img()
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    c = load_chain_config(predecision_path(tmp_path))
    return replace(
        c, training=replace(c.training, expected_gpu="", train_image_digest=""),
        netconfig=replace(c.netconfig, owner_hotkey=""),
        v2=replace(c.v2, launch_at_utc="", commit_open_at_utc="", commit_from_block=0,
                   prelaunch_burn_readiness="", image_decoupled_from_block=0),
        generator=replace(c.generator, corpus_target_points=0, min_channels=1,
                          sandbox_strict=False),
        round=replace(c.round, commit_floor_block=0, dedup_mode="off", epoch_blocks_prev=0,
                      epoch_activation_block=0, funded_mode="off", funded_pods="off",
                      funded_king_rent=False),
        scoring=replace(c.scoring, win_margin_start_prev=0.0, margin_activation_block=0,
                        forfeit_from_block=0, margin_v2_from_block=0),
        manifest=replace(c.manifest, validator_hotkey=""))


def v2_setup(tmp_path, *, rule: bool, decoupled: bool = True):
    """A validator on a v2 config whose first v2 era starts at ``n``, the
    DEC-CA-0067 decoupled check in force from ``n`` and (``rule``) the W-IMG
    one-image-per-seed-epoch rule from ``n``. Returns ``(runner, manifest of
    the first v2 settlement, v2 contract, cfg)``."""
    from tests.unit.test_era_validator import _runner
    from tests.unit.test_v2_golive_validator import _moe_manifest, _v2

    v2cfg, n = _v2(base_cfg(tmp_path))
    v2 = v2cfg.v2
    if decoupled:
        v2 = replace(v2, image_decoupled_from_block=n)
    if rule:
        v2 = replace(v2, image_epoch_from_block=n)
    # mainnet gates the DECLARED contract (DEC-CA-0036, armed 8942400): the
    # locked terms only — the worker image is never a locked term
    v2cfg = replace(v2cfg, v2=v2, scoring=replace(v2cfg.scoring, declared_contract_from_block=1))
    block = n + v2cfg.round.epoch_blocks
    m = _moe_manifest(v2cfg, block)
    return _runner(v2cfg), m, v2cfg.training.v2_contract, v2cfg


def stamp(m, contract, *, images=None, body_image=None):
    """Stamp a manifest like the trainer does post-switch: numerics on the
    manifest and every entry, each entry's OWN image (``images`` per entry
    index, default the era image) and its full digest re-derived from it.
    ``body_image`` publishes a declared ``contract_body`` with that image."""
    from cascade.shared.manifest import (
        contract_digest,
        contract_payload,
        entry_contract_payload,
        numerics_contract_digest,
    )

    era_contract = contract if body_image is None else replace(
        contract, train_image_digest=body_image)
    era_image = str(era_contract.train_image_digest)
    num = numerics_contract_digest(era_contract)
    proto = str(era_contract.continuation_protocol or "")
    entries = []
    for i, e in enumerate(m.entries):
        img = (images or {}).get(i, era_image)
        digest = contract_digest(entry_contract_payload(era_contract, img, proto))
        entries.append(replace(e, numerics_contract_digest=num, train_image_digest=img,
                               continuation_protocol=proto, training_contract_digest=digest))
    out = replace(m, entries=entries, numerics_contract_digest=num)
    if body_image is not None:
        out = replace(out, contract_body=contract_payload(era_contract),
                      contract_digest=contract_digest(era_contract))
    return out


def receipt(m, block):
    """A receipt stand-in carrying the signed manifest body (what the audit reads)."""
    return SimpleNamespace(manifest=json.loads(m.canonical_body()), epoch_start_block=block,
                           image_decoupled_block=0, status="scored",
                           round_id=str(m.round_id))


def with_keys(tmp_path, **kv):
    """The shipped chain.toml with ``[v2]`` keys set (or appended)."""
    require_w_img()
    text = (REPO / "chain.toml").read_text()
    extra = []
    for k, v in kv.items():
        text, n = re.subn(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", text)
        if not n:
            extra.append(f"{k} = {v}")
    if extra:
        text = re.sub(r"(?m)^king_ladder = true$", "king_ladder = true\n" + "\n".join(extra),
                      text, count=1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    return p
