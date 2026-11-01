"""DEC-CA-0067 automatic image repin at the switch: the ``[v2] *_after`` pins.

* Loader: all-or-none, one image digest, formats, the measured-wall companion;
  round-trip; shipped empty; bound onto the v2 contract as a schedule.
* Before the switch (vote unresolved, or no after pins): every pin, the
  effective contract and the dispatcher are byte-identical.
* From the switch: legs of eras from it get the after image / pod image /
  fingerprint / wall image; earlier eras keep the current ones; the numerics
  digest is unchanged and the pre-switch pin proves an old-era leg.
* The dispatch hook refuses a host on the other image and pushes the
  rendered post-switch chain.toml.
"""
from __future__ import annotations

import re
import threading
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import load_chain_config
from cascade.shared.config.chain_config import bind_image_after
from cascade.shared.config.v2 import parse_image_after
from cascade.shared.era import round_pins_at
from cascade.shared.manifest import TrainedEntry, contract_digest, numerics_contract_digest

REPO = Path(__file__).resolve().parents[2]
B0 = 9046800 + 3600                     # an era start on the 900 grid (era = 3600)
SW = B0 + 7200                          # the switch era start
RC16 = "sha256:34375845b2c4412ccd8462a6e9ca5ad78837567115bc7f9b56320458ba991409"
RC16_REF = "ghcr.io/tensorlink-ai/cascade-worker:worker-v2.0.0-rc16@" + RC16
FP16 = "5fe6ed7c0f0d5305b6101e589aab50b2ca5a76573cbc7afb3f44bc7354ce8fd0"
WALL16 = "sha256:" + "c" * 64
AFTER = {"train_image_digest_after": RC16, "funded_pod_image_after": RC16_REF,
         "worker_code_fingerprint_after": FP16}


# ── the knobs ────────────────────────────────────────────────────────────────


def test_shipped_empty_and_unbound():
    for path in ("chain.toml", "chain.testnet.toml"):
        cfg = load_chain_config(REPO / path)
        assert (cfg.v2.train_image_digest_after, cfg.v2.funded_pod_image_after,
                cfg.v2.worker_code_fingerprint_after,
                cfg.v2.funded_sku_wall_measured_image_after) == ("", "", "", "")
        c = cfg.training.v2_contract
        if c is not None:
            assert (c.train_image_digest_after, c.train_image_after_block) == ("", 0)
        assert round_pins_at(cfg, 10**9) is cfg.round


def test_parse_is_all_or_none_and_one_digest():
    assert parse_image_after({})["train_image_digest_after"] == ""
    ok = parse_image_after({**AFTER, "funded_sku_wall_measured_image_after": WALL16})
    assert ok["worker_code_fingerprint_after"] == FP16
    for bad in ({"train_image_digest_after": RC16},
                {**AFTER, "funded_pod_image_after": RC16},                  # not a ref
                {**AFTER, "funded_pod_image_after": "x@sha256:" + "d" * 64},  # other digest
                {**AFTER, "train_image_digest_after": "rc16"},
                {**AFTER, "worker_code_fingerprint_after": "abc"},
                {**AFTER, "funded_sku_wall_measured_image_after": "wall"},
                {"funded_sku_wall_measured_image_after": WALL16},
                {**AFTER, "train_image_digest_after": 5}):
        with pytest.raises(ValueError, match=r"\[v2\]"):
            parse_image_after(bad)


def _with(tmp_path, **kv):
    text = (REPO / "chain.toml").read_text()
    for k, v in kv.items():
        text, n = re.subn(rf"(?m)^{k}\s*=.*$", f'{k} = "{v}"', text)
        assert n == 1, k
    p = tmp_path / "chain.toml"
    p.write_text(text)
    return p


def test_round_trips_and_binds_the_schedule(tmp_path):
    cfg = load_chain_config(_with(tmp_path, **AFTER))
    assert cfg.v2.funded_pod_image_after == RC16_REF
    c = cfg.training.v2_contract
    assert c.train_image_digest_after == RC16
    assert c.train_image_after_block == 0                  # vote unresolved: no schedule


def test_measured_wall_gate_needs_its_after_image(tmp_path):
    text = _with(tmp_path, **AFTER).read_text()
    text = re.sub(r"(?m)^funded_pod_image = ", f'funded_sku_wall_measured_image = "{WALL16}"\n'
                  "funded_pod_image = ", text, count=1)
    p = tmp_path / "gated.toml"
    p.write_text(text)
    with pytest.raises(ValueError, match="funded_sku_wall_measured_image_after"):
        load_chain_config(p)


# ── the switch ───────────────────────────────────────────────────────────────


@pytest.fixture
def acfg():
    """The shipped config with the rollover typed, v2 live from B0, the after
    pins armed (the vote still unresolved)."""
    c = load_chain_config(REPO / "chain.toml")
    c = replace(c, round=replace(c.round, rolling_from_block=B0, epoch_blocks=900,
                                 epoch_blocks_prev=3600, epoch_activation_block=B0,
                                 funded_sku_wall_measured_image=""),
                scoring=replace(c.scoring, era_king_from_block=B0, tenure_blocks_from_block=B0,
                                cohort_maxt_increment_from_block=B0),
                v2=replace(c.v2, from_block=B0, launch_at_utc="", **AFTER),
                training=replace(c.training, v2_from_block=B0))
    return bind_image_after(c)


def test_unresolved_vote_is_byte_identical(acfg):
    base = acfg.training.v2_contract
    for era in (B0, SW, SW + 3600):
        eff = acfg.training.at_block(era)
        assert eff.train_image_digest == base.train_image_digest
        assert contract_digest(eff) == contract_digest(replace(base, train_image_digest_after=""))
        assert round_pins_at(acfg, era) is acfg.round


def test_legs_either_side_of_the_switch_get_their_era_pins(acfg):
    armed = A.apply_image_decoupled_activation(acfg, SW)
    v2c = armed.training.v2_contract
    assert (v2c.train_image_digest_after, v2c.train_image_after_block) == (RC16, SW)
    before, after = armed.training.at_block(SW - 3600), armed.training.at_block(SW)
    assert before.train_image_digest == acfg.training.v2_contract.train_image_digest
    assert after.train_image_digest == RC16
    assert armed.throne_contracts_at(SW + 3600)[0].train_image_digest == RC16
    assert contract_digest(after) != contract_digest(before)
    assert numerics_contract_digest(after) == numerics_contract_digest(before)
    assert round_pins_at(armed, SW - 3600) is armed.round
    post = round_pins_at(armed, SW)
    assert (post.funded_pod_image, post.worker_code_fingerprint) == (RC16_REF, FP16)


def test_leg_round_follows_the_thread_local_contract(acfg):
    from cascade.trainer.leg_pins import leg_contract_scope, leg_round

    armed = A.apply_image_decoupled_activation(acfg, SW)
    runner = SimpleNamespace(cfg=armed)
    assert leg_round(runner) is armed.round                      # no leg on this thread
    with leg_contract_scope(runner, armed.training.at_block(SW)):
        assert leg_round(runner).funded_pod_image == RC16_REF
        seen = []
        t = threading.Thread(target=lambda: seen.append(leg_round(runner)))
        t.start()
        t.join()
        assert seen[0] is armed.round                            # thread-local
    with leg_contract_scope(runner, armed.training.at_block(SW - 3600)):
        assert leg_round(runner) is armed.round
    assert leg_round(runner) is armed.round


def test_rendered_chain_toml_pins_the_after_release(acfg):
    from cascade.trainer.leg_pins import render_after_toml

    armed = A.apply_image_decoupled_activation(acfg, SW)
    text = (REPO / "chain.toml").read_text()
    out = render_after_toml(text, armed)
    raw = tomllib.loads(out)
    assert raw["v2"]["training"]["train_image_digest"] == RC16
    assert raw["round"]["funded_pod_image"] == RC16_REF
    assert raw["round"]["worker_code_fingerprint"] == FP16
    assert raw["v2"]["image_decoupled_from_block"] == SW
    # everything else is untouched
    assert tomllib.loads(text)["v2"]["training"]["base_lr"] == raw["v2"]["training"]["base_lr"]
    assert out.count("\n") == text.count("\n")


def test_dispatch_hook_refuses_the_other_image_and_pushes_the_after_config(acfg, tmp_path):
    from cascade.trainer.leg_pins import AFTER_TOML_NAME, LegImageMismatch, dispatch_host_hook

    armed = A.apply_image_decoupled_activation(acfg, SW)
    src = tmp_path / "chain.toml"
    src.write_text((REPO / "chain.toml").read_text())
    pushed = []

    def push(host, *, src=None, dest_name="chain.deployed.toml"):
        pushed.append((Path(src).read_text(), dest_name))
        return host

    runner = SimpleNamespace(cfg=armed, deployed_chain_toml=src, work_root=tmp_path,
                             _push_deployed_chain_toml=push)
    from cascade.trainer.remote import RemoteHost

    host = RemoteHost(name="lane", host="10.0.0.1", workdir="/w",
                      chain_toml="/w/chain.deployed.toml")
    on = {"fp": FP16}
    hook = dispatch_host_hook(runner, fingerprint_fn=lambda h: on["fp"])
    out = hook(host, SW)                                        # post-switch leg, rc16 host
    assert out.chain_toml == f"/w/{AFTER_TOML_NAME}"
    assert pushed and pushed[0][1] == AFTER_TOML_NAME and RC16 in pushed[0][0]
    with pytest.raises(LegImageMismatch):
        hook(host, SW - 3600)                                   # pre-switch leg on rc16
    on["fp"] = armed.round.worker_code_fingerprint
    assert hook(host, SW - 3600) is host                         # current image: unchanged
    with pytest.raises(LegImageMismatch):
        hook(host, SW)                                          # post-switch leg on old image
    assert hook(host, None) is host


def test_dispatcher_hook_absent_until_armed(cfg):
    from cascade.trainer.runner_rolling import _RollingMixin

    r = SimpleNamespace(cfg=cfg)
    assert _RollingMixin._leg_pins_hook(r) == {}


# ── the trainer keeps an old-era leg ─────────────────────────────────────────


def test_pre_switch_king_is_proven_without_a_manual_prior(cfg, tmp_path):
    from cascade.trainer.rolling import EraState
    from tests.unit.test_rolling import Clock, _ptr, _sched
    from tests.unit.test_v2_golive_validator import _v2

    v2cfg, n = _v2(cfg)
    v2cfg = replace(v2cfg, v2=replace(v2cfg.v2, **AFTER, image_decoupled_from_block=n))
    v2cfg = bind_image_after(v2cfg)
    sched, _ = _sched(v2cfg, tmp_path, Clock())
    era = EraState(10, n, n, 1)
    base = v2cfg.training.v2_contract
    king = TrainedEntry("king", 1, "king", "fixture/gen", _ptr("king"), "d", n,
                        init_pointer="",
                        training_contract_digest=contract_digest(
                            replace(base, train_image_digest_after="", train_image_after_block=0)))
    assert v2cfg.training.at_block(n).train_image_digest == RC16
    assert not sched._stale_contract(era)(king.training_contract_digest)
    assert sched._numerics_of(era, king.training_contract_digest) == numerics_contract_digest(
        v2cfg.training.at_block(n))
