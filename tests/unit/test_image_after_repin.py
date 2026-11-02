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


@pytest.fixture(autouse=True)
def _qualified(monkeypatch):
    """These tests exercise the repin mechanics with a QUALIFIED after release
    (the owner's qualification rule itself is tested in
    test_image_qualification.py)."""
    monkeypatch.setattr("cascade.trainer.leg_pins.after_image_refusal", lambda cfg: "")


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
    on = {"fp": FP16, "img": RC16}
    hook = dispatch_host_hook(runner, fingerprint_fn=lambda h, fresh=False: on["fp"],
                              image_fn=lambda h, fresh=False: on["img"])
    out = hook(host, SW)                                        # post-switch leg, rc16 host
    assert out.chain_toml == f"/w/{AFTER_TOML_NAME}"
    assert pushed and pushed[0][1] == AFTER_TOML_NAME and RC16 in pushed[0][0]
    # review 3: a pre-switch leg is handed through untouched (the worker's own
    # FINAL image check still guards it)
    assert hook(host, SW - 3600) is host
    on["img"] = armed.training.at_block(SW - 3600).train_image_digest   # base pin injected
    with pytest.raises(LegImageMismatch, match="runtime image"):
        hook(host, SW)                       # after code, but the worker would refuse FINAL
    on["fp"] = armed.round.worker_code_fingerprint
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


# ── review fixes: lane selection, cache, provisioner, audit replay ───────────


class _LaneHost:
    def __init__(self, name, host):
        self.name, self.host, self.port, self.stage = name, host, 22, "final"
        self.workdir, self.pinned_host_key, self.allocation_id = "/w", "", ""


class _Hook:
    def __init__(self, image_of):
        self.image_of = image_of

    def filters(self, block):
        return block is not None and block >= SW

    def match_state(self, host, block):
        state = self.image_of[host.name]
        if state is None:
            return None                                   # unreachable right now
        return state == ("after" if block >= SW else "current")

    def matches(self, host, block):
        return self.match_state(host, block) is True

    def __call__(self, host, block):
        from cascade.trainer.leg_pins import LegImageMismatch

        if not self.matches(host, block):
            raise LegImageMismatch("wrong image")
        return host


class _LaneDisp:
    def __init__(self, hook):
        self.host_hook, self.calls = hook, []

    def dispatch(self, host, **kw):
        host = self.host_hook(host, kw.get("contract_block"))
        self.calls.append(host.name)
        return SimpleNamespace(host=host.name)


def test_lane_pool_picks_the_lane_on_the_legs_era_image():
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    old, new = _LaneHost("old", "10.0.0.1"), _LaneHost("new", "10.0.0.2")
    hosts = [old, new]
    pool = _FinalLanePool(hosts, lambda: hosts)
    d = _LaneDisp(_Hook({"old": "current", "new": "after"}))
    out = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x",
                                               role="challenger", contract_block=SW)
    assert out.host == "new" and d.calls == ["new"]
    out = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x",
                                               role="challenger", contract_block=SW - 3600)
    assert out.host == "old"
    assert sorted(h.name for h in pool.known_hosts()) == ["new", "old"]   # nobody tripped


def test_no_lane_on_the_era_image_refuses_cleanly_without_the_breaker():
    from cascade.trainer.leg_pins import LegImageMismatch
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    hosts = [_LaneHost("old", "10.0.0.1")]
    pool = _FinalLanePool(hosts, lambda: hosts)
    d = _LaneDisp(_Hook({"old": "current"}))
    with pytest.raises(LegImageMismatch, match="no lane runs this leg's era image"):
        TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x",
                                             role="challenger", contract_block=SW)
    assert [h.name for h in pool.known_hosts()] == ["old"] and d.calls == []


def test_split_fleet_two_eras_never_starve_each_other():
    """Review 2: a post-switch leg waiting for the busy after-image lane must
    NOT hold the idle current-image lane — a concurrent pre-switch leg takes
    it at once; the waiter gets its lane when it is handed back."""
    import time as _t

    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    old, new = _LaneHost("old", "10.0.0.1"), _LaneHost("new", "10.0.0.2")
    hosts = [old, new]
    pool = _FinalLanePool(hosts, lambda: hosts)
    first, second = pool.get(), pool.get()
    assert {first.name, second.name} == {"old", "new"}
    pool.put(old)                                    # "new" is busy with another leg
    d = _LaneDisp(_Hook({"old": "current", "new": "after"}))
    got: dict = {}

    def post_leg():
        got["post"] = TrainerRunner._dispatch_on_free_lane(
            d, pool, hosts, describe="post", role="challenger", contract_block=SW)

    t = threading.Thread(target=post_leg, daemon=True)
    t.start()
    _t.sleep(0.3)
    assert "post" not in got                         # waiting for "new"
    pre = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="pre",
                                               role="challenger", contract_block=SW - 3600)
    assert pre.host == "old"                         # not held by the waiting leg
    pool.put(new)                                    # the busy leg hands it back
    t.join(10)
    assert got["post"].host == "new"
    assert sorted(h.name for h in pool.known_hosts()) == ["new", "old"]


def test_filtered_take_respects_the_deadline():
    import time as _t

    from cascade.trainer.faults import _LaneDeadlinePassed
    from cascade.trainer.lane_pool import _FinalLanePool

    old = _LaneHost("old", "10.0.0.1")
    new = _LaneHost("new", "10.0.0.2")
    pool = _FinalLanePool([old, new], lambda: [old, new])
    pool.get(), pool.get()
    pool.put(old)                                    # only the other era's lane is idle
    with pytest.raises(_LaneDeadlinePassed):
        pool.get(deadline=_t.time() - 1, match=lambda h: h.name == "new")
    assert pool.get(block=False).name == "old"       # never taken by the refused leg


def test_fingerprint_cache_is_keyed_per_container_and_expires(monkeypatch):
    from cascade.trainer import leg_pins

    calls = []
    monkeypatch.setattr("cascade.provision.codeprint.remote_code_fingerprint",
                        lambda run, workdir: (calls.append(1) or (state["fp"], 3, "")))
    state = {"fp": "a" * 64}
    leg_pins._FINGERPRINTS.clear()
    h1 = _LaneHost("lane", "10.0.0.9")
    h1.pinned_host_key = "key-1"
    t = [0.0]
    assert leg_pins.host_code_fingerprint(h1, clock=lambda: t[0]) == "a" * 64
    assert leg_pins.host_code_fingerprint(h1, clock=lambda: t[0]) == "a" * 64
    assert len(calls) == 1                                       # cached
    state["fp"] = "b" * 64
    h2 = _LaneHost("lane", "10.0.0.9")
    h2.pinned_host_key = "key-2"                                 # same ip:port, new container
    assert leg_pins.host_code_fingerprint(h2, clock=lambda: t[0]) == "b" * 64
    t[0] = leg_pins.FINGERPRINT_TTL_SECONDS + 1
    assert leg_pins.host_code_fingerprint(h1, clock=lambda: t[0]) == "b" * 64   # expired
    leg_pins._FINGERPRINTS.clear()


def test_hook_remeasures_a_rebootstrapped_lane_before_refusing(acfg, tmp_path):
    from cascade.trainer.leg_pins import dispatch_host_hook

    armed = A.apply_image_decoupled_activation(acfg, SW)
    seen = []

    def fp(host, fresh=False):
        seen.append(fresh)
        return FP16 if fresh else armed.round.worker_code_fingerprint   # stale cache

    src = tmp_path / "chain.toml"
    src.write_text((REPO / "chain.toml").read_text())
    runner = SimpleNamespace(cfg=armed, deployed_chain_toml=src, work_root=tmp_path,
                             _push_deployed_chain_toml=lambda h, **kw: h)
    from cascade.trainer.remote import RemoteHost

    host = RemoteHost(name="lane", host="10.0.0.1", workdir="/w")
    hook = dispatch_host_hook(runner, fingerprint_fn=fp, image_fn=lambda h, fresh=False: RC16)
    assert hook(host, SW).chain_toml.endswith("chain.deployed.after.toml")
    assert seen == [False, True]
    runner.deployed_chain_toml = None                           # no config to render
    from cascade.trainer.leg_pins import LegImageMismatch

    with pytest.raises(LegImageMismatch, match="deployed_chain_toml"):
        hook(host, SW)


def test_provisioner_gate_accepts_the_after_release():
    from cascade.provision.health import HealthGate

    gate = HealthGate(sku="H100", image_digest="sha256:" + "1" * 64, code_fingerprint="1" * 64,
                      alt_image_digest=RC16, alt_code_fingerprint=FP16)

    def run_for(digest, fp):
        def run(argv):
            if argv[0] == "printenv":
                return SimpleNamespace(returncode=0, stdout=digest, stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return run

    assert gate._check_image_digest(run_for(RC16, FP16))[0]
    assert not gate._check_image_digest(run_for("sha256:" + "9" * 64, FP16))[0]
    plain = HealthGate(sku="H100", image_digest="sha256:" + "1" * 64)
    assert not plain._check_image_digest(run_for(RC16, FP16))[0]          # byte-identical


def test_audit_replays_each_leg_under_its_own_runtime_terms():
    from cascade.audit.rederive import entry_replay_contract

    c = load_chain_config(REPO / "chain.toml").training.v2_contract
    e = SimpleNamespace(train_image_digest=RC16, continuation_protocol="cascade-continuation/1")
    assert entry_replay_contract(c, e, SimpleNamespace(numerics_contract_digest="")) is c
    out = entry_replay_contract(c, e, SimpleNamespace(numerics_contract_digest="n" * 64))
    assert (out.train_image_digest, out.continuation_protocol) == (
        RC16, "cascade-continuation/1")
    assert numerics_contract_digest(out) == numerics_contract_digest(c)


def test_tier1_rederives_with_the_legs_own_protocol(tmp_path, monkeypatch):
    from cascade.audit import rederive as R
    from cascade.shared.manifest import TrainingManifest, format_trained_pointer

    cfg = load_chain_config(REPO / "chain.toml")
    cfg = replace(cfg, training=replace(cfg.training, v2_from_block=1))
    v2c = cfg.training.v2_contract
    entry = TrainedEntry("hk", 1, "king", "ns/g@sha256:" + "1" * 64,
                         format_trained_pointer("ns/c@sha256:" + "2" * 64), "digest-x", 5,
                         size=v2c.arch_preset, numerics_contract_digest="n" * 64,
                         train_image_digest=RC16, continuation_protocol="cascade-continuation/1")
    manifest = TrainingManifest(round_id="1", created_block=5, contract_digest="c" * 64,
                                base_arch_digest="a" * 64, eval_dataset="e", entries=[entry],
                                numerics_contract_digest="n" * 64)
    receipt = SimpleNamespace(status="scored", manifest={}, epoch_start_block=SW + 900,
                              generation_seed=7, load_embedded_manifest=lambda: manifest)
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return "digest-x"

    monkeypatch.setattr(R, "_rederive_digest", fake)
    monkeypatch.setattr(R, "_fetch_generator", lambda *a, **k: None)
    root = tmp_path / "corpora"
    (root / "hk").mkdir(parents=True)
    (root / "hk" / "corpus.json").write_text("{}")
    out = R.run_tier1(receipt, cfg, workdir=tmp_path, full_stream=True, corpus_root=root)
    assert out[0].status == "PASS", out
    assert seen.get("continuation_protocol") == "cascade-continuation/1"


# ── review 2: transport vs image, billing ────────────────────────────────────


@pytest.mark.parametrize("mode", ["rc255", "timeout"])
def test_unreachable_pod_during_the_probe_is_a_transport_failure(monkeypatch, mode):
    import subprocess

    from cascade.trainer import leg_pins
    from cascade.trainer.faults import _transport_failure
    from cascade.trainer.remote import RemoteDispatchError

    def fake_ssh(argv, timeout, stdin_text=None):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 255, "", "ssh: connect: Connection timed out")

    monkeypatch.setattr("cascade.trainer.remote.run_ssh", fake_ssh)
    monkeypatch.setattr("cascade.trainer.remote.build_ssh_argv", lambda h, cmd: ["ssh", cmd])
    leg_pins._FINGERPRINTS.clear()
    with pytest.raises(RemoteDispatchError) as ei:
        leg_pins.host_code_fingerprint(_LaneHost("lane", "10.0.0.7"))
    assert ei.value.returncode == 255 and _transport_failure(ei.value)
    assert not isinstance(ei.value, leg_pins.LegImageMismatch)


def test_a_failed_fingerprint_probe_is_unverified_never_wrong(monkeypatch):
    import subprocess

    from cascade.trainer import leg_pins

    monkeypatch.setattr("cascade.trainer.remote.run_ssh",
                        lambda argv, timeout, stdin_text=None:
                        subprocess.CompletedProcess(argv, 1, "", "find: no such dir"))
    monkeypatch.setattr("cascade.trainer.remote.build_ssh_argv", lambda h, cmd: ["ssh", cmd])
    leg_pins._FINGERPRINTS.clear()
    # review 4: remote python / OOM / disk / missing dir = UNKNOWN (no blame)
    with pytest.raises(leg_pins.LegImageUnverified):
        leg_pins.host_code_fingerprint(_LaneHost("lane", "10.0.0.8"))


def test_unreachable_lane_does_not_match_and_the_dispatch_drains_it(acfg, tmp_path):
    from cascade.trainer.leg_pins import dispatch_host_hook
    from cascade.trainer.remote import RemoteDispatchError

    armed = A.apply_image_decoupled_activation(acfg, SW)

    def fp(host, fresh=False):
        raise RemoteDispatchError("pod unreachable", returncode=255)

    runner = SimpleNamespace(cfg=armed, deployed_chain_toml=None, work_root=tmp_path)
    hook = dispatch_host_hook(runner, fingerprint_fn=fp)
    host = _LaneHost("lane", "10.0.0.9")
    assert hook.matches(host, SW) is False
    with pytest.raises(RemoteDispatchError):
        hook(host, SW)


def test_image_refusal_never_moves_a_payer_legs_billing():
    from cascade.trainer.faults import _OperatorFault
    from cascade.trainer.leg_pins import ImageAfterUnqualified, LegImageMismatch
    from cascade.trainer.runner_funded_legs import bill_operator_after_fault

    flips = []
    q = SimpleNamespace(set_operator_billed=lambda hk, flag: flips.append((hk, flag)))
    runner = SimpleNamespace(_funded_queue=lambda: q)
    assert not bill_operator_after_fault(runner, "hk", LegImageMismatch("other image"))
    assert not bill_operator_after_fault(runner, "hk", ImageAfterUnqualified("no record"))
    assert flips == []
    assert bill_operator_after_fault(runner, "hk", _OperatorFault("upload lost"))
    assert flips == [("hk", True)]


def test_restart_reattaches_an_old_image_leg_under_the_decoupled_rule(cfg, tmp_path):
    """Review 2: after a post-lock-in restart an in-flight leg whose digest
    differs from the era's only in its image (dispatched before the repin) is
    RE-ATTACHED; a leg with other numerics is still requeued."""
    from cascade.trainer.rolling import EraState
    from tests.unit.test_rolling import Clock, _sched
    from tests.unit.test_v2_golive_validator import _v2

    v2cfg, n = _v2(cfg)
    v2cfg = bind_image_after(replace(v2cfg, v2=replace(v2cfg.v2, **AFTER,
                                                        image_decoupled_from_block=n)))
    sched, _ = _sched(v2cfg, tmp_path, Clock())
    era = EraState(10, n, n, 1)
    base = v2cfg.training.v2_contract
    old_image = contract_digest(replace(base, train_image_digest_after="", train_image_after_block=0))
    assert old_image != sched._contract_identity(era)
    launched, requeued = [], []
    sched._era_state_for = lambda idx: era
    sched._uid_of = lambda client, hk: 1
    sched._flight_stamp = lambda e, era, block, q: (0, 0)
    sched._launch_leg = lambda gen, *a, **k: launched.append(gen.hotkey)
    q = SimpleNamespace(requeue=lambda hk, **kw: requeued.append(hk))

    def flight(hk, digest):
        return SimpleNamespace(hotkey=hk, ref="r", reveal_block=1, era_index=10,
                               training_init=era.warm_start_ckpt,
                               training_contract_digest=digest, label="")

    sched._reattach_flights(None, n + 5, q, [flight("old-image", old_image),
                                            flight("numerics", "f" * 64)], set(), {})
    assert launched == ["old-image"] and requeued == ["numerics"]


def test_receipt_of_a_reattached_old_image_leg_is_accepted(acfg):
    from cascade.trainer.leg_pins import receipt_contract_ok

    armed = A.apply_image_decoupled_activation(acfg, SW)
    post = armed.training.at_block(SW)
    old = contract_digest(replace(acfg.training.v2_contract, train_image_digest_after=""))
    assert receipt_contract_ok(armed, post, contract_digest(post), SW)
    assert receipt_contract_ok(armed, post, old, SW)                 # image-only: proven
    assert not receipt_contract_ok(armed, post, old, None)           # no era named: strict
    assert not receipt_contract_ok(armed, post, "f" * 64, SW)
    pre = armed.training.at_block(SW - 3600)
    assert not receipt_contract_ok(armed, pre, contract_digest(post), SW - 3600)  # old rule


def test_marketcast_post_switch_leg_carries_its_contract_block(acfg):
    from cascade.trainer.marketcast_runner import mkt_contract_block_kw

    armed = A.apply_image_decoupled_activation(acfg, SW)
    assert mkt_contract_block_kw(armed, SimpleNamespace(start_block=SW)) == {"contract_block": SW}
    assert mkt_contract_block_kw(armed, SimpleNamespace(start_block=SW - 3600)) == {}
    assert mkt_contract_block_kw(acfg, SimpleNamespace(start_block=SW)) == {}   # unresolved


# ── review 3: matching is a no-op unless the leg is post-switch; F5 image ────


def _boom(*a, **k):
    raise AssertionError("probed a lane for a leg that needs no matching")


def test_armed_but_unresolved_hands_lanes_out_as_today(acfg):
    from cascade.trainer.leg_pins import dispatch_host_hook
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    runner = SimpleNamespace(cfg=acfg, deployed_chain_toml=None)
    hook = dispatch_host_hook(runner, fingerprint_fn=_boom, image_fn=_boom)
    assert not hook.filters(SW)
    hosts = [_LaneHost("a", "10.0.0.1")]
    pool = _FinalLanePool(hosts, lambda: hosts)
    d = _LaneDisp(hook)
    out = TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x",
                                               role="challenger", contract_block=SW)
    assert out.host == "a"                                       # no probe, no refusal


def test_pre_switch_leg_is_never_probed(acfg):
    from cascade.trainer.leg_pins import dispatch_host_hook

    armed = A.apply_image_decoupled_activation(acfg, SW)
    hook = dispatch_host_hook(SimpleNamespace(cfg=armed), fingerprint_fn=_boom, image_fn=_boom)
    host = _LaneHost("a", "10.0.0.1")
    assert hook.match_state(host, SW - 3600) is True and hook(host, SW - 3600) is host


def test_post_switch_match_needs_the_runtime_image_too(acfg):
    from cascade.trainer.leg_pins import dispatch_host_hook

    armed = A.apply_image_decoupled_activation(acfg, SW)
    base_img = armed.training.at_block(SW - 3600).train_image_digest
    img = {"v": base_img}
    hook = dispatch_host_hook(SimpleNamespace(cfg=armed), fingerprint_fn=lambda h, fresh=False: FP16,
                              image_fn=lambda h, fresh=False: img["v"])
    host = _LaneHost("a", "10.0.0.1")
    assert hook.match_state(host, SW) is False       # after code, base pin injected
    img["v"] = RC16
    hook._reprobed.clear()
    assert hook.match_state(host, SW) is True


def test_other_image_lane_is_reprobed_at_most_once_a_minute(acfg):
    from cascade.trainer.leg_pins import dispatch_host_hook

    armed = A.apply_image_decoupled_activation(acfg, SW)
    calls = []
    t = [0.0]

    def fp(h, fresh=False):
        calls.append(fresh)
        return armed.round.worker_code_fingerprint             # the current image

    hook = dispatch_host_hook(SimpleNamespace(cfg=armed), fingerprint_fn=fp,
                              image_fn=lambda h, fresh=False: RC16, clock=lambda: t[0])
    host = _LaneHost("a", "10.0.0.1")
    for _ in range(5):
        assert hook.match_state(host, SW) is False
    assert calls.count(True) == 1                               # one fresh re-probe
    t[0] = hook.REPROBE_SECONDS + 1
    assert hook.match_state(host, SW) is False
    assert calls.count(True) == 2


def test_unreachable_lane_means_wait_never_no_lane():
    import time as _t

    from cascade.trainer.faults import _LaneDeadlinePassed
    from cascade.trainer.leg_pins import LegImageMismatch
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    hosts = [_LaneHost("down", "10.0.0.1")]
    pool = _FinalLanePool(hosts, lambda: hosts)
    d = _LaneDisp(_Hook({"down": None}))
    with pytest.raises(_LaneDeadlinePassed):                    # waited, then the deadline
        TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x", role="challenger",
                                             contract_block=SW, deadline=_t.time() - 1)
    try:
        pool.get(deadline=_t.time() - 1, match=lambda h: None)
    except _LaneDeadlinePassed:
        pass
    except LegImageMismatch:                                    # pragma: no cover
        pytest.fail("an unreachable lane must not read as 'no lane on this image'")
    assert [h.name for h in pool.known_hosts()] == ["down"]


# ── owner rule 2026-11-02: never the wrong image ─────────────────────────────


def _armed_post(acfg):
    return A.apply_image_decoupled_activation(acfg, SW)


def test_worker_image_refusal_needs_rc2_and_the_workers_own_last_line():
    """Review 4 CRITICAL: never free text. A generator raising the old marker
    (worker exit 3 / an rc-1 stderr tail) is never an image refusal."""
    from cascade.trainer.contract import IMAGE_REFUSAL_SENTINEL as S
    from cascade.trainer.leg_pins import worker_refused_image
    from cascade.trainer.remote import RemoteDispatchError, worker_image_refusal

    assert worker_image_refusal(2, f"log line\n{S}\n")
    assert not worker_image_refusal(3, S)                         # miner rejection exit
    assert not worker_image_refusal(1, S)
    assert not worker_image_refusal(2, f"{S}\nmore output")       # not the last line
    assert not worker_image_refusal(2, "refusing FINAL run: runtime image sha256:a")
    e = RemoteDispatchError("miner submission rejected: ValueError('refusing FINAL run: x "
                            f"{S}')", returncode=3)
    assert not worker_refused_image(e)
    e2 = RemoteDispatchError("rc=2", returncode=2)
    assert not worker_refused_image(e2)                           # flag only from stdout
    e2.image_refusal = True
    assert worker_refused_image(e2)


def test_image_refusal_retry_cap_is_per_leg_and_persists(tmp_path):
    from cascade.trainer.leg_pins import (
        IMAGE_RETRY_CAP,
        IMAGE_RETRY_WINDOW_S,
        image_retry_exhausted,
        leg_key,
        note_image_refusal,
    )

    r = SimpleNamespace(work_root=tmp_path)
    k = leg_key("hk", "challenger", 4242)
    for i in range(IMAGE_RETRY_CAP):
        assert not image_retry_exhausted(r, k, now=100.0)
        note_image_refusal(r, k, now=100.0 + i)
    assert image_retry_exhausted(r, k, now=200.0)
    assert not image_retry_exhausted(r, leg_key("hk", "challenger", 4243), now=200.0)
    assert not image_retry_exhausted(r, leg_key("hk", "king", 4242), now=200.0)
    restarted = SimpleNamespace(work_root=tmp_path)                 # a trainer restart
    assert image_retry_exhausted(restarted, k, now=200.0)
    assert not image_retry_exhausted(restarted, k, now=100.0 + IMAGE_RETRY_WINDOW_S + 5)


def test_wrong_image_pod_blames_only_on_measured_code_evidence(tmp_path):
    from cascade.provision.core import quarantined_hosts
    from cascade.provision.lium_identity import load_identities
    from cascade.trainer.leg_pins import (
        ImageAfterUnqualified,
        LegImageMismatch,
        LegImageUnverified,
        image_retry_exhausted,
    )
    from cascade.trainer.runner_funded_legs import _wrong_image_pod

    r = SimpleNamespace(work_root=tmp_path,
                        _funded_claim_meta={"exec-1": (0.0, "pod-1"), "exec-2": (0.0, "pod-9")})
    host = SimpleNamespace(host="203.0.113.7", name="funded")
    pod = SimpleNamespace(instance_id="pod-1")
    _wrong_image_pod(r, "k", host, pod, ImageAfterUnqualified("our config"))
    _wrong_image_pod(r, "k", host, pod, LegImageUnverified("probe failed"))
    _wrong_image_pod(r, "k", host, pod, LegImageMismatch("our injected env differs"))
    assert load_identities()["bad"] == {} and "203.0.113.7" not in quarantined_hosts()
    _wrong_image_pod(r, "k", host, pod, LegImageMismatch("pod code differs",
                                                         code_evidence=True))
    bad = load_identities()["bad"]
    assert "exec-1" in bad and "exec-2" not in bad
    assert "203.0.113.7" in quarantined_hosts()
    assert image_retry_exhausted(r, "k")          # env mismatch + code mismatch = 2 pods


def test_capped_post_switch_leg_waits_without_renting(acfg, tmp_path):
    from cascade.trainer.leg_pins import (
        IMAGE_RETRY_CAP,
        ImageRetryCapped,
        leg_key,
        note_image_refusal,
    )
    from cascade.trainer.runner_funded_legs import _FundedLegMixin

    armed = _armed_post(acfg)
    seen = []
    runner = SimpleNamespace(
        cfg=armed, work_root=tmp_path, _funded_leg_failures={},
        _record_funded_failure=lambda hk, msg, **kw: seen.append(kw),
        _rent_funded_host=lambda *a, **k: pytest.fail("rented past the wrong-image cap"))
    for _ in range(IMAGE_RETRY_CAP):
        note_image_refusal(runner, leg_key("hk", "challenger", 1))
    gen = SimpleNamespace(hotkey="hk", uid=1, ref="ns/g@sha256:" + "1" * 64)
    with pytest.raises(ImageRetryCapped):                         # waits, not dropped
        _FundedLegMixin._run_funded_leg_scoped(
            runner, None, gen, SimpleNamespace(base_seed=1), SW, armed.training.at_block(SW),
            "", warm_start_ref="")
    assert seen == [{"miner_fault": False, "error_class": "no_capacity", "burn": False}]


def test_pre_switch_leg_ignores_the_cap(acfg, tmp_path):
    """Review 4 HIGH 2: pre-switch behaviour is exactly the release's."""
    from cascade.trainer.leg_pins import IMAGE_RETRY_CAP, leg_key, note_image_refusal
    from cascade.trainer.runner_funded_legs import _FundedLegMixin

    armed = _armed_post(acfg)
    runner = SimpleNamespace(cfg=armed, work_root=tmp_path, _funded_leg_failures={},
                             _record_funded_failure=lambda *a, **k: None)
    for _ in range(IMAGE_RETRY_CAP):
        note_image_refusal(runner, leg_key("hk", "challenger", 1))
    sentinel = RuntimeError("reached the reupload step")

    def _reupload(*a, **k):
        raise sentinel

    import cascade.trainer.runner_funded_legs as RFL

    orig = RFL._reupload_pending
    RFL._reupload_pending = _reupload
    try:
        with pytest.raises(RuntimeError, match="reached the reupload step"):
            _FundedLegMixin._run_funded_leg_scoped(
                runner, None, SimpleNamespace(hotkey="hk", uid=1, ref="r"),
                SimpleNamespace(base_seed=1), SW - 3600,
                armed.training.at_block(SW - 3600), "", warm_start_ref="")
    finally:
        RFL._reupload_pending = orig


class _RefusingDisp:
    def __init__(self, hook, *, rc=2, sentinel=True, text="worker exit 2"):
        self.host_hook, self.rc, self.sentinel, self.text = hook, rc, sentinel, text

    def dispatch(self, host, **kw):
        from cascade.trainer.remote import RemoteDispatchError

        e = RemoteDispatchError(self.text, returncode=self.rc)
        e.image_refusal = self.sentinel and self.rc == 2
        raise e


def _real_hook(cfg, fp):
    from cascade.trainer.leg_pins import dispatch_host_hook

    return dispatch_host_hook(SimpleNamespace(cfg=cfg, deployed_chain_toml=None),
                              fingerprint_fn=lambda h, fresh=False: fp["v"],
                              image_fn=lambda h, fresh=False: RC16)


def test_post_switch_lane_refusal_holds_the_lane_only_on_measured_code(acfg):
    from cascade.trainer.leg_pins import LegImageMismatch
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    armed = _armed_post(acfg)
    fp = {"v": FP16}
    hook = _real_hook(armed, fp)
    hosts = [_LaneHost("lane", "10.0.0.1")]
    pool = _FinalLanePool(hosts, lambda: hosts, breaker_threshold=1)
    # matching passed (after code), the worker refused: our probe now sees the
    # same release => our config skew => lane stays in rotation, no breaker
    with pytest.raises(LegImageMismatch) as ei:
        TrainerRunner._dispatch_on_free_lane(_RefusingDisp(hook), pool, hosts, describe="x",
                                             role="challenger", contract_block=SW)
    assert not ei.value.code_evidence and [h.name for h in pool.known_hosts()] == ["lane"]
    # the lane flips to another release between the match and the refusal
    calls = {"n": 0}

    def flip(h, fresh=False):
        calls["n"] += 1
        return FP16 if not fresh or calls["n"] < 3 else armed.round.worker_code_fingerprint
    hook.fingerprint_fn = flip
    with pytest.raises(LegImageMismatch) as ei:
        TrainerRunner._dispatch_on_free_lane(_RefusingDisp(hook), pool, hosts, describe="x",
                                             role="challenger", contract_block=SW)
    assert ei.value.code_evidence and pool.known_hosts() == []


def test_marker_text_from_a_generator_is_a_plain_failure(acfg):
    """A miner's exception carrying the old marker never holds a lane."""
    from cascade.trainer.leg_pins import LegImageMismatch
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    armed = _armed_post(acfg)
    hook = _real_hook(armed, {"v": FP16})
    hosts = [_LaneHost("lane", "10.0.0.1")]
    pool = _FinalLanePool(hosts, lambda: hosts)
    d = _RefusingDisp(hook, rc=3, sentinel=False,
                      text="miner submission rejected: ValueError('refusing FINAL run: x')")
    with pytest.raises(Exception) as ei:
        TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x", role="challenger",
                                             contract_block=SW)
    assert not isinstance(ei.value, LegImageMismatch)
    assert [h.name for h in pool.known_hosts()] == ["lane"]


def test_pre_switch_lane_refusal_keeps_the_release_behaviour(acfg):
    from cascade.trainer.leg_pins import LegImageMismatch
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    armed = _armed_post(acfg)
    hook = _real_hook(armed, {"v": FP16})
    hosts = [_LaneHost("lane", "10.0.0.1")]
    pool = _FinalLanePool(hosts, lambda: hosts)
    with pytest.raises(Exception) as ei:
        TrainerRunner._dispatch_on_free_lane(_RefusingDisp(hook), pool, hosts, describe="x",
                                             role="challenger", contract_block=SW - 3600)
    assert not isinstance(ei.value, LegImageMismatch)
    assert [h.name for h in pool.known_hosts()] == ["lane"]


def test_runtime_image_probe_reads_what_the_worker_sees(monkeypatch):
    import subprocess

    from cascade.trainer import leg_pins

    host = _LaneHost("lane", "10.0.0.5")
    host.static_env = (("CASCADE_TRAIN_IMAGE_DIGEST", RC16),)
    assert leg_pins.host_runtime_image(host) == RC16           # sourced into the worker
    host.static_env = ()
    out = {"rc": 0, "stdout": RC16 + "\n"}
    monkeypatch.setattr("cascade.trainer.remote.run_ssh", lambda argv, timeout, stdin_text=None:
                        subprocess.CompletedProcess(argv, out["rc"], out["stdout"], ""))
    monkeypatch.setattr("cascade.trainer.remote.build_ssh_argv", lambda h, cmd: ["ssh", cmd])
    leg_pins._RUNTIME_IMAGES.clear()
    assert leg_pins.host_runtime_image(host, fresh=True) == RC16
    out.update(rc=1, stdout="")
    assert leg_pins.host_runtime_image(host, fresh=True) == ""     # unset: a real answer
    out.update(rc=127)
    with pytest.raises(leg_pins.LegImageUnverified):
        leg_pins.host_runtime_image(host, fresh=True)
    leg_pins._RUNTIME_IMAGES.clear()


def test_unverified_probe_is_unknown_in_matching(acfg):
    from cascade.trainer.leg_pins import LegImageUnverified, dispatch_host_hook

    armed = _armed_post(acfg)

    def fp(h, fresh=False):
        raise LegImageUnverified("remote python failed")

    hook = dispatch_host_hook(SimpleNamespace(cfg=armed), fingerprint_fn=fp,
                              image_fn=lambda h, fresh=False: RC16)
    assert hook.match_state(_LaneHost("a", "10.0.0.1"), SW) is None
