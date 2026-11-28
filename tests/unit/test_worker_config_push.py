"""Leg pods get a WORKER CONFIG their pinned release parses (launch blocker
2026-11-24).

rc15 (5d41d94b, src 5e3fade; lane A + lane B) raises on main's chain.toml
([marketcast] suites, then [lane_c] target mkt:market) and mkt-rc2 (0205f761,
src 6c9cab9; [marketcast] pod_image) raises [marketcast] unscorable. The
trainer / provisioner now push the deployed file cut to the tables that
release's worker reads — every kept table verbatim, so contract_digest,
base_arch_digest and every image pin are the deployed ones.

Config-parse only: nothing here imports torch.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from cascade.shared.config import load_chain_config
from cascade.shared.manifest import contract_digest
from cascade.shared.worker_config import (
    LEG_TABLES,
    LEGACY_WORKER_PROFILES,
    derive_worker_toml,
    profile_for_image,
    worker_toml_for_image,
)
from cascade.trainer.contract import compute_base_arch_digest

REPO = Path(__file__).resolve().parents[2]
CHAIN = REPO / "chain.toml"
RC15 = "sha256:5d41d94bc0c0d3ae3dfd042b0cc7e4d900c7d1f48173808436c12bf764a67641"
RC2 = "sha256:0205f761fde14da71b9a90b4f1ecdc862d657f982accb11450e0ce22c61d2f12"
SOURCES = {RC15: "5e3fade3d5706b1cbb615c1020fd7e2fae617f3f",
           RC2: "6c9cab969e36440df49c72e408852f0760f86aff"}
BLOCKS = (None, 9_000_000, 9_586_800, 9_600_000, 10_000_000)


def _text() -> str:
    return CHAIN.read_text(encoding="utf-8")


def _digests(cfg) -> dict:
    out = {"pin.base_arch_digest": cfg.training.base_arch_digest}
    for b in BLOCKS:
        t = cfg.training.at_block(b)
        out[f"{b}.contract"] = contract_digest(t.primary_size)
        out[f"{b}.arch"] = compute_base_arch_digest(t.primary_size)
        out[f"{b}.image"] = t.train_image_digest
        for s in t.extra_sizes:
            out[f"{b}.{s.arch_preset}"] = contract_digest(t.for_size(s))
    if cfg.training.v2_contract is not None:
        out["v2.contract"] = contract_digest(cfg.training.v2_contract)
    return out


def _load(text: str, tmp_path: Path, name: str):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return load_chain_config(p)


# ── the derivation ───────────────────────────────────────────────────────────


def test_profiles_are_keyed_by_the_live_pins():
    raw = tomllib.loads(_text())
    assert profile_for_image(raw["round"]["funded_pod_image"]).name == "worker-v2.0.0-rc15"
    assert profile_for_image(raw["v2"]["training"]["train_image_digest"]) is \
        LEGACY_WORKER_PROFILES[RC15]
    assert profile_for_image(raw["marketcast"]["pod_image"]).name == "worker-v2.1.0-mkt-rc2"


def test_unknown_image_gets_the_deployed_file_unchanged():
    text = _text()
    other = "reg/x@sha256:" + "b" * 64
    assert worker_toml_for_image(text, other) == (text, "")
    assert worker_toml_for_image(text, "") == (text, "")
    assert derive_worker_toml(text, None) is text


@pytest.mark.parametrize("image", [RC15, RC2])
def test_kept_tables_are_verbatim_and_the_rest_dropped(image):
    prof = LEGACY_WORKER_PROFILES[image]
    raw = tomllib.loads(_text())
    got = tomllib.loads(worker_toml_for_image(_text(), image)[0])
    converted = {(t, k) for t, k, _fn in prof.down_convert}
    for k, v in raw.items():
        if not isinstance(v, dict):
            assert got[k] == v                       # schema_version & co
        elif k in prof.keep:
            want = {kk: vv for kk, vv in v.items() if (k, kk) not in converted}
            assert {kk: vv for kk, vv in got[k].items() if (k, kk) not in converted} == want
        else:
            assert k not in got
    # the contract tables are never down-converted
    assert not {t for t, _k in converted} & {"training", "v2", "round", "lane_b"}


def test_rc15_profile_drops_exactly_the_tables_rc15_cannot_or_need_not_load():
    got = tomllib.loads(worker_toml_for_image(_text(), RC15)[0])
    raw = tomllib.loads(_text())
    assert set(raw) - set(got) == {k for k in raw if isinstance(raw[k], dict)
                                   and k not in LEG_TABLES}
    assert "marketcast" not in got and "lane_c" not in got
    for t in ("training", "v2", "round", "lane_b", "storage", "generator"):
        assert got[t] == raw[t]


def test_rc2_profile_keeps_marketcast_and_lane_c_with_only_the_wait_down_converted():
    raw = tomllib.loads(_text())
    got = tomllib.loads(worker_toml_for_image(_text(), RC2)[0])
    assert got["lane_c"] == raw["lane_c"]
    diff = {k for k in set(raw["marketcast"]) | set(got["marketcast"])
            if raw["marketcast"].get(k) != got["marketcast"].get(k)}
    assert diff <= {"verdict_wait_blocks"}
    # what the leg reads is the deployed value
    for k in ("from_block", "base_ref", "base_sha256", "suites", "horizons", "frequencies",
              "assets", "data_sources", "embargo_days", "max_leg_hours", "pod_image",
              "continuation_protocol", "event_times", "crps_scale", "native_intervals"):
        assert got["marketcast"].get(k) == raw["marketcast"].get(k)
    # the down-converted wait satisfies mkt-rc2's forward-window scorability
    mc = got["marketcast"]
    wait_s = int(mc["verdict_wait_blocks"]) * 12 - int(mc.get("start_margin_seconds", 3600))
    from cascade.shared.config.marketcast import duration_minutes

    longest = max(60 * duration_minutes(h) for s in mc["suites"] for h in mc["horizons"][s])
    assert wait_s >= 2 * longest
    assert wait_s >= int(mc.get("min_time_blocks", 8)) * int(mc.get("cluster_block_hours", 6)) \
        * 3600


@pytest.mark.parametrize("image", [RC15, RC2])
def test_current_loader_digests_are_identical_on_the_worker_config(image, tmp_path):
    full = _load(_text(), tmp_path, "full.toml")
    derived = _load(worker_toml_for_image(_text(), image)[0], tmp_path, "w.toml")
    assert _digests(derived) == _digests(full)
    for t in ("round", "storage", "generator", "lane_b", "v2", "generation"):
        assert getattr(derived, t) == getattr(full, t), t


def test_derivation_refuses_down_converting_a_dropped_table():
    from cascade.shared import worker_config as wc

    bad = wc.WorkerConfigProfile(name="x", source_commit="", keep=("training",),
                                 down_convert=(("marketcast", "k", lambda raw: 1),))
    with pytest.raises(ValueError, match="drops"):
        wc.derive_worker_toml(_text(), bad)


# ── the old releases themselves (needs the git history; skipped without) ─────


_PROBE = r"""
import json, sys
sys.modules["torch"] = None              # config-parse only: torch can never load
sys.path.insert(0, sys.argv[1])
from cascade.shared.config import load_chain_config
from cascade.shared.manifest import contract_digest
from cascade.trainer.contract import compute_base_arch_digest
cfg = load_chain_config(sys.argv[2])
out = {"pin.base_arch_digest": cfg.training.base_arch_digest}
for b in json.loads(sys.argv[3]):
    t = cfg.training.at_block(b)
    out[f"{b}.contract"] = contract_digest(t.primary_size)
    out[f"{b}.arch"] = compute_base_arch_digest(t.primary_size)
    out[f"{b}.image"] = t.train_image_digest
    for s in t.extra_sizes:
        out[f"{b}.{s.arch_preset}"] = contract_digest(t.for_size(s))
if cfg.training.v2_contract is not None:
    out["v2.contract"] = contract_digest(cfg.training.v2_contract)
extra = {"lane_b.leg_hours": cfg.lane_b.leg_hours,
         "heat": [cfg.round.heat_guard_factor, cfg.round.heat_guard_floor_seconds]}
if len(sys.argv) > 4:
    mc = cfg.marketcast
    extra["mkt"] = {k: repr(getattr(mc, k)) for k in sys.argv[4].split(",")}
    extra["lane_c.leg_hours"] = cfg.lane_c.leg_hours
print(json.dumps({"digests": out, "extra": extra}, sort_keys=True))
"""

_MKT_LEG_FIELDS = ("from_block,base_ref,base_sha256,base_arch,quantile_knots,suites,assets,"
                   "frequencies,horizons,data_sources,max_context_length,covariates,"
                   "augmentations,embargo_days,max_synthetic_fraction,max_leg_hours,max_lr,"
                   "max_steps,max_batch,max_lora_rank,continuation_protocol,crps_scale,"
                   "micro_batch_tokens,native_intervals,native_history_days,event_times,"
                   "composite,composite_suite_weights,pod_image,worker_code_fingerprint,"
                   "bar_relay")


def _old_tree(sha: str, tmp_path: Path) -> Path:
    if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"],
                      capture_output=True).returncode != 0:
        pytest.skip(f"release source {sha[:7]} not in this checkout's history")
    dest = tmp_path / sha[:7]
    dest.mkdir()
    arch = subprocess.run(["git", "-C", str(REPO), "archive", sha, "cascade"],
                          capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arch, check=True)
    return dest


def _probe(tree: Path, toml: Path, fields: str = ""):
    argv = [sys.executable, "-I", "-c", _PROBE, str(tree), str(toml),
            json.dumps(list(BLOCKS))] + ([fields] if fields else [])
    return subprocess.run(argv, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("image", [RC15, RC2])
def test_the_pinned_release_refuses_the_full_file_and_loads_its_worker_config(image, tmp_path):
    tree = _old_tree(SOURCES[image], tmp_path)
    full_p = tmp_path / "chain.deployed.toml"
    full_p.write_text(_text(), encoding="utf-8")
    refused = _probe(tree, full_p)
    assert refused.returncode != 0 and "ValueError" in refused.stderr   # today's failure

    w = tmp_path / "worker.toml"
    w.write_text(worker_toml_for_image(_text(), image)[0], encoding="utf-8")
    mkt = image == RC2
    ok = _probe(tree, w, _MKT_LEG_FIELDS if mkt else "")
    assert ok.returncode == 0, ok.stderr[-2000:]
    old = json.loads(ok.stdout)
    full = load_chain_config(full_p)
    assert old["digests"] == _digests(full)
    assert old["extra"]["lane_b.leg_hours"] == full.lane_b.leg_hours
    assert old["extra"]["heat"] == [full.round.heat_guard_factor,
                                    full.round.heat_guard_floor_seconds]
    if mkt:
        mc = full.marketcast
        assert old["extra"]["mkt"] == {k: repr(getattr(mc, k)) for k in
                                       _MKT_LEG_FIELDS.split(",")}
        assert old["extra"]["lane_c.leg_hours"] == full.lane_c.leg_hours


# ── the push sites ───────────────────────────────────────────────────────────


def _fake_runner(tmp_path):
    from cascade.trainer.loop import TrainerRunner

    full = load_chain_config(CHAIN)
    fake = types.SimpleNamespace(cfg=full, deployed_chain_toml=CHAIN, work_root=tmp_path,
                                 _scp_calls=[])
    fake._scp_runner = lambda argv: fake._scp_calls.append(list(argv))
    fake._push_deployed_chain_toml = TrainerRunner._push_deployed_chain_toml.__get__(fake)
    return fake


def _host():
    from cascade.trainer.remote import RemoteHost

    return RemoteHost(name="funded-x", host="10.0.0.7", port=2222, user="root",
                      key_path="~/.ssh/k", workdir="/root/cascade")


def test_trainer_push_sends_the_fleet_release_its_worker_config(tmp_path):
    fake = _fake_runner(tmp_path)
    out = fake._push_deployed_chain_toml(_host())
    assert out.chain_toml == "/root/cascade/chain.deployed.toml"
    src = Path(fake._scp_calls[-1][-2])
    assert src != CHAIN and "worker-v2.0.0-rc15" in str(src)
    assert src.read_text(encoding="utf-8") == worker_toml_for_image(_text(), RC15)[0]


def test_trainer_push_sends_a_marketcast_leg_the_mkt_release_config(tmp_path, monkeypatch):
    from cascade.trainer.leg_pins import mkt_leg_scope

    # a QUALIFIED MarketCast image (the rule: test_image_qualification.py)
    monkeypatch.setattr("cascade.trainer.leg_pins.mkt_image_refusal", lambda cfg: "")
    fake = _fake_runner(tmp_path)
    with mkt_leg_scope(fake):
        fake._push_deployed_chain_toml(_host())
    src = Path(fake._scp_calls[-1][-2])
    assert "worker-v2.1.0-mkt-rc2" in str(src)
    assert src.read_text(encoding="utf-8") == worker_toml_for_image(_text(), RC2)[0]


def test_trainer_push_without_a_resolvable_image_sends_the_deployed_file(tmp_path):
    fake = _fake_runner(tmp_path)
    del fake.cfg            # e.g. a bare test double: behaviour exactly as before
    fake._push_deployed_chain_toml(_host())
    assert fake._scp_calls[-1][-2] == str(CHAIN)


def test_provisioner_push_derives_per_stage(monkeypatch, tmp_path):
    from cascade.provision.loop import PodAddress, RenderSettings
    from cascade.provision.main import make_config_push

    calls: list[list[str]] = []
    monkeypatch.setattr("cascade.provision.main.subprocess.run",
                        lambda argv, **kw: calls.append(list(argv)) or types.SimpleNamespace(
                            returncode=0, stdout="", stderr=""))
    render = RenderSettings(image="reg/x@" + RC15, ssh_pubkey="ssh-ed25519 AAAA o",
                            key_path="~/.ssh/k")
    hook = make_config_push(render, box_chain_toml=str(CHAIN), pod_user="root",
                            stage_images={"*": "reg/x@" + RC15, "mkt": "reg/m@" + RC2},
                            work_dir=str(tmp_path))
    hook(PodAddress("10.0.0.5", 2222), "eval", "")
    hook(PodAddress("10.0.0.6", 2222), "mkt", "")
    scps = [c for c in calls if c[0] == "scp"]
    assert Path(scps[0][-2]).read_text(encoding="utf-8") == worker_toml_for_image(_text(), RC15)[0]
    assert Path(scps[1][-2]).read_text(encoding="utf-8") == worker_toml_for_image(_text(), RC2)[0]
    # no stage images (or an unknown release): the full file, as before
    calls.clear()
    make_config_push(render, box_chain_toml=str(CHAIN), pod_user="root")(
        PodAddress("10.0.0.7", 22), "eval", "")
    assert [c for c in calls if c[0] == "scp"][0][-2] == str(CHAIN)


def test_provisioner_main_wires_the_stage_images():
    import inspect

    from cascade.provision import main as main_mod

    src = inspect.getsource(main_mod)
    assert 'stage_images={"*": cfg.round.funded_pod_image' in src
    assert '"mkt": cfg.marketcast.pod_image' in src
