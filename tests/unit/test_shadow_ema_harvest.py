"""#400 power-EMA shadow files (weights_ema005 / weights_ema010 / weights_raw)
are operator analysis data: verified with the checkpoint, kept on R2 (r2
transport) or — only with ``[round] funded_keep_shadow_ema`` — on the box
(ssh transport), and never part of the Hippius push."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cascade.trainer.loop as loop_module
import cascade.trainer.r2_transfer as r2
import tests.unit.test_funded_pod_wiring as wiring
from cascade.shared.config import RoundConfig
from cascade.trainer import runner_common
from tests.unit.test_r2_direct_harvest import _ckpt_files, _leg, _r2_runner, _safetensors

SHADOW = ("weights_ema005.safetensors", "weights_ema010.safetensors", "weights_raw.safetensors")


def test_shadow_names():
    for name in SHADOW:
        assert r2.shadow_ema_file(name) and r2.checkpoint_file_allowed(name)
    for name in ("weights.safetensors", "weights_stable.safetensors", "optimizer.safetensors",
                 "power_ema.safetensors", "config.json"):
        assert not r2.shadow_ema_file(name)


# ── r2 transport ─────────────────────────────────────────────────────────────


def _files_with_shadows():
    files = _ckpt_files()
    for i, name in enumerate(SHADOW):
        files[name] = _safetensors(64 + i)
    return files


def test_r2_carries_the_shadow_files_to_r2_but_never_to_hippius(tmp_path, monkeypatch):
    t = _r2_runner(tmp_path, monkeypatch, files=_files_with_shadows())
    _leg(t)
    # verified against the receipt like every tensor file, uploaded to R2 ...
    put = {e.split(" ", 1)[1] for e in t.events if e.startswith("put ")}
    assert set(SHADOW) <= put
    # ... but the published checkpoint is exactly the pre-shadow file set
    assert t.pushed == [{n: b for n, b in _ckpt_files().items()
                         if n not in (".train_complete", "notes.bin")}]
    # the published objects are deleted; the shadow files stay on R2
    left = sorted(k.rsplit("/", 1)[1] for k in t.store.s3.objects)
    assert left == sorted(SHADOW)
    assert all(k.startswith("ckpt/777/hkA/") for k in t.store.s3.objects)
    assert len(t.store.s3.deleted) == 4
    assert not any(k.rsplit("/", 1)[1] in SHADOW for k in t.store.s3.deleted)


def test_relay_push_never_fetches_a_shadow_file(tmp_path, monkeypatch):
    fetched = []

    def fetch(name, meta, dest):
        fetched.append(name)
        Path(dest).write_bytes(b"x")

    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint", lambda d, **kw: None)
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_weights_finite", lambda d: None)
    pushed = []

    def push(d, repo, hub, hf_repo=None):
        pushed.append(sorted(p.name for p in Path(d).iterdir()))
        return SimpleNamespace(ref=SimpleNamespace(immutable_ref=f"{repo}@sha256:" + "d" * 64),
                               size_bytes=1)

    monkeypatch.setattr(r2, "upload_dir_to_hub_or_hf", push)
    names = ("config.json", "weights.safetensors", *SHADOW)
    plan = {"repo": "cascade/ckpt", "files": {n: {"size": 1, "sha256": "0" * 64} for n in names}}
    assert r2.relay_push(plan, workdir=tmp_path, fetch=fetch)["ok"]
    assert sorted(fetched) == ["config.json", "weights.safetensors"]
    assert pushed == [["config.json", "weights.safetensors"]]


def test_published_and_shadow_parts_split_the_record():
    ckpt = r2.R2Checkpoint("b", "ckpt/1/hk/j/", {
        n: {"key": f"ckpt/1/hk/j/{n}", "size": 2, "sha256": "0" * 64}
        for n in ("weights.safetensors", "config.json", *SHADOW)})
    assert sorted(ckpt.published().files) == ["config.json", "weights.safetensors"]
    assert sorted(ckpt.shadow().files) == sorted(SHADOW)
    assert ckpt.published().total_bytes == 4


# ── ssh transport ────────────────────────────────────────────────────────────


def _ssh_leg(tmp_path, monkeypatch, **round_kw):
    pod_dir = tmp_path / "pod_ckpt"
    pod_dir.mkdir()
    files = {"weights.safetensors": b"w" * 8, "config.json": b"{}",
             **{n: n.encode() for n in SHADOW}}
    for n, b in files.items():
        (pod_dir / n).write_bytes(b)
    digests = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()
               if n.endswith(".safetensors")}
    disp = wiring._FakeLocalDispDigests(digests)
    runner, torn, seeds, contract = wiring._harvest_runner(tmp_path, monkeypatch, disp=disp, **round_kw)

    def _harvest(host, rdir, dest, **kw):
        import shutil
        shutil.copytree(pod_dir, dest, dirs_exist_ok=True)
        return dest

    monkeypatch.setattr("cascade.trainer.remote.harvest_remote_dir", _harvest)
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_checkpoint",
                        lambda d, contract=None, **kw: None)
    monkeypatch.setattr("cascade.eval.checkpoint_guard.verify_weights_finite", lambda d: None)
    uploaded = []

    def _up(d, repo, hub, hf_repo=None):
        uploaded.append(sorted(p.name for p in Path(d).iterdir()))
        return SimpleNamespace(ref=SimpleNamespace(
            immutable_ref="cascade/ckpt-r777-challenger-toto2-4m@sha256:" + "d" * 64))

    monkeypatch.setattr(loop_module, "upload_dir_to_hub_or_hf", _up)
    runner._hf_ckpt_repo = lambda repo: None
    entry = runner._run_funded_leg(disp, wiring._challenger("hkA"), seeds, 100, contract, "",
                                   warm_start_ref=None)
    assert entry.miner_hotkey == "hkA" and "hkA" not in runner._funded_leg_failures
    return uploaded, files


def _kept(root: Path):
    d = root / runner_common.SHADOW_EMA_DIRNAME
    return sorted(p.name for p in d.rglob("*") if p.is_file()) if d.exists() else []


def test_ssh_harvest_verifies_then_deletes_the_shadow_files_by_default(tmp_path, monkeypatch):
    uploaded, files = _ssh_leg(tmp_path, monkeypatch)
    assert uploaded == [["config.json", "weights.safetensors"]]      # the pre-shadow set
    assert _kept(tmp_path) == []
    assert not any(p.name in SHADOW for p in tmp_path.rglob("*")
                   if p.is_file() and "pod_ckpt" not in p.parts)


def test_ssh_harvest_keeps_the_shadow_files_aside_with_the_knob(tmp_path, monkeypatch):
    uploaded, files = _ssh_leg(tmp_path, monkeypatch, funded_keep_shadow_ema=True)
    assert uploaded == [["config.json", "weights.safetensors"]]      # still never published
    assert _kept(tmp_path) == sorted(SHADOW)
    kept = next((tmp_path / runner_common.SHADOW_EMA_DIRNAME).rglob("weights_raw.safetensors"))
    assert kept.read_bytes() == files["weights_raw.safetensors"]


def test_set_aside_is_a_no_op_without_shadow_files(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "weights.safetensors").write_bytes(b"w")
    cfg = SimpleNamespace(round=RoundConfig(funded_keep_shadow_ema=True))
    assert runner_common.set_aside_shadow_ema(cfg, tmp_path, d, 1, "hk", job_id="j") == []
    assert [p.name for p in d.iterdir()] == ["weights.safetensors"]


# ── the knob ─────────────────────────────────────────────────────────────────


def test_keep_shadow_knob_ships_dark_and_round_trips_through_the_loader(tmp_path):
    from cascade.shared.config import load_chain_config

    assert RoundConfig().funded_keep_shadow_ema is False
    repo_toml = Path(__file__).resolve().parents[2] / "chain.toml"
    assert load_chain_config(repo_toml).round.funded_keep_shadow_ema is False
    src = repo_toml.read_text(encoding="utf-8")
    edited = src.replace("\n[round]\n", "\n[round]\nfunded_keep_shadow_ema = true\n", 1)
    assert edited != src
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    assert load_chain_config(tmp_path / "chain.toml").round.funded_keep_shadow_ema is True
