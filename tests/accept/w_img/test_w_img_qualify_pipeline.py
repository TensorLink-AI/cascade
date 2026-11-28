"""W-IMG retro item 8 (owner 2026-11-28, "work through the retrospective
plan"): a worker image is BUILT, QUALIFIED and SMOKE-TESTED by the pipeline
before any pin can point at it.

Incidents this closes (OPSLOG 2026-11-26..28): rc15 vs the new image was not
bitwise and the repin ran under a dateless ``image_after_unqualified_override
= true``; the rc4 bf16-autocast crash surfaced only on a paid leg; mkt-rc3 was
hand-built; ``uv pip install -e`` drifted 46 packages off the lock (#999).

* publish-worker emits a build manifest (digest, commit, lock hash, in-image
  fingerprint, torch build + CUDA arch list) without pulling the image;
* a qualification record is /2: bitwise vs the previous image's golden file +
  the build manifest + a passing 30-step smoke per role + measured walls; a
  bitwise-only /1 record no longer qualifies;
* the trainer refuses an ``*_after`` image and a ``[marketcast] pod_image``
  without one, unless a DATED override (at most 14 days ahead) is set — and
  the override stops being honoured when it expires;
* the job definition + driver plan the run and emit ONE config patch;
* docs/OPERATOR.md orders the release: build -> qualify -> walls -> PR -> repin.

Torch-free: nothing here builds a model, loads a checkpoint or imports torch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared import image_qualification as Q
from cascade.shared.config import load_chain_config
from tests.unit.test_image_after_repin import SW, acfg  # noqa: F401 — the shared fixture

from .helpers import REPO

SKUS = ("8xH200", "8xH100")


def _script(name: str):
    path = REPO / "scripts" / name
    assert path.is_file(), f"scripts/{name} is missing"
    spec = importlib.util.spec_from_file_location(f"_w_img_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def armed(acfg, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(Q, "QUALIFICATIONS_DIR", tmp_path / "qualifications")
    cfg = A.apply_image_decoupled_activation(acfg, SW)
    return replace(cfg, round=replace(cfg.round, funded_pod_skus=SKUS))


def _pair(cfg):
    return (cfg.training.v2_contract.train_image_digest, cfg.v2.train_image_digest_after,
            cfg.round.worker_code_fingerprint, cfg.v2.worker_code_fingerprint_after)


def _bitwise_only(cfg) -> dict:
    """A record that qualified an image before item 8: bitwise + resume only."""
    frm, to, ffp, tfp = _pair(cfg)
    comp = {"manifest_sha256": "a" * 64, "tensors": 3, "equal": True,
            "resume_manifest_sha256": "b" * 64, "resume_equal": True}
    return {"schema": "cascade-image-qualification/1", "from_image": "sha256:" + Q.image_hex(frm),
            "to_image": "sha256:" + Q.image_hex(to), "from_worker_code_fingerprint": ffp,
            "to_worker_code_fingerprint": tfp, "date_utc": "2026-11-27T00:10:00Z",
            "run_ids": ["qual-1"], "world_sizes": [8], "resume_tested": True,
            "comparisons": {"8": comp}}


def _write(cfg, obj):
    frm, to, _, _ = _pair(cfg)
    Q.QUALIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (Q.QUALIFICATIONS_DIR / Q.record_filename(frm, to)).write_text(json.dumps(obj))


# ── the guard ────────────────────────────────────────────────────────────────


def test_a_bitwise_only_record_no_longer_qualifies_the_after_image(armed):
    _write(armed, _bitwise_only(armed))
    st = Q.qualification_status(armed)
    assert not st.usable, "a record without build manifest, smoke and walls qualified the image"
    for part in ("build", "smoke", "measured_walls"):
        assert part in st.reason, st.reason


def test_full_record_needs_every_role_and_the_measured_walls(armed):
    assert hasattr(Q, "validate_smoke") and hasattr(Q, "validate_walls"), "no /2 sections"
    _, to, _, _ = _pair(armed)
    roles = {r: {"passed": True, "run_id": "q", "gpus": 8, "steps": 30, "final_loss": 6.3,
                 "bf16_autocast": True} for r in Q.FLEET_SMOKE_ROLES}
    smoke = {"image": to, "roles": roles}
    assert Q.validate_smoke(smoke, image=to, roles=Q.FLEET_SMOKE_ROLES) == []
    # the rc4 class: an expert smoke that never ran the bf16 autocast backward
    rc4 = json.loads(json.dumps(smoke))
    rc4["roles"]["lane_b_expert_bf16"]["bf16_autocast"] = False
    assert any("bf16" in b for b in Q.validate_smoke(rc4, image=to, roles=Q.FLEET_SMOKE_ROLES))
    short = json.loads(json.dumps(smoke))
    short["roles"]["lane_a_leg"]["steps"] = 12
    assert any("optimizer steps" in b for b in
               Q.validate_smoke(short, image=to, roles=Q.FLEET_SMOKE_ROLES))
    walls = {"image": to, "skus": {"8xH200": {"gpus": 8, "steps_per_s": 1.9,
                                              "measured_steps": 58, "wall_seconds": 13300,
                                              "run_id": "w"}}}
    assert Q.validate_walls(walls, image=to, skus=SKUS, require_all=False) == []
    assert any("8xH100" in b for b in Q.validate_walls(walls, image=to, skus=SKUS,
                                                       require_all=True))
    pcie = json.loads(json.dumps(walls))
    pcie["skus"]["8xH200"]["gpus"] = 2      # a 2-GPU speed test never stands in for 8x
    assert Q.validate_walls(pcie, image=to, skus=SKUS, require_all=False)
    assert Q.wall_seconds_from_rate(23841, 1.94, 300.0) == 13219


def test_an_override_needs_a_date_and_lapses_when_it_passes(armed, tmp_path):
    text = (REPO / "chain.toml").read_text()
    on = re.sub(r"(?m)^image_after_unqualified_override\s*=.*$",
                "image_after_unqualified_override = true", text)
    p = tmp_path / "chain.toml"
    p.write_text(on)
    try:
        load_chain_config(p)
        refused = ""
    except ValueError as e:
        refused = str(e)
    assert "image_after_unqualified_override_until" in refused, \
        "a dateless override still loads"
    assert hasattr(armed.v2, "image_after_unqualified_override_until")
    until = "2026-11-30T00:00Z"
    over = replace(armed, v2=replace(armed.v2, image_after_unqualified_override=True,
                                     image_after_unqualified_override_until=until))
    before = Q.parse_until(until) - 3600
    assert Q.qualification_status(over, now=before).usable
    lapsed = Q.qualification_status(over, now=Q.parse_until(until) + 1)
    assert not lapsed.usable and "EXPIRED" in lapsed.reason
    try:
        Q.check_override_until("k", "2026-12-31", now=Q.parse_until("2026-11-28"))
        far = ""
    except ValueError as e:
        far = str(e)
    assert "days ahead" in far, "an override a month ahead was accepted"


def test_marketcast_pod_image_is_refused_without_a_record(acfg, tmp_path, monkeypatch):  # noqa: F811
    from cascade.trainer.leg_pins import leg_round, mkt_leg_scope

    monkeypatch.setattr(Q, "QUALIFICATIONS_DIR", tmp_path / "qualifications")
    cfg = load_chain_config(REPO / "chain.toml")          # pins mkt-rc2, never qualified
    assert cfg.marketcast.pod_image
    runner = SimpleNamespace(cfg=cfg)
    try:
        with mkt_leg_scope(runner):
            leg_round(runner)
        err = None
    except Exception as e:  # noqa: BLE001
        err = e
    assert err is not None and "not qualified" in str(err), \
        "a MarketCast leg rented on an unqualified pod_image"
    from cascade.trainer.leg_pins import LegImageMismatch, image_qualification_condition

    assert isinstance(err, LegImageMismatch)              # operator fault, unburned
    assert image_qualification_condition(cfg)["code"] == "mkt_image_unqualified"
    now = Q.parse_until("2026-11-28T12:00Z")
    monkeypatch.setattr(Q.time, "time", lambda: now)
    over = replace(cfg, marketcast=replace(cfg.marketcast,
                                           pod_image_unqualified_override_until="2026-11-30"))
    with mkt_leg_scope(runner := SimpleNamespace(cfg=over)):
        assert leg_round(runner).funded_pod_image == cfg.marketcast.pod_image
    assert image_qualification_condition(over)["code"] == "mkt_image_override"


def test_mkt_record_needs_the_flat_context_forward_bitwise():
    assert hasattr(Q, "validate_mkt_record"), "no MarketCast qualification record"
    img = "sha256:" + "7" * 63 + "8"
    fp = "9" * 63 + "0"
    fwd = "1" * 63 + "2"
    rec = {"schema": Q.MKT_SCHEMA, "to_image": img, "to_worker_code_fingerprint": fp,
           "from_image": "sha256:" + "3" * 63 + "4", "from_forward_sha256": fwd,
           "date_utc": "2026-11-28T12:00:00Z", "run_ids": ["m"],
           "build": {"schema": Q.BUILD_SCHEMA, "builder": Q.BUILDER, "run_id": "1",
                     "image": "ghcr.io/x/cascade-worker@" + img, "source_commit": "a" * 39 + "b",
                     "lock_sha256": "c" * 63 + "d", "lock_frozen": True,
                     "worker_code_fingerprint": fp, "torch": "2.4.1+cu124", "cuda": "12.4",
                     "cuda_arch_list": ["sm_89"]},
           "smoke": {"image": img, "roles": {
               "mkt_finetune": {"passed": True, "run_id": "m", "gpus": 1, "steps": 30,
                                "final_loss": 0.4},
               "mkt_flat_context": {"passed": True, "run_id": "m", "gpus": 1,
                                    "grads_finite": True, "forward_sha256": fwd}}},
           "measured_walls": {"image": img, "skus": {"RTX4090": {
               "gpus": 1, "steps_per_s": 3.0, "measured_steps": 28, "wall_seconds": 3700,
               "run_id": "m"}}}}
    assert Q.validate_mkt_record(rec, to_image=img, to_fingerprint=fp, skus=("RTX4090",)) == []
    rec["from_forward_sha256"] = "5" * 63 + "6"
    assert any("numerics changed" in b for b in
               Q.validate_mkt_record(rec, to_image=img, to_fingerprint=fp, skus=("RTX4090",)))
    rec["build"]["builder"] = "laptop"                     # the hand-built mkt-rc3 class
    assert any("by hand" in b for b in
               Q.validate_mkt_record(rec, to_image=img, to_fingerprint=fp, skus=("RTX4090",)))


# ── the build ────────────────────────────────────────────────────────────────


def _mini_repo(root: Path) -> Path:
    (root / "cascade" / "shared").mkdir(parents=True)
    (root / "cascade" / "__init__.py").write_text("")
    (root / "cascade" / "shared" / "x.py").write_text("X = 1\n")
    (root / "uv.lock").write_text("version = 1\n")
    return root


def test_the_build_manifest_is_the_checkouts_lock_frozen_build(tmp_path):
    wim = _script("worker_image_manifest.py")
    repo = _mini_repo(tmp_path / "repo")
    fp, n = wim.code_fingerprint(repo)
    # the canonical rule of cascade.provision.codeprint, byte for byte
    from cascade.provision.codeprint import canonical_fingerprint

    pairs = [(str(p.relative_to(repo)), hashlib.sha256(p.read_bytes()).hexdigest())
             for p in (repo / "cascade").rglob("*.py")]
    assert (fp, n) == (canonical_fingerprint(pairs), 2)
    probe = {"schema": wim.PROBE_SCHEMA, "torch": "2.4.1+cu124", "cuda": "12.4",
             "cuda_arch_list": ["sm_80", "sm_90"], "python": "3.11.10",
             "worker_code_fingerprint": fp, "lock_sha256": wim.lock_sha256(repo),
             "lock_verified": True}
    image = "ghcr.io/tensorlink-ai/cascade-worker@sha256:" + "e" * 63 + "f"
    man, bad = wim.build_manifest(repo=repo, image=image, probe_doc=probe,
                                  source_commit="a" * 40, run_id="45910848579")
    assert bad == [] and Q.validate_build(man, image=image, fingerprint=fp) == []
    for mutate, why in (({"worker_code_fingerprint": "0" * 64}, "not this source"),
                        ({"lock_verified": False}, "lock-drift marker"),
                        ({"cuda_arch_list": []}, "arch list")):
        _, bad = wim.build_manifest(repo=repo, image=image, probe_doc={**probe, **mutate},
                                    source_commit="a" * 40, run_id="1")
        assert any(why in b for b in bad), (mutate, bad)


def test_publish_worker_emits_the_manifest_without_pulling_the_image():
    wf = (REPO / ".github" / "workflows" / "publish-worker.yml").read_text()
    df = (REPO / "deploy" / "Dockerfile").read_text()
    assert "worker_image_manifest.py emit" in wf and "target: image-probe" in wf
    assert "actions/upload-artifact" in wf and "image-manifest.json" in wf
    assert "worker_image_manifest.py probe" in df
    stages = re.findall(r"(?m)^FROM\s+(\S+)(?:\s+AS\s+(\S+))?", df)
    assert stages[0][1] == "worker" and stages[-1] == ("worker", ""), \
        "the default target must stay the worker image"
    assert ("scratch", "image-probe") in stages
    # the probe runs after the lock-drift guard wrote its marker
    assert df.index("cascade.shared.lockdrift") < df.index("worker_image_manifest.py probe")


# ── the job ──────────────────────────────────────────────────────────────────


def test_the_smoke_verdicts_catch_the_paid_leg_classes():
    sm = _script("image_smoke.py")
    assert hasattr(sm, "role_result"), "no smoke verdict"
    ok_events = [{"t": 10.0 + i, "event": "step"} for i in range(30)] + [
        {"t": 41.0, "event": "train_exit",
         "metrics": {"steps": 30, "final_loss": 6.3, "gpu_n": 8,
                     "expert_training_backend": "compact_rows"}},
        {"t": 41.0, "event": "expert_backward_total", "grad": {"torch.bfloat16": 10}}]
    log = "__CASCADE_RECEIPT__{}"
    good = sm.role_result("lane_b_expert_bf16", run_id="r", rc=0, events=ok_events,
                          log_text=log, gpus_visible=8, t_start=0.0)
    assert good["passed"] and good["bf16_autocast"] and good["steps_per_s"] == pytest.approx(1.0)
    one_gpu = sm.role_result("lane_a_leg", run_id="r", rc=0, events=[
        *ok_events[:-2], {"t": 41.0, "event": "train_exit",
                          "metrics": {"steps": 30, "final_loss": 6.3, "gpu_n": 1}}],
        log_text=log, gpus_visible=8, t_start=0.0)
    assert not one_gpu["passed"] and "ranks" in one_gpu["detail"]   # the rc15 1-of-8 class
    crash = sm.role_result("lane_b_expert_bf16", run_id="r", rc=1, events=[
        {"t": 1.0, "event": "train_error",
         "error": "RuntimeError: expected scalar type Float but found BFloat16"}],
        log_text="", gpus_visible=8, t_start=0.0)
    assert not crash["passed"] and "BFloat16" in crash["detail"]   # the rc4 class
    nan = sm.role_result("mkt_flat_context", run_id="r", rc=0, events=[], log_text="",
                         gpus_visible=1, t_start=0.0,
                         extra={"ok": True, "grads_finite": False, "forward_sha256": "0" * 64})
    assert not nan["passed"]                                         # the lane C NaN class
    assert set(Q.FLEET_SMOKE_ROLES) | set(Q.MKT_SMOKE_ROLES) == set(sm.ROLES)


def test_the_job_plans_one_cheap_pod_and_emits_one_patch(tmp_path):
    drv = _script("qualify_worker_image.py")
    job = drv.load_job(REPO / "deploy" / "qualify" / "worker_image_job.toml")
    assert job["pod"]["provider"] == "lium"                          # GPU = Lium only
    fp = "f" * 63 + "e"
    image = "ghcr.io/tensorlink-ai/cascade-worker@sha256:" + "d" * 63 + "c"
    manifest = {"schema": Q.BUILD_SCHEMA, "builder": Q.BUILDER, "run_id": "1", "image": image,
                "source_commit": "a" * 40, "lock_sha256": "b" * 64, "lock_frozen": True,
                "worker_code_fingerprint": fp, "torch": "2.4.1+cu124", "cuda": "12.4",
                "cuda_arch_list": ["sm_89", "sm_90"]}
    plan = drv.plan(job, REPO / "chain.toml", manifest)
    assert 8 in plan["worlds"] and plan["roles"] == list(Q.FLEET_SMOKE_ROLES)
    assert any("king_ref" in p for p in plan["problems"])           # shipped job: fill per release
    raw = tomllib.loads((REPO / "chain.toml").read_text())
    frm = raw["v2"]["training"]["train_image_digest"] if "training" in raw.get("v2", {}) \
        else raw["training"]["train_image_digest"]
    record = {"schema": Q.SCHEMA, "from_image": frm,
              "measured_walls": {"skus": {"8xH200": {"wall_seconds": 12999}}}}
    patch = drv.config_patch(record, manifest, REPO / "chain.toml")
    for pin in ("train_image_digest_after", "funded_pod_image_after",
                "worker_code_fingerprint_after", "funded_sku_wall_measured_image_after",
                "ops_continuation_image_digest", "prior_train_image_digests",
                "image_after_unqualified_override = false"):
        assert pin in patch, pin
    assert '"8xH200" = 12999' in patch and "seed-epoch" in patch
    lane_a = {"passed": True, "steps_per_s": 1.94, "startup_s": 300.0, "steps": 60,
              "gpus": 8, "run_id": "w"}
    w = drv.walls_entry(sku="8xH200", lane_a=lane_a, training_steps=23841, gpu_name="H200")
    assert w["wall_seconds"] == 13219 and w["measured_steps"] == 58


def test_the_harness_runs_the_candidate_alone_against_a_golden_file():
    h = _script("qualify_pyramid_bitwise.py")
    assert hasattr(h, "harness_terms"), "qualify_pyramid_bitwise.py has no --solo mode"
    w = _script("write_image_qualification.py")
    assert hasattr(w, "build_golden") and hasattr(w, "comparison_rows")


def test_the_operator_checklist_orders_the_release():
    doc = (REPO / "docs" / "OPERATOR.md").read_text()
    assert "## Releasing a worker image" in doc, "no release checklist in docs/OPERATOR.md"
    sec = doc.split("## Releasing a worker image", 1)[1].split("\n## ", 1)[0]
    steps = ["publish-worker", "qualify_worker_image.py pod", "--walls-only",
             "write_image_qualification.py", "qualify_worker_image.py patch",
             "image_epoch_from_block"]
    at = [sec.find(s) for s in steps]
    assert -1 not in at and at == sorted(at), dict(zip(steps, at, strict=True))
