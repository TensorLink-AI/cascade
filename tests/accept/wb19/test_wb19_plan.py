"""WB19: `run` preflight, the dry-run plan record and every `run` refusal.

All in-process: ``tool.main(argv, env=..., fs_root=...)`` with a clean
throwaway fs root, so these run anywhere (the prod box included) and never
touch a GPU, torch or the network. Contract: README.md "`run`".
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    CHAIN_TOML,
    COMBINE_REFUSALS,
    HEX64,
    IMAGE_ENV,
    LEG_SCHEMA,
    PROD_MARKERS,
    RUN_REFUSALS,
    RUN_SCHEMA,
    SECRET,
    SECRET_ENV,
    SELECTION,
    SELECTION_CANON,
    STEPS,
    chain_cfg,
    clean_env,
    contract_block,
    dry_run_argv,
    expected_leg_config_digest,
    fs_root,
    gpu_fixture,
    leg_contract,
    leg_contract_digest,
    load_tool,
    pinned_image,
    refusal_code,
    run_tool,
    write_json,
)


def _dry(tool, tmp_path, capsys, sku="8xH100", *, env=None, root=None, **kw):
    out = kw.pop("out", tmp_path / f"rec-{sku}.json")
    argv = dry_run_argv(tmp_path, sku, out=out, **kw)
    rc, so, se = run_tool(tool, argv, env or clean_env(tmp_path), root or fs_root(tmp_path),
                          capsys)
    return rc, so, se, out


def _refused(tool, tmp_path, capsys, code, sku="8xH100", **kw):
    out = tmp_path / "out" / "rec.json"
    out.parent.mkdir(exist_ok=True)
    before = b'{"preexisting": true}\n'
    out.write_bytes(before)
    rc, so, se, _ = _dry(tool, tmp_path, capsys, sku, out=out, **kw)
    assert rc == 2, f"expected exit 2 ({code}), got {rc}; stderr: {se}"
    assert refusal_code(se) == code, f"expected REFUSED {code}, stderr was: {se!r}"
    assert out.read_bytes() == before, f"a {code} refusal must leave --out byte-identical"
    return se


# ── pinned constants ───────────────────────────────────────────────────────

def test_refusal_code_sets_pinned():
    tool = load_tool()
    assert set(tool.RUN_REFUSALS) == RUN_REFUSALS
    assert set(tool.COMBINE_REFUSALS) == COMBINE_REFUSALS
    assert tool.RUN_SCHEMA == RUN_SCHEMA
    assert tuple(tool.PROD_MARKERS) == PROD_MARKERS


# ── the prod guard ─────────────────────────────────────────────────────────

def test_prod_reasons_clean_host_is_empty(tmp_path):
    tool = load_tool()
    assert tool.prod_reasons({}, fs_root(tmp_path)) == []


@pytest.mark.parametrize("marker", PROD_MARKERS)
def test_prod_reasons_each_marker_trips_and_never_reads_it(tmp_path, marker):
    tool = load_tool()
    root = fs_root(tmp_path)
    p = root / marker
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"CASCADE_OPS_WEBHOOK_URL={SECRET}\n")
    reasons = tool.prod_reasons({}, root)
    assert reasons, f"{marker} present must mark the host as prod"
    assert any(marker in r for r in reasons), reasons
    assert SECRET not in json.dumps(reasons), "the guard must stat markers, never read them"


@pytest.mark.parametrize("role", ["prod", "PROD", " prod "])
def test_prod_reasons_host_role_env(tmp_path, role):
    tool = load_tool()
    assert tool.prod_reasons({"CASCADE_HOST_ROLE": role}, fs_root(tmp_path))
    assert tool.prod_reasons({"CASCADE_HOST_ROLE": "pod"}, fs_root(tmp_path)) == []


def test_run_refuses_prod_host_first(tmp_path, capsys):
    tool = load_tool()
    root = fs_root(tmp_path)
    (root / "root" / "cascade").mkdir(parents=True)
    (root / "root" / "cascade" / "validator_state.db").write_bytes(b"")
    env = clean_env(tmp_path)
    env.pop(IMAGE_ENV)   # wrong image too: prod_host still wins
    _refused(tool, tmp_path, capsys, "prod_host", sku="L40S", env=env, root=root)


def test_run_refuses_prod_role_even_in_dry_run(tmp_path, capsys):
    tool = load_tool()
    _refused(tool, tmp_path, capsys, "prod_host",
             env=clean_env(tmp_path, CASCADE_HOST_ROLE="prod"))


# ── the dry-run plan record ────────────────────────────────────────────────

def test_dry_run_expert_record(tmp_path, capsys):
    tool = load_tool()
    rc, so, se, out = _dry(tool, tmp_path, capsys, "8xH100")
    assert rc == 0, se
    rec = json.loads(out.read_text())
    cfg = chain_cfg()
    block = contract_block(cfg)
    lc = leg_contract(cfg, block, "expert")
    assert rec["schema"] == RUN_SCHEMA and rec["wp"] == "WB19"
    assert rec["source"] == "dry_run" and rec["mode"] == "expert"
    assert rec["sku"] == "8xH100" and rec["contract_block"] == block
    assert rec["image_digest"] == pinned_image(cfg)
    c = rec["contract"]
    assert c["digest"] == leg_contract_digest(cfg, block, "expert")
    assert c["training_steps"] == lc.training_steps
    assert c["max_train_seconds"] == lc.max_train_seconds
    assert c["micro_batches"] == lc.micro_batches
    assert c["batch_size"] == lc.batch_size and c["context_length"] == lc.context_length
    assert c["leg_hours"] == pytest.approx(cfg.lane_b.leg_hours)
    assert rec["selection"] == SELECTION_CANON
    assert rec["steps"] == STEPS
    g = rec["gpu"]
    fx = gpu_fixture("8xH100")
    assert g["name"] == fx["gpus"][0]["name"] and g["count"] == 8
    assert g["driver"] == fx["driver"] and g["cuda"] == fx["cuda"]
    assert g["memory_total_bytes"] == fx["gpus"][0]["memory_total_bytes"]
    for k in ("wall", "steps_per_s", "tokens_per_s", "peak_memory_bytes", "finite", "torch"):
        assert rec[k] is None, f"a dry run measures nothing: {k} must be null"
    assert isinstance(rec["tool"]["git_sha"], str) and isinstance(rec["tool"]["git_dirty"], bool)
    assert HEX64.fullmatch(rec["leg_config_digest"])
    assert rec["body_sha256"] == tool.body_sha256(rec)
    json.dumps(rec, allow_nan=False)


def test_dry_run_stdout_summary(tmp_path, capsys):
    tool = load_tool()
    rc, so, se, out = _dry(tool, tmp_path, capsys, "4xH200")
    assert rc == 0, se
    rec = json.loads(out.read_text())
    last = [ln for ln in so.splitlines() if ln.strip()][-1]
    summary = json.loads(last)
    assert summary["leg_config_digest"] == rec["leg_config_digest"]
    assert summary["contract_digest"] == rec["contract"]["digest"]
    assert summary["sku"] == "4xH200" and summary["steps"] == STEPS


def test_body_sha256_is_canonical_json(tmp_path, capsys):
    tool = load_tool()
    from .helpers import body_sha256

    rc, _, se, out = _dry(tool, tmp_path, capsys)
    assert rc == 0, se
    rec = json.loads(out.read_text())
    assert rec["body_sha256"] == body_sha256(rec)
    assert tool.body_sha256(rec) == body_sha256(rec)


def test_leg_config_digest_formula(tmp_path, capsys):
    tool = load_tool()
    rc, _, se, out = _dry(tool, tmp_path, capsys, "8xH200")
    assert rc == 0, se
    rec = json.loads(out.read_text())
    assert rec["leg_config_digest"] == expected_leg_config_digest(
        mode="expert", sku="8xH200", contract_digest=rec["contract"]["digest"],
        image_digest=rec["image_digest"], steps=STEPS, selection=SELECTION_CANON)
    assert LEG_SCHEMA == "laned_wb19_leg/1"


def test_leg_config_digest_ignores_selection_order(tmp_path, capsys):
    tool = load_tool()
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    shuffled = {"0": [4, 1], "3": [5, 9, 2]}
    rc1, _, e1, o1 = _dry(tool, a, capsys, selection=SELECTION)
    rc2, _, e2, o2 = _dry(tool, b, capsys, selection=shuffled)
    assert rc1 == 0 and rc2 == 0, (e1, e2)
    r1, r2 = json.loads(o1.read_text()), json.loads(o2.read_text())
    assert r1["selection"] == r2["selection"] == SELECTION_CANON
    assert r1["leg_config_digest"] == r2["leg_config_digest"]


@pytest.mark.parametrize("change", [
    {"steps": STEPS + 1},
    {"sku": "8xH200"},
    {"selection": {"0": [1, 4], "3": [2, 5]}},
])
def test_leg_config_digest_moves_with_each_input(tmp_path, capsys, change):
    tool = load_tool()
    base_dir, other_dir = tmp_path / "base", tmp_path / "other"
    base_dir.mkdir()
    other_dir.mkdir()
    rc1, _, e1, o1 = _dry(tool, base_dir, capsys)
    change = dict(change)
    sku = change.pop("sku", "8xH100")
    rc2, _, e2, o2 = _dry(tool, other_dir, capsys, sku, **change)
    assert rc1 == 0 and rc2 == 0, (e1, e2)
    assert (json.loads(o1.read_text())["leg_config_digest"]
            != json.loads(o2.read_text())["leg_config_digest"])


def test_dry_run_records_are_byte_identical(tmp_path, capsys):
    tool = load_tool()
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    rc1, _, e1, o1 = _dry(tool, one, capsys, out=one / "r.json")
    rc2, _, e2, o2 = _dry(tool, two, capsys, out=two / "r.json")
    assert rc1 == 0 and rc2 == 0, (e1, e2)
    assert o1.read_bytes() == o2.read_bytes(), "a dry-run record carries no clock or host noise"


def test_dry_run_full_lane_a_leg(tmp_path, capsys):
    tool = load_tool()
    rc, _, se, out = _dry(tool, tmp_path, capsys, "8xH100", mode="full", steps=None)
    assert rc == 0, se
    rec = json.loads(out.read_text())
    cfg = chain_cfg()
    full = leg_contract(cfg, contract_block(cfg), "full")
    assert rec["mode"] == "full" and rec["selection"] is None
    assert rec["contract"]["digest"] == leg_contract_digest(cfg, contract_block(cfg), "full")
    assert rec["contract"]["leg_hours"] is None
    assert rec["steps"] == full.training_steps, "a full lane A leg runs the contract's steps"
    assert rec["leg_config_digest"] == expected_leg_config_digest(
        mode="full", sku="8xH100", contract_digest=rec["contract"]["digest"],
        image_digest=rec["image_digest"], steps=full.training_steps, selection=None)


def test_full_mode_refuses_steps(tmp_path, capsys):
    tool = load_tool()
    _refused(tool, tmp_path, capsys, "bad_steps", mode="full", steps=100)


@pytest.mark.parametrize("steps", [0, -5])
def test_bad_steps(tmp_path, capsys, steps):
    tool = load_tool()
    _refused(tool, tmp_path, capsys, "bad_steps", steps=steps)


def test_no_micro_batch_override_flag(tmp_path, capsys):
    tool = load_tool()
    out = tmp_path / "r.json"
    argv = dry_run_argv(tmp_path, "8xH100", out=out, extra=["--micro-batches", "4"])
    rc, _, _ = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc != 0 and not out.exists(), "micro-batching is the contract's; no flag may change it"


# ── wrong contract / image ─────────────────────────────────────────────────

def test_expect_contract_digest_match_and_mismatch(tmp_path, capsys):
    tool = load_tool()
    cfg = chain_cfg()
    good = leg_contract_digest(cfg, contract_block(cfg), "expert")
    rc, _, se, _ = _dry(tool, tmp_path, capsys, extra=["--expect-contract-digest", good])
    assert rc == 0, se
    bad = good[::-1]
    _refused(tool, tmp_path, capsys, "wrong_contract", extra=["--expect-contract-digest", bad])


def test_wrong_contract_for_a_stale_chain_toml(tmp_path, capsys):
    """The pod's baked chain.toml (pre-repin) must not pass for the deployed one."""
    tool = load_tool()
    cfg = chain_cfg()
    good = leg_contract_digest(cfg, contract_block(cfg), "expert")
    stale = tmp_path / "chain.stale.toml"
    text = CHAIN_TOML.read_text()
    assert "training_steps = 23841" in text
    stale.write_text(text.replace("training_steps = 23841", "training_steps = 23840", 1))
    argv = dry_run_argv(tmp_path, "8xH100", out=tmp_path / "r.json",
                        extra=["--expect-contract-digest", good])
    argv[argv.index("--chain-toml") + 1] = stale
    rc, _, se = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc == 2 and refusal_code(se) == "wrong_contract", se


@pytest.mark.parametrize("image", [None, "", "latest", "sha256:" + "ab" * 32,
                                   "ghcr.io/x/y@sha256:" + "cd" * 32])
def test_wrong_image(tmp_path, capsys, image):
    tool = load_tool()
    env = clean_env(tmp_path)
    if image is None:
        env.pop(IMAGE_ENV)
    else:
        env[IMAGE_ENV] = image
    _refused(tool, tmp_path, capsys, "wrong_image", env=env)


def test_image_as_full_ref_is_accepted(tmp_path, capsys):
    tool = load_tool()
    env = clean_env(tmp_path)
    env[IMAGE_ENV] = "ghcr.io/tensorlink-ai/cascade-worker:rc@" + pinned_image()
    rc, _, se, out = _dry(tool, tmp_path, capsys, env=env)
    assert rc == 0, se
    assert json.loads(out.read_text())["image_digest"] == pinned_image()


# ── the candidate GPU list ─────────────────────────────────────────────────

def test_every_lane_b_sku_is_a_candidate(tmp_path, capsys):
    tool = load_tool()
    for sku in chain_cfg().lane_b.funded_pod_skus:
        d = tmp_path / sku
        d.mkdir()
        rc, _, se, _ = _dry(tool, d, capsys, sku)
        assert rc == 0, f"{sku}: {se}"


def test_gpu_not_candidate(tmp_path, capsys):
    tool = load_tool()
    _refused(tool, tmp_path, capsys, "gpu_not_candidate", sku="L40S")


def test_owner_candidates_file_adds_gpu_types(tmp_path, capsys):
    tool = load_tool()
    oa8 = write_json(tmp_path / "oa8.json", {"oa": "OA8", "status": "approved",
                                             "gpu_types": ["L40S", "RTX4090"],
                                             "cost_cap_usd": 400})
    rc, _, se, out = _dry(tool, tmp_path, capsys, "L40S", extra=["--candidates", oa8])
    assert rc == 0, se
    assert json.loads(out.read_text())["sku"] == "L40S"
    _refused(tool, tmp_path, capsys, "gpu_not_candidate", sku="L40",
             extra=["--candidates", oa8])


def test_candidate_match_is_case_insensitive_catalog_spelling(tmp_path, capsys):
    tool = load_tool()
    argv = dry_run_argv(tmp_path, "8xH100", out=tmp_path / "r.json")
    argv[argv.index("--sku") + 1] = "8xh100"
    rc, _, se = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc == 0, se
    assert json.loads((tmp_path / "r.json").read_text())["sku"] == "8xH100"


def test_candidate_refusal_precedes_image_check(tmp_path, capsys):
    tool = load_tool()
    env = clean_env(tmp_path)
    env.pop(IMAGE_ENV)
    _refused(tool, tmp_path, capsys, "gpu_not_candidate", sku="L40S", env=env)


# ── the GPU actually present ───────────────────────────────────────────────

@pytest.mark.parametrize("sku,fixture", [
    ("8xH100", gpu_fixture("8xH100", count=4)),
    ("4xH100", gpu_fixture("4xH100", count=8)),
    ("8xH100", gpu_fixture("8xH100", name="NVIDIA H200")),
    ("4xH200", {"gpus": [{"name": "NVIDIA H200", "memory_total_bytes": 1}] * 3
                + [{"name": "NVIDIA H100 80GB HBM3", "memory_total_bytes": 1}],
                "driver": "570.86.15", "cuda": "12.4"}),
    ("8xH100", {"gpus": [], "driver": "570.86.15", "cuda": "12.4"}),
])
def test_gpu_mismatch(tmp_path, capsys, sku, fixture):
    tool = load_tool()
    _refused(tool, tmp_path, capsys, "gpu_mismatch", sku=sku, fixture=fixture)


def test_gpu_name_token_rule(tmp_path, capsys):
    """`L40` is not `L40S`; `RTX4090` matches "NVIDIA GeForce RTX 4090"."""
    tool = load_tool()
    oa8 = write_json(tmp_path / "oa8.json", {"gpu_types": ["L40", "L40S", "RTX4090", "A6000"]})
    cand = ["--candidates", oa8]
    _refused(tool, tmp_path, capsys, "gpu_mismatch", sku="L40",
             fixture=gpu_fixture("L40", name="NVIDIA L40S"), extra=cand)
    _refused(tool, tmp_path, capsys, "gpu_mismatch", sku="L40S",
             fixture=gpu_fixture("L40S", name="NVIDIA L40"), extra=cand)
    for sku in ("RTX4090", "A6000", "L40S", "L40"):
        d = tmp_path / sku
        d.mkdir()
        rc, _, se, _ = _dry(tool, d, capsys, sku, extra=cand)
        assert rc == 0, f"{sku}: {se}"


def test_fixture_not_allowed_outside_dry_run(tmp_path, capsys):
    tool = load_tool()
    out = tmp_path / "r.json"
    argv = [a for a in dry_run_argv(tmp_path, "8xH100", out=out) if a != "--dry-run"]
    rc, _, se = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc == 2 and refusal_code(se) == "fixture_not_allowed", se
    assert not out.exists()


def test_real_run_without_gpu_refuses_before_training(tmp_path, capsys):
    """No nvidia-smi on PATH (CI, the prod box): no_gpu, nothing written, no torch."""
    tool = load_tool()
    import sys

    out = tmp_path / "r.json"
    argv = ["run", "--chain-toml", CHAIN_TOML, "--contract-block", contract_block(),
            "--sku", "8xH100", "--mode", "expert", "--steps", STEPS,
            "--selection-json", json.dumps(SELECTION), "--out", out]
    had_torch = "torch" in sys.modules
    rc, _, se = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc == 2 and refusal_code(se) == "no_gpu", se
    assert not out.exists()
    if not had_torch:
        assert "torch" not in sys.modules, "preflight must refuse before importing torch"


def test_image_refusal_precedes_gpu_probe(tmp_path, capsys):
    tool = load_tool()
    argv = ["run", "--chain-toml", CHAIN_TOML, "--contract-block", contract_block(),
            "--sku", "8xH100", "--mode", "expert", "--steps", STEPS,
            "--selection-json", json.dumps(SELECTION), "--out", tmp_path / "r.json"]
    env = clean_env(tmp_path)
    env[IMAGE_ENV] = "sha256:" + "ef" * 32
    rc, _, se = run_tool(tool, argv, env, fs_root(tmp_path), capsys)
    assert rc == 2 and refusal_code(se) == "wrong_image", se


# ── the expert selection ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "not json", "{}", '{"0": []}', '{"0": [-1]}',
                                 '{"x": [1]}', '{"0": [1, 1]}', '{"0": ["a"]}', "[1, 2]",
                                 '{"0": [1.5]}'])
def test_bad_selection(tmp_path, capsys, raw):
    tool = load_tool()
    out = tmp_path / "out" / "rec.json"
    out.parent.mkdir()
    argv = dry_run_argv(tmp_path, "8xH100", out=out, selection=None)
    if raw is not None:
        argv += ["--selection-json", raw]
    rc, _, se = run_tool(tool, argv, clean_env(tmp_path), fs_root(tmp_path), capsys)
    assert rc == 2 and refusal_code(se) == "bad_selection", (raw, se)
    assert not out.exists()


# ── secrets ────────────────────────────────────────────────────────────────

def test_never_prints_or_records_secrets(tmp_path, capsys):
    tool = load_tool()
    env = clean_env(tmp_path, **{k: f"{SECRET}-{k}" for k in SECRET_ENV})
    rc, so, se, out = _dry(tool, tmp_path, capsys, env=env)
    assert rc == 0, se
    blob = so + se + out.read_text()
    d = tmp_path / "refusal"
    d.mkdir()
    rc2, so2, se2, _ = _dry(tool, d, capsys, env=env,
                            extra=["--expect-contract-digest", "0" * 63 + "1"])
    assert rc2 == 2
    rc3, so3, se3, _ = _dry(tool, d, capsys, "L40S", env=env)
    assert rc3 == 2
    blob += so2 + se2 + so3 + se3
    assert SECRET not in blob, "no secret env value may reach stdout, stderr or a record"
