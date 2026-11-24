"""W4 (plan rev6.18 B5, owner 2026-11-24: lane D's eval host is a 1x RTX PRO
6000 Blackwell): the qualification kit of a scoring environment
(``cascade.eval.lane_d_env_qual``) and the committed ``sm120-torch212``
entry's lock. Contract: README.md "Qualifying a scoring environment (B5)".

No torch: the forecasts are a deterministic numpy stand-in, the W0b junit
and the A9 ``laned_t7/1`` document are built in the test. Every test's first
statement imports a name this feature adds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
PRO6000 = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
SM120 = "sm120-torch212"
LANE_D = {"alpha": 0.05, "bootstrap_B": 200, "d_margin": 0.01, "d_margin_start": 0.03}
HORIZON = 64


def _windows(half: str, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        length = int(rng.integers(200, 600))
        walk = 20.0 + np.cumsum(rng.normal(0, 0.3, length + HORIZON))
        out.append({"series_id": f"{half}-{i}", "window_id": f"w-{half}-{i}",
                    "source": f"{half}-src{i % 25}", "seasonal_period": 1,
                    "history": walk[:length], "target": walk[length:]})
    return out


def _fake_forecast(scale: float):
    from cascade.eval.lane_d_scorer import SCORING_LEVELS

    z = np.array([-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28])
    assert len(z) == len(SCORING_LEVELS)

    def forecast(ckpt, hists, *, context, horizon, device):
        spread = float(json.loads((Path(ckpt) / "config.json").read_text())["spread"])
        rows = []
        for h in hists:
            last = float(h[-min(len(h), context // 64):].mean())
            rows.append(np.full((horizon, 9), last) + spread * z[None, :])
        return np.stack(rows) * scale
    return forecast


def _probe(env, gpu, driver, torch_build):
    def probe(_env):
        return {"env": env, "gpu_model": gpu, "driver": driver, "torch_build": torch_build,
                "fingerprint": hashlib.sha256(gpu.encode()).hexdigest()}
    return probe


def _t7(gpu=PRO6000, n_windows=15000, wall_s=600.0):
    from cascade.eval.lane_d_t7 import LAUNCH_VARIANTS, build_evidence, settlement_budget, variant_row

    rows = [variant_row(host="laned-eval-sm120", gpu=gpu, variant=v, input_len=v, history_min=v,
                        history_max=v, wall_s=wall_s, peak_mem_gb=20.0,
                        peak_mem_source="cuda_max_allocated", budget=settlement_budget(96.0),
                        n_models=10, n_windows=n_windows) for v in LAUNCH_VARIANTS]
    return build_evidence(rows, short_window={"base_context": 4096, "declared_context": 131072,
                                              "n_windows": 100, "max_abs_diff": 0.0},
                          tool={"command": "cascade.eval.lane_d_t7_pod", "git_sha": "a" * 40},
                          torch_version="2.12.1+cu130", source="pod")


def _junit(path: Path, *, fail: str | None = None, drop: str | None = None) -> Path:
    from cascade.eval.lane_d_env_qual import W0B_REQUIRED

    cases = []
    for n in W0B_REQUIRED + ("test_w0b_iterates_exactly_the_scorer_contexts",):
        if n == drop:
            continue
        inner = '<failure message="x"/>' if n == fail else ""
        cases.append(f'<testcase classname="tests.accept.w0b" name="{n}">{inner}</testcase>')
    path.write_text(f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>')
    return path


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    from cascade.eval.lane_d_env_qual import forecast_run, pick_golden, write_golden
    from cascade.shared.config.lane_d import load_scoring_envs

    base = tmp_path_factory.mktemp("qual")
    ws = pick_golden({"live": _windows("live", 520, 1), "public": _windows("public", 510, 2)},
                     horizon=HORIZON, per_half=500, seed="golden-seed")
    golden = base / "golden"
    write_golden(ws, golden, horizon=HORIZON, seed="golden-seed",
                 pools={"live": {"snapshot": "s", "sha256": "0" * 64},
                        "public": {"snapshot": "p", "sha256": "1" * 64}})
    ckpts = {}
    for name, spread in (("king", 0.4), ("house", 0.5), ("entry", 0.3)):
        d = base / "ckpt" / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"spread": spread}))
        (d / "weights.safetensors").write_bytes(name.encode())
        ckpts[name] = d
    envs = load_scoring_envs()
    forecast_run(golden, ckpts, env="cu124-torch241", out_dir=base / "ref",
                 forecast=_fake_forecast(1.0), envs=envs,
                 probe=_probe("cu124-torch241", "NVIDIA H200", "570.133.20",
                              "2.4.1+cu124/cuda-12.4"))
    forecast_run(golden, ckpts, env=SM120, out_dir=base / "cand",
                 forecast=_fake_forecast(1.0 + 1e-6), envs=envs,
                 probe=_probe(SM120, PRO6000, "595.99.02", "2.12.1+cu130/cuda-13.0"))
    t7 = base / "t7.json"
    t7.write_text(json.dumps(_t7()))
    return {"base": base, "golden": golden, "envs": envs, "t7": t7, "ckpts": ckpts}


def _evidence(kit, **kw):
    from cascade.eval.lane_d_env_qual import compare

    junit = kw.pop("junit", None) or _junit(kit["base"] / "w0b.xml")
    return compare(kit["golden"], kit["base"] / "ref", kit["base"] / "cand", lane_d=LANE_D,
                   seed="replay-seed", w0b_junit=junit, a9_t7_path=kw.pop("t7", kit["t7"]))


def _sm120(kit):
    return {e.name: e for e in kit["envs"]}[SM120]


@pytest.fixture(scope="module")
def good(kit):
    return _evidence(kit)


# ── the committed entry ─────────────────────────────────────────────────────


def test_sm120_entry_pins_its_committed_lock_and_the_real_torch_build():
    from cascade.eval.lane_d_env_qual import REFERENCE_ENV
    from cascade.shared.config.lane_d import load_scoring_envs

    envs = {e.name: e for e in load_scoring_envs()}
    assert REFERENCE_ENV in envs
    e = envs[SM120]
    assert PRO6000 in e.gpu_models
    lock = REPO / e.lock
    assert e.lock == "deploy/laned/envs/sm120-torch212/uv.lock" and lock.is_file()
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == e.lock_sha256
    pkgs = {p["name"]: p["version"] for p in tomllib.loads(lock.read_text())["package"]}
    # The PyPI wheel's torch/version.py: __version__ "2.12.1+cu130", cuda "13.0".
    assert pkgs["torch"] == "2.12.1" and e.torch_build == "2.12.1+cu130/cuda-13.0"
    main = {p["name"]: p["version"] for p in tomllib.loads((REPO / "uv.lock").read_text())["package"]}
    assert pkgs["numpy"] == main["numpy"] and pkgs["safetensors"] == main["safetensors"]
    side = tomllib.loads((REPO / "benchmarks" / "uv.lock").read_text())
    assert {p["name"]: p["version"] for p in side["package"]}["torch"] == pkgs["torch"]
    # Lane A's training contract is not this lock.
    assert 'version = "2.4.1+cu124"' in (REPO / "uv.lock").read_text(encoding="utf-8")


def test_every_qualified_non_reference_entry_has_evidence_that_rederives():
    from cascade.eval.lane_d_env_qual import evidence_problem
    from cascade.shared.config.lane_d import load_scoring_envs

    envs = load_scoring_envs()
    for e in envs:
        path = REPO / e.evidence if e.evidence and e.evidence != "reference" else None
        if e.status != "qualified":
            assert path is None or not path.exists() or path.suffix == ".json"
            continue
        if e.evidence == "reference":
            continue
        assert path.is_file(), f"{e.name}: evidence {e.evidence} is committed"
        why = evidence_problem(json.loads(path.read_text()), e, envs=envs)
        assert why is None, f"{e.name}: {why}"


# ── golden set ──────────────────────────────────────────────────────────────


def test_golden_set_is_deterministic_and_tamper_evident(kit, tmp_path):
    from cascade.eval.lane_d_env_qual import QualError, load_golden, pick_golden

    meta, hists, tgts = load_golden(kit["golden"])
    assert [w["half"] for w in meta["windows"]].count("live") == 500
    again = pick_golden({"live": _windows("live", 520, 1), "public": _windows("public", 510, 2)},
                        horizon=HORIZON, per_half=500, seed="golden-seed")
    assert [w["series_id"] for w in again] == [w["series_id"] for w in meta["windows"]]
    with pytest.raises(QualError):
        pick_golden({"live": _windows("live", 10, 1)}, horizon=HORIZON, per_half=500, seed="s")
    bad = tmp_path / "g"
    bad.mkdir()
    m = copy.deepcopy(meta)
    m["windows"][0]["source"] = "other"
    (bad / "golden.json").write_text(json.dumps(m))
    (bad / "golden.npz").write_bytes((kit["golden"] / "golden.npz").read_bytes())
    with pytest.raises(QualError):
        load_golden(bad)


def test_forecast_run_refuses_a_host_that_does_not_run_the_env(kit, tmp_path):
    from cascade.eval.lane_d_env_qual import QualError, forecast_run

    with pytest.raises(QualError):
        forecast_run(kit["golden"], kit["ckpts"], env=SM120, out_dir=tmp_path / "x",
                     forecast=_fake_forecast(1.0), envs=kit["envs"],
                     probe=_probe(SM120, PRO6000, "595.99.02", "2.12.1/cuda-13.0"))


# ── evidence ────────────────────────────────────────────────────────────────


def test_agreeing_runs_qualify(kit, good):
    from cascade.eval.lane_d_env_qual import REL_TOL, evidence_problem

    v = good["verdict"]
    assert v["passed"] and v["crowns_identical"] and v["max_rel_diff"] <= REL_TOL
    assert evidence_problem(json.loads(json.dumps(good)), _sm120(kit), envs=kit["envs"]) is None


def _mut_score(doc, factor):
    doc["scores"]["candidate"]["house@4096"]["crps"][3] *= factor


def _mut_crown(doc):
    for k in doc["scores"]["candidate"]:
        if k.startswith("entry@"):
            for m in ("crps", "mase"):
                doc["scores"]["candidate"][k][m] = [x * 0.5 for x in doc["scores"]["candidate"][k][m]]


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: _mut_score(d, 1.002), "rel. diff"),
    (_mut_crown, "rel. diff"),
    (lambda d: d["candidate"].update(lock_sha256="f" * 64), "another lock"),
    (lambda d: d["candidate"]["host"].update(gpu_model="NVIDIA H200"), "candidate host"),
    (lambda d: d["reference"].update(env="sm120-torch212"), "reference"),
    (lambda d: d["golden"].update(halves=["live"] * len(d["golden"]["halves"])), "public"),
    (lambda d: d.update(contexts=[4096]), "contexts"),
    (lambda d: d["w0b"]["cases"][0].update(status="skipped"), "W0b"),
    (lambda d: d["a9"]["rows"][0].update(n_windows=5000), "A9"),
    (lambda d: d["verdict"].update(passed=True) or d["a9"].update(source="local"), "A9"),
])
def test_any_defect_refuses_and_no_flag_is_trusted(kit, good, mutate, needle):
    from cascade.eval.lane_d_env_qual import evidence_problem

    doc = json.loads(json.dumps(good))
    mutate(doc)
    why = evidence_problem(doc, _sm120(kit), envs=kit["envs"])
    assert why is not None and needle in why, why


def test_a_flipped_crown_inside_tolerance_is_still_refused(kit, good, monkeypatch):
    from cascade.eval import lane_d_env_qual as q

    calls = {"n": 0}
    real = q.replay_crowns

    def flip(scores, *a, **kw):
        out = real(scores, *a, **kw)
        calls["n"] += 1
        if calls["n"] % 2 == 0:       # the candidate's replay
            out["4096"]["d_margin"][0]["cleared"] = not out["4096"]["d_margin"][0]["cleared"]
        return out

    monkeypatch.setattr(q, "replay_crowns", flip)
    why = q.evidence_problem(json.loads(json.dumps(good)), _sm120(kit), envs=kit["envs"])
    assert why is not None and "crown" in why


def test_w0b_must_run_every_required_torch_test(kit, tmp_path):
    from cascade.eval.lane_d_env_qual import W0B_REQUIRED, evidence_problem

    doc = _evidence(kit, junit=_junit(tmp_path / "a.xml", drop=W0B_REQUIRED[0]))
    assert not doc["verdict"]["passed"]
    assert "did not run" in evidence_problem(doc, _sm120(kit), envs=kit["envs"])
    doc = _evidence(kit, junit=_junit(tmp_path / "b.xml", fail=W0B_REQUIRED[1]))
    assert "not passed" in evidence_problem(doc, _sm120(kit), envs=kit["envs"])


def test_a9_must_fit_half_a_settlement_on_that_gpu(kit, tmp_path):
    from cascade.eval.lane_d_env_qual import evidence_problem

    slow = tmp_path / "slow.json"
    slow.write_text(json.dumps(_t7(wall_s=6000.0)))          # 10 models > 1.5 h
    assert "A9" in evidence_problem(_evidence(kit, t7=slow), _sm120(kit), envs=kit["envs"])
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_t7(gpu="NVIDIA H200")))
    assert "gpu" in evidence_problem(_evidence(kit, t7=other), _sm120(kit), envs=kit["envs"])


def test_check_cli_reports_a_missing_evidence_file(capsys):
    from cascade.eval.lane_d_env_qual import main

    assert main(["check", "--env", SM120, "--evidence", "/nonexistent/e.json"]) == 1
