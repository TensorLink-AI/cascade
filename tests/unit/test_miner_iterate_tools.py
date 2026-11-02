"""Miner local-iteration tools: ``--pool-hf`` (retired HF eval-pool snapshots),
``--keep-ckpt``, ``cascade bench`` / ``iterate --bench`` (the public bench-gate
preview) and the geometry ``cascade verify`` reports. HTTP and the benchmark
sidecar are faked; nothing here touches the network or loads a model."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.bench_gate import RULE_V1, RULE_V2, SUITES_V2, evaluate_bench_gate
from cascade.eval.benchmarks import extract_bench_scores, item_rows_from_report
from cascade.miner import bench as bench_mod
from cascade.miner import cli
from cascade.miner import hf_pool as hf
from cascade.miner import iterate as iterate_mod
from cascade.miner.output import OK, REJECTED, USAGE
from cascade.shared.hippius import pack_dir_to_tar

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts" / "example_generator"
SNAP_OLD = "2026-10-27-block-9361800"
SNAP_NEW = "2026-10-31-block-9388800"
REVISION = "c0ffee" + "0" * 34


def run_json(capsys, argv):
    rc = cli.main([*argv, "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["exit_code"] == rc
    return rc, doc


# ── fake HF ──────────────────────────────────────────────────────────────────


class _Resp(io.BytesIO):
    def __init__(self, body: bytes, headers=None):
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _pool_tar(tmp_path: Path, n: int = 4, length: int = 200) -> bytes:
    src = tmp_path / "pool-src"
    src.mkdir(exist_ok=True)
    for i in range(n):
        np.save(src / f"s{i}.npy", np.sin(np.arange(length) / (3.0 + i)))
    (src / "metadata.json").write_text(json.dumps({f"s{i}": {"freq": "H"} for i in range(n)}))
    return pack_dir_to_tar(src)


class FakeHF:
    """Serves the tree listing (paginated), POOL_SHA256 and the tar."""

    def __init__(self, tar: bytes, *, sha: str | None = None, folders=None):
        self.tar = tar
        self.sha = sha or hashlib.sha256(tar).hexdigest()
        self.folders = folders if folders is not None else [
            "2026-10-30", SNAP_OLD, "2026-11-01", SNAP_NEW]
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        if url.endswith("/revision/main"):
            return _Resp(json.dumps({"sha": REVISION}).encode())
        if "/api/datasets/" in url:
            assert f"/tree/{REVISION}/" in url  # the listing is pinned to one commit
            rows = [{"type": "directory", "path": f"snapshots/{f}"} for f in self.folders]
            if "page=2" in url:
                return _Resp(json.dumps(rows[2:]).encode())
            nxt = url + "?cursor=x&page=2"
            return _Resp(json.dumps(rows[:2]).encode(), {"Link": f'<{nxt}>; rel="next"'})
        assert f"/resolve/{REVISION}/" in url  # downloads pinned to the same commit
        if url.endswith(hf.SHA_MARKER):
            return _Resp((self.sha + "\n").encode())
        if url.endswith(".tar"):
            return _Resp(self.tar)
        raise AssertionError(f"unexpected url {url}")


def offline(*_a, **_k):
    raise OSError("network is unreachable")


def test_latest_never_picks_a_legacy_folder():
    assert hf.pick_latest(["2026-12-31", SNAP_OLD, "snapshots/2026-11-30", SNAP_NEW]) == SNAP_NEW
    with pytest.raises(hf.HfPoolError):
        hf.pick_latest(["2026-07-23", "2026-08-01"])


def test_spec_validation_refuses_legacy_and_swallowed_paths():
    assert hf.validate_spec("latest") == hf.LATEST
    assert hf.validate_spec(f"snapshots/{SNAP_NEW}/") == SNAP_NEW
    # full-width / Arabic-Indic digits are Unicode \\d but never a snapshot
    for bad in ("2026-07-23", "./my-gen", "block-9388800",
                "\uff12\uff10\uff12\uff16-10-31-block-1", "2026-10-31-block-\u0661"):
        with pytest.raises(hf.HfPoolError):
            hf.validate_spec(bad)


def test_latest_follows_pagination_verifies_and_extracts_from_tar(tmp_path, monkeypatch):
    fake = FakeHF(_pool_tar(tmp_path))
    monkeypatch.setattr(hf, "_open", fake)
    pool = hf.resolve_hf_pool("latest", tmp_path / "work")
    assert pool.snapshot == SNAP_NEW and pool.block == 9388800 and pool.sha256 == fake.sha
    assert sorted(p.name for p in pool.path.glob("*.npy")) == [f"s{i}.npy" for i in range(4)]
    assert pool.path == tmp_path / "work" / "hf-pool" / fake.sha / "pool"
    assert pool.revision == REVISION and pool.as_dict()["revision"] == REVISION
    # the verified tar is deleted after extraction: one copy on disk
    assert not list((tmp_path / "work" / "hf-pool").rglob("*.tar"))
    assert any(u.endswith(f"{SNAP_NEW}/block-9388800.tar") for u in fake.calls)
    # only the verified tar is downloaded, never the unpacked convenience copies
    assert not any(u.endswith(".npy") for u in fake.calls)


def test_sha_mismatch_refuses_and_caches_nothing(tmp_path, monkeypatch):
    fake = FakeHF(_pool_tar(tmp_path), sha="0" * 64)
    monkeypatch.setattr(hf, "_open", fake)
    with pytest.raises(hf.HfPoolIntegrityError):
        hf.resolve_hf_pool(SNAP_NEW, tmp_path / "work")
    cache = tmp_path / "work" / "hf-pool"
    assert not list(cache.glob(f"*/{hf.COMPLETE_MARKER}"))
    assert not list(cache.glob(".incoming-*"))


def test_cache_hit_needs_no_network(tmp_path, monkeypatch):
    fake = FakeHF(_pool_tar(tmp_path))
    monkeypatch.setattr(hf, "_open", fake)
    first = hf.resolve_hf_pool(SNAP_NEW, tmp_path / "work")
    monkeypatch.setattr(hf, "_open", offline)
    assert hf.resolve_hf_pool(SNAP_NEW, tmp_path / "work") == first
    warnings = []
    assert hf.resolve_hf_pool("latest", tmp_path / "work", warn=warnings.append) == first
    assert warnings and "cached" in warnings[0]
    with pytest.raises(hf.HfPoolError):
        hf.resolve_hf_pool("latest", tmp_path / "empty")


def test_hf_pool_scores_through_the_validator_loader(cfg, tmp_path, monkeypatch):
    from cascade.miner import score as score_mod

    monkeypatch.setattr(hf, "_open", FakeHF(_pool_tar(tmp_path)))
    pool = hf.resolve_hf_pool("latest", tmp_path / "work")
    windows, label = score_mod._load_pool_windows(
        cfg, pool_dir=None, pool_ref="", n_windows=3, seed=0, cache_dir=tmp_path, pool_hf=pool)
    assert 0 < len(windows) <= 3 and SNAP_NEW in label and "retired" in label


# ── iterate settings / cache key ─────────────────────────────────────────────


def _args(**kw):
    base = dict(pool_dir=None, pool_ref="", train_hours=0.01, n_windows=4, device="cpu",
                warm_start=None, _cache_context=None, _pool_stamp=None)
    return SimpleNamespace(**{**base, **kw})


def test_settings_key_changes_with_snapshot_and_keeps_old_keys():
    plain = iterate_mod.settings_key(_args(), 0)
    assert "pool_hf" not in plain  # caches of other pools keep their keys
    a = iterate_mod.settings_key(_args(_pool_hf=SimpleNamespace(snapshot=SNAP_OLD, sha256="a" * 64)), 0)
    b = iterate_mod.settings_key(_args(_pool_hf=SimpleNamespace(snapshot=SNAP_NEW, sha256="b" * 64)), 0)
    assert a != b != plain
    assert iterate_mod.ScoreCache.key("d", a) != iterate_mod.ScoreCache.key("d", b)


# ── --keep-ckpt ──────────────────────────────────────────────────────────────


def test_keep_ckpt_keeps_the_trained_checkpoint(cfg, tmp_path, monkeypatch):
    from cascade.miner import score as score_mod
    from tests.unit.test_miner_score import _stub_train_eval

    _stub_train_eval(monkeypatch)
    keep = tmp_path / "kept"
    r = score_mod.score_generator(EXAMPLE, cfg, n_windows=3, cache_dir=tmp_path / "w",
                                  keep_ckpt=keep)
    assert (keep / "model.safetensors").is_file() and r.checkpoint_dir == str(keep)
    with pytest.raises(ValueError, match="not an empty directory"):
        score_mod.score_generator(EXAMPLE, cfg, n_windows=3, cache_dir=tmp_path / "w",
                                  keep_ckpt=keep)
    r = score_mod.score_generator(EXAMPLE, cfg, n_windows=3, cache_dir=tmp_path / "w")
    assert r.checkpoint_dir == ""


# ── bench verdict ────────────────────────────────────────────────────────────


def _report(scale: dict[str, float] | None = None, *, n: int = 40, seed: int = 0,
            skip=(), suites=SUITES_V2) -> dict:
    rng = np.random.default_rng(seed)
    out = []
    for key in suites:
        name = bench_mod.SIDECAR_SUITES[key]
        if key in skip:
            out.append({"suite": name, "status": "skipped", "metrics": {}, "n_series": 0,
                        "detail": "data not staged"})
            continue
        k = (scale or {}).get(key, 1.0)
        base = rng.uniform(0.5, 1.5, size=(n, 2))
        rows = [{"full": f"{key}/{i}", "CRPS": float(c * k), "MASE": float(m * k)}
                for i, (c, m) in enumerate(base)]
        out.append({"suite": name, "status": "ok", "n_series": n, "n_expected": n,
                    "metrics": {"crps": float(base[:, 0].mean() * k),
                                "mase": float(base[:, 1].mean() * k)},
                    "rows": rows, "detail": ""})
    return {"checkpoint": "/ckpt", "suites": out}


SETTINGS_V2 = {"rule": RULE_V2, "suites": list(SUITES_V2), "tolerance": 0.01,
               "min_win_rate": 0.5, "min_items": 10, "alpha": 0.05, "B": 400}


@pytest.mark.parametrize("scale,expect", [(0.95, "pass"), (1.05, "block")])
def test_verdict_is_the_gate_function_on_fixture_numbers(scale, expect):
    king = _report(seed=1)
    chal = _report(dict.fromkeys(SUITES_V2, scale), seed=1)
    v = bench_mod.judge(king, chal, SETTINGS_V2)
    gate = evaluate_bench_gate(
        item_rows_from_report(king), item_rows_from_report(chal),
        extract_bench_scores(king, require_fev=True), extract_bench_scores(chal, require_fev=True),
        alpha=0.05, B=400, seed=bench_mod.LOCAL_SEED, min_items=10, rule=RULE_V2,
        tolerance=0.01, min_win_rate=0.5)
    assert v["verdict"] == expect and v["would_block"] is (expect == "block")
    assert v["gate"] == gate.detail() and v["reason"] == gate.reason
    assert not v["partial"]
    assert any(expect.upper() in line for line in bench_mod.render(v))


def test_unstaged_suite_makes_the_verdict_partial():
    king = _report(seed=2, skip=("fev",))
    chal = _report(dict.fromkeys(SUITES_V2, 0.9), seed=2, skip=("fev",))
    v = bench_mod.judge(king, chal, SETTINGS_V2)
    assert v["verdict"] == "uncomputable" and v["would_block"] is None and v["partial"]
    assert set(v["suites_not_ok"]) == {"fev"}
    assert set(v["staged_preview"]) == {"gift", "boom", "time"}
    assert v["staged_preview"]["gift"]["ratio"] == pytest.approx(0.9)
    assert any("PARTIAL" in line for line in bench_mod.render(v))


def test_gate_settings_read_the_config(cfg):
    from dataclasses import replace

    s = bench_mod.gate_settings(replace(cfg, scoring=replace(
        cfg.scoring, bench_gate_tolerance=0.03, bench_gate_min_win_rate=0.6)))
    assert s["tolerance"] == 0.03 and s["min_win_rate"] == 0.6
    assert s["configured_mode"]["scoring"] == cfg.scoring.bench_gate_mode
    assert "v2" in s["configured_mode"]
    v = bench_mod.judge(_report(seed=5), _report(seed=5), {**SETTINGS_V2, "configured_mode": s["configured_mode"]})
    assert any("IGNORED" in line and cfg.scoring.bench_gate_mode in line for line in bench_mod.render(v))
    assert s["rule"] in (RULE_V1, RULE_V2)
    assert s["suites"] == (list(SUITES_V2) if s["rule"] == RULE_V2 else ["gift", "boom", "time"])


def test_missing_sidecar_env_is_actionable(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setattr(bench_mod.shutil, "which", lambda _n: "/usr/bin/uv")
    fatal, _ = bench_mod.sidecar_problems(tmp_path, ["gift", "time", "fev"])
    assert "uv sync --project benchmarks --frozen --extra time --extra fev" in fatal
    (tmp_path / ".venv").mkdir()
    fatal, warnings = bench_mod.sidecar_problems(tmp_path, ["gift", "time", "fev"])
    assert fatal is None and len(warnings) == 2
    assert any("--extra time" in w for w in warnings) and any("--extra fev" in w for w in warnings)


# ── the CLI wiring ───────────────────────────────────────────────────────────


def _fake_scoring(monkeypatch, calls):
    def fake_score(repo, cfg, *, seed, keep_ckpt=None, pool_hf=None, **kw):
        calls.append({"repo": Path(repo).name, "seed": seed, "keep": keep_ckpt, "pool_hf": pool_hf})
        if keep_ckpt is not None:
            Path(keep_ckpt).mkdir(parents=True)
            (Path(keep_ckpt) / "forecast_wrapper.py").write_text("")
        base = 1.0 if Path(repo).name == "king" else 0.9
        return SimpleNamespace(
            geomean=base + seed / 100, n_windows=4, corpus_digest="d" * 64, n_series=10,
            train_seconds=1.0, pool_label=pool_hf.label if pool_hf else "synthetic",
            init_label="random init", pool_snapshot=pool_hf.snapshot if pool_hf else "",
            pool_sha256=pool_hf.sha256 if pool_hf else "",
            checkpoint_dir=str(keep_ckpt) if keep_ckpt is not None else "")

    monkeypatch.setattr("cascade.miner.score.score_generator", fake_score)
    monkeypatch.setattr("cascade.miner.verify.verify_repo",
                        lambda *a, **k: SimpleNamespace(ok=True, to_dict=lambda: {"ok": True},
                                                        render=lambda: "OK"))


def _repos(tmp_path):
    cand, king = tmp_path / "cand", tmp_path / "king"
    shutil.copytree(EXAMPLE, cand)
    shutil.copytree(EXAMPLE, king)
    (king / "config.json").write_text('{"king": true}')
    return cand, king


def test_iterate_bench_uses_kept_checkpoints_and_the_gate(capsys, monkeypatch, tmp_path):
    cand, king = _repos(tmp_path)
    calls, benched = [], []
    _fake_scoring(monkeypatch, calls)
    monkeypatch.setattr(bench_mod, "sidecar_problems", lambda *a: (None, []))
    monkeypatch.setattr(bench_mod, "_guard", lambda c: None)

    def fake_reports(ckpts, **kw):
        benched.append([str(c) for c in ckpts])
        suites = [k for k, v in bench_mod.SIDECAR_SUITES.items() if v in kw["suites"]]
        return [_report(dict.fromkeys(suites, 0.95), seed=3, suites=suites),
                _report(seed=3, suites=suites)]

    monkeypatch.setattr(bench_mod, "_run_reports", fake_reports)
    argv = ["iterate", str(cand), "--baseline", str(king), "--seeds", "0,1", "--bench",
            "--work-dir", str(tmp_path / "work")]
    rc, doc = run_json(capsys, argv)
    assert rc == OK, doc
    b = doc["data"]["bench"]
    assert b["verdict"] == "pass" and b["seed_benched"] == 0
    cand_ckpt, base_ckpt = benched[0]
    assert Path(cand_ckpt).is_dir() and Path(base_ckpt).is_dir()
    assert Path(cand_ckpt).parent == tmp_path / "work" / "checkpoints"
    assert Path(cand_ckpt).name.startswith("candidate-") and Path(base_ckpt).name.startswith("baseline-")
    history = (tmp_path / "work" / "history.jsonl").read_text().splitlines()
    assert json.loads(history[-1])["bench_verdict"] == "pass"
    # unchanged re-run: cached scores AND kept checkpoints, so nothing retrains
    rc, _ = run_json(capsys, argv)
    assert rc == OK and len(calls) == 4


def test_iterate_bench_without_sidecar_fails_before_training(capsys, monkeypatch, tmp_path):
    cand, king = _repos(tmp_path)
    calls = []
    _fake_scoring(monkeypatch, calls)
    monkeypatch.setattr(bench_mod, "sidecar_problems", lambda *a: ("install it: uv sync ...", []))
    rc, doc = run_json(capsys, ["iterate", str(cand), "--baseline", str(king), "--bench",
                                "--work-dir", str(tmp_path / "w")])
    assert rc == USAGE and doc["error"]["code"] == "bench_unavailable" and calls == []
    rc, doc = run_json(capsys, ["iterate", str(cand), "--baseline", "none", "--bench",
                                "--work-dir", str(tmp_path / "w")])
    assert rc == USAGE and doc["error"]["code"] == "bench_needs_baseline"


def test_iterate_keep_ckpt_dir_exists_after_run(capsys, monkeypatch, tmp_path):
    cand, _ = _repos(tmp_path)
    calls = []
    _fake_scoring(monkeypatch, calls)
    keep = tmp_path / "kept"
    rc, doc = run_json(capsys, ["iterate", str(cand), "--baseline", "none", "--keep-ckpt",
                                str(keep), "--work-dir", str(tmp_path / "w")])
    assert rc == OK
    kept = [Path(c["keep"]) for c in calls]
    assert kept and all(k.is_dir() and k.parent == keep for k in kept)


def test_iterate_pool_hf_records_snapshot_and_rekeys_cache(capsys, monkeypatch, tmp_path):
    cand, king = _repos(tmp_path)
    calls = []
    _fake_scoring(monkeypatch, calls)
    snap = [SNAP_OLD]

    def fake_resolve(spec, cache_root, warn=None):
        assert spec == "latest"
        return hf.HfPool(snapshot=snap[0], sha256=hashlib.sha256(snap[0].encode()).hexdigest(),
                         block=int(snap[0].rsplit("-", 1)[1]), path=tmp_path)

    monkeypatch.setattr(hf, "resolve_hf_pool", fake_resolve)
    argv = ["iterate", str(cand), "--baseline", str(king), "--pool-hf",
            "--work-dir", str(tmp_path / "work")]
    rc, doc = run_json(capsys, argv)
    assert rc == OK and len(calls) == 2 and all(c["pool_hf"].snapshot == SNAP_OLD for c in calls)
    assert doc["data"]["pool_hf"]["snapshot"] == SNAP_OLD
    line = json.loads((tmp_path / "work" / "history.jsonl").read_text().splitlines()[-1])
    assert line["pool_hf"]["sha256"] == doc["data"]["pool_hf"]["sha256"]
    rc, _ = run_json(capsys, argv)
    assert rc == OK and len(calls) == 2           # same snapshot: cached
    snap[0] = SNAP_NEW
    rc, doc = run_json(capsys, argv)
    assert rc == OK and len(calls) == 4           # new snapshot: both sides retrain
    assert doc["data"]["history"]["runs"] == 1    # history compares within a snapshot


def test_pool_flags_are_mutually_exclusive(capsys, tmp_path):
    rc, doc = run_json(capsys, ["score", str(tmp_path), "--pool-hf", "--pool-dir", str(tmp_path)])
    assert rc == USAGE


def test_pool_hf_integrity_failure_is_rejected(capsys, monkeypatch, tmp_path):
    cand, _ = _repos(tmp_path)
    _fake_scoring(monkeypatch, [])

    def bad(*a, **k):
        raise hf.HfPoolIntegrityError("sha mismatch")

    monkeypatch.setattr(hf, "resolve_hf_pool", bad)
    rc, doc = run_json(capsys, ["iterate", str(cand), "--baseline", "none", "--pool-hf",
                                "--work-dir", str(tmp_path / "w")])
    assert rc == REJECTED and doc["error"]["code"] == "pool_hf_integrity"


def test_bench_command_numbers_only_and_verdict(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(bench_mod, "sidecar_problems", lambda *a: (None, []))
    monkeypatch.setattr(bench_mod, "_guard", lambda c: None)

    def fake_reports(ckpts, **kw):
        suites = [k for k, v in bench_mod.SIDECAR_SUITES.items() if v in kw["suites"]]
        out = [_report(dict.fromkeys(suites, 1.05), seed=4, suites=suites)]
        return out + ([_report(seed=4, suites=suites)] if len(ckpts) > 1 else [])

    monkeypatch.setattr(bench_mod, "_run_reports", fake_reports)
    rc, doc = run_json(capsys, ["bench", str(tmp_path / "c")])
    assert rc == OK and "bench_gate" not in doc["data"]
    rc, doc = run_json(capsys, ["bench", str(tmp_path / "c"), "--king", str(tmp_path / "k")])
    assert rc == OK and doc["data"]["bench_gate"]["verdict"] == "block"


# ── verify reports the geometry it checks ────────────────────────────────────




def _hostile_tar(tmp_path: Path, kind: str) -> bytes:
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        ok = io.BytesIO(b"x")
        info = tarfile.TarInfo("ok.npy")
        info.size = 1
        tar.addfile(info, ok)
        if kind == "symlink":
            link = tarfile.TarInfo("evil.npy")
            link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(link)
        else:
            esc = tarfile.TarInfo("../escape.npy")
            esc.size = 1
            tar.addfile(esc, io.BytesIO(b"y"))
    return buf.getvalue()


@pytest.mark.parametrize("kind", ["symlink", "traversal"])
def test_hostile_tar_is_refused_and_nothing_is_cached(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(hf, "_open", FakeHF(_hostile_tar(tmp_path, kind)))
    with pytest.raises(hf.HfPoolError, match="unsafe tar member"):
        hf.resolve_hf_pool(SNAP_NEW, tmp_path / "work")
    cache = tmp_path / "work" / "hf-pool"
    assert not list(cache.glob(f"*/{hf.COMPLETE_MARKER}")) and not list(cache.glob(".incoming-*"))
    assert not (tmp_path / "work" / "escape.npy").exists()


def test_corrupt_tar_with_matching_sha_is_a_clean_pool_error(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "_open", FakeHF(b"not a tar at all" * 64))
    with pytest.raises(hf.HfPoolError) as e:
        hf.resolve_hf_pool(SNAP_NEW, tmp_path / "work")
    assert not isinstance(e.value, hf.HfPoolIntegrityError)


# ── verify reports the geometry it checks (CLI layer, not verify_repo) ───────


def test_verify_cli_reports_effective_geometry(capsys):
    from cascade.miner.geometry import geometry
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(Path(__file__).resolve().parents[2] / "chain.toml")
    g = geometry(cfg, True)
    assert g["max_channels"] == cfg.generator.max_channels
    v2 = cfg.training.v2_contract
    if v2 is not None and cfg.generator.min_channels == cfg.generator.max_channels:
        assert g["min_channels"] == g["max_channels"]  # the v2 width rule is checked
    if v2 is not None and v2.batch_denomination == "sequences":
        row = next(t for t in g["training"] if t["contract"] == "v2")
        assert row["series_per_step"] == max(1, v2.batch_size // cfg.generator.max_channels)
        assert row["long_series"] == v2.long_series
    rc, doc = run_json(capsys, ["verify", str(EXAMPLE), "--skip-runtime"])
    assert doc["data"]["geometry"] == g


def test_verify_repo_report_is_unchanged_by_geometry():
    """The trainer's baseline qualification consumes verify_repo; the display
    geometry must not ride on its report (launch_preflight hashes verify.py)."""
    from cascade.miner.verify import VerifyReport

    assert "geometry" not in VerifyReport.__dataclass_fields__
