"""W4 (plan rev6.18 B5 + B7, owner 2026-11-23): qualified scoring
environments and the environment handshake. Contract: README.md "Scoring
environments (rev6.18 B5/B7)".

Lane D may be scored on any GPU running a QUALIFIED scoring environment of
the committed list ``deploy/laned/scoring_envs.toml`` (incl. RTX PRO 6000
Blackwell once its entry is qualified). The eval host's hello carries its
environment; ``check_worker`` verifies code identity AND environment and
refuses an unqualified one. Lane A's pinned training contract is untouched.

Every test's first statement imports a rev6.18 name (absent on main).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GIT_HEAD, PACKAGE_HASH = "a" * 40, "b" * 64
RTX4090, H100, PRO6000 = ("NVIDIA GeForce RTX 4090", "NVIDIA H100 80GB HBM3",
                          "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
CU124 = "2.4.1+cu124/cuda-12.4"


def _env(**kw) -> dict:
    e = {"name": "cu124-torch241", "status": "qualified", "from_block": 0, "until_block": 0,
         "lock": "uv.lock", "lock_sha256": "c" * 64, "torch_build": CU124,
         "gpu_models": [RTX4090, H100], "driver_min": "550.54.14", "driver_max": "",
         "evidence": "reference"}
    e.update(kw)
    return e


def _envs(*entries):
    from cascade.shared.config.lane_d import parse_scoring_envs

    return parse_scoring_envs({"env": [copy.deepcopy(e) for e in (entries or (_env(),))]})


def _hello(**env) -> dict:
    from cascade.eval.lane_d_handshake import HANDSHAKE_SCHEMA

    h = {"schema": HANDSHAKE_SCHEMA, "git_head": GIT_HEAD, "package_hash": PACKAGE_HASH,
         "env": "cu124-torch241", "torch_build": CU124, "gpu_model": RTX4090,
         "driver": "550.127.05", "fingerprint": hashlib.sha256(b"GPU-1").hexdigest()}
    h.update(env)
    return h


def _expected():
    from cascade.eval.lane_d_handshake import CodeIdentity

    return CodeIdentity(git_head=GIT_HEAD, package_hash=PACKAGE_HASH)


# ── B5: the committed list ──────────────────────────────────────────────────


def test_the_committed_list_loads_with_the_launch_entries():
    from cascade.shared.config.lane_d import SCORING_ENVS_FILE, load_scoring_envs

    assert SCORING_ENVS_FILE == "deploy/laned/scoring_envs.toml"
    envs = {e.name: e for e in load_scoring_envs()}
    ref = envs["cu124-torch241"]
    assert ref.status == "qualified" and ref.qualified_at(1) and ref.evidence == "reference"
    assert ref.torch_build == CU124 and RTX4090 in ref.gpu_models
    blackwell = [e for e in envs.values() if PRO6000 in e.gpu_models]
    assert blackwell, "B5 (owner): an RTX PRO 6000 Blackwell environment is listed"
    for e in blackwell:
        assert e.status == "qualified" or not e.qualified_at(10**9), \
            "a Blackwell entry scores only once qualified (evidence + owner merge)"
        assert "2.4.1" not in e.torch_build, "sm_120 cannot run torch 2.4.1+cu124"


def test_a_qualified_entry_pins_its_lock_and_lane_a_is_untouched():
    from cascade.shared.config.lane_d import load_scoring_envs

    for e in load_scoring_envs():
        if e.status != "qualified":
            continue
        lock = REPO / e.lock
        assert lock.is_file(), f"{e.name}: lock {e.lock} is committed"
        assert hashlib.sha256(lock.read_bytes()).hexdigest() == e.lock_sha256, \
            f"{e.name}: lock_sha256 is the committed lock's (re-qualify on a lock change)"
    # Lane A's training contract stays torch 2.4.1+cu124 (never moved by B5).
    assert 'version = "2.4.1+cu124"' in (REPO / "uv.lock").read_text(encoding="utf-8")


@pytest.mark.parametrize("mutate,why", [
    (lambda r: r["env"].append(_env()), "duplicate name"),
    (lambda r: r["env"][0].update(status="maybe"), "unknown status"),
    (lambda r: r["env"][0].update(evidence=""), "qualified without evidence"),
    (lambda r: r["env"][0].update(lock_sha256="xyz"), "qualified without a lock hash"),
    (lambda r: r["env"][0].update(driver_min="r550"), "driver not dotted ints"),
    (lambda r: r["env"][0].update(driver_max="500"), "driver_max below driver_min"),
    (lambda r: r["env"][0].update(gpu_models=[]), "no GPU model"),
    (lambda r: r["env"][0].update(torch_build="2.4.1"), "torch build without cuda"),
    (lambda r: r["env"][0].update(from_block=10, until_block=5), "until before from"),
    (lambda r: r["env"][0].update(extra=1), "unknown key"),
    (lambda r: r.update(env=[]), "empty list"),
])
def test_a_malformed_list_is_refused(mutate, why):
    from cascade.shared.config.lane_d import ScoringEnvError, parse_scoring_envs

    raw = {"env": [_env()]}
    mutate(raw)
    with pytest.raises(ScoringEnvError):
        parse_scoring_envs(raw)
    assert issubclass(ScoringEnvError, ValueError), why


def test_qualification_is_block_scoped():
    (e,) = _envs(_env(from_block=100, until_block=200))
    assert not e.qualified_at(100), "boundary must be > from_block"
    assert e.qualified_at(101) and e.qualified_at(200)
    assert not e.qualified_at(201)
    (c,) = _envs(_env(status="candidate", evidence="", lock="", lock_sha256=""))
    assert not c.qualified_at(10**9), "a candidate never scores"


def test_the_committed_file_is_valid_toml_with_only_env_tables():
    from cascade.shared.config.lane_d import scoring_envs_path

    raw = tomllib.loads(scoring_envs_path().read_text(encoding="utf-8"))
    assert set(raw) == {"env"} and len(raw["env"]) >= 2


# ── B7: the hello carries the environment ───────────────────────────────────


def test_handshake_fields_name_the_environment():
    from cascade.eval.lane_d_handshake import ENV_FIELDS, HANDSHAKE_FIELDS

    assert {"env", "torch_build"} <= set(HANDSHAKE_FIELDS), "plan B7"
    assert set(ENV_FIELDS) == {"env", "torch_build", "gpu_model", "driver", "fingerprint"}


def test_probe_reads_gpu_driver_torch_and_hashes_the_uuid(monkeypatch):
    from cascade.eval.lane_d_handshake import ENV_VAR, probe_environment

    monkeypatch.delenv(ENV_VAR, raising=False)
    smi = lambda: [(RTX4090, "550.127.05", "GPU-1234")]  # noqa: E731
    got = probe_environment(envs=_envs(), smi=smi, torch_build=lambda: CU124)
    assert got == {"env": "cu124-torch241", "torch_build": CU124, "gpu_model": RTX4090,
                   "driver": "550.127.05",
                   "fingerprint": hashlib.sha256(b"GPU-1234").hexdigest()}
    assert "GPU-1234" not in json.dumps(got), "the clear UUID never leaves the host"
    monkeypatch.setenv(ENV_VAR, "named-env")
    assert probe_environment(envs=_envs(), smi=smi, torch_build=lambda: CU124)["env"] == \
        "named-env", "the operator's name wins"
    assert probe_environment("cli-env", envs=_envs(), smi=smi,
                             torch_build=lambda: CU124)["env"] == "cli-env"


def test_probe_without_a_gpu_or_torch_reports_empty_fields(monkeypatch):
    from cascade.eval.lane_d_handshake import ENV_VAR, probe_environment

    monkeypatch.delenv(ENV_VAR, raising=False)
    got = probe_environment(envs=_envs(), smi=lambda: [], torch_build=lambda: "")
    assert got == {"env": "", "torch_build": "", "gpu_model": "", "driver": "",
                   "fingerprint": ""}


def test_worker_hello_carries_the_environment_when_probed(tmp_path):
    from cascade.eval.lane_d_handshake import ENV_FIELDS, worker_hello

    (tmp_path / "cascade").mkdir()
    (tmp_path / "cascade" / "__init__.py").write_text("")
    env = {k: f"v-{k}" for k in ENV_FIELDS}
    h = worker_hello(tmp_path, environment=env)
    assert all(h[k] == env[k] for k in ENV_FIELDS)
    assert set(worker_hello(tmp_path)) == {"schema", "git_head", "package_hash"}


def test_cli_hello_prints_the_probed_environment(monkeypatch, capsys, tmp_path):
    import cascade.eval.lane_d_handshake as hs
    from cascade.eval.lane_d_forecasts import main
    from cascade.eval.lane_d_handshake import probe_environment  # noqa: F401 (rev6.18 name)

    (tmp_path / "cascade").mkdir()
    (tmp_path / "cascade" / "__init__.py").write_text("")
    seen = {}

    def probe(env=None, **_):
        seen["env"] = env
        return {"env": env or "", "torch_build": CU124, "gpu_model": RTX4090,
                "driver": "550.127.05", "fingerprint": "f" * 64}

    monkeypatch.setattr(hs, "probe_environment", probe)
    assert main(["hello", "--root", str(tmp_path), "--env", "cu124-torch241"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert seen["env"] == "cu124-torch241"
    assert out["env"] == "cu124-torch241" and out["gpu_model"] == RTX4090


# ── B7: check_worker verifies and refuses ───────────────────────────────────


def test_check_worker_accepts_a_qualified_host_and_returns_its_record():
    from cascade.eval.lane_d_handshake import check_worker
    from cascade.shared.config.lane_d import HOST_KEYS, host_identity

    ident, host = check_worker(_hello(), envs=_envs(), boundary=1000, expected=_expected())
    assert ident == _expected()
    assert tuple(sorted(host)) == tuple(sorted(HOST_KEYS))
    assert host["identity"] == host_identity(host["env"], host["gpu_model"], host["driver"],
                                             host["torch_build"], host["fingerprint"])
    _, other = check_worker(_hello(fingerprint=hashlib.sha256(b"GPU-2").hexdigest()),
                            envs=_envs(), boundary=1000, expected=_expected())
    assert other["identity"] != host["identity"], "another physical GPU is another host"


@pytest.mark.parametrize("hello_kw,envs_kw,why", [
    ({"env": "nope"}, {}, "env not in the list"),
    ({"gpu_model": PRO6000}, {}, "GPU not allowed in the env"),
    ({"driver": "535.104.05"}, {}, "driver below the floor"),
    ({"driver": "garbage"}, {}, "driver unparseable"),
    ({"torch_build": "2.12.1/cuda-13.0"}, {}, "torch build of another env"),
    ({}, {"status": "candidate", "evidence": "", "lock": "", "lock_sha256": ""},
     "candidate env"),
    ({}, {"from_block": 5000}, "not yet activated at the boundary"),
    ({}, {"until_block": 500}, "retired before the boundary"),
    ({}, {"driver_max": "550.100"}, "driver above the ceiling"),
    ({"env": ""}, {}, "no env named"),
    ({"fingerprint": ""}, {}, "no fingerprint"),
])
def test_check_worker_refuses_an_unqualified_environment(hello_kw, envs_kw, why, caplog):
    from cascade.eval.lane_d_handshake import CODE_FIELDS, HandshakeMismatch, check_worker

    with pytest.raises(HandshakeMismatch) as e:
        check_worker(_hello(**hello_kw), envs=_envs(_env(**envs_kw)), boundary=1000,
                     expected=_expected())
    assert e.value.fields and not set(e.value.fields) & set(CODE_FIELDS), why
    assert any(r.levelno == logging.CRITICAL and "lane_d_handshake_env_refused" in r.getMessage()
               for r in caplog.records), why


def test_check_worker_still_refuses_stale_code_first():
    from cascade.eval.lane_d_handshake import HandshakeMismatch, check_worker

    with pytest.raises(HandshakeMismatch) as e:
        check_worker(_hello(package_hash="d" * 64), envs=_envs(), boundary=1000,
                     expected=_expected())
    assert e.value.fields == ("package_hash",)


def test_a_blackwell_host_scores_once_its_env_is_qualified():
    """Owner 2026-11-23: lane D may be scored on RTX PRO 6000 Blackwell."""
    from cascade.eval.lane_d_handshake import HandshakeMismatch, check_worker

    sm120 = _env(name="sm120-torch212", torch_build="2.12.1/cuda-13.0", gpu_models=[PRO6000],
                 driver_min="580", evidence="evidence/lane_d_env_sm120-torch212.json")
    hello = _hello(env="sm120-torch212", torch_build="2.12.1/cuda-13.0", gpu_model=PRO6000,
                   driver="595.99.02")
    _, host = check_worker(hello, envs=_envs(_env(), sm120), boundary=1000,
                           expected=_expected())
    assert host["gpu_model"] == PRO6000
    cand = dict(sm120, status="candidate")
    with pytest.raises(HandshakeMismatch):
        check_worker(hello, envs=_envs(_env(), cand), boundary=1000, expected=_expected())
