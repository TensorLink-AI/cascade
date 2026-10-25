"""The agent-facing CLI contract: --json documents, exit codes, spend
guardrails, ``cascade iterate`` and ``cascade direct``."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from bittensor.sp_core import Keypair

from cascade.miner import cli
from cascade.miner import direct as direct_mod
from cascade.miner import iterate as iterate_mod
from cascade.miner.output import NEEDS_CONFIRM, OK, REJECTED, USAGE
from tests.unit.test_direct_moe import admit, environment, guard  # noqa: F401  (pytest fixtures)

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts" / "example_generator"


def run_json(capsys, argv):
    rc = cli.main([*argv, "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "cascade-cli/1" and doc["exit_code"] == rc
    assert doc["ok"] == (rc == OK) and (doc["error"] is None) == (rc == OK)
    return rc, doc


def no_network(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("a refused command must not touch the network")

    monkeypatch.setattr("urllib.request.urlopen", refuse)


def test_json_verify_is_one_document_on_stdout(capsys, tmp_path):
    rc, doc = run_json(capsys, ["verify", str(EXAMPLE), "--skip-runtime"])
    assert rc == OK and doc["command"] == "verify" and doc["data"]["ok"] is True

    rc, doc = run_json(capsys, ["verify", str(tmp_path), "--skip-runtime"])
    assert rc == REJECTED and doc["error"]["code"] == "verify_failed"
    assert doc["data"]["failures"][0]["step"] == "repo_layout"


def test_json_flag_goes_before_or_after_the_command(capsys):
    rc = cli.main(["--json", "verify", str(EXAMPLE), "--skip-runtime"])
    assert rc == OK and json.loads(capsys.readouterr().out)["data"]["ok"] is True


def test_json_usage_errors_still_answer_json(capsys):
    rc, doc = run_json(capsys, ["fund", "https://intake.example"])
    assert rc == USAGE and doc["error"]["code"] == "usage"


def test_env_var_turns_on_json(capsys, monkeypatch):
    monkeypatch.setenv("CASCADE_JSON", "1")
    assert cli.main(["verify", str(EXAMPLE), "--skip-runtime"]) == OK
    assert json.loads(capsys.readouterr().out)["command"] == "verify"


def test_fund_refuses_without_yes(capsys, monkeypatch):
    no_network(monkeypatch)
    monkeypatch.setenv("LIUM_API_KEY", "sk-test")
    rc, doc = run_json(capsys, ["fund", "https://intake.example", "--ref", "a/b@sha256:" + "0" * 64,
                                "--wallet-name", "w", "--wallet-hotkey", "h"])
    assert rc == NEEDS_CONFIRM and doc["error"]["code"] == "confirmation_required"


def test_submit_with_a_key_refuses_without_yes(capsys, monkeypatch):
    no_network(monkeypatch)
    monkeypatch.setenv("LIUM_API_KEY", "sk-test")
    rc, doc = run_json(capsys, ["submit", str(EXAMPLE), "https://intake.example",
                                "--wallet-name", "w", "--wallet-hotkey", "h", "--skip-verify"])
    assert rc == NEEDS_CONFIRM and "sk-test" not in json.dumps(doc)


@pytest.mark.parametrize("extra,code", [
    ([], "max_spend_required"),
    (["--max-spend", "5"], "over_max_spend"),
    (["--max-spend", "50"], "confirmation_required"),
])
def test_direct_fund_spend_guardrails(capsys, monkeypatch, extra, code):
    no_network(monkeypatch)
    rc, doc = run_json(capsys, ["direct", "fund", "https://intake.example", "--ref", "a@sha256:b",
                                "--provider", "vast", "--profile", "p", "--world-size", "8",
                                "--spend-cap", "10", "--wallet-name", "w", "--wallet-hotkey", "h",
                                *extra])
    assert rc == NEEDS_CONFIRM and doc["error"]["code"] == code and doc["command"] == "direct fund"


def network_for(environment, guard_digest, **over):  # noqa: F811
    return {"trainer": "T", "seq": 1, "burn_all": False, "guards": [guard_digest],
            "journal": {"bucket": "j"}, "ledger_policy": environment[4], **over}


def test_direct_fund_builds_terms_the_real_intake_admits(environment, guard, monkeypatch, capsys):  # noqa: F811
    from cascade.direct.intake import DirectFundingIntake
    from cascade.shared.direct_moe import digest

    ctl, cohort, _, _, policy, path = environment
    admit(environment, guard)  # the incumbent, so the cohort is live
    network = network_for(environment, digest(guard.to_dict()))
    alice = Keypair.create_from_uri("//Alice")
    state = json.loads(json.dumps(ctl.ledger.state))
    monkeypatch.setattr(direct_mod, "load_direct",
                        lambda args, out: (OK, network, None, state, None))
    monkeypatch.setattr(direct_mod, "_open_wallet", lambda args, out: alice)
    intake = DirectFundingIntake(path, policy, registered=lambda _: True,
                                 revealed=lambda *_: True, funding_authorized=lambda *_: True)
    sent = {}

    class Resp:
        def __init__(self, status, body):
            self.status, self.body = status, body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def urlopen(req, timeout=0):
        sent["envelope"] = json.loads(req.data)
        status, body = intake.fund(sent["envelope"], now=2)
        return Resp(status, body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    ctl.ledger.close()
    argv = ["direct", "fund", "https://intake.example", "--ref", "gen-alice", "--provider", "fake",
            "--profile", "fake-8", "--world-size", "8", "--spend-cap", "100",
            "--max-spend", "100", "--yes", "--wallet-name", "w", "--wallet-hotkey", "h",
            "--owner-hotkey", "O", "--config-bucket", "B", "--offline"]
    monkeypatch.setattr(direct_mod, "PROVIDERS", (*direct_mod.PROVIDERS, "fake"))
    rc, doc = run_json(capsys, argv)
    assert rc == OK, doc
    terms = sent["envelope"]["body"]["terms"]
    assert terms["cohort"] == cohort.key and terms["payer"] == alice.ss58_address
    assert terms["anchor"] == cohort.anchor and terms["role"] == "challenger"
    assert doc["data"]["status"] == "admitted"
    # the default job id is deterministic, so retrying is idempotent
    rc, doc = run_json(capsys, argv)
    assert rc == OK and doc["data"]["job"] == sent["envelope"]["body"]["job"]


def test_direct_build_terms_refusals(environment, guard):  # noqa: F811
    from cascade.shared.direct_moe import digest

    admit(environment, guard)
    state = environment[0].ledger.state
    g = digest(guard.to_dict())
    args = SimpleNamespace(cohort=None, guard=None, role="challenger", reward=0, world_size=8,
                           spend_cap=10, job=None, ref="gen", provider=["vast"],
                           profile=["p"], reference=None)
    job, terms = direct_mod.build_terms(args, network_for(environment, g), state, "me")
    assert terms["guard_contract"] == g and terms["reference"] == terms["anchor"]
    with pytest.raises(ValueError, match="pick one"):
        direct_mod.build_terms(args, network_for(environment, g, guards=[g, "f" * 64]), state, "me")
    with pytest.raises(ValueError, match="only the king"):
        direct_mod.build_terms(SimpleNamespace(**{**vars(args), "role": "incumbent"}),
                               network_for(environment, g), state, "me")


def test_direct_status_reports_a_stale_king(environment, guard):  # noqa: F811
    state = {"king": "K", "king_history": [{"payer": "K", "round": 2}], "anchor": "a",
             "phase": "broad", "cohorts": {}, "advanced": [], "jobs": {}, "spent": 0,
             "reserved_spend": 0, "payout_boundaries": {}, "paid": {},
             "policy": {"spend_pool": 1, "round_origin": 0, "round_blocks": 600, "era_blocks": 100},
             "rounds": {"opened": {"4": "c"}}}
    s = direct_mod.summarise({"trainer": "T", "seq": 3, "burn_all": False, "guards": []}, state,
                             now_block=4 * 600 + 10, hotkey="K")
    assert s["clock"]["round"] == 4 and s["clock"]["cohort_opened"] == "c"
    assert s["king"]["stale"] is True and s["you"]["is_king"] is True
    s = direct_mod.summarise({"trainer": "T", "seq": 3, "burn_all": False, "guards": []}, state,
                             now_block=3 * 600)
    assert s["king"]["stale"] is False


def test_iterate_caches_the_baseline_and_logs_history(capsys, monkeypatch, tmp_path):
    cand, king = tmp_path / "cand", tmp_path / "king"
    shutil.copytree(EXAMPLE, cand)
    shutil.copytree(EXAMPLE, king)
    (king / "config.json").write_text('{"king": true}')
    calls = []

    def fake_score(repo, cfg, *, seed, **kw):
        calls.append((Path(repo).name, seed))
        base = 1.0 if Path(repo).name == "king" else 0.9
        return SimpleNamespace(geomean=base + seed / 100, n_windows=4, corpus_digest="d" * 64,
                               n_series=10, train_seconds=1.0, pool_label="synthetic",
                               init_label="random init")

    monkeypatch.setattr("cascade.miner.score.score_generator", fake_score)
    monkeypatch.setattr("cascade.miner.verify.verify_repo",
                        lambda *a, **k: SimpleNamespace(ok=True, to_dict=lambda: {"ok": True},
                                                        render=lambda: "OK"))
    argv = ["iterate", str(cand), "--baseline", str(king), "--seeds", "0,1",
            "--work-dir", str(tmp_path / "work")]
    rc, doc = run_json(capsys, argv)
    assert rc == OK and len(calls) == 4
    c = doc["data"]["comparison"]
    assert c["beats_baseline"] and c["seed_wins"] == 2 and c["relative_improvement"] > 0.09

    (cand / "config.json").write_text('{"v": 2}')
    rc, doc = run_json(capsys, argv)
    assert rc == OK and len(calls) == 6  # only the changed candidate retrains
    assert doc["data"]["history"]["runs"] == 2
    assert doc["data"]["history"]["best"]["mean"] == pytest.approx(0.905)


def test_iterate_reports_verify_failures_without_scoring(capsys, tmp_path):
    rc, doc = run_json(capsys, ["iterate", str(tmp_path), "--baseline", "none",
                                "--work-dir", str(tmp_path / "w")])
    assert rc == REJECTED and doc["error"]["code"] == "verify_failed"


def test_iterate_pins_live_init_across_seeds_and_invalidates_cache_on_promotion(capsys, monkeypatch, tmp_path):
    cand, king = tmp_path / "candidate", tmp_path / "king"
    shutil.copytree(EXAMPLE, cand)
    shutil.copytree(EXAMPLE, king)
    (king / "config.json").write_text('{"king":true}')
    init = ["cascade/init@sha256:" + "a" * 64]
    calls = []
    monkeypatch.setattr("cascade.miner.training_context.live_context", lambda cfg: (cfg, init[0], {}))

    def score(repo, cfg, *, seed, warm_start, **kwargs):
        calls.append((Path(repo).name, seed, warm_start))
        return SimpleNamespace(geomean=.9 if Path(repo).name == "candidate" else 1.,
                               n_windows=4, corpus_digest="d" * 64, n_series=10,
                               train_seconds=1., pool_label="synthetic", init_label=warm_start)

    monkeypatch.setattr("cascade.miner.score.score_generator", score)
    monkeypatch.setattr("cascade.miner.verify.verify_repo", lambda *a, **k:
                        SimpleNamespace(ok=True, to_dict=lambda: {"ok": True}, render=lambda: "OK"))
    argv = ["iterate", str(cand), "--baseline", str(king), "--seeds", "0,1",
            "--warm-start", "live", "--work-dir", str(tmp_path / "work")]
    rc, _ = run_json(capsys, argv)
    assert rc == OK and len(calls) == 4 and {c[2] for c in calls} == {init[0]}
    rc, _ = run_json(capsys, argv)
    assert rc == OK and len(calls) == 4
    init[0] = "cascade/init@sha256:" + "b" * 64
    rc, doc = run_json(capsys, argv)
    assert rc == OK and len(calls) == 8 and {c[2] for c in calls[4:]} == {init[0]}
    assert doc["data"]["history"]["runs"] == 1


def test_iterate_seed_parsing():
    assert iterate_mod.parse_seeds("0, 1,2") == [0, 1, 2]
    with pytest.raises(ValueError):
        iterate_mod.parse_seeds("1,1")
