"""`cascade mine` — the optimisation loop, the tune proposer, the agent
proposer's plumbing and the mine-ui API, with training/scoring mocked."""

from __future__ import annotations

import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cascade.miner import optimize as opt
from cascade.miner import ui as ui_mod

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "scripts" / "example_generator"


def _gen(tmp_path: Path, name: str, cfg: dict) -> Path:
    d = tmp_path / name
    shutil.copytree(EXAMPLE, d, dirs_exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _score_by_weight(d: Path, seed: int) -> float:
    """Deterministic fake score: lower when weights.a grows (a clear hill)."""
    cfg = json.loads((d / "config.json").read_text())
    return 1.0 / (1.0 + cfg["weights"]["a"]) + 0.001 * seed


def _loop(tmp_path, **kw) -> opt.OptimizationLoop:
    start = _gen(tmp_path, "start", {"name": "x", "weights": {"a": 0.2, "b": 0.8}, "scale": 1.5})
    king = _gen(tmp_path, "king", {"name": "k", "weights": {"a": 0.5, "b": 0.5}, "scale": 1.0})
    cfg = opt.LoopConfig(workdir=tmp_path / "run", start_dir=start, king_dir=king,
                         iterations=kw.pop("iterations", 12), tune_sigma=0.6, **kw)
    return opt.OptimizationLoop(cfg, score_fn=_score_by_weight,
                                verify_fn=lambda d: (True, "OK"),
                                proposer=opt.TuneProposer(cfg, run_seed=7))


# -- tune proposer ---------------------------------------------------------- #

def test_tunable_paths_mixtures_and_floats_only():
    cfg = {"name": "n", "min_length": 128, "batch_size": 256, "clip": 0.5, "seed_offset": 0.3,
           "family_weights": {"a": 0.1, "b": 0.9}, "flag": True, "nested": {"lam": 2.5, "k": 3}}
    paths = set(opt.tunable_paths(cfg))
    assert ("family_weights", "a") in paths and ("family_weights", "b") in paths
    assert ("clip",) in paths and ("nested", "lam") in paths and ("min_length",) in paths
    assert ("batch_size",) not in paths and ("nested", "k") not in paths     # ints: structural
    assert ("seed_offset",) not in paths and ("flag",) not in paths and ("name",) not in paths


def test_tune_config_renormalises_mixture_and_clamps_lengths():
    cfg = {"min_length": 4000, "max_length": 4096, "w": {"a": 0.25, "b": 0.25, "c": 0.5}}
    for i in range(40):
        new, note = opt.tune_config(cfg, np.random.default_rng(i), sigma=1.0, max_keys=3)
        assert note.startswith("tune: ")
        assert sum(new["w"].values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in new["w"].values())
        assert opt.MIN_SERIES_LEN <= new["min_length"] <= new["max_length"] <= opt.MAX_SERIES_LEN
    assert cfg["w"] == {"a": 0.25, "b": 0.25, "c": 0.5}       # input untouched


def test_tune_is_deterministic_per_iteration():
    cfg = {"w": {"a": 0.3, "b": 0.7}, "x": 1.25}
    a = opt.tune_config(cfg, np.random.default_rng([3, 5]), sigma=0.3, max_keys=2)
    b = opt.tune_config(cfg, np.random.default_rng([3, 5]), sigma=0.3, max_keys=2)
    assert a == b


def test_tune_without_knobs_says_so():
    new, note = opt.tune_config({"name": "x"}, np.random.default_rng(0), sigma=0.3, max_keys=2)
    assert new == {"name": "x"} and "no tunable keys" in note


# -- the loop ---------------------------------------------------------------- #

def test_loop_hill_climbs_and_keeps_best(tmp_path):
    loop = _loop(tmp_path, seeds=(0, 1))
    state = loop.run()
    hist = opt.read_history(tmp_path / "run")
    assert hist[0]["status"] == "baseline" and hist[0]["accepted"]
    assert hist[1]["status"] == "king" and not hist[1]["accepted"]
    assert len(hist) == 2 + 12
    assert state["status"] == "finished"
    # accepted scores are strictly improving; best/ holds the best config
    acc = [h["score"] for h in hist if h["accepted"]]
    assert acc == sorted(acc, reverse=True) and len(acc) >= 2
    best_cfg = json.loads((tmp_path / "run" / "best" / "config.json").read_text())
    assert _score_by_weight(tmp_path / "run" / "best", 0) + 0.0005 == pytest.approx(
        state["best"]["score"])
    assert best_cfg["weights"]["a"] > 0.2
    assert state["beats_king"] is not None
    # per-seed mean
    assert hist[0]["per_seed"] == pytest.approx([1 / 1.2, 1 / 1.2 + 0.001])


def test_loop_resumes_from_workdir(tmp_path):
    _loop(tmp_path, iterations=3).run()
    loop2 = _loop(tmp_path, iterations=2)
    loop2.run()
    hist = opt.read_history(tmp_path / "run")
    assert [h["status"] for h in hist].count("baseline") == 1       # baseline not redone
    assert [h["status"] for h in hist].count("king") == 1
    assert [h["iteration"] for h in hist] == list(range(len(hist)))


def test_rejected_and_erroring_candidates_never_become_best(tmp_path):
    loop = _loop(tmp_path, iterations=4)
    calls = {"n": 0}

    def verify(d):
        calls["n"] += 1
        return (calls["n"] <= 2, "FAIL: blocked_import")   # baseline + king pass, rest fail

    loop._verify_fn = verify
    loop.run()
    hist = opt.read_history(tmp_path / "run")
    assert {h["status"] for h in hist[2:]} == {"rejected"}
    assert not any(h["accepted"] for h in hist[2:])
    assert hist[2]["detail"].startswith("FAIL")

    loop3 = _loop(tmp_path / "b", iterations=2)
    loop3._score_fn = lambda d, s: (_ for _ in ()).throw(RuntimeError("nan loss")) \
        if "candidates/0000" not in str(d) and "king" not in str(d) else 0.5
    loop3.run()
    h3 = opt.read_history(tmp_path / "b" / "run")
    assert [h["status"] for h in h3[2:]] == ["error", "error"]
    assert "nan loss" in h3[2]["detail"]


def test_noop_proposal_is_rejected_without_scoring(tmp_path):
    class Noop:
        name = "noop"

        def propose(self, d, it, hist):
            return "nothing"

    loop = _loop(tmp_path, iterations=2)
    loop.proposer = Noop()
    scored = []
    base = loop._score_fn
    loop._score_fn = lambda d, s: scored.append(d) or base(d, s)
    loop.run()
    hist = opt.read_history(tmp_path / "run")
    assert [h["detail"] for h in hist[2:]] == ["proposal changed nothing"] * 2
    assert len(scored) == 2                                 # baseline + king only


def test_stop_file_stops_between_candidates(tmp_path):
    loop = _loop(tmp_path, iterations=50)
    orig = loop.step

    def step():
        c = orig()
        if c.iteration == 4:
            opt.request_stop(tmp_path / "run")
        return c

    loop.step = step
    state = loop.run()
    assert state["status"] == "stopped"
    assert max(h["iteration"] for h in opt.read_history(tmp_path / "run")) == 4


def test_baseline_failure_fails_the_run(tmp_path):
    loop = _loop(tmp_path)
    loop._verify_fn = lambda d: (False, "FAIL: repo_layout")
    with pytest.raises(RuntimeError, match="baseline"):
        loop.run()
    assert opt.read_state(tmp_path / "run")["status"] == "failed"


def test_auto_submit_gated_on_margin_and_wallet(tmp_path):
    sent = []
    loop = _loop(tmp_path, iterations=15, auto_submit=True, submit_margin=0.5)
    loop._submit_fn = lambda d: sent.append(d) or 0
    st = loop.run()
    assert not sent and st["submit"].startswith("skipped: best does not beat")

    loop2 = _loop(tmp_path / "b", iterations=15, auto_submit=True, submit_margin=-1.0)
    loop2._submit_fn = lambda d: sent.append(d) or 0
    st2 = loop2.run()
    assert not sent and "wallet" in st2["submit"]

    loop3 = _loop(tmp_path / "c", iterations=15, auto_submit=True, submit_margin=-1.0,
                  intake="https://x", wallet_name="w", wallet_hotkey="h")
    loop3._submit_fn = lambda d: sent.append(d) or 0
    st3 = loop3.run()
    assert sent == [tmp_path / "c" / "run" / "best"] and st3["submit"] == "submit exited 0"


def test_no_auto_submit_by_default(tmp_path):
    loop = _loop(tmp_path, iterations=2, submit_margin=-1.0, intake="https://x",
                 wallet_name="w", wallet_hotkey="h")
    loop._submit_fn = lambda d: pytest.fail("submitted without --auto-submit")
    assert "submit" not in loop.run()


# -- agent proposer ----------------------------------------------------------- #

def test_agent_proposer_runs_command_in_candidate_dir(tmp_path):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import json, sys, pathlib\n"
        "prompt = sys.stdin.read()\n"
        "assert 'cascade-mine skill (proposer mode)' in prompt\n"
        "cfg = json.loads(pathlib.Path('config.json').read_text())\n"
        "cfg['weights']['a'] *= 3\n"
        "pathlib.Path('config.json').write_text(json.dumps(cfg))\n"
        f"pathlib.Path('{opt.AGENT_NOTE}').write_text('weights.a x3\\n')\n"
    )
    loop = _loop(tmp_path, iterations=1)
    loop.cfg.agent_cmd = f"{sys.executable} {script}"
    loop.proposer = opt.AgentProposer(loop.cfg, loop)
    loop.run()
    hist = opt.read_history(tmp_path / "run")
    last = hist[-1]
    assert last["note"] == "agent: weights.a x3" and last["accepted"]
    cand = Path(last["dir"])
    assert not (cand / opt.AGENT_NOTE).exists()             # never ships
    assert (cand.parent / f"{cand.name}.note.md").is_file()


def test_agent_failure_is_recorded_not_fatal(tmp_path):
    loop = _loop(tmp_path, iterations=2)
    loop.cfg.agent_cmd = f"{sys.executable} -c \"import sys; sys.exit(3)\""
    loop.proposer = opt.AgentProposer(loop.cfg, loop)
    assert loop.run()["status"] == "finished"
    hist = opt.read_history(tmp_path / "run")
    assert [h["status"] for h in hist[2:]] == ["error", "error"]
    assert "agent exited 3" in hist[2]["detail"]


# -- mine-ui ------------------------------------------------------------------ #

@pytest.fixture
def ui_server(tmp_path):
    app = ui_mod.MineUI(tmp_path / "run", token="tok", require_page_token=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui_mod.make_handler(app))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield app, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _req(url, *, body=None, token="tok", headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if token is not None:
        h["X-Cascade-Token"] = token
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers=h, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_ui_page_embeds_token_and_api_needs_it(ui_server):
    app, base = ui_server
    code, html = _req(base + "/", token=None)
    assert code == 200 and b'const TOKEN = "tok";' in html
    assert _req(base + "/api/status", token=None)[0] == 403
    assert _req(base + "/api/status", token="nope")[0] == 403
    assert _req(base + "/api/start", body={}, token=None)[0] == 403
    code, raw = _req(base + "/api/status")
    doc = json.loads(raw)
    assert code == 200 and doc["running"] is False and doc["history"] == []
    assert set(doc["env"]) >= {"lium_api_key", "gpu", "claude_cli", "wallets"}
    # the key's value is never exposed, only presence
    assert all(isinstance(v, bool) for k, v in doc["env"].items() if k != "wallets")


def test_ui_rejects_foreign_host_on_loopback(ui_server):
    _, base = ui_server
    assert _req(base + "/", token=None, headers={"Host": "evil.example:8765"})[0] == 403
    assert _req(base + "/api/status", headers={"Host": "evil.example"})[0] == 403


def test_ui_page_token_required_off_loopback(tmp_path):
    app = ui_mod.MineUI(tmp_path / "run", token="tok", require_page_token=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui_mod.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert _req(base + "/", token=None)[0] == 401

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        with pytest.raises(urllib.error.HTTPError) as ei:
            opener.open(base + "/?token=tok", timeout=5)
        assert ei.value.code == 302 and "cascade_ui=tok" in ei.value.headers["Set-Cookie"]
        code, _ = _req(base + "/", token=None, headers={"Cookie": "cascade_ui=tok"})
        assert code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ui_start_builds_safe_argv(ui_server, monkeypatch):
    app, base = ui_server
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen["argv"], seen["kw"] = argv, kw
            self.pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(ui_mod.subprocess, "Popen", FakePopen)
    code, raw = _req(base + "/api/start", body={
        "proposer": "tune", "iterations": "5", "pool_dir": "rel/pool", "warm_start": "live",
        "evil": "--auto-submit", "seeds": "0,1", "king": "none"})
    assert code == 200, raw
    argv = seen["argv"]
    assert argv[1:4] == ["-m", "cascade.miner.cli", "mine"]
    assert "--auto-submit" not in argv and "evil" not in " ".join(argv)
    assert argv[argv.index("--iterations") + 1] == "5"
    assert Path(argv[argv.index("--pool-dir") + 1]).is_absolute()
    assert argv[argv.index("--king") + 1] == "none"
    assert seen["kw"]["start_new_session"] is True
    # second start while running → 409
    assert _req(base + "/api/start", body={"proposer": "tune"})[0] == 409
    assert _req(base + "/api/start", body={"proposer": "rm -rf"})[0] == 409


def test_ui_bad_proposer_and_submit_guards(ui_server, monkeypatch):
    app, base = ui_server
    assert _req(base + "/api/start", body={"proposer": "rm -rf"})[0] == 400
    code, raw = _req(base + "/api/submit", body={"intake": "https://x"})
    assert code == 400 and b"SUBMIT" in raw
    code, raw = _req(base + "/api/submit", body={"confirm": "SUBMIT", "intake": "https://x",
                                                  "wallet_name": "w", "wallet_hotkey": "h"})
    assert code == 400 and b"no best" in raw
    best = app.workdir / "best"
    best.mkdir(parents=True)
    (best / "generator.py").write_text("")
    code, raw = _req(base + "/api/submit", body={"confirm": "SUBMIT", "intake": "http://x",
                                                  "wallet_name": "w", "wallet_hotkey": "h"})
    assert code == 400 and b"https" in raw
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen["argv"] = argv

        def poll(self):
            return None

    monkeypatch.setattr(ui_mod.subprocess, "Popen", FakePopen)
    code, _ = _req(base + "/api/submit", body={"confirm": "SUBMIT", "intake": "https://x",
                                                "wallet_name": "w", "wallet_hotkey": "h",
                                                "label": "v1"})
    assert code == 200
    assert seen["argv"][3:6] == ["submit", str(best), "https://x"]
    assert seen["argv"][-2:] == ["--label", "v1"]


def test_ui_stop_writes_stop_file(ui_server):
    app, base = ui_server
    assert _req(base + "/api/stop", body={})[0] == 200
    assert (app.workdir / opt.STOP_FILE).exists()


# -- cmd proposer (bring-your-own strategy) ------------------------------------ #

def test_cmd_proposer_contract_stdin_env_and_note(tmp_path):
    script = tmp_path / "strategy.py"
    script.write_text(
        "import json, os, sys, pathlib\n"
        "ctx = json.load(sys.stdin)\n"
        "assert pathlib.Path.cwd() == pathlib.Path(os.environ['CASCADE_CANDIDATE_DIR'])\n"
        "assert ctx['candidate_dir'] == os.environ['CASCADE_CANDIDATE_DIR']\n"
        "assert int(os.environ['CASCADE_ITERATION']) == ctx['iteration']\n"
        "assert float(os.environ['CASCADE_BEST_SCORE']) == ctx['best']['score']\n"
        "assert ctx['king']['status'] == 'king' and os.environ['CASCADE_KING_SCORE']\n"
        "assert pathlib.Path(os.environ['CASCADE_HISTORY']).is_file()\n"
        "assert len(ctx['history']) >= 2\n"
        "cfg = json.loads(pathlib.Path('config.json').read_text())\n"
        "cfg['weights']['a'] += 0.5\n"
        "pathlib.Path('config.json').write_text(json.dumps(cfg))\n"
        "pathlib.Path(os.environ['CASCADE_NOTE_FILE']).write_text('a += 0.5')\n"
    )
    loop = _loop(tmp_path, iterations=2, proposer="cmd",
                 propose_cmd=f"{sys.executable} {script}")
    loop.proposer = opt.CommandProposer(loop.cfg, loop)
    loop.run()
    hist = opt.read_history(tmp_path / "run")
    assert [h["note"] for h in hist[2:]] == ["cmd: a += 0.5"] * 2
    assert all(h["accepted"] for h in hist[2:])
    assert not (Path(hist[-1]["dir"]) / opt.AGENT_NOTE).exists()


def test_cmd_proposer_falls_back_to_last_stdout_line(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        "import json, pathlib\n"
        "c = json.loads(pathlib.Path('config.json').read_text()); c['weights']['a'] *= 2\n"
        "pathlib.Path('config.json').write_text(json.dumps(c))\n"
        "print('thinking...'); print('doubled a'); print()\n")
    loop = _loop(tmp_path, iterations=1, proposer="cmd",
                 propose_cmd=f"{sys.executable} {{dir}}/../../../s.py")
    loop.proposer = opt.CommandProposer(loop.cfg, loop)
    loop.run()
    assert opt.read_history(tmp_path / "run")[-1]["note"] == "cmd: doubled a"


def test_cmd_proposer_requires_a_command(tmp_path):
    cfg = opt.LoopConfig(workdir=tmp_path / "run", start_dir=EXAMPLE, proposer="cmd")
    with pytest.raises(ValueError, match="--propose-cmd"):
        opt.OptimizationLoop(cfg, score_fn=_score_by_weight, verify_fn=lambda d: (True, ""))


def test_example_strategy_coordinate_search(tmp_path):
    """scripts/example_strategy.py through the real cmd proposer: walks one
    family at a time, never repeats a move from the same parent, and pushes a
    winning move again."""
    strat = REPO / "scripts" / "example_strategy.py"
    loop = _loop(tmp_path, iterations=6, proposer="cmd",
                 propose_cmd=f"{sys.executable} {strat}")
    loop.proposer = opt.CommandProposer(loop.cfg, loop)
    loop.run()
    hist = opt.read_history(tmp_path / "run")
    moves = [h for h in hist[2:]]
    assert all(h["status"] == "scored" for h in moves), [h["detail"] for h in moves]
    assert moves[0]["note"] == "cmd: weights.a x2" and moves[0]["accepted"]
    assert moves[1]["note"] == "cmd: weights.a x2"                 # pushes the winner
    pairs = [(h["parent"], h["note"]) for h in moves]
    assert len(pairs) == len(set(pairs))                            # never repeats
    cfg = json.loads((tmp_path / "run" / "best" / "config.json").read_text())
    assert sum(cfg["weights"].values()) == pytest.approx(1.0)


def test_ui_ralph_start_uses_ralph_subcommand_with_provider(ui_server, monkeypatch):
    app, base = ui_server
    seen = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen["argv"] = argv
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(ui_mod.subprocess, "Popen", FakePopen)
    assert _req(base + "/api/start", body={"proposer": "ralph", "llm_provider": "openai"})[0] == 400
    code, _ = _req(base + "/api/start", body={
        "proposer": "ralph", "llm_provider": "chutes", "llm_model": "org/Coder",
        "agent_max_turns": "40", "llm_key": "sk-should-be-ignored"})
    assert code == 200
    argv = seen["argv"]
    assert argv[3] == "ralph"
    assert argv[argv.index("--llm-provider") + 1] == "chutes"
    assert argv[argv.index("--llm-model") + 1] == "org/Coder"
    assert "sk-should-be-ignored" not in " ".join(argv)           # keys never via the UI
