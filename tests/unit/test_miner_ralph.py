"""`cascade ralph` — provider resolution / Claude Code env, the preflight against a
fake Anthropic endpoint, and the Ralph proposer's prompt + notebook plumbing
through the real mine loop (agent and scoring stubbed)."""

from __future__ import annotations

import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cascade.miner import optimize as opt
from cascade.miner import ralph

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "scripts" / "example_generator"


def _cfg(tmp_path, **kw) -> opt.LoopConfig:
    return opt.LoopConfig(workdir=tmp_path / "run", start_dir=EXAMPLE, **kw)


# -- provider ---------------------------------------------------------------- #

def test_anthropic_provider_needs_nothing_and_sets_no_endpoint(tmp_path):
    p = ralph.resolve_provider(_cfg(tmp_path))
    assert p.name == "anthropic" and p.claude_env() == {}
    p2 = ralph.resolve_provider(_cfg(tmp_path, llm_model="claude-x"))
    assert p2.claude_env()["ANTHROPIC_MODEL"] == "claude-x"
    assert "ANTHROPIC_BASE_URL" not in p2.claude_env()


@pytest.mark.parametrize("name,key_env", [("chutes", "CHUTES_API_KEY"),
                                          ("saygm", "SAYGM_API_KEY")])
def test_presets_route_claude_code_to_the_provider(tmp_path, monkeypatch, name, key_env):
    monkeypatch.setenv(key_env, "sk-test")
    p = ralph.resolve_provider(_cfg(tmp_path, llm_provider=name, llm_model="org/Coder-1"))
    env = p.claude_env()
    assert env["ANTHROPIC_BASE_URL"] == ralph.PRESETS[name].base_url
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test" and env["ANTHROPIC_API_KEY"] == ""
    for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
        assert env[var] == "org/Coder-1"                 # no tier falls back to a Claude id
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "sk-test" not in p.describe()                 # the key is never printed


def test_provider_fails_fast_with_actionable_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("CHUTES_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"\$CHUTES_API_KEY is not set"):
        ralph.resolve_provider(_cfg(tmp_path, llm_provider="chutes", llm_model="m"))
    monkeypatch.setenv("CHUTES_API_KEY", "k")
    with pytest.raises(ValueError, match="--llm-model is required"):
        ralph.resolve_provider(_cfg(tmp_path, llm_provider="chutes"))
    with pytest.raises(ValueError, match="--llm-base-url"):
        ralph.resolve_provider(_cfg(tmp_path, llm_provider="custom", llm_model="m"))
    with pytest.raises(ValueError, match="expected one of"):
        ralph.resolve_provider(_cfg(tmp_path, llm_provider="openai"))


def test_custom_provider_and_x_api_key_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc")
    p = ralph.resolve_provider(_cfg(
        tmp_path, llm_provider="custom", llm_model="m", llm_base_url="https://gw.example/anthropic/",
        llm_key_env="MY_KEY", llm_auth="x-api-key"))
    env = p.claude_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example/anthropic"       # trailing / dropped
    assert env["ANTHROPIC_API_KEY"] == "abc" and env["ANTHROPIC_AUTH_TOKEN"] == ""


# -- preflight ----------------------------------------------------------------- #

class _FakeAnthropic(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.mode == "openai":
            out, code = {"choices": [{"message": {"content": "OK"}}]}, 200
        elif self.mode == "404":
            out, code = {"error": "nope"}, 404
        else:
            out, code = {"type": "message", "content": [{"type": "text", "text": "OK"}]}, 200
        raw = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def fake_endpoint(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAnthropic)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _FakeAnthropic.seen = []
    monkeypatch.setattr(ralph.shutil, "which", lambda name: "/usr/bin/claude")
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("mode,ok,needle", [
    ("ok", True, "answered 'OK'"),
    ("404", False, "no Anthropic Messages API"),
    ("openai", False, "OpenAI-compatible only"),
])
def test_preflight_diagnoses_the_endpoint(fake_endpoint, monkeypatch, mode, ok, needle):
    _FakeAnthropic.mode = mode
    monkeypatch.setenv("K", "sk-1")
    p = ralph.Provider("custom", base_url=fake_endpoint, model="org/M", key_env="K")
    got_ok, msg = ralph.preflight(p, timeout=5)
    assert got_ok is ok and needle in msg
    path, headers, body = _FakeAnthropic.seen[-1]
    assert path == "/v1/messages" and body["model"] == "org/M"
    assert headers["authorization"] == "Bearer sk-1"


def test_preflight_requires_the_claude_cli(monkeypatch):
    monkeypatch.setattr(ralph.shutil, "which", lambda name: None)
    ok, msg = ralph.preflight(ralph.Provider("anthropic"))
    assert not ok and "claude" in msg


# -- the Ralph proposer through the real loop ----------------------------------- #

FAKE_CLAUDE = r'''
import os, pathlib, sys
prompt = sys.stdin.read()
log = pathlib.Path(os.environ["CASCADE_FAKE_LOG"])
assert "Ralph: improve this cascade generator" in prompt
assert "This iteration: #" in prompt
assert os.environ["ANTHROPIC_BASE_URL"] == "https://claude.chutes.ai"
assert os.environ["ANTHROPIC_MODEL"] == "org/Coder-1"
assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "sk-test"
assert "--max-turns" in sys.argv and "7" in sys.argv
leaked = [k for k in ("CLAUDE_CODE_OAUTH_TOKEN", "LIUM_API_KEY", "GH_TOKEN", "SAYGM_API_KEY",
                      "CASCADE_UI_TOKEN", "AWS_SECRET_ACCESS_KEY",
                      "CLAUDE_SESSION_INGRESS_TOKEN_FILE") if k in os.environ]
assert not leaked, leaked
cfgdir = pathlib.Path(os.environ["CLAUDE_CONFIG_DIR"])
assert cfgdir.name == ".claude-agent" and not (cfgdir / ".credentials.json").exists()
assert (cfgdir / "skills" / "cascade-mine" / "SKILL.md").is_file()
assert os.environ["PATH"]
nb = pathlib.Path(".ralph-notes.md")
assert nb.is_file()                                  # notebook handed in
notes = nb.read_text()
it = os.environ["CASCADE_ITERATION"]
log.write_text(log.read_text() + f"iter {it} saw_results={'Results log' in notes and '- #' in notes}\n"
               if log.exists() else f"iter {it} saw_results=False\n")
# a CODE change to the generator, not config
g = pathlib.Path("generator.py")
g.write_text(g.read_text().replace("sigma = rng.uniform(0.1, 0.5)",
                                    "sigma = rng.uniform(0.1, 0.5) * 0.9", 1)
             + f"\n# ralph iter {it}\n")
nb.write_text(notes.replace("## Plan / next ideas\n", f"## Plan / next ideas\n- idea from {it}\n", 1))
pathlib.Path(".mine-note.md").write_text(f"shrink AR noise (iter {it})\n")
'''


def _ralph_loop(tmp_path, monkeypatch, iterations=2):
    monkeypatch.setenv("CHUTES_API_KEY", "sk-test")
    # An executable named `claude`: provider routing / env isolation / --max-turns
    # apply to Claude Code only, so the fake must look like it.
    script = tmp_path / "bin" / "claude"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n" + FAKE_CLAUDE)
    script.chmod(0o755)
    monkeypatch.setenv("CASCADE_FAKE_LOG", str(tmp_path / "agent.log"))
    # Secrets that must never reach an agent talking to a third-party provider.
    for k, v in {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-SECRET", "LIUM_API_KEY": "lium",
                 "GH_TOKEN": "gh", "SAYGM_API_KEY": "other", "CASCADE_UI_TOKEN": "ui",
                 "AWS_SECRET_ACCESS_KEY": "aws", "CLAUDE_SESSION_INGRESS_TOKEN_FILE": "/x"}.items():
        monkeypatch.setenv(k, v)
    start = tmp_path / "start"
    shutil.copytree(EXAMPLE, start)
    cfg = opt.LoopConfig(
        workdir=tmp_path / "run", start_dir=start, proposer="ralph", iterations=iterations,
        agent_cmd=str(script), agent_max_turns=7,
        llm_provider="chutes", llm_model="org/Coder-1")
    scores = iter([0.9, 0.8, 0.85, 0.7, 0.75])
    return opt.OptimizationLoop(cfg, score_fn=lambda d, s: next(scores),
                                verify_fn=lambda d: (True, "OK"))


def test_ralph_edits_code_keeps_notebook_and_logs_results(tmp_path, monkeypatch):
    loop = _ralph_loop(tmp_path, monkeypatch, iterations=2)
    assert isinstance(loop.proposer, ralph.RalphProposer)
    state = loop.run()
    run = tmp_path / "run"
    hist = opt.read_history(run)
    assert [h["status"] for h in hist] == ["baseline", "scored", "scored"]
    assert hist[1]["note"] == "ralph: shrink AR noise (iter 1)" and hist[1]["accepted"]
    assert not hist[2]["accepted"]                       # 0.85 > best 0.8
    assert state["best"]["iteration"] == 1
    # the code change is what got kept; loop files never ship
    best_gen = (run / "best" / "generator.py").read_text()
    assert "* 0.9" in best_gen and "# ralph iter 1" in best_gen
    for c in (run / "candidates").iterdir():
        if c.is_dir():
            assert not (c / ralph.CAND_NOTES).exists() and not (c / opt.AGENT_NOTE).exists()
    # notebook: agent edits persisted + loop-appended results; prompt file seeded
    notes = (run / ralph.NOTES_FILE).read_text()
    assert "- idea from 1" in notes and "- idea from 2" in notes
    assert "- #1 ACCEPTED (new best) score 0.80000" in notes
    assert "- #2 scored score 0.85000" in notes
    assert (run / ralph.PROMPT_FILE).read_text() == ralph.DEFAULT_PROMPT
    # iteration 2's fresh context saw iteration 1's outcome via the notebook
    assert (tmp_path / "agent.log").read_text().splitlines()[-1] == "iter 2 saw_results=True"


def test_edited_prompt_file_steers_the_next_iteration(tmp_path, monkeypatch):
    loop = _ralph_loop(tmp_path, monkeypatch, iterations=1)
    loop.run()
    pf = tmp_path / "run" / ralph.PROMPT_FILE
    pf.write_text(ralph.DEFAULT_PROMPT + "\nFOCUS: energy load families only.\n")
    seen = loop.proposer.stdin_for(tmp_path / "x", 9, loop.history)
    assert "FOCUS: energy load families only." in seen and "This iteration: #9" in seen


def test_agent_failure_still_recovers_notebook_and_logs_it(tmp_path, monkeypatch):
    loop = _ralph_loop(tmp_path, monkeypatch, iterations=1)
    loop.proposer.command = (f"{sys.executable} -c \"import pathlib,sys; "
                             "pathlib.Path('.ralph-notes.md').write_text('# kept\\\\n'); sys.exit(4)\"")
    loop.run()
    run = tmp_path / "run"
    h = opt.read_history(run)[-1]
    assert h["status"] == "error" and "ralph exited 4" in h["detail"]
    notes = (run / ralph.NOTES_FILE).read_text()
    assert notes.startswith("# kept") and "- #1 error" in notes and "ralph exited 4" in notes


def test_anthropic_agent_env_keeps_login_drops_unrelated_secrets(tmp_path):
    src = {"PATH": "/bin", "HOME": "/root", "CLAUDE_CODE_OAUTH_TOKEN": "oat",
           "ANTHROPIC_API_KEY": "sk-ant", "LIUM_API_KEY": "lium", "HIPPIUS_HUB_TOKEN": "h",
           "CHUTES_API_KEY": "c", "CASCADE_UI_TOKEN": "ui", "MY_FLAG": "1"}
    env = ralph.agent_base_env(ralph.Provider("anthropic"), tmp_path, environ=src)
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oat" and env["ANTHROPIC_API_KEY"] == "sk-ant"
    assert env["MY_FLAG"] == "1" and env["PATH"] == "/bin"
    for k in ("LIUM_API_KEY", "HIPPIUS_HUB_TOKEN", "CHUTES_API_KEY", "CASCADE_UI_TOKEN"):
        assert k not in env


def test_third_party_agent_env_is_an_allowlist(tmp_path):
    src = {"PATH": "/bin", "HTTPS_PROXY": "http://p", "CLAUDE_CODE_OAUTH_TOKEN": "oat",
           "ANTHROPIC_API_KEY": "sk-ant", "CASCADE_WORKDIR": "/w", "MY_FLAG": "1",
           "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1"}
    p = ralph.Provider("chutes", base_url="https://x", model="m", key_env="K")
    env = ralph.agent_base_env(p, tmp_path, environ=src)
    assert set(env) == {"PATH", "HTTPS_PROXY", "CASCADE_WORKDIR", "CLAUDE_CONFIG_DIR"}
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / ".claude-agent")


def test_sse_stream_counts_as_a_messages_reply():
    raw = (b'event: message_start\ndata: {"type":"message_start","message":{"type":"message",'
           b'"content":[]}}\n\nevent: content_block_start\ndata: {"type":"content_block_start",'
           b'"index":0,"content_block":{"type":"text","text":""}}\n\nevent: content_block_delta\n'
           b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta",'
           b'"text":"OK"}}\n\n')
    doc = ralph._sse_message(raw)
    assert doc["type"] == "message" and doc["content"][0]["text"] == "OK"


def test_engy_and_chutes_presets(monkeypatch):
    monkeypatch.setenv("ENGY_API_KEY", "k")
    p = ralph.resolve_provider(opt.LoopConfig(workdir=Path("x"), start_dir=Path("y"),
                                              llm_provider="engy", llm_model="kimi-k3"))
    env = p.claude_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://api.engy.ai"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "k" and env["ANTHROPIC_MODEL"] == "kimi-k3"
    assert ralph.PRESETS["chutes"].base_url == "https://claude.chutes.ai"
