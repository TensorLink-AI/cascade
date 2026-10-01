"""``cascade mine-ui`` — a local web front end for the ``cascade mine`` loop.

Stdlib only (``http.server``): one page (``mine_ui.html``) plus a small JSON
API. The UI never runs the loop in-process. **Start** spawns
``cascade mine --workdir <workdir> …`` as a child process (log in
``<workdir>/loop.log``), and the page polls the workdir's ``state.json`` /
``history.jsonl``. Because the loop's state lives on disk, the UI can be
restarted without losing the run, and a loop started from the terminal or by
the agent skill shows up here too.

Security model. Anyone who can reach the UI can spend compute, and through
**Submit** also a hotkey and a Lium-funded leg, so:

* every ``/api/*`` call needs the per-process token in ``X-Cascade-Token``.
  It is a custom header, so a cross-site page cannot send it without a CORS
  preflight, which this server never answers.
* on a loopback bind the page itself is open (and embeds the token), but only
  for a loopback ``Host:`` header, which defeats DNS rebinding;
* on any other bind (the Docker image) the page needs ``?token=`` once (then a
  cookie). The token comes from ``--token`` / ``$CASCADE_UI_TOKEN``, else it is
  generated and printed with the URL at startup.
* secrets never pass through the UI. ``LIUM_API_KEY`` and the wallet stay in
  the environment / ``~/.bittensor``, and the page only shows whether they are
  present. Submitting takes a typed confirmation.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .optimize import (
    BEST_DIR,
    PROPOSERS,
    read_history,
    read_state,
    request_stop,
    workdir_lock_holder,
)
from .ralph import PROVIDERS

PAGE = Path(__file__).with_name("mine_ui.html")
LOOP_LOG = "loop.log"
SUBMIT_LOG = "submit.log"
LOG_TAIL_BYTES = 16_000
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}

# Start-form fields → `cascade mine` flags. Only these reach argv; every value
# is passed as its own argv element (never through a shell).
_START_FLAGS = {
    "proposer": "--proposer", "iterations": "--iterations", "seeds": "--seeds",
    "train_hours": "--train-hours", "n_windows": "--n-windows", "device": "--device",
    "pool_dir": "--pool-dir", "pool_ref": "--pool", "warm_start": "--warm-start",
    "min_improvement": "--min-improvement", "start": "--start", "king": "--king",
    "agent_cmd": "--agent-cmd", "propose_cmd": "--propose-cmd",
    "llm_provider": "--llm-provider", "llm_model": "--llm-model",
    "llm_base_url": "--llm-base-url", "agent_max_turns": "--agent-max-turns",
    "llm_key_env": "--llm-key-env", "llm_auth": "--llm-auth",
}
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")

_PATH_KEYS = {"pool_dir", "start", "king"}


def _tail(path: Path, n: int = LOG_TAIL_BYTES) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _environment() -> dict:
    """What the miner has wired up — presence only, never values."""
    # Device nodes, not `import torch`: the UI must answer in milliseconds and
    # never pay a multi-second torch import on a status poll.
    gpu = Path("/dev/nvidiactl").exists() or shutil.which("nvidia-smi") is not None
    wallets = Path(os.environ.get("BT_WALLET_PATH", Path.home() / ".bittensor" / "wallets"))
    names = sorted(p.name for p in wallets.iterdir() if p.is_dir()) if wallets.is_dir() else []
    return {
        "lium_api_key": bool(os.environ.get("LIUM_API_KEY")),
        "chutes_api_key": bool(os.environ.get("CHUTES_API_KEY")),
        "saygm_api_key": bool(os.environ.get("SAYGM_API_KEY")),
        "anthropic_auth": bool(os.environ.get("ANTHROPIC_API_KEY")
                               or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")),
        "claude_cli": shutil.which("claude") is not None,
        "gpu": gpu,
        "wallets": names,
    }


class MineUI:
    """The server's shared state: the workdir, the child loop, the token."""

    def __init__(self, workdir: Path, *, token: str, require_page_token: bool,
                 chain_toml: Path | None = None) -> None:
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.require_page_token = require_page_token
        self.chain_toml = chain_toml
        self.proc: subprocess.Popen | None = None
        self.submit_proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self._env_cache: tuple[float, dict] = (0.0, {})

    # -- status ------------------------------------------------------------- #
    def running(self) -> bool:
        # A child still starting up has not taken the lock yet; after that the
        # workdir lock is the truth (released by the kernel however the loop
        # dies), never a pid read back from state.json.
        if self.proc is not None and self.proc.poll() is None:
            return True
        return workdir_lock_holder(self.workdir) is not None

    def environment(self) -> dict:
        ts, env = self._env_cache
        if time.time() - ts > 30:
            env = _environment()
            self._env_cache = (time.time(), env)
        return env

    def status(self) -> dict:
        return {
            "workdir": str(self.workdir),
            "running": self.running(),
            "submitting": self.submit_proc is not None and self.submit_proc.poll() is None,
            "state": read_state(self.workdir),
            "history": read_history(self.workdir),
            "has_best": (self.workdir / BEST_DIR / "generator.py").is_file(),
            "log": _tail(self.workdir / LOOP_LOG),
            "submit_log": _tail(self.workdir / SUBMIT_LOG, 6000),
            "env": self.environment(),
        }

    # -- actions ------------------------------------------------------------ #
    def start(self, body: dict) -> tuple[int, dict]:
        with self.lock:
            if self.running():
                return 409, {"error": "a loop is already running in this workdir"}
            # `ralph` = `mine --proposer ralph` + an LLM preflight whose diagnosis
            # (bad URL / key / model) lands in loop.log before anything is spent.
            sub = "ralph" if body.get("proposer") == "ralph" else "mine"
            argv = [sys.executable, "-m", "cascade.miner.cli", sub,
                    "--workdir", str(self.workdir)]
            if self.chain_toml:
                argv += ["--chain-toml", str(self.chain_toml)]
            for key, flag in _START_FLAGS.items():
                v = body.get(key)
                if v is None or v == "" or isinstance(v, dict | list):
                    continue
                if key == "proposer" and v not in PROPOSERS:
                    return 400, {"error": f"bad proposer {v!r}"}
                if key == "llm_provider" and v not in PROVIDERS:
                    return 400, {"error": f"bad llm provider {v!r}"}
                if key == "llm_auth" and v not in ("bearer", "x-api-key"):
                    return 400, {"error": f"bad llm auth {v!r}"}
                if key == "llm_key_env" and not _ENV_NAME.fullmatch(str(v)):
                    return 400, {"error": "llm key env must be an env var NAME (the key "
                                          "itself stays in the environment)"}
                if key in _PATH_KEYS and str(v).lower() != "none":
                    # The child runs with cwd=workdir; anchor paths to where the UI runs.
                    v = Path(str(v)).expanduser().resolve()
                argv += [flag, str(v)]
            log = open(self.workdir / LOOP_LOG, "ab")  # noqa: SIM115 — owned by the child
            log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
            log.flush()
            self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=self.workdir, start_new_session=True)
            log.close()
            return 200, {"ok": True, "pid": self.proc.pid}

    def stop(self, body: dict) -> tuple[int, dict]:
        request_stop(self.workdir)
        if body.get("force"):
            if self.proc is not None and self.proc.poll() is None:
                pid = self.proc.pid            # our child: its own session/group
                with contextlib.suppress(OSError):
                    os.killpg(pid, signal.SIGTERM)
                return 200, {"ok": True, "stopping": "killed"}
            pid = workdir_lock_holder(self.workdir)
            if pid and pid > 0 and pid != os.getpid():
                with contextlib.suppress(OSError):
                    # Only a process that leads its own group gets the group
                    # signal (a terminal-started loop may share ours).
                    if os.getpgid(pid) == pid and pid != os.getpgrp():
                        os.killpg(pid, signal.SIGTERM)
                    else:
                        os.kill(pid, signal.SIGTERM)
                return 200, {"ok": True, "stopping": "killed"}
            return 200, {"ok": True, "stopping": "nothing running"}
        return 200, {"ok": True, "stopping": "after the current candidate"}

    def submit(self, body: dict) -> tuple[int, dict]:
        if body.get("confirm") != "SUBMIT":
            return 400, {"error": "type SUBMIT to confirm — this spends the hotkey"}
        best = self.workdir / BEST_DIR
        if not (best / "generator.py").is_file():
            return 400, {"error": "no best generator yet"}
        if self.running():
            return 409, {"error": "stop the loop before submitting"}
        intake = str(body.get("intake") or "").strip()
        wname = str(body.get("wallet_name") or "").strip()
        whot = str(body.get("wallet_hotkey") or "").strip()
        if not (intake.startswith("https://") and wname and whot):
            return 400, {"error": "need an https:// intake, wallet name and hotkey"}
        with self.lock:
            if self.submit_proc is not None and self.submit_proc.poll() is None:
                return 409, {"error": "a submission is already running"}
            argv = [sys.executable, "-m", "cascade.miner.cli", "submit", str(best), intake,
                    "--wallet-name", wname, "--wallet-hotkey", whot]
            if body.get("label"):
                argv += ["--label", str(body["label"])]
            if self.chain_toml:
                argv += ["--chain-toml", str(self.chain_toml)]
            log = open(self.workdir / SUBMIT_LOG, "ab")  # noqa: SIM115
            self.submit_proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                                cwd=self.workdir)
            log.close()
        return 200, {"ok": True}


def make_handler(app: MineUI):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cascade-mine-ui"

        def log_message(self, fmt, *args):  # quiet: the page polls every few seconds
            pass

        # -- auth ----------------------------------------------------------- #
        def _host_ok(self) -> bool:
            if app.require_page_token:
                return True
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in _LOOPBACK

        def _cookie_token(self) -> str:
            c = SimpleCookie(self.headers.get("Cookie") or "")
            return c["cascade_ui"].value if "cascade_ui" in c else ""

        def _api_ok(self) -> bool:
            got = self.headers.get("X-Cascade-Token") or ""
            return self._host_ok() and hmac.compare_digest(got, app.token)

        # -- io ------------------------------------------------------------- #
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, doc: dict) -> None:
            self._send(code, json.dumps(doc).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 64_000:
                return {}
            try:
                doc = json.loads(self.rfile.read(n))
            except ValueError:
                return {}
            return doc if isinstance(doc, dict) else {}

        # -- routes --------------------------------------------------------- #
        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path == "/":
                if not self._host_ok():
                    return self._send(403, b"forbidden host", "text/plain")
                if app.require_page_token:
                    q = parse_qs(u.query).get("token", [""])[0]
                    if q and hmac.compare_digest(q, app.token):
                        return self._send(302, b"", "text/plain", {
                            "Location": "/",
                            "Set-Cookie": f"cascade_ui={app.token}; HttpOnly; SameSite=Strict; Path=/",
                        })
                    if not hmac.compare_digest(self._cookie_token(), app.token):
                        return self._send(401, b"open the URL with ?token=... printed at startup",
                                          "text/plain")
                html = PAGE.read_text(encoding="utf-8").replace(
                    "__CASCADE_TOKEN__", json.dumps(app.token))
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if u.path == "/api/status":
                if not self._api_ok():
                    return self._json(403, {"error": "bad token"})
                return self._json(200, app.status())
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            if not self._api_ok():
                return self._json(403, {"error": "bad token"})
            body = self._body()
            route = {"/api/start": app.start, "/api/stop": app.stop,
                     "/api/submit": app.submit}.get(u.path)
            if route is None:
                return self._json(404, {"error": "not found"})
            code, doc = route(body)
            return self._json(code, doc)

    return Handler


def serve(workdir: Path | str, *, host: str = "127.0.0.1", port: int = 8765,
          token: str | None = None, chain_toml: Path | None = None) -> int:
    token = token or os.environ.get("CASCADE_UI_TOKEN") or secrets.token_urlsafe(18)
    loopback = host in _LOOPBACK
    app = MineUI(Path(workdir), token=token, require_page_token=not loopback,
                 chain_toml=chain_toml)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    shown = "localhost" if host in ("0.0.0.0", "::") or loopback else host
    url = f"http://{shown}:{port}/" + ("" if loopback else f"?token={token}")
    print(f"cascade mine-ui on {url}\n  workdir: {app.workdir}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0

