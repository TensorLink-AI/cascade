"""Ralph loop: Claude Code, on any Anthropic-compatible model, rewriting the generator.

A Ralph loop runs the SAME prompt again and again, each time in a FRESH agent
context, and carries memory between iterations in files. Here it is a
proposer for the ``cascade mine`` loop (``--proposer ralph`` /
``cascade ralph``):

* ``<workdir>/RALPH_PROMPT.md``: the standing instruction, written from
  :data:`DEFAULT_PROMPT` on first run. Edit it while the loop runs to steer.
* ``<workdir>/RALPH_NOTES.md``: the agent's notebook (plan, hypotheses,
  lessons). It is copied into each candidate as ``.ralph-notes.md``, the agent
  updates it, and it is moved back out before verify so it never ships. The
  loop appends every outcome (score, accepted or not, verify failure) to its
  results log, so the next fresh context sees what actually happened.
* Each iteration, Claude Code edits the **generator's code** in a copy of the
  current best. The loop then verifies, scores and keeps it or not as for any
  proposer. The agent never scores itself, so it cannot grade its own work.

The model behind Claude Code is chosen with ``--llm-provider``:

* ``anthropic``: Claude Code's own login / ``ANTHROPIC_API_KEY``.
* ``chutes`` / ``saygm``: their Anthropic-compatible endpoints, keys from
  ``$CHUTES_API_KEY`` / ``$SAYGM_API_KEY``.
* ``custom``: any Anthropic Messages endpoint (``--llm-base-url`` +
  ``--llm-key-env``).

Claude Code speaks the Anthropic Messages API, and only to
``$ANTHROPIC_BASE_URL``. For a provider that only offers OpenAI-compatible
chat, put a translating proxy in front (LiteLLM, claude-code-router) and use
``custom``. Run ``cascade ralph --check`` first: one tiny request proves the
URL, key and model before the loop spends anything.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .optimize import AGENT_NOTE, AgentProposer, LoopConfig, format_history_rows

PROMPT_FILE = "RALPH_PROMPT.md"
NOTES_FILE = "RALPH_NOTES.md"
CAND_NOTES = ".ralph-notes.md"


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    key_env: str
    auth: str                # "bearer" → ANTHROPIC_AUTH_TOKEN; "x-api-key" → ANTHROPIC_API_KEY


# Best-known Anthropic-compatible roots: Claude Code appends /v1/messages.
# Providers move these. `cascade ralph --check` tells you if one is wrong, and
# --llm-base-url overrides it.
PRESETS: dict[str, ProviderPreset] = {
    # Verified 2026-10-01: Chutes serves Anthropic Messages on claude.chutes.ai
    # (llm.chutes.ai is its OpenAI-compatible host and 404s /v1/messages). It
    # always answers as an SSE stream, which Claude Code consumes natively.
    "chutes": ProviderPreset("https://claude.chutes.ai", "CHUTES_API_KEY", "bearer"),
    "saygm": ProviderPreset("https://api.saygm.com", "SAYGM_API_KEY", "bearer"),
    # Engy (engy.ai/docs): an Anthropic-compatible gateway (Kimi, GLM, Qwen,
    # DeepSeek, …) documented for Claude Code as ANTHROPIC_BASE_URL + a bearer
    # ANTHROPIC_AUTH_TOKEN; model ids such as "kimi-k3".
    "engy": ProviderPreset("https://api.engy.ai", "ENGY_API_KEY", "bearer"),
}
PROVIDERS = ("anthropic", "chutes", "saygm", "engy", "custom")


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str = ""
    model: str = ""
    key_env: str = ""
    auth: str = "bearer"

    @property
    def key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""

    def claude_env(self) -> dict[str, str]:
        """Env that points headless Claude Code at this provider (empty = its own login)."""
        env: dict[str, str] = {}
        if self.name != "anthropic":
            env["ANTHROPIC_BASE_URL"] = self.base_url
            if self.auth == "x-api-key":
                env["ANTHROPIC_API_KEY"], env["ANTHROPIC_AUTH_TOKEN"] = self.key, ""
            else:
                env["ANTHROPIC_AUTH_TOKEN"], env["ANTHROPIC_API_KEY"] = self.key, ""
            # Third-party endpoints are slower and serve no Anthropic side traffic.
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
            env.setdefault("API_TIMEOUT_MS", "600000")
        if self.model:
            # Every tier (main, background, subagents) uses the one model, so no
            # request goes to a model name the provider does not serve.
            for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                        "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                        "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL"):
                env[var] = self.model
        return env

    def describe(self) -> str:
        if self.name == "anthropic":
            return f"anthropic ({self.model or 'Claude Code default model'})"
        return f"{self.name} {self.model} @ {self.base_url} (key ${self.key_env})"


# --------------------------------------------------------------------------- #
# The agent's environment. Claude Code authenticates with whatever it finds: a
# logged-in ~/.claude, CLAUDE_CODE_OAUTH_TOKEN, a host-managed session token.
# Pointed at a third-party endpoint, any of those would be SENT TO THAT PROVIDER
# (observed: a host OAuth token outranked ANTHROPIC_AUTH_TOKEN). So:
#   * third-party provider: a strict allowlist + a fresh CLAUDE_CONFIG_DIR, so
#     the provider's key is the only credential Claude Code can see;
#   * anthropic: the miner's own Claude login stays, but secrets the agent has
#     no use for (Lium, Hippius, cloud, git, other providers, the UI token) go.
# --------------------------------------------------------------------------- #

_ALLOW_ENV = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TZ", "TMPDIR",
    "LC_ALL", "LC_CTYPE", "VIRTUAL_ENV", "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
}
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "SESSION")
# Kept for the anthropic provider: how Claude Code finds the miner's own login.
_ANTHROPIC_AUTH_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                       "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR"}


def agent_base_env(provider: Provider, workdir: Path,
                   environ: dict[str, str] | None = None) -> dict[str, str]:
    """The env a Claude Code proposer starts from (provider vars are added on top)."""
    src = dict(os.environ if environ is None else environ)
    if provider.name == "anthropic":
        def keep(k: str) -> bool:
            return k in _ALLOW_ENV or k in _ANTHROPIC_AUTH_ENV or not any(
                m in k.upper() for m in _SECRET_MARKERS)
        return {k: v for k, v in src.items() if keep(k)}
    env = {k: v for k, v in src.items() if k in _ALLOW_ENV}
    env.update({k: v for k, v in src.items() if k.startswith("CASCADE_")
                and not any(m in k for m in _SECRET_MARKERS)})
    env["CLAUDE_CONFIG_DIR"] = str(_agent_config_dir(workdir))
    return env


def _agent_config_dir(workdir: Path) -> Path:
    """A Claude Code config dir with no login in it, plus the cascade-mine skill."""
    d = Path(workdir) / ".claude-agent"
    skills = d / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "cascade-mine"
    if src.is_dir() and not (skills / "cascade-mine").exists():
        shutil.copytree(src, skills / "cascade-mine")
    return d


def resolve_provider(cfg: LoopConfig) -> Provider:
    """Build the provider from the loop config, failing fast on a missing piece."""
    name = (cfg.llm_provider or "anthropic").lower()
    if name not in PROVIDERS:
        raise ValueError(f"--llm-provider {name!r}: expected one of {', '.join(PROVIDERS)}")
    if name == "anthropic":
        return Provider("anthropic", model=cfg.llm_model)
    preset = PRESETS.get(name)
    base = (cfg.llm_base_url or (preset.base_url if preset else "")).rstrip("/")
    key_env = cfg.llm_key_env or (preset.key_env if preset else "")
    auth = cfg.llm_auth or (preset.auth if preset else "bearer")
    if auth not in ("bearer", "x-api-key"):
        raise ValueError(f"--llm-auth {auth!r}: expected bearer or x-api-key")
    if not base:
        raise ValueError("--llm-provider custom needs --llm-base-url (Anthropic Messages root)")
    if not key_env:
        raise ValueError(f"--llm-provider {name} needs --llm-key-env (the env var holding the key)")
    if not os.environ.get(key_env):
        raise ValueError(f"${key_env} is not set: export your {name} API key (env only, "
                         "never a command-line argument)")
    if not cfg.llm_model:
        raise ValueError(f"--llm-model is required for {name}: give a model id from your "
                         "provider's model list (a strong coding model with tool use, e.g. a "
                         "GLM, Kimi, Qwen-Coder or DeepSeek release)")
    return Provider(name, base_url=base, model=cfg.llm_model, key_env=key_env, auth=auth)


def _sse_message(raw: bytes) -> dict:
    """Fold an Anthropic Messages SSE stream into the non-streaming shape."""
    msg: dict = {}
    blocks: dict[int, dict] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            continue
        t = ev.get("type")
        if t == "message_start":
            msg = dict(ev.get("message") or {})
        elif t == "content_block_start":
            blocks[ev.get("index", 0)] = dict(ev.get("content_block") or {})
        elif t == "content_block_delta":
            b = blocks.setdefault(ev.get("index", 0), {"type": "text", "text": ""})
            d = ev.get("delta") or {}
            if d.get("type") == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
    if msg:
        msg["content"] = [blocks[i] for i in sorted(blocks)]
    return msg


def preflight(provider: Provider, *, timeout: float = 60.0) -> tuple[bool, str]:
    """One tiny Messages request through the exact URL/auth/model Claude Code will
    use, with a diagnosis instead of a stack trace. ``anthropic`` only checks the CLI."""
    if shutil.which("claude") is None:
        return False, ("the `claude` CLI is not on PATH (use the :oneclick image, or "
                       "install Claude Code: curl -fsSL https://claude.ai/install.sh | bash)")
    if provider.name == "anthropic":
        return True, "claude CLI found; using Claude Code's own login / ANTHROPIC_API_KEY"
    url = provider.base_url + "/v1/messages"
    # Claude Code strips a "[1m]"-style context suffix before sending; so do we.
    model = __import__("re").sub(r"\[[0-9]+[km]\]$", "", provider.model)
    body = json.dumps({
        "model": model, "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the word OK."}],
    }).encode()
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if provider.auth == "x-api-key":
        headers["x-api-key"] = provider.key
    else:
        headers["authorization"] = f"Bearer {provider.key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read() or b"{}"
            ctype = r.headers.get("content-type", "")
        # Some endpoints (Chutes) stream even without "stream": true; Claude Code
        # streams anyway, so an SSE reply is a valid Messages endpoint.
        doc = _sse_message(raw) if "event-stream" in ctype else json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "replace")
        hint = {
            401: f"key rejected: check ${provider.key_env}, or try --llm-auth "
                 f"{'x-api-key' if provider.auth == 'bearer' else 'bearer'}",
            403: f"key rejected or not entitled to {provider.model!r}",
            404: f"no Anthropic Messages API at {url}: check the provider's docs for its "
                 "Anthropic-compatible root and pass --llm-base-url (Claude Code appends "
                 "/v1/messages). If it only offers OpenAI chat, front it with LiteLLM / "
                 "claude-code-router and use --llm-provider custom",
            400: f"request rejected (often an unknown model id {provider.model!r})",
            429: "rate-limited or out of credit",
        }.get(e.code, "unexpected response")
        return False, f"HTTP {e.code} from {url}: {hint}\n  body: {detail}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"cannot reach {url}: {e}"
    except ValueError:
        return False, f"{url} answered with non-JSON: not an Anthropic Messages endpoint"
    if doc.get("type") != "message" or not isinstance(doc.get("content"), list):
        return False, (f"{url} answered, but not in the Anthropic Messages shape "
                       f"(got keys {sorted(doc)[:6]}); it is probably OpenAI-compatible only")
    text = "".join(b.get("text", "") for b in doc["content"] if isinstance(b, dict))
    return True, f"OK: {provider.describe()} answered {text.strip()[:40]!r}"


DEFAULT_PROMPT = """\
# Ralph: improve this cascade generator

You are one iteration of a long-running loop. Every iteration gets this same
prompt and a fresh context. Your memory is the notebook `.ralph-notes.md` in
your working directory. READ IT FIRST and UPDATE IT before you finish.

## The game
The working directory is a copy of the current best cascade data generator
(`generator.py` + `config.json`). The operator trains a fixed small
forecaster (Toto2-4M) on this generator's synthetic series and scores it on
real held-out windows: geomean of CRPS and MASE, lower is better. The loop
scores your change after you exit and keeps it only if it beats the best.

## Your job this iteration
Make ONE focused change to the generator's CODE that you expect to make the
trained forecaster better on REAL data (energy, nature, sales, web,
transport, finance, epidemiology, sensors). Good directions:
- realism of the observation process: publication resolution and rounding,
  sticky or held values, missing bins, reporting cadence, count data,
  saturation and clipping at physical limits;
- a new series family covering behaviour the corpus lacks (check the family
  registry first; do not duplicate one);
- fixing a family whose output is unrealistic (scale, noise level,
  seasonality periods, regime switches);
- speed. Data is streamed under a fixed wall clock, so a slower generator
  trains the model on fewer tokens. Never add per-series work without a reason.
Pure mixture-weight reweighting is the `tune` proposer's job. Here, change
what the series ARE.
The eval rewards BREADTH: past winners spread their gain over most domains and all
three horizons, while near misses packed a similar gain into one feed and lost.

## Hard rules (violations are rejected before scoring)
- Deterministic in `seed` only: derive every RNG from it; no `hash()`, clock,
  `os.urandom`, network.
- No blocked imports (socket, subprocess, pickle, multiprocessing, …) and no
  code packed into strings. Only allowlisted deps (numpy, scipy, pandas,
  numba, scikit-learn, statsmodels, …).
- Yields finite float arrays `(L,)` or `(C, L)`, 64 <= L <= 4096, C <= 32,
  magnitudes in a sane float32 range.
- Keep the layout: generator.py / config.json / requirements.txt. Edit only
  files in your working directory.

## How
1. Read `.ralph-notes.md`. Do not retry an idea its results log shows failed
   unless you have a specific reason it will differ now.
2. Find the relevant code with Grep (the file may be very large; do not read
   it end to end).
3. Make the change. Run `cascade verify .` and fix anything it reports.
4. Update `.ralph-notes.md`: what you tried and why, what you expect, and
   the next ideas worth trying. Keep it under ~150 lines and prune stale notes.
5. Write ONE line describing the change to `.mine-note.md`.
Do not train or score; the loop does that.
"""


def _dynamic_section(loop, cand_dir: Path, iteration: int, history: list[dict]) -> str:
    best, king = loop.best_record(), loop.king_record()
    rows = format_history_rows(history, 15, with_reason=True)
    best_s = "n/a" if not best else f"{best['score']:.5f} (#{best['iteration']})"
    king_s = "n/a" if not king or king.get("score") is None else f"{king['score']:.5f}"
    return (f"\n\n## This iteration: #{iteration}\n"
            f"Working directory: {cand_dir}\n"
            f"Current best: {best_s}. Reference king (same pool, seeds, init): {king_s}.\n"
            f"Recent results (most recent last):\n{rows}\n")


class RalphProposer(AgentProposer):
    """Same prompt each iteration + a persistent notebook; see the module docstring."""

    name = "ralph"
    label = "ralph"

    def __init__(self, cfg: LoopConfig, loop) -> None:
        super().__init__(cfg, loop)
        self.prompt_path = loop.workdir / PROMPT_FILE
        self.notes_path = loop.workdir / NOTES_FILE
        if not self.prompt_path.is_file():
            self.prompt_path.write_text(DEFAULT_PROMPT, encoding="utf-8")
        if not self.notes_path.is_file():
            self.notes_path.write_text(
                "# Ralph notebook\n\n## Plan / next ideas\n\n## Learned\n\n## Results log\n",
                encoding="utf-8")

    def stdin_for(self, cand_dir: Path, iteration: int, history: list[dict]) -> str:
        return (self.prompt_path.read_text(encoding="utf-8")
                + _dynamic_section(self.loop, cand_dir, iteration, history))

    def prepare(self, cand_dir: Path, iteration: int) -> None:
        shutil.copyfile(self.notes_path, cand_dir / CAND_NOTES)

    def finish(self, cand_dir: Path, iteration: int) -> None:
        nb = cand_dir / CAND_NOTES
        if nb.is_file():
            text = nb.read_text(encoding="utf-8")
            if text.strip():
                self.notes_path.write_text(text, encoding="utf-8")
            nb.unlink()

    def on_result(self, rec: dict, best: dict | None) -> None:
        s = "—" if rec.get("score") is None else f"{rec['score']:.5f}"
        verdict = "ACCEPTED (new best)" if rec.get("accepted") else rec.get("status", "?")
        detail = (rec.get("detail") or "").strip().splitlines()
        why = f" — {detail[0][:200]}" if detail and rec.get("status") != "scored" else ""
        best_s = "" if not best else f" | best {best['score']:.5f}"
        line = f"- #{rec['iteration']} {verdict} score {s}{best_s}: {rec.get('note', '')}{why}\n"
        with open(self.notes_path, "a", encoding="utf-8") as f:
            f.write(line)


__all__ = ["AGENT_NOTE", "DEFAULT_PROMPT", "PRESETS", "PROVIDERS", "Provider",
           "RalphProposer", "preflight", "resolve_provider"]
