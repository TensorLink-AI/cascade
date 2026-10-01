# Ralph loop: an LLM rewrites your generator

`cascade ralph` runs Claude Code in a loop on your generator's **code**. The
model can be Anthropic's or any Anthropic-compatible one: Chutes, SayGM, or
a gateway of your own. Every iteration uses the same standing prompt and a
fresh agent context, and memory lives in a notebook file (the Ralph
pattern). After each iteration the existing `cascade mine` loop verifies,
scores and keeps or discards the change. The model never grades its own
work.

```
            ┌───────────── RALPH_PROMPT.md (same every time; edit to steer)
            │   + this iteration: best / king / last 15 results
            ▼
best/ ──copy──▶ candidates/NNNN/ ──claude -p (fresh context)──▶ edited generator.py
                 + .ralph-notes.md  ◀── reads & updates ──▶ RALPH_NOTES.md (memory)
                                                              ▲
verify ─▶ score (fixed pool/seeds/init) ─▶ keep if better ──┘ result appended
```

## Quick start

```bash
export CHUTES_API_KEY=...          # or SAYGM_API_KEY=..., env only
cascade ralph --llm-provider chutes --llm-model <model-id> --check     # 1 tiny request
cascade ralph --llm-provider chutes --llm-model <model-id> \
    --pool-dir ./my-heldout --warm-start live --agent-max-turns 40
```

Docker (the `oneclick` image has Claude Code):

```bash
docker run --gpus all -v "$PWD:/work" -e CHUTES_API_KEY \
    ghcr.io/tensorlink-ai/cascade-miner:oneclick \
    ralph --llm-provider chutes --llm-model <model-id> --pool-dir ./my-heldout
```

In the UI, pick **ralph: LLM rewrites generator code**, then a provider and a
model id.

Defaults: starts from `champions/king`, 100 iterations, and every other
`cascade mine` flag works (`--seeds`, `--train-hours`, `--iterations`, …).
`touch mine-run/STOP` stops after the current candidate. Re-running on the
same workdir resumes, including the notebook.

## Providers

| `--llm-provider` | Endpoint (Claude Code appends `/v1/messages`) | Key |
|---|---|---|
| `anthropic` | Claude Code's own login | `claude login` or `ANTHROPIC_API_KEY` |
| `chutes` | `https://llm.chutes.ai` | `$CHUTES_API_KEY` |
| `saygm` | `https://api.saygm.com` | `$SAYGM_API_KEY` |
| `custom` | `--llm-base-url` | `--llm-key-env NAME` |

The chutes and saygm URLs are best-known presets, not verified from this
repo. **Run `--check` first.** It sends one Messages request through the
exact URL, auth and model Claude Code will use, and names the problem: a
wrong path (404), a bad key (401/403, or try `--llm-auth x-api-key`), an
unknown model (400), no credit (429), or an endpoint that only speaks
OpenAI. Override a preset with `--llm-base-url`.

Claude Code speaks only the Anthropic Messages API. If a provider offers
only OpenAI-style chat, run a translating proxy in front of it (LiteLLM,
claude-code-router) and point `custom` at the proxy.

**Model choice matters more than anything else here.** Use a strong coding
model with reliable tool calling. The agent has to grep a large file, edit
it, and run `cascade verify`. Claude Code warns that a non-Claude id "isn't
described by this version's model catalog" and assumes a 200k context for
auto-compaction. That is harmless for these short, single-change iterations.

## Credentials are isolated

With a third-party provider, the agent starts from a **strict env
allowlist**: `PATH`, locale, proxy/CA variables and non-secret `CASCADE_*`
vars, plus the provider's base URL, model and key. It also gets a **fresh
`CLAUDE_CONFIG_DIR`** (`mine-run/.claude-agent/`). A logged-in
`~/.claude`, `CLAUDE_CODE_OAUTH_TOKEN`, a host-managed session token,
`LIUM_API_KEY`, Hippius or cloud keys can therefore never be sent to the
provider. This was observed, not hypothetical: without the isolation, an
Anthropic OAuth token outranked `ANTHROPIC_AUTH_TOKEN` and went to the
third-party endpoint. With `anthropic`, your own Claude login is kept, and
everything secret the agent has no use for is still stripped.

The agent may only Read, Edit, Write, Glob, Grep and run `cascade verify`,
with edits confined to the candidate dir. `--agent-max-turns` bounds the
cost of each iteration.

## Steering it

- `mine-run/RALPH_PROMPT.md` is the standing prompt, seeded from
  `cascade/miner/ralph.py: DEFAULT_PROMPT`. Edit it while the loop runs,
  e.g. "focus on energy and transport families" or "no new families, fix
  realism of existing ones". The next iteration picks up the change.
- `mine-run/RALPH_NOTES.md` is the agent's memory: plan, lessons, and a
  results log the loop appends to after every candidate. You can write
  into it too; the agent reads it first every time.
- History, scores and the change notes are in `history.jsonl` and the UI,
  as for every proposer.

## Honest expectations

Each iteration costs one model session plus one local training run, so it
is minutes to tens of minutes on a GPU. The same caveats as every local
score apply ([ONE_CLICK_MINING.md](ONE_CLICK_MINING.md#reading-the-numbers-honestly)):
directional, pool-specific, noisy at one seed. An LLM that writes plausible
code is not evidence that the data got better. Only the score decides, and
only a margin over the king that survives more seeds and a longer budget is
worth a hotkey.
