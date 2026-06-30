"""LLM-as-a-judge boundary for the submission screen (OpenRouter / GLM).

The judge runs *after* the deterministic checks (static guard, determinism) as a
second opinion on the things a parser can't decide:

* **distillation** — is this generator *sampling a prior*, or *replaying fitted
  weights*? Fed generator.py + config + requirements + the code-only interface
  rule, plus the mechanical literal-density evidence from
  :mod:`cascade.screen.signals`. Hard-fail.
* **benchmark-targeting** — given that the eval is *private and rotating*, is this
  a *general prior* or one *shaped to a specific distribution* (hardcoded periods/
  scales/lengths matching public benchmarks)? Hard-fail if blatant, warn
  otherwise — the rotating eval handles the rest.

Both share **one** judge call (``judge_static``) to keep cost down. The
copy-of-king check is mechanical (:mod:`cascade.screen.canonicalize`); the LLM is
only consulted for the middle band (``judge_copy_of_king``), fed both generators.

The client is a tiny injectable :class:`JudgeClient` protocol so the screen is
testable offline with a fake; the shipped implementation is
:class:`OpenRouterJudge` (defaults to ``z-ai/glm-5.2``). cascade's own code may
use the network freely — the import blocklist only constrains submitted
generators.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

# Verdicts the screen understands. "fail" → hard reject; "warn" → record, don't
# block; "pass" → clean. The model is asked to emit one of these directly.
VERDICTS = ("pass", "warn", "fail")


class JudgeError(RuntimeError):
    """The judge could not be reached or returned an unusable response."""


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: str          # one of VERDICTS
    reason: str
    raw: str | None = None

    @property
    def failed(self) -> bool:
        return self.verdict == "fail"

    @property
    def warned(self) -> bool:
        return self.verdict == "warn"


class JudgeClient(Protocol):
    """Minimal chat boundary: a single system+user turn returning model text.

    Implemented by :class:`OpenRouterJudge` in production and by a fake in tests.
    """

    def complete(self, system: str, user: str) -> str: ...


# ── OpenRouter client (z-ai / GLM by default) ────────────────────────────────


@dataclass
class OpenRouterJudge:
    """Chat-completions client for OpenRouter, defaulting to GLM (z-ai).

    Uses ``urllib`` from the standard library so the core install pulls no new
    dependency. The API key is read from ``api_key_env`` (default
    ``OPENROUTER_API_KEY``) unless passed explicitly.
    """

    model: str = "z-ai/glm-5.2"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: int = 60
    max_tokens: int = 1024
    temperature: float = 0.0

    def _key(self) -> str:
        key = self.api_key or os.environ.get(self.api_key_env, "")
        if not key:
            raise JudgeError(
                f"no OpenRouter API key (set ${self.api_key_env} or pass api_key=...)"
            )
        return key

    def complete(self, system: str, user: str) -> str:
        import urllib.error
        import urllib.request

        url = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
                # Optional OpenRouter attribution headers — harmless if ignored.
                "HTTP-Referer": "https://github.com/tensorlink-ai/cascade",
                "X-Title": "cascade-submission-screen",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # noqa: PERF203
            detail = e.read().decode("utf-8", "replace")[:500]
            raise JudgeError(f"OpenRouter HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise JudgeError(f"OpenRouter request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise JudgeError(f"OpenRouter returned non-JSON: {e}") from e

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise JudgeError(f"OpenRouter response missing content: {payload}") from e


# ── response parsing ─────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or ```` ```json ```` fences; we scan for the first
    balanced ``{...}``. Raises :class:`JudgeError` if none is parseable.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break  # try the next '{'
        start = text.find("{", start + 1)
    raise JudgeError(f"no JSON object in judge reply: {text[:200]!r}")


def _verdict_from(obj: dict, key: str, raw: str) -> JudgeVerdict:
    """Read ``{<key>: 'pass'|'warn'|'fail', <key>_reason: ...}`` into a verdict."""
    v = str(obj.get(key, "")).strip().lower()
    if v not in VERDICTS:
        raise JudgeError(f"judge returned invalid {key} verdict {v!r}")
    reason = str(obj.get(f"{key}_reason") or obj.get("reason") or "").strip()
    return JudgeVerdict(verdict=v, reason=reason, raw=raw)


# ── prompts ──────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a strict security reviewer for a Bittensor subnet that pays miners "
    "for synthetic time-series DATA GENERATORS. A valid submission is PURELY "
    "ALGORITHMIC: it samples from a prior (Gaussian-process / kernel families, "
    "causal DAGs, parametric trend / seasonality / noise). It must NOT ship or "
    "replay fitted model weights, and must NOT be shaped to any specific eval "
    "distribution. Reason carefully, then answer ONLY with one JSON object."
)


def _static_user_prompt(files: dict[str, str], evidence: dict, interface_rule: str) -> str:
    parts = [
        "Judge this generator submission on TWO questions and return JSON.",
        "",
        "Interface rule the submission must obey:",
        interface_rule.strip(),
        "",
        "(1) DISTILLATION — Is this code SAMPLING A PRIOR, or REPLAYING FITTED "
        "WEIGHTS? Blatant tells: large literal numeric arrays, base64/compressed "
        "blobs decoded at runtime, libraries used only to load a pretrained "
        "model, fake 'kernel params' that are actually trained weights. Honest "
        "priors are mostly logic; distilled ones are mostly numbers. Verdict "
        "'fail' if it replays fitted weights, else 'pass'.",
        "",
        "Mechanical evidence (computed from the source, not your guess):",
        json.dumps(evidence, indent=2),
        "",
        "(2) BENCHMARK-TARGETING — The held-out eval is PRIVATE and ROTATES every "
        "round; the miner never sees it. Is this a GENERAL prior, or one SHAPED "
        "to a specific distribution (hardcoded periods/scales/lengths matching "
        "public benchmarks like GIFT-Eval / Chronos / M4)? Verdict 'fail' if "
        "blatantly targeted, 'warn' if mildly suspicious, else 'pass'.",
        "",
        "Respond with ONLY this JSON object:",
        '{"distillation": "pass|fail", "distillation_reason": "...", '
        '"benchmark_targeting": "pass|warn|fail", "benchmark_targeting_reason": "..."}',
        "",
        "=== SUBMISSION FILES ===",
    ]
    for name, content in files.items():
        parts.append(f"\n--- {name} ---\n{content}")
    return "\n".join(parts)


def _copy_user_prompt(challenger: str, king: str, similarity: float) -> str:
    return "\n".join(
        [
            "Two data generators below. The first is a CHALLENGER; the second is "
            "the reigning KING (its source is public). Mechanical canonical-source "
            f"similarity is {similarity:.3f} — high enough to be suspicious, not "
            "high enough to auto-reject. Decide whether the challenger is a "
            "RESTRUCTURED COPY of the king (same algorithm, re-laid-out to dodge a "
            "similarity threshold) versus a genuinely independent generator.",
            "",
            "Verdict 'fail' if it is essentially the king's generator restructured, "
            "else 'pass'.",
            "",
            "Respond with ONLY this JSON object:",
            '{"copy_of_king": "pass|fail", "copy_of_king_reason": "..."}',
            "",
            "=== CHALLENGER generator.py ===",
            challenger,
            "",
            "=== KING generator.py ===",
            king,
        ]
    )


# ── judge calls ──────────────────────────────────────────────────────────────


def judge_static(
    files: dict[str, str],
    evidence: dict,
    client: JudgeClient,
    *,
    interface_rule: str,
) -> tuple[JudgeVerdict, JudgeVerdict]:
    """One LLM call covering distillation + benchmark-targeting.

    Returns ``(distillation_verdict, benchmark_targeting_verdict)``. Raises
    :class:`JudgeError` on transport or parse failure (the caller decides whether
    that is fail-closed or fail-open).
    """
    raw = client.complete(_SYSTEM, _static_user_prompt(files, evidence, interface_rule))
    obj = _extract_json(raw)
    distill = _verdict_from(obj, "distillation", raw)
    if distill.verdict == "warn":  # distillation is binary; treat a stray warn as fail-safe pass
        distill = JudgeVerdict("pass", distill.reason, raw)
    bench = _verdict_from(obj, "benchmark_targeting", raw)
    return distill, bench


def judge_copy_of_king(
    challenger_src: str,
    king_src: str,
    similarity: float,
    client: JudgeClient,
) -> JudgeVerdict:
    """LLM adjudication for the copy-of-king *middle band* only (both sources fed)."""
    raw = client.complete(_SYSTEM, _copy_user_prompt(challenger_src, king_src, similarity))
    obj = _extract_json(raw)
    v = _verdict_from(obj, "copy_of_king", raw)
    if v.verdict == "warn":  # binary decision; a stray warn is not a hard reject
        return JudgeVerdict("pass", v.reason, raw)
    return v


def build_judge(cfg) -> JudgeClient | None:
    """Construct the configured judge client, or ``None`` when judging is off.

    ``None`` when ``[judge] enabled = false`` or no API key is available, so a
    caller can pass the result straight through and the screen no-ops offline.
    """
    jc = getattr(cfg, "judge", None)
    if jc is None or not jc.enabled:
        return None
    if jc.provider != "openrouter":
        raise JudgeError(f"unsupported judge provider {jc.provider!r} (only 'openrouter')")
    if not (jc.api_key or os.environ.get(jc.api_key_env, "")):
        return None
    return OpenRouterJudge(
        model=jc.model,
        base_url=jc.base_url,
        api_key=jc.api_key or None,
        api_key_env=jc.api_key_env,
        timeout_seconds=jc.timeout_seconds,
        max_tokens=jc.max_tokens,
    )
