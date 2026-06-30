"""LLM-judge submission screen: mechanical signals, canonical-source similarity,
verdict parsing, and the orchestrator — all offline with a fake judge."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cascade.screen import (
    canonical_source,
    numeric_evidence,
    screen_repo,
    source_similarity,
)
from cascade.screen.judge import (
    JudgeError,
    OpenRouterJudge,
    _extract_json,  # noqa: PLC2701 — unit under test
    build_judge,
    judge_copy_of_king,
    judge_static,
)

# ── fakes ─────────────────────────────────────────────────────────────────────


class FakeJudge:
    """Returns a canned reply (str) or raises (Exception) per call."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _repo(tmp_path, generator_src: str, *, config: str = "{}", reqs: str = ""):
    (tmp_path / "generator.py").write_text(generator_src, encoding="utf-8")
    (tmp_path / "config.json").write_text(config, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(reqs, encoding="utf-8")
    return tmp_path


# ── mechanical signals: numeric evidence ───────────────────────────────────────


def test_numeric_evidence_honest_prior_is_mostly_logic(tmp_path):
    src = (
        "import numpy as np\n"
        "def gen(rng, n):\n"
        "    period = rng.choice([7, 12, 24])\n"
        "    amp = rng.uniform(0.2, 2.0)\n"
        "    return amp * np.sin(period)\n"
    )
    _repo(tmp_path, src)
    ev = numeric_evidence(tmp_path)
    assert ev.numeric_literal_count > 0
    assert ev.total_literal_bytes < 64          # a handful of tiny constants
    assert ev.largest_numeric_array_len == 3    # the [7, 12, 24] choice list


def test_numeric_evidence_flags_big_high_entropy_array(tmp_path):
    weights = [round(0.1234567 * i % 1.0, 9) for i in range(1, 400)]
    src = "W = " + repr(weights) + "\n"
    _repo(tmp_path, src)
    ev = numeric_evidence(tmp_path)
    assert ev.largest_numeric_array_len >= 399
    assert ev.total_literal_bytes > 2000        # the array dominates the source
    assert ev.constant_entropy > 3.0            # mantissa digits look like noise


def test_numeric_evidence_counts_string_blob_bytes(tmp_path):
    blob = "QUJD" * 500  # base64-ish payload
    _repo(tmp_path, f'BLOB = "{blob}"\n')
    ev = numeric_evidence(tmp_path)
    assert ev.max_string_literal_bytes >= 2000
    assert ev.total_literal_bytes == 0          # it's a string, not numeric


def test_numeric_evidence_ignores_bools_and_syntax_errors(tmp_path):
    (tmp_path / "generator.py").write_text("flag = True\nx = (\n", encoding="utf-8")
    ev = numeric_evidence(tmp_path)
    assert ev.numeric_literal_count == 0        # True is a bool, file is broken
    assert ev.total_literal_bytes == 0


# ── mechanical: canonical source + similarity ──────────────────────────────────


def test_canonical_source_strips_comments_docstrings_and_whitespace():
    a = '"""Doc."""\nx = 1  # comment\n\n\ny   =   2\n'
    b = "x = 1\ny = 2\n"
    assert canonical_source(a) == canonical_source(b)


def test_source_similarity_identical_after_reflow():
    a = "def f(x):\n    # add one\n    return x + 1\n"
    b = '"""different docstring"""\ndef f(x):\n        return x+1\n'
    assert source_similarity(a, b) == pytest.approx(1.0)


def test_source_similarity_independent_generators_low():
    a = "def f(x):\n    return x + 1\n"
    b = "import numpy as np\n\nclass G:\n    def run(self):\n        return np.zeros(10)\n"
    assert source_similarity(a, b) < 0.6


# ── verdict parsing ────────────────────────────────────────────────────────────


def test_extract_json_from_fenced_prose():
    text = 'Here is my call:\n```json\n{"distillation": "pass"}\n```\nthanks'
    assert _extract_json(text) == {"distillation": "pass"}


def test_extract_json_skips_unbalanced_prefix():
    text = 'note {oops not json} then {"verdict": "fail"}'
    assert _extract_json(text) == {"verdict": "fail"}


def test_extract_json_raises_when_absent():
    with pytest.raises(JudgeError):
        _extract_json("no object here")


def test_judge_static_parses_both_verdicts():
    reply = json.dumps({
        "distillation": "fail", "distillation_reason": "10k float array",
        "benchmark_targeting": "warn", "benchmark_targeting_reason": "period 24 hardcoded",
    })
    distill, bench = judge_static({}, {}, FakeJudge(reply), interface_rule="rule")
    assert distill.failed and "array" in distill.reason
    assert bench.warned


def test_judge_static_rejects_invalid_verdict():
    with pytest.raises(JudgeError):
        judge_static({}, {}, FakeJudge('{"distillation": "maybe", "benchmark_targeting": "pass"}'),
                     interface_rule="rule")


def test_judge_copy_of_king_warn_is_not_a_hard_fail():
    v = judge_copy_of_king("a", "b", 0.8, FakeJudge('{"copy_of_king": "warn"}'))
    assert v.verdict == "pass"  # binary decision; a stray warn is not a reject


# ── OpenRouter client config ───────────────────────────────────────────────────


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(JudgeError):
        OpenRouterJudge().complete("sys", "user")


def test_build_judge_off_when_disabled(cfg):
    assert build_judge(cfg) is None  # shipped chain.toml has [judge] enabled = false


def test_build_judge_none_without_key(cfg, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    enabled = replace(cfg, judge=replace(cfg.judge, enabled=True))
    assert build_judge(enabled) is None  # enabled but no key ⇒ no-op


def test_build_judge_constructs_client_with_key(cfg, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    enabled = replace(cfg, judge=replace(cfg.judge, enabled=True))
    judge = build_judge(enabled)
    assert isinstance(judge, OpenRouterJudge)
    assert judge.model == "z-ai/glm-5.2"


# ── orchestrator: screen_repo ───────────────────────────────────────────────────


@pytest.fixture()
def judge_cfg(cfg):
    return replace(cfg, judge=replace(cfg.judge, enabled=True))


def test_screen_repo_clean_submission_passes(tmp_path, judge_cfg):
    _repo(tmp_path, "x = 1\n")
    reply = json.dumps({"distillation": "pass", "benchmark_targeting": "pass"})
    report = screen_repo(tmp_path, judge_cfg, FakeJudge(reply))
    assert report.ok
    assert not report.failures


def test_screen_repo_distillation_fail_rejects(tmp_path, judge_cfg):
    _repo(tmp_path, "W = [0.1, 0.2, 0.3]\n")
    reply = json.dumps({
        "distillation": "fail", "distillation_reason": "replays fitted weights",
        "benchmark_targeting": "pass",
    })
    report = screen_repo(tmp_path, judge_cfg, FakeJudge(reply))
    assert not report.ok
    assert [c.name for c in report.failures] == ["distillation"]


def test_screen_repo_benchmark_warn_is_not_a_failure(tmp_path, judge_cfg):
    _repo(tmp_path, "x = 1\n")
    reply = json.dumps({"distillation": "pass", "benchmark_targeting": "warn",
                        "benchmark_targeting_reason": "period 24"})
    report = screen_repo(tmp_path, judge_cfg, FakeJudge(reply))
    assert report.ok
    assert [c.name for c in report.warnings] == ["benchmark_targeting"]


def test_screen_repo_passes_evidence_to_judge(tmp_path, judge_cfg):
    _repo(tmp_path, "W = [0.123456789, 0.987654321]\n")
    judge = FakeJudge(json.dumps({"distillation": "pass", "benchmark_targeting": "pass"}))
    screen_repo(tmp_path, judge_cfg, judge)
    _, user_prompt = judge.calls[0]
    assert "total_numeric_literal_bytes" in user_prompt
    assert "private" in user_prompt.lower() and "rotat" in user_prompt.lower()


def test_screen_repo_fail_closed_on_judge_error(tmp_path, judge_cfg):
    _repo(tmp_path, "x = 1\n")
    report = screen_repo(tmp_path, judge_cfg, FakeJudge(JudgeError("boom")))
    assert not report.ok                        # fail_closed (default) ⇒ reject
    assert {c.name for c in report.failures} == {"distillation", "benchmark_targeting"}


def test_screen_repo_fail_open_on_judge_error(tmp_path, cfg):
    _repo(tmp_path, "x = 1\n")
    open_cfg = replace(cfg, judge=replace(cfg.judge, enabled=True, fail_closed=False))
    report = screen_repo(tmp_path, open_cfg, FakeJudge(JudgeError("boom")))
    assert report.ok                            # fail_open ⇒ warn, don't block
    assert {c.name for c in report.warnings} == {"distillation", "benchmark_targeting"}


# ── copy-of-king band ───────────────────────────────────────────────────────────


def test_screen_repo_copy_mechanical_reject_no_llm(tmp_path, cfg):
    king = "def f(x):\n    return x + 1\n"
    _repo(tmp_path, '"""reflowed"""\ndef f(x):\n        return x+1  # copy\n')
    pub = replace(cfg, judge=replace(cfg.judge, enabled=True, publish_king=True))
    # A judge that would explode if called — proves the reject was mechanical.
    judge = FakeJudge(JudgeError("must not be called for static or copy"))
    # Make the static call succeed but the copy band must not invoke the LLM.
    judge.reply = json.dumps({"distillation": "pass", "benchmark_targeting": "pass"})
    report = screen_repo(tmp_path, pub, judge, king_source=king)
    copy = next(c for c in report.checks if c.name == "copy_of_king")
    assert copy.failed and "mechanical" in copy.reason
    assert len(judge.calls) == 1                 # only the static call, no copy call


def test_screen_repo_copy_middle_band_asks_llm(tmp_path, cfg):
    king = "def f(x):\n    y = x + 1\n    return y\n"
    # Restructured enough to drop below reject but above review.
    _repo(tmp_path, "def f(z):\n    total = z\n    total = total + 1\n    return total\n")
    pub = replace(
        cfg,
        judge=replace(cfg.judge, enabled=True, publish_king=True,
                      copy_reject_similarity=0.99, copy_review_similarity=0.30),
    )
    replies = iter([
        json.dumps({"distillation": "pass", "benchmark_targeting": "pass"}),
        json.dumps({"copy_of_king": "fail", "copy_of_king_reason": "same algo restructured"}),
    ])

    class SeqJudge:
        calls = 0

        def complete(self, system, user):
            SeqJudge.calls += 1
            return next(replies)

    report = screen_repo(tmp_path, pub, SeqJudge(), king_source=king)
    copy = next(c for c in report.checks if c.name == "copy_of_king")
    assert copy.failed and SeqJudge.calls == 2   # static + copy call


def test_screen_repo_skips_copy_when_not_published(tmp_path, judge_cfg):
    _repo(tmp_path, "def f(x):\n    return x + 1\n")
    judge = FakeJudge(json.dumps({"distillation": "pass", "benchmark_targeting": "pass"}))
    report = screen_repo(tmp_path, judge_cfg, judge, king_source="def f(x):\n    return x + 1\n")
    assert all(c.name != "copy_of_king" for c in report.checks)  # publish_king is false
