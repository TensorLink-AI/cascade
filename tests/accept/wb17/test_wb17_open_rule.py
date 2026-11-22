"""WB17: the lane B opening rule (plan §3.0 lane B track, §12b.4 LB4, D41;
T-rows TB17 and TB18).

The WB17 row, verbatim:

    | WB17 | Opening rule (D41, LB4): `open_rule` implementation; driver state
    records the rule and the satisfying evidence | `lane_b_driver` state
    `gate_rule`, `gate_evidence` | WB10 | ci + testnet |
    `scripts/laned/accept WB17` (TB17–TB18) |

TB17 / TB18, verbatim (§4):

    | TB17 | **(rev4)** Opening rule (D41 default) on fixtures: tokens met +
    king-check pass | Gate opens; driver state records rule + evidence |
    | TB18 | **(rev4)** Tokens met, king check failing (or tokens short) |
    Gate stays closed |

D41 (§6): "Lane B opening rule (LB4) | (a) `open_tokens` + a passing lane A
king check (**recommended**); (b) `open_tokens` alone; (c) co-gate with
re-measured thresholds". §17.4 register: "| D41 | decided | (a) `open_tokens`
+ a passing lane A king check | S | owner (default), 2026-11-21 |".

LB4 (§12b.4): "The new key is `[lane_b] open_rule` (`"tokens_king_check"` |
`"tokens"` | `"cogate"`; default `"cogate"`, so unarmed digests don't move)."

The plan does not say what happens when the evidence later stops satisfying
the rule. The driver's gate is documented as a one-way latch
(``LaneBDriverState.gate_opened_block``: "One-way: lane B never re-closes —
in-flight legs anchor on the king's routing, and later drift is caught per
leg by the rebase ``routing_stale`` check"), and D45 puts lane B live at full
share from the block the rule is satisfied. WB17 keeps the latch for every
rule. Contract: README.md.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from . import helpers as h
from .helpers import HK_A, HK_B, KING, OPEN_TOKENS, PREV_KING


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _assert_closed_uncharged(sc, started: list, hotkey: str) -> None:
    """A closed gate dispatches nothing and charges nothing: the funded entry
    stays ``queued`` with no attempt, requeue or error class, and the gate
    record stays empty."""
    assert started == [], f"a closed gate started legs: {[x.sub.hotkey[:9] for x in started]}"
    assert sc.ops.dispatched == [], "a closed gate dispatched a leg"
    assert hotkey not in sc.open_legs(), "a closed gate opened a leg"
    assert int(sc.driver.st.gate_opened_block) == 0, (
        f"the gate opened at block {sc.driver.st.gate_opened_block} while its rule was unmet")
    e = sc.entry(hotkey)
    assert e.status == "queued", f"entry status {e.status!r} behind a closed gate (stays queued)"
    assert (e.attempts, e.requeues, e.last_error_class) == (0, 0, ""), (
        f"entry charged behind a closed gate: attempts={e.attempts} requeues={e.requeues} "
        f"class={e.last_error_class!r}")
    assert h.state_field(sc.driver.st, "gate_rule") == "", "gate_rule is '' while closed"
    assert h.state_field(sc.driver.st, "gate_evidence") == {}, "gate_evidence is {} while closed"


def _closed_reason(sc) -> str:
    reason = h.state_field(sc.driver.st, "gate_closed_reason")
    disk = sc.on_disk()
    assert disk.get("gate_closed_reason") == reason, (
        f"gate_closed_reason not persisted: memory {reason!r}, disk "
        f"{disk.get('gate_closed_reason')!r}")
    return reason


def _open(sc, *, tokens: int = 150, check: dict | None = None, era: int = 0) -> list:
    """Fund HK_A, meet the rule in ``era`` with one king, tick once."""
    sc.fund(HK_A, 10)
    sc.tokens(tokens)
    sc.ops.check = check
    return sc.tick(era, KING)


def _assert_opened(sc, started: list, rule: str) -> None:
    assert HK_A in {x.sub.hotkey for x in started}, (
        f"[lane_b] open_rule = {rule!r} met: the gate opens and the funded entry starts in "
        f"the same tick; started={[x.sub.hotkey[:9] for x in started]}, "
        f"gate_opened_block={sc.driver.st.gate_opened_block}, routing={sc.driver.st.routing}")
    assert int(sc.driver.st.gate_opened_block) == sc.block, (
        f"gate_opened_block {sc.driver.st.gate_opened_block} != the opening tick's block "
        f"{sc.block}")
    assert h.state_field(sc.driver.st, "gate_rule") == rule, (
        f"gate_rule records the configured rule {rule!r}; got {sc.driver.st.gate_rule!r}")
    assert _closed_reason(sc) == "", "gate_closed_reason is '' once the gate is open"


# ── plan text ────────────────────────────────────────────────────────────────


def test_plan_rows_pinned():
    text = h.PLAN.read_text(encoding="utf-8")
    assert ("| WB17 | Opening rule (D41, LB4): `open_rule` implementation; driver state "
            "records the rule and the satisfying evidence | `lane_b_driver` state "
            "`gate_rule`, `gate_evidence` | WB10 | ci + testnet | "
            "`scripts/laned/accept WB17` (TB17–TB18) |") in text
    assert ("| TB17 | **(rev4)** Opening rule (D41 default) on fixtures: tokens met + "
            "king-check pass | Gate opens; driver state records rule + evidence |") in text
    assert ("| TB18 | **(rev4)** Tokens met, king check failing (or tokens short) | "
            "Gate stays closed |") in text
    assert "| TB17, TB18 | WB17 |" in text
    assert ("| D41 | **(rev4)** Lane B opening rule (LB4) | (a) `open_tokens` + a passing "
            "lane A king check (**recommended**); (b) `open_tokens` alone; (c) co-gate with "
            "re-measured thresholds |") in text
    assert ("| D41 | decided | (a) `open_tokens` + a passing lane A king check | S | "
            "owner (default), 2026-11-21 |") in text
    flat = _flat(text)
    assert ("The new key is `[lane_b] open_rule` (`\"tokens_king_check\"` | `\"tokens\"` | "
            "`\"cogate\"`; default `\"cogate\"`, so unarmed digests don't move).") in flat
    assert ("`open_rule` and the `[lane_b]` thresholds are overlay-settable "
            "(`ALLOWED[\"lane_b\"] = None`).") in flat


# ── config (WB10's parse; the rule forms and refusal at load) ────────────────


@pytest.mark.parametrize("rule", h.RULES)
def test_rule_forms_parse(rule):
    lb = h.lane_b_cfg(rule)
    assert lb.open_rule == rule


def test_shipped_default_is_cogate():
    assert h.lane_b_cfg(None).open_rule == h.RULE_COGATE


@pytest.mark.parametrize("bad", ["tokens+king_check", "TOKENS", "", "open", 1, True, None,
                                 ["tokens"]])
def test_invalid_rule_refused_at_load(bad):
    from cascade.shared.config import parse_lane_b

    raw = h.lane_b_raw(None)
    raw["open_rule"] = bad
    with pytest.raises(ValueError, match="open_rule"):
        parse_lane_b(raw)


# ── TB17: the rule met opens the gate and records rule + evidence ────────────


def test_tb17_tokens_king_check_opens_and_records_rule_and_evidence(tmp_path):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    check = h.king_check("pass", round_id="v2-r0100")
    started = _open(sc, tokens=150, check=check)
    _assert_opened(sc, started, h.RULE_TOKENS_KING_CHECK)
    assert sc.ops.check_reads >= 1, "tokens_king_check reads the lane A king check"
    ev = h.state_field(sc.driver.st, "gate_evidence")
    assert isinstance(ev, dict) and ev, f"gate_evidence is a non-empty object; got {ev!r}"
    assert ev.get("lineage_tokens") == 150, (
        f"gate_evidence.lineage_tokens is the lineage count that met the rule; got {ev!r}")
    assert ev.get("open_tokens") == OPEN_TOKENS, (
        f"gate_evidence.open_tokens is the configured threshold; got {ev!r}")
    kc = ev.get("king_check")
    assert isinstance(kc, dict), f"gate_evidence.king_check records the passing check; {ev!r}"
    assert kc.get("result") == "pass", kc
    assert kc.get("round_id") == "v2-r0100", kc
    assert kc.get("king_pointer") == KING, kc
    disk = sc.on_disk()
    assert disk.get("gate_rule") == h.RULE_TOKENS_KING_CHECK, disk.get("gate_rule")
    assert disk.get("gate_evidence") == ev, (
        f"gate_evidence persisted as written: disk {disk.get('gate_evidence')!r} vs {ev!r}")
    assert disk.get("gate_opened_block") == sc.block


def test_tb17_tokens_rule_opens_on_tokens_alone(tmp_path):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS)
    started = _open(sc, tokens=OPEN_TOKENS, check=None)
    _assert_opened(sc, started, h.RULE_TOKENS)
    ev = h.state_field(sc.driver.st, "gate_evidence")
    assert isinstance(ev, dict) and ev.get("lineage_tokens") == OPEN_TOKENS \
        and ev.get("open_tokens") == OPEN_TOKENS, (
            f"gate_evidence records lineage_tokens and open_tokens (met at equality); {ev!r}")


@pytest.mark.parametrize("rule", [h.RULE_TOKENS_KING_CHECK, h.RULE_TOKENS])
def test_tb17_evidence_strict_json_and_deterministic(tmp_path, rule):
    evs = []
    for i, t in enumerate((1_000_000.0, 2_345_678.5)):
        sc = h.Scenario(tmp_path / f"run{i}", rule, clock_t=t)
        _assert_opened(sc, _open(sc, tokens=150, check=h.king_check("pass")), rule)
        ev = h.state_field(sc.driver.st, "gate_evidence")
        assert h.is_strict_json(ev), (
            f"gate_evidence is strict JSON (plain dicts / lists / str / int / finite "
            f"float / bool, string keys): {ev!r}")
        text = sc.state_path.read_text(encoding="utf-8")
        assert "NaN" not in text and "Infinity" not in text, "state file is not strict JSON"
        evs.append(json.dumps(ev, sort_keys=True, allow_nan=False))
    assert evs[0] == evs[1], (
        "gate_evidence is deterministic: the same chain facts in another work dir and at "
        f"another wall-clock time give the same bytes:\n  {evs[0]}\n  {evs[1]}")


def test_tb17_closed_then_passing_check_opens_with_that_check(tmp_path):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    started = _open(sc, tokens=150, check=h.king_check("fail", round_id="v2-r0100"))
    _assert_closed_uncharged(sc, started, HK_A)
    assert _closed_reason(sc) == h.KING_CHECK_NOT_PASSED
    sc.ops.check = h.king_check("pass", round_id="v2-r0101", block=1900)
    started = sc.tick(1, KING)
    _assert_opened(sc, started, h.RULE_TOKENS_KING_CHECK)
    kc = sc.driver.st.gate_evidence.get("king_check") or {}
    assert kc.get("round_id") == "v2-r0101" and kc.get("result") == "pass", (
        f"gate_evidence records the check that satisfied the rule, not an earlier one: {kc!r}")


# ── TB18: tokens met + king check failing, or tokens short ⇒ closed ──────────


@pytest.mark.parametrize("result", ["fail", "deferred"])
def test_tb18_king_check_not_passed_stays_closed(tmp_path, result):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    started = _open(sc, tokens=150, check=h.king_check(result))
    _assert_closed_uncharged(sc, started, HK_A)
    assert _closed_reason(sc) == h.KING_CHECK_NOT_PASSED, (
        f"a king check with result {result!r} is not a passing one: typed reason "
        f"{h.KING_CHECK_NOT_PASSED!r}; got {sc.driver.st.gate_closed_reason!r}")
    started = sc.tick(1, KING)          # repeated ticks: still closed, still uncharged
    _assert_closed_uncharged(sc, started, HK_A)


def test_tb18_king_check_unavailable_stays_closed(tmp_path):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    started = _open(sc, tokens=150, check=None)
    _assert_closed_uncharged(sc, started, HK_A)
    assert sc.ops.check_reads >= 1, "tokens met: the rule reads the king check"
    assert _closed_reason(sc) == h.KING_CHECK_UNAVAILABLE, (
        f"no readable king check: typed reason {h.KING_CHECK_UNAVAILABLE!r}; "
        f"got {sc.driver.st.gate_closed_reason!r}")


@pytest.mark.parametrize("rule", [h.RULE_TOKENS_KING_CHECK, h.RULE_TOKENS])
def test_tb18_tokens_short_stays_closed(tmp_path, rule):
    sc = h.Scenario(tmp_path, rule)
    started = _open(sc, tokens=OPEN_TOKENS - 1, check=h.king_check("pass"))
    _assert_closed_uncharged(sc, started, HK_A)
    assert _closed_reason(sc) == h.TOKENS_SHORT, (
        f"lineage {OPEN_TOKENS - 1} < open_tokens {OPEN_TOKENS}: typed reason "
        f"{h.TOKENS_SHORT!r} (checked before the king check); "
        f"got {sc.driver.st.gate_closed_reason!r}")


def test_tb18_reason_is_typed_enum(tmp_path):
    seen = set()
    for i, (tokens, check) in enumerate(((10, h.king_check("pass")),
                                         (150, h.king_check("fail")), (150, None))):
        sc = h.Scenario(tmp_path / f"s{i}", h.RULE_TOKENS_KING_CHECK)
        _open(sc, tokens=tokens, check=check)
        seen.add(_closed_reason(sc))
    assert seen == set(h.CLOSED_REASONS), (
        f"closed reasons are exactly {sorted(h.CLOSED_REASONS)}; got {sorted(seen)}")


# ── the latch: evidence that stops satisfying the rule ───────────────────────


def test_gate_stays_open_when_evidence_stops_satisfying(tmp_path):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    _assert_opened(sc, _open(sc, tokens=150, check=h.king_check("pass")),
                   h.RULE_TOKENS_KING_CHECK)
    opened, rule = int(sc.driver.st.gate_opened_block), sc.driver.st.gate_rule
    ev = json.loads(json.dumps(sc.driver.st.gate_evidence))
    sc.ops.check = h.king_check("fail", round_id="v2-r0102", block=1900)
    sc.fund(HK_B, 20, "web")
    started = sc.tick(1, KING)
    assert HK_B in {x.sub.hotkey for x in started}, (
        "the gate is a one-way latch: a later failing king check does not re-close it")
    assert int(sc.driver.st.gate_opened_block) == opened
    assert sc.driver.st.gate_rule == rule
    assert sc.driver.st.gate_evidence == ev, (
        "gate_evidence keeps the facts that opened the gate; later ticks never rewrite it")
    assert sc.driver.st.gate_closed_reason == ""


def test_gate_record_survives_restart(tmp_path):
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    _assert_opened(sc, _open(sc, tokens=150, check=h.king_check("pass")),
                   h.RULE_TOKENS_KING_CHECK)
    before = (int(sc.driver.st.gate_opened_block), sc.driver.st.gate_rule,
              json.loads(json.dumps(sc.driver.st.gate_evidence)))
    sc.restart()
    after = (int(sc.driver.st.gate_opened_block), h.state_field(sc.driver.st, "gate_rule"),
             h.state_field(sc.driver.st, "gate_evidence"))
    assert after == before, f"the gate record survives a restart: {before!r} -> {after!r}"
    sc.ops.check = None
    sc.fund(HK_B, 20, "web")
    started = sc.tick(1, KING)
    assert HK_B in {x.sub.hotkey for x in started}, (
        "after a restart the latched gate stays open without re-reading the rule's facts")
    assert sc.on_disk().get("gate_evidence") == before[2]


# ── production wiring ────────────────────────────────────────────────────────


def test_runner_ops_king_check_reads_the_verified_receipt():
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    rec = h.king_check("pass", round_id="v2-r0100")
    runner = SimpleNamespace(_receipt_king_check=lambda: (dict(rec), 900, "v2-r0100"))
    ops = RunnerLaneBOps(runner, None, None)
    assert callable(getattr(ops, "king_check", None)), (
        "RunnerLaneBOps.king_check() is the driver's read of the latest lane A king check")
    assert ops.king_check() == rec, (
        "RunnerLaneBOps.king_check() returns the record of the runner's verified receipt "
        "read (_receipt_king_check()[0])")
    none_runner = SimpleNamespace(_receipt_king_check=lambda: None)
    assert RunnerLaneBOps(none_runner, None, None).king_check() is None


# ── invariant 1 (paired per invariant 10) ────────────────────────────────────


def test_inv1_open_rule_unchanged_when_unarmed(tmp_path):
    """``open_rule`` at its default ("cogate"): today's gate exactly — one
    king is never enough (the co-gate compares the previous era's king), the
    second era's king opens it once the tokens are met, and the king check is
    never read."""
    sc = h.Scenario(tmp_path, None)
    assert sc.lb.open_rule == h.RULE_COGATE
    started = _open(sc, tokens=150, check=h.king_check("pass"), era=0)
    assert started == [] and int(sc.driver.st.gate_opened_block) == 0
    started = sc.tick(1, KING.replace("king@", "king2@"))
    assert HK_A in {x.sub.hotkey for x in started}, sc.driver.st.routing
    assert sc.ops.check_reads == 0, "the co-gate never reads the lane A king check"

    short = h.Scenario(tmp_path / "short", None)
    short.fund(HK_A, 10)
    short.tokens(OPEN_TOKENS - 1)
    assert short.tick(0, PREV_KING) == []
    assert short.tick(1, KING) == [] and int(short.driver.st.gate_opened_block) == 0
    assert short.ops.check_reads == 0


def test_inv1_open_rule_armed_nonempty_lane_d_effect(tmp_path):
    """Armed (D41 a): the same facts open the gate on the FIRST king — the
    rule needs no previous king — and the king check is read."""
    sc = h.Scenario(tmp_path, h.RULE_TOKENS_KING_CHECK)
    started = _open(sc, tokens=150, check=h.king_check("pass"), era=0)
    _assert_opened(sc, started, h.RULE_TOKENS_KING_CHECK)
    assert sc.ops.check_reads >= 1
