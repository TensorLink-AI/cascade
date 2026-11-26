"""Lane B domain tags: which domain evals each judged generator ACTUALLY improved.

Owner 2026-11-25. RECORDS AND DISPLAY ONLY: nothing here feeds weights,
scoring, verdicts, merge acceptance, a digest, a contract or any consensus
value. The tags are a pure function of the validator's SIGNED lane B verdict
rows (the receipt's ``lane_b`` list, :func:`cascade.validator.expert_lane.
judgements_record`) — nothing is re-evaluated. They are written only into
unsigned, presentational documents: the ``lane_b_domains`` column of a
``receipts/index.json`` row and ``status/v2.json`` ``lane_b.domain_tags``.
Every reader tolerates their absence; the signed receipt never carries them.

**"Improved"** is exactly the slot gain test of
:func:`cascade.eval.expert_lane.judge_expert_slots` (rev6.22): a slot is
improved iff it was conclusive and, for CRPS or MASE, the max-T family LCB is
``>= gain_bar`` (``gain_margin`` for a declared target,
``undeclared_gain_margin`` for every other slot) with a strictly positive
point gain, AND the other metric is non-inferior. The verdict records the
non-inferiority outcome in its reason: a slot whose gain test failed (either
part) carries ``reason`` ``"no gain: ..."``, every other conclusive slot
cleared it. So a row is ``improved`` iff it is a judged (``stage ==
"verdict"``), conclusive row whose reason is not ``"no gain"`` AND some
metric's recorded ``metric_lcbs >= gain_bar`` with ``metric_gains > 0``.
Guards are NOT part of "improved": a slot that cleared its gain test but was
blocked by a leg guard is ``improved`` with ``claimed: false`` (the guard
reason is kept). ``margin = target_lcb - gain_bar`` — the verdict's own best
metric LCB (the clearing metric when one cleared) over the bar applied.

Statuses per slot: ``improved`` | ``not_improved`` | ``inconclusive`` (too few
windows / clusters / items: never a pass) | ``not_evaluated`` (an entry
refused before judging: stale / targets / selection / merge). Slots include
``horizon:<h>`` and ``suite:<s>`` targets alongside the GIFT-Eval domains.
A pre-rev6.22 row (no ``gain_bar``) uses the same reason rule; its margin is
``None`` (the bar was not recorded).

Stdlib only, so the receipt-index builder can import it cheaply.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

STATUSES = ("improved", "not_improved", "inconclusive", "not_evaluated")
METRICS = ("crps", "mase")


def _f(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(x, nd)


def slot_tag(row: dict) -> dict:
    """The tag of ONE verdict row: ``{status, declared, gain_bar, lcb, gain,
    margin, metric, claimed, reason}`` (see the module docstring)."""
    stage = str(row.get("stage") or "")
    reason = str(row.get("reason") or "")
    lcb, bar, gain = _f(row.get("target_lcb")), _f(row.get("gain_bar")), _f(row.get("target_gain"))
    tag = {"status": "not_evaluated", "declared": bool(row.get("declared", True)),
           "gain_bar": _r(bar), "lcb": _r(lcb), "gain": _r(gain), "margin": None,
           "metric": None, "claimed": bool(row.get("passed")), "reason": reason}
    if stage != "verdict":
        tag["claimed"] = False
        return tag
    if row.get("inconclusive") or reason.startswith("inconclusive"):
        tag["status"] = "inconclusive"
        return tag
    gain_failed = reason.startswith("no gain")
    if bar is None:
        # pre-rev6.22 single-target row: the reason rule alone, no bar recorded
        improved = not gain_failed and lcb is not None
    else:
        lcbs, gains = (d if isinstance(d := row.get(k), dict) else {}
                       for k in ("metric_lcbs", "metric_gains"))
        cleared = [m for m in METRICS
                   if (lo := _f(lcbs.get(m))) is not None and lo >= bar
                   and (g := _f(gains.get(m))) is not None and g > 0.0]
        improved = not gain_failed and bool(cleared)
        if lcb is not None:
            tag["margin"] = _r(lcb - bar)
        metric = str(row.get("gain_metric") or "")
        if not metric and lcb is not None:
            # a guard-blocked slot records no gain_metric: the metric whose
            # family LCB the verdict reported (its best)
            metric = next((m for m in METRICS if _f(lcbs.get(m)) == lcb), "")
        tag["metric"] = metric or None
    tag["status"] = "improved" if improved else "not_improved"
    return tag


def lane_b_domain_tags(rows: Iterable[dict] | None) -> list[dict]:
    """Per judged generator (``hotkey`` + ``leg`` pointer), from the receipt's
    signed lane B verdict rows: ``{hotkey, leg, targets, improved,
    not_improved, inconclusive, not_evaluated, domains: {slot: tag}}``.
    ``[]`` for ``None`` / no rows. Never mutates ``rows``."""
    groups: dict[tuple[str, str], dict] = {}
    for row in rows or ():
        if not isinstance(row, dict) or "error" in row or not row.get("domain"):
            continue
        hk, leg = str(row.get("hotkey") or ""), str(row.get("leg") or "")
        ts = row.get("targets")
        ts = ts if isinstance(ts, list) and ts else [row.get("domain")]
        g = groups.setdefault((hk, leg), {
            "hotkey": hk, "leg": leg, "targets": [str(t) for t in ts],
            **{s: [] for s in STATUSES}, "domains": {}})
        slot = str(row["domain"])
        if slot in g["domains"]:
            continue
        tag = slot_tag(row)
        g["domains"][slot] = tag
        g[tag["status"]].append(slot)
    return [groups[k] for k in sorted(groups)]


def tags_line(gen: dict) -> str:
    """``energy +0.0123 (crps), h720 +0.0040 · not improved web_cloudops -0.0100``
    — one human line of a generator's tags (dashboard / ``cascade status``)."""
    doms = gen.get("domains") or {}

    def one(s: str) -> str:
        t = doms.get(s) or {}
        m = _f(t.get("margin"))
        num = "" if m is None else f" {m:+.4f}"
        star = "" if t.get("declared", True) else "*"
        return f"{s}{star}{num}"

    parts = []
    for status, label in (("improved", "improved"), ("not_improved", "not improved"),
                          ("inconclusive", "inconclusive"), ("not_evaluated", "not evaluated")):
        slots = list(gen.get(status) or [])
        if slots:
            parts.append(f"{label} " + ", ".join(one(s) for s in slots))
    return " · ".join(parts) or "no slots"


__all__ = ["METRICS", "STATUSES", "lane_b_domain_tags", "slot_tag", "tags_line"]
