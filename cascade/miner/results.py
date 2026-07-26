"""``cascade results`` — pull YOUR result out of a round's public receipt.

The signed :class:`~cascade.shared.receipt.RoundReceipt` already carries
everything a miner needs to see how their submission fared — the heat
standings (rank, relative score, raw CRPS/MASE, failure notes) ride the
embedded manifest, and a finalist's full per-window duel scores are in
``entry_scores`` — but until this command the only consumers were the web
dashboard and ``cascade-audit``. This module turns one receipt + one identity
(a UID or hotkey) into a miner-centric report: where you placed, how far you
were from advancing, why you dropped out (when the trainer attached a note),
and — for a finalist — where the duel was won or lost, per eval domain and
per upstream source feed.

Everything here is *presentational*: it reads the public receipt and never
touches consensus. Pure functions over a parsed receipt, so the report logic
is unit-testable offline; the CLI wiring (fetch + render) lives in
:mod:`cascade.miner.cli`.

Two deliberate reading aids baked into the diagnosis text:

* Heat rank is judged against the *cutoff* (the last advancing entrant), not
  just the leader — "0.4% behind the last finalist" is the actionable number
  for a screened-out miner.
* A duel loss distinguishes "worse than the king" (lcb ≤ 0) from "better but
  not by the required margin" (0 < lcb ≤ margin) — the second is a
  keep-going signal the bare verdict hides.
"""

from __future__ import annotations

import math
from typing import Any

from ..shared.receipt import EntryScores, RoundReceipt

# Cap on per-source rows in the duel breakdown: enough to see a pattern,
# few enough to read in a terminal.
WORST_SOURCES_SHOWN = 5
BEST_SOURCES_SHOWN = 3


def _finite(x: object) -> float | None:
    """None unless ``x`` is a finite number (report dicts stay strict-JSON)."""
    if x is None:
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _matches(who: str, uid: int | None, hotkey: str | None) -> bool:
    """Does the identity query ``who`` (uid digits or ss58 hotkey) name this
    (uid, hotkey) pair?"""
    if who.isdigit():
        return uid is not None and int(who) == int(uid)
    return hotkey is not None and who == hotkey


def resolve_identity(receipt: RoundReceipt, who: str) -> tuple[int | None, str | None]:
    """Resolve ``who`` to a ``(uid, hotkey)`` pair from the receipt's own
    records (participants, heat entrants, entry scores) — no chain needed.
    Either side may stay None when the receipt never mentions the identity.
    """
    for p in receipt.participants:
        if _matches(who, p.uid, p.hotkey):
            return p.uid, p.hotkey
    heat = _heat_block(receipt)
    for e in (heat.get("entrants") or []) if heat else []:
        if _matches(who, e.get("uid"), e.get("hotkey")):
            return e.get("uid"), e.get("hotkey")
    for es in receipt.entry_scores:
        if _matches(who, es.uid, es.hotkey):
            return es.uid, es.hotkey
    return (int(who), None) if who.isdigit() else (None, who)


def _heat_block(receipt: RoundReceipt) -> dict | None:
    """The manifest's informational heat block as a plain dict, or None."""
    heat = receipt.manifest.get("heat") if isinstance(receipt.manifest, dict) else None
    return heat if isinstance(heat, dict) else None


# ── heat section ─────────────────────────────────────────────────────────────


def _heat_section(receipt: RoundReceipt, who: str) -> dict | None:
    """This miner's heat standing + derived gap analysis, or None when the
    round ran no screen (or the receipt predates heat standings)."""
    heat = _heat_block(receipt)
    if heat is None:
        return None
    entrants = [e for e in (heat.get("entrants") or []) if isinstance(e, dict)]
    me = next((e for e in entrants if _matches(who, e.get("uid"), e.get("hotkey"))), None)
    scored = sorted((e for e in entrants if e.get("rank") is not None),
                    key=lambda e: int(e["rank"]))
    finalists = int(heat.get("finalists") or 0)
    # The cutoff is the LAST entrant that advanced: beat their score and you
    # are in the duel. (min() guards a screen where fewer scored than the
    # finalist budget — then everyone scored advanced and there is no cutoff.)
    cutoff = scored[min(finalists, len(scored)) - 1] if scored and finalists else None
    out: dict[str, Any] = {
        "screen_size": heat.get("screen_size"),
        "finalists": finalists,
        "n_entrants": len(entrants),
        "n_scored": len(scored),
        "n_windows": heat.get("n_windows"),
        "leader_lcb": _finite(heat.get("leader_lcb")),
        "entrant": None,
    }
    if me is None:
        return out
    rel = _finite(me.get("rel_score"))
    cutoff_rel = _finite(cutoff.get("rel_score")) if cutoff else None
    entry: dict[str, Any] = {
        "status": me.get("status"),
        "rank": me.get("rank"),
        "rel_score": rel,
        "p_best": _finite(me.get("p_best")),
        "crps": _finite(me.get("crps")),
        "mase": _finite(me.get("mase")),
        "note": me.get("note"),
        "gen_ref": me.get("gen_ref"),
        "pct_behind_leader": (rel - 1.0) * 100.0 if rel is not None else None,
        "pct_behind_cutoff": ((rel - cutoff_rel) * 100.0
                              if rel is not None and cutoff_rel is not None
                              and me.get("status") == "screened" else None),
        "cutoff_rel_score": cutoff_rel,
    }
    entry["diagnosis"] = _heat_diagnosis(entry, out)
    out["entrant"] = entry
    return out


def _heat_diagnosis(entry: dict, heat: dict) -> str:
    """One human sentence saying what the heat outcome MEANS for this miner."""
    status = entry.get("status")
    note = entry.get("note")
    if status == "advanced":
        return (f"advanced to the duel (rank {entry['rank']} of {heat['n_scored']} "
                "scored — see the duel section for how the final went)")
    if status == "screened":
        behind_leader = entry.get("pct_behind_leader")
        behind_cut = entry.get("pct_behind_cutoff")
        parts = [f"screened out at rank {entry['rank']} of {heat['n_scored']}"]
        if behind_leader is not None:
            parts.append(f"{behind_leader:.1f}% behind the leader")
        if behind_cut is not None:
            parts.append(f"{behind_cut:.1f}% behind the last finalist — that is "
                         "the gap to close")
        lcb = heat.get("leader_lcb")
        if lcb is not None and lcb <= 0:
            parts.append("(the screen itself was not decisive: leader_lcb ≤ 0, so "
                         "the top places were statistically interchangeable on this "
                         "round's evidence)")
        return "; ".join(parts)
    if status == "failed_train":
        return ("your generator never produced a trained checkpoint" +
                (f" — trainer note: {note}" if note else
                 " (crash, timeout, or unreachable artifact; no reason was recorded "
                 "for this round)"))
    if status == "failed_screen":
        return ("your generator trained, but scoring the checkpoint failed" +
                (f" — trainer note: {note}" if note else
                 " (typically degenerate output: NaNs, constant series, or an "
                 "output-contract violation)"))
    if status == "duplicate":
        return ("dropped as a duplicate" +
                (f" — {note}" if note else
                 ": your corpus was byte-identical to an earlier reveal, and the "
                 "earliest reveal keeps the slot"))
    return f"status: {status}"


# ── duel section ─────────────────────────────────────────────────────────────


def _geomean_of(records: list) -> float | None:
    """``global_geomean`` over WindowScoreRecords — the exact aggregate the duel
    reports (numpy import stays local so the report logic imports without the
    eval stack)."""
    if not records:
        return None
    from ..eval.scoring import global_geomean

    return _finite(global_geomean([r.to_score() for r in records]))


def _source_breakdown(mine: EntryScores, theirs: EntryScores) -> list[dict]:
    """Per-upstream-feed comparison of two paired score rows, worst-first.

    King and challenger score the same windows in the same order, so grouping
    both by the window's ``source`` (its cluster key; ``series_id`` when the
    pool is unlabeled) yields paired groups. ``ratio`` is your geomean over
    theirs on that feed — above 1.0 you lost that feed, below you won it.
    """
    if len(mine.scores) != len(theirs.scores):
        return []  # differently-shaped rows can't be paired; skip, don't guess
    groups: dict[str, tuple[list, list]] = {}
    for m, t in zip(mine.scores, theirs.scores, strict=True):
        key = m.source or m.series_id
        groups.setdefault(key, ([], []))[0].append(m)
        groups[key][1].append(t)
    rows = []
    for key, (ms, ts) in groups.items():
        own, opp = _geomean_of(ms), _geomean_of(ts)
        if own is None or opp is None or opp <= 0:
            continue
        rows.append({"source": key, "n_windows": len(ms), "own": own, "opp": opp,
                     "ratio": own / opp})
    rows.sort(key=lambda r: -r["ratio"])
    return rows


def _duel_section(receipt: RoundReceipt, who: str) -> dict | None:
    """This miner's duel outcome — verdict headline, per-size geomeans, and the
    per-domain / per-source breakdown against the opponent. None unless the
    miner is one of the scored duel entries (king or the finalist)."""
    mine = [es for es in receipt.entry_scores if _matches(who, es.uid, es.hotkey)]
    if not mine:
        return None
    role = mine[0].role
    other_role = "challenger" if role == "king" else "king"
    theirs_by_size = {es.size: es for es in receipt.entry_scores if es.role == other_role}
    sizes = []
    paired_sources: list[dict] = []
    for es in mine:
        opp = theirs_by_size.get(es.size)
        sizes.append({
            "size": es.size,
            "own_geomean": _geomean_of(list(es.scores)),
            "opp_geomean": _geomean_of(list(opp.scores)) if opp else None,
            "n_windows": len(es.scores),
        })
        if opp is not None:
            paired_sources.extend(_source_breakdown(es, opp))
    # Re-pool per-source rows across sizes (a feed can appear once per size).
    pooled: dict[str, dict] = {}
    for r in paired_sources:
        p = pooled.setdefault(r["source"], {"source": r["source"], "n_windows": 0,
                                            "_own": 0.0, "_opp": 0.0})
        p["n_windows"] += r["n_windows"]
        # Weight each size's geomean by its window count in log space so the
        # pooled ratio stays a geometric quantity, like everything else here.
        p["_own"] += math.log(max(r["own"], 1e-12)) * r["n_windows"]
        p["_opp"] += math.log(max(r["opp"], 1e-12)) * r["n_windows"]
    source_rows = []
    for p in pooled.values():
        n = p["n_windows"]
        own = math.exp(p["_own"] / n)
        opp = math.exp(p["_opp"] / n)
        source_rows.append({"source": p["source"], "n_windows": n,
                            "own": own, "opp": opp, "ratio": own / opp})
    source_rows.sort(key=lambda r: -r["ratio"])

    v = receipt.verdict
    verdict = None
    if v is not None:
        verdict = {
            "lcb": _finite(v.lcb), "margin": _finite(v.margin),
            "challenger_wins_round": v.challenger_wins_round,
            "inconclusive": v.inconclusive, "dethroned": v.dethroned,
            "king_geomean": _finite(v.king_geomean),
            "chal_geomean": _finite(v.chal_geomean),
            "win_rate": _finite(v.win_rate), "wilcoxon_p": _finite(v.wilcoxon_p),
            "boot_p50": _finite(v.boot_p50), "boot_p95": _finite(v.boot_p95),
            "gift_gate_passed": v.gift_gate_passed,
            "per_domain_win_rate": v.per_domain_win_rate,
            "note": v.note,
        }
    out = {
        "role": role,
        "sizes": sizes,
        "verdict": verdict,
        # source falls back to series_id on an unlabeled pool — then the rows
        # are per-window, not per-feed; the renderer labels them accordingly.
        "sources_labeled": any(s.source for es in mine for s in es.scores),
        "worst_sources": source_rows[:WORST_SOURCES_SHOWN],
        "best_sources": list(reversed(source_rows[-BEST_SOURCES_SHOWN:]))
        if len(source_rows) > WORST_SOURCES_SHOWN else [],
    }
    out["diagnosis"] = _duel_diagnosis(role, verdict)
    return out


def _duel_diagnosis(role: str, verdict: dict | None) -> str:
    """One human sentence on what the duel verdict means from THIS side."""
    if verdict is None:
        return "no verdict was recorded on this receipt"
    lcb, margin = verdict.get("lcb"), verdict.get("margin") or 0.0
    win = bool(verdict.get("challenger_wins_round"))
    dethroned = bool(verdict.get("dethroned"))
    if verdict.get("inconclusive"):
        return ("the duel was inconclusive (not enough paired evidence to bound the "
                "difference) — the throne is unchanged and no one 'lost' on merit")
    if role == "challenger":
        if win:
            return ("you WON the duel: your relative improvement's lower bound "
                    f"({lcb:+.4f}) cleared the {margin:.3f} margin"
                    + (" — you took the throne" if dethroned else
                       " — win recorded (dethronement needs consecutive wins)"))
        if lcb is not None and lcb > 0:
            return (f"close loss: you measured BETTER than the king (lcb {lcb:+.4f} > 0) "
                    f"but did not clear the {margin:.3f} win margin — a genuinely "
                    "improved fork of this generator can finish the job")
        return (f"you lost the duel: the lower bound on your improvement was {lcb:+.4f} "
                f"(needed > {margin:.3f}) — see the per-domain/per-source rows for "
                "where the king beat you" if lcb is not None else
                "you lost the duel — see the per-domain/per-source rows")
    # king's perspective
    if win and dethroned:
        return "you were DETHRONED: the challenger's improvement cleared the margin"
    if win:
        return ("the challenger won this round (dethronement needs consecutive wins) — "
                "your reign continues but is under pressure")
    return "you HELD the throne: the challenger did not clear the win margin"


# ── report assembly ──────────────────────────────────────────────────────────


def _advice(report: dict) -> list[str]:
    """Short, concrete next steps derived from the outcome — the feedback-loop
    part of the report. Ordered most-specific first; capped so it stays read."""
    tips: list[str] = []
    heat = report.get("heat") or {}
    entrant = heat.get("entrant") or {}
    status = entrant.get("status")
    note = entrant.get("note") or ""
    duel = report.get("duel") or {}

    if "generator_artifact_unreachable" in note:
        tips.append("your Hub repo was not publicly fetchable (HTTP 401/403/404): open "
                    "the Hippius Hub UI and set the project visibility to PUBLIC, then "
                    "re-deploy — see docs/MINER.md §5")
    elif status == "failed_train":
        tips.append("reproduce locally before the next round: `cascade verify ./my-gen` "
                    "(contract + determinism), then `cascade score ./my-gen` at the "
                    "same heat budget the screen uses")
    if status == "failed_screen":
        tips.append("check your generator's output for NaNs/constant series at the "
                    "round's scale: `cascade score` runs the same scorer locally")
    if status == "duplicate":
        tips.append("byte-identical copies can never win a slot (earliest reveal keeps "
                    "it) — fork and genuinely improve instead: `cascade fetch king` "
                    "and beat its number on your own held-out pool")
    if "deadline_hit" in note:
        tips.append("your generator hit the wall-clock cap before the token budget: "
                    "throughput is a compute multiplier, so profile `generate()` and "
                    "make it faster rather than richer")
    if status == "screened":
        tips.append("iterate offline against the reigning best: `cascade fetch king` + "
                    "`cascade score` both on the SAME local pool, and only re-deploy "
                    "once you beat its number")
        tips.append("the private eval pool is broad real-world data — widen your prior "
                    "(mix series families) rather than tuning one shape harder")
    v = (duel.get("verdict") or {}) if duel else {}
    lost_feeds = [r for r in (duel.get("worst_sources") or []) if r["ratio"] > 1.0] \
        if duel else []
    if duel and duel.get("role") == "challenger" \
            and not v.get("challenger_wins_round") and lost_feeds:
        worst = ", ".join(r["source"] for r in lost_feeds[:3])
        tips.append(f"your duel losses concentrate in: {worst} — target these regimes "
                    "in your generator's mixture")
    if report.get("status") == "rejected":
        tips.append("a rejected round is a round-level gate (trainer/validator "
                    "contract), not a judgement of your submission — it simply "
                    "re-runs; nothing to fix on your side")
    if not report.get("participated"):
        tips.append("confirm your reveal landed before the boundary with "
                    "`cascade reveal-status <hotkey> --watch` and time deploys with "
                    "`cascade round`")
    return tips


def build_report(receipt: RoundReceipt, who: str) -> dict:
    """The full miner-centric report for one receipt, as a JSON-safe dict.

    ``who`` is a UID (digits) or hotkey (ss58). Never raises on a receipt the
    miner isn't in — absence is itself the feedback, and the report says why
    that can happen.
    """
    uid, hotkey = resolve_identity(receipt, who)
    heat = _heat_section(receipt, who)
    duel = _duel_section(receipt, who)
    in_participants = any(_matches(who, p.uid, p.hotkey) for p in receipt.participants)
    in_heat = bool(heat and heat.get("entrant"))
    participated = in_participants or in_heat or duel is not None

    participation_note = None
    if receipt.status == "rejected":
        participation_note = (
            f"the validator REJECTED this round's manifest ({receipt.reject_reason}); "
            "no one was scored — this is a round-level gate on the trainer's claim, "
            "not on any miner's submission")
    elif not participated:
        if receipt.participants:
            participation_note = (
                "not in this round's participant set — the commitment was not revealed "
                "strictly before the epoch boundary (it may have rolled into the next "
                "round), or the hotkey never committed")
        else:
            participation_note = (
                "this receipt records no participant set (the chain was unreachable "
                "when it was written), so participation can't be confirmed from it")
    elif in_participants and not in_heat and duel is None:
        if heat is not None:
            participation_note = (
                "you were an eligible participant but appear in neither the heat "
                "standings nor the duel — the submission was dropped before the heat "
                "(a ref identical to an earlier reveal, or the content-dedup screen)")
        else:
            participation_note = (
                "you were an eligible participant but no screen ran and you are not in "
                "the duel entries — either the field auto-advanced without you being "
                "trained (final-stage drop) or the round predates heat standings")

    report = {
        "round_id": receipt.round_id,
        "status": receipt.status,
        "epoch_start_block": receipt.epoch_start_block,
        "reject_reason": receipt.reject_reason,
        "validator_hotkey": receipt.validator_hotkey or None,
        "who": {"query": who, "uid": uid, "hotkey": hotkey},
        "participated": participated,
        "participation_note": participation_note,
        "heat": heat,
        "duel": duel,
    }
    report["advice"] = _advice(report)
    return report


# ── rendering ────────────────────────────────────────────────────────────────


def _short(hotkey: str | None) -> str:
    if not hotkey:
        return "?"
    return hotkey if len(hotkey) <= 13 else f"{hotkey[:6]}…{hotkey[-4:]}"


def _fmt(x: float | None, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def render_report(report: dict) -> str:
    """The terminal view of :func:`build_report` — same aesthetic as the
    ``cascade round`` dashboard (two-space keys, one section per stage)."""
    who = report["who"]
    ident = f"uid {who['uid']}" if who["uid"] is not None else "?"
    lines = [
        f"cascade results — {ident} ({_short(who['hotkey'])})",
        f"  round           {report['round_id']}  ·  status: {report['status']}",
    ]
    if report.get("participation_note"):
        lines.append(f"  note            {report['participation_note']}")
    if report["status"] == "rejected":
        # A rejected round scored no one; heat/duel sections would only mislead.
        for i, tip in enumerate(report.get("advice") or []):
            lines.append(("  next steps      - " if i == 0
                          else "                  - ") + tip)
        return "\n".join(lines)

    heat = report.get("heat")
    if heat and heat.get("entrant"):
        e = heat["entrant"]
        lines.append(
            f"  heat            {e['status']}"
            + (f" — rank {e['rank']} / {heat['n_scored']} scored"
               if e.get("rank") is not None else "")
            + f"  ({heat['n_entrants']} entrants · {heat['finalists']} advance"
            + (f" · screened at {heat['screen_size']}" if heat.get("screen_size") else "")
            + ")")
        if e.get("rel_score") is not None:
            gap = f"    score         {e['pct_behind_leader']:.1f}% behind the leader"
            if e.get("pct_behind_cutoff") is not None:
                gap += f" · {e['pct_behind_cutoff']:.1f}% behind the last finalist"
            lines.append(gap)
        if e.get("crps") is not None and e.get("mase") is not None:
            geo = math.sqrt(e["crps"] * e["mase"])
            lines.append(f"    components    crps {e['crps']:.4f} · mase {e['mase']:.4f}"
                         f"  (geomean {geo:.4f}, lower is better)")
        if e.get("p_best") is not None:
            lines.append(f"    p_best        {e['p_best']:.3f} — bootstrap probability "
                         "this entry was truly the round's best")
        if e.get("note"):
            lines.append(f"    trainer note  {e['note']}")
        lines.append(f"    read          {e['diagnosis']}")
    elif heat is not None:
        lines.append(f"  heat            you are not in this round's standings "
                     f"({heat['n_entrants']} entrants recorded)")
    else:
        lines.append("  heat            no screen ran this round (field fit within the "
                     "finalist budget, or the round predates heat standings)")

    duel = report.get("duel")
    if duel:
        v = duel.get("verdict") or {}
        lines.append(f"  duel            entered as {duel['role'].upper()}")
        for s in duel["sizes"]:
            lines.append(
                f"    {s['size'] or 'primary':<12}  you {_fmt(s['own_geomean'])} vs "
                f"opponent {_fmt(s['opp_geomean'])}  ({s['n_windows']} windows)")
        if v:
            # lcb / win_rate / boot_p50 are all challenger-relative (the duel's
            # statistic), whichever side this report is for — label them so.
            lines.append(
                f"    verdict       challenger lcb {_fmt(v.get('lcb'), '+.4f')} vs margin "
                f"{_fmt(v.get('margin'), '.3f')} · chal win_rate "
                f"{_fmt(v.get('win_rate'), '.3f')} · boot p50 "
                f"{_fmt(v.get('boot_p50'), '+.4f')}")
            pdwr = v.get("per_domain_win_rate") or {}
            if pdwr:
                worst = sorted(pdwr.items(), key=lambda kv: (kv[1][0] is None, kv[1][0]))
                row = " · ".join(f"{dom or '(none)'} {wr:.2f}"
                                 for dom, (wr, _n) in worst[:4] if wr is not None)
                if row:
                    lines.append(f"    by domain     challenger win share: {row}")
        weak_label = "weakest feeds" if duel.get("sources_labeled") else "worst windows"
        for label, rows in ((weak_label, duel.get("worst_sources") or []),
                            ("strongest    ", duel.get("best_sources") or [])):
            for i, r in enumerate(rows):
                head = f"    {label}" if i == 0 else "                 "
                lines.append(f"{head}  {r['source']}: you/them "
                             f"{r['ratio']:.3f} ({r['n_windows']} windows)")
        lines.append(f"    read          {duel['diagnosis']}")
    elif report["status"] == "scored" and heat and heat.get("entrant") \
            and heat["entrant"].get("status") == "advanced":
        lines.append("  duel            you advanced but have no scored duel entry — "
                     "dropped at the final stage (content clone of the king or a "
                     "failed full-budget train)")

    for i, tip in enumerate(report.get("advice") or []):
        lines.append(("  next steps      - " if i == 0 else "                  - ") + tip)
    return "\n".join(lines)


def compact_row(report: dict) -> str:
    """One line per round for ``--history`` mode."""
    heat = report.get("heat") or {}
    e = heat.get("entrant")
    duel = report.get("duel")
    if report["status"] == "rejected":
        outcome = f"rejected ({report.get('reject_reason')})"
    elif duel:
        v = duel.get("verdict") or {}
        role = duel["role"]
        if role == "challenger":
            outcome = ("duel WON" if v.get("challenger_wins_round")
                       else f"duel lost (lcb {_fmt(v.get('lcb'), '+.4f')})")
        else:
            outcome = ("throne LOST" if v.get("dethroned") else "throne held")
    elif e:
        outcome = f"heat {e['status']}" + (
            f" (rank {e['rank']}/{heat.get('n_scored')}, "
            f"{e['pct_behind_leader']:.1f}% behind)"
            if e.get("rank") is not None and e.get("pct_behind_leader") is not None
            else "")
        if e.get("note"):
            outcome += f" — {e['note'][:60]}"
    elif report.get("participated"):
        outcome = "participated (no standing recorded)"
    else:
        outcome = "not in this round"
    return f"round {report['round_id']:<22}  {outcome}"
