"""Event detection for the Discord announcer — pure diffing, no I/O.

The dashboard answers questions only for people who open it; most miner
questions land in chat. ``scripts/announce_events.py`` closes that gap by
polling the same public-read JSON feeds the web dashboard consumes and posting
an announcement when something actually happens:

* **round settled** — a new round appears in ``receipts/index.json``;
* **stage transition** — ``status/round.json`` moves heat → duel → validation;
* **submission revealed** — a new commitment shows up in ``status/chain.json``;
* **benchmark refreshed** — ``status/bench.json`` (public GIFT-Eval/BOOM/TIME
  telemetry) carries a new report.

This module is the announcer's brain: :func:`diff_events` compares the fetched
feed snapshots against the persisted :class:`AnnouncerState` and returns the
events to post plus the advanced state. Everything here is presentational and
read-only — the feeds are the unsigned dashboard documents, never audit or
weight inputs, and a wrong or missed announcement affects nothing but chat.

State priming: a feed seen for the first time (fresh state file) seeds the
baseline silently instead of replaying history — a new deployment must not
flood the channel with 400 archived rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Public-read key the benchmark telemetry publisher is expected to write
# (sibling of CHAIN_STATUS_KEY / ROUND_STATUS_KEY in ``chain_status``). The
# announcer tolerates its absence — no publisher, no bench events.
BENCH_STATUS_KEY = "status/bench.json"

ANNOUNCER_STATE_SCHEMA = 1

# Rolling caps on the dedupe sets — the index itself keeps 400 entries, and
# chain.json holds one commitment per hotkey, so these bound state growth
# without ever forgetting anything still visible in a feed.
MAX_SEEN_ROUNDS = 500
MAX_SEEN_SUBMISSIONS = 1000

# Discord hard-caps message content at 2000 characters.
DISCORD_CONTENT_LIMIT = 2000


@dataclass
class AnnouncerState:
    """What the announcer has already announced. JSON-serializable so a cron
    or restarted loop never re-posts. ``None`` sets mean "feed never seen" —
    the next snapshot primes them silently."""

    seen_rounds: list[str] | None = None            # round_ids announced (or primed)
    stage: dict | None = None                       # {"round_id", "stage"} last seen
    seen_submissions: list[list] | None = None      # [hotkey, commit_block] pairs
    bench_mark: str | None = None                   # identity of the last bench report
    extra: dict = field(default_factory=dict)       # forward-compat passthrough

    @classmethod
    def from_json(cls, text: str) -> AnnouncerState:
        """Parse a persisted state file; malformed/alien content resets to a
        fresh (silent-priming) state rather than crashing the loop."""
        try:
            doc = json.loads(text)
        except (ValueError, TypeError):
            return cls()
        if not isinstance(doc, dict):
            return cls()
        seen_rounds = doc.get("seen_rounds")
        subs = doc.get("seen_submissions")
        stage = doc.get("stage")
        bench = doc.get("bench_mark")
        return cls(
            seen_rounds=[str(r) for r in seen_rounds] if isinstance(seen_rounds, list) else None,
            stage=stage if isinstance(stage, dict) else None,
            seen_submissions=[[str(s[0]), int(s[1])] for s in subs
                              if isinstance(s, (list, tuple)) and len(s) == 2]
            if isinstance(subs, list) else None,
            bench_mark=str(bench) if isinstance(bench, str) else None,
        )

    def to_json(self) -> str:
        return json.dumps({
            "schema": ANNOUNCER_STATE_SCHEMA,
            "seen_rounds": self.seen_rounds,
            "stage": self.stage,
            "seen_submissions": self.seen_submissions,
            "bench_mark": self.bench_mark,
        }, indent=2, sort_keys=True)


@dataclass(frozen=True)
class Event:
    """One announcement: ``kind`` for filtering/logging, ``text`` for Discord."""

    kind: str   # "round_settled" | "stage" | "submission" | "bench"
    text: str


# ── helpers ──────────────────────────────────────────────────────────────────


def _short_ref(ref: object) -> str:
    """``repo@sha256:<64hex>`` → ``repo@1234abcd…`` (chat-width)."""
    s = str(ref or "")
    repo, sep, digest = s.partition("@")
    if not sep:
        return s[:40]
    digest = digest.removeprefix("sha256:").removeprefix("hf:")
    return f"{repo}@{digest[:8]}…"


def _short_hotkey(hotkey: object) -> str:
    s = str(hotkey or "")
    return s if len(s) <= 13 else f"{s[:6]}…{s[-4:]}"


def _fmt_num(value: object, digits: int = 4) -> str | None:
    try:
        return f"{float(value):.{digits}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def clamp_content(text: str, limit: int = DISCORD_CONTENT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _index_rounds(index_doc: object) -> list[dict]:
    if not isinstance(index_doc, dict):
        return []
    rounds = index_doc.get("rounds")
    return [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else []


def _best_entry_per_round(rounds: list[dict]) -> dict[str, dict]:
    """Collapse the (round_id, validator_hotkey)-keyed index to one entry per
    round, preferring a scored entry over a rejected one (the same precedence
    the dashboard and receipt index apply)."""
    best: dict[str, dict] = {}
    for r in rounds:
        rid = str(r.get("round_id", ""))
        if not rid:
            continue
        prev = best.get(rid)
        if prev is None or (str(prev.get("status")) != "scored"
                            and str(r.get("status")) == "scored"):
            best[rid] = r
    return best


# ── message formatting ───────────────────────────────────────────────────────


def format_settled(entry: dict) -> str:
    """One Discord line for a settled round's index entry."""
    rid = str(entry.get("round_id", "?"))
    if str(entry.get("status")) == "rejected":
        reason = str(entry.get("reject_reason") or "see receipt")
        return clamp_content(f"⚠️ Round `{rid}` settled — **rejected** ({reason})")
    parts: list[str] = []
    if entry.get("dethroned"):
        who = entry.get("chal_uid")
        ref = _short_ref(entry.get("chal_gen_ref"))
        head = (f"♛ Round `{rid}` settled — **DETHRONED**: challenger"
                f"{f' uid {who}' if who is not None else ''}"
                f"{f' (`{ref}`)' if ref else ''} took the throne")
    else:
        king = entry.get("post_round_king_uid")
        held = f"king held{f' (uid {king})' if king is not None else ''}"
        head = f"♛ Round `{rid}` settled — {held}"
    parts.append(head)
    detail: list[str] = []
    wr = _fmt_num(entry.get("win_rate"), 2)
    if wr is not None:
        detail.append(f"win rate {wr}")
    lcb = _fmt_num(entry.get("lcb"))
    if lcb is not None:
        detail.append(f"LCB {lcb}")
    nw = entry.get("n_windows")
    if nw is not None:
        detail.append(f"{nw} windows")
    if entry.get("inconclusive"):
        detail.append("inconclusive")
    if detail:
        parts.append(" · ".join(detail))
    return clamp_content(" — ".join(parts) if len(parts) > 1 else parts[0])


def format_stage(doc: dict) -> str:
    rid = str(doc.get("round_id", "?"))
    stage = str(doc.get("stage", "?"))
    if stage == "heat":
        total = doc.get("heat_total")
        extra = f" — screening {int(total)} challenger(s)" if total is not None else ""
        return f"🔥 Round `{rid}`: heat started{extra}"
    if stage == "duel":
        n = doc.get("finalists")
        who = f"{int(n)} finalist(s)" if n is not None else "finalists"
        return f"⚔️ Round `{rid}`: duel — king vs {who} at the full budget"
    return f"🧪 Round `{rid}`: validation — manifest published, validators scoring"


def format_submissions(new_subs: list[dict], chain_doc: dict) -> str:
    """One batched line for all newly revealed commitments in a snapshot."""
    epoch_start = chain_doc.get("epoch_start_block")
    lines = [f"📡 {len(new_subs)} new submission(s) revealed:"]
    for s in new_subs[:8]:
        target = ""
        try:
            if epoch_start is not None:
                nxt = int(s.get("commit_block", 0)) >= int(epoch_start)
                target = " → next round" if nxt else " → this round"
        except (TypeError, ValueError):
            target = ""
        lines.append(f"• uid {s.get('uid', '?')} {_short_hotkey(s.get('hotkey'))} "
                     f"`{_short_ref(s.get('gen_ref'))}`"
                     f" (block {s.get('commit_block', '?')}){target}")
    if len(new_subs) > 8:
        lines.append(f"… and {len(new_subs) - 8} more")
    return clamp_content("\n".join(lines))


_BENCH_SUITE_LABELS = (("gift_eval", "GIFT-Eval"), ("boom", "BOOM"), ("time", "TIME"))


def format_bench(doc: dict) -> str:
    rid = doc.get("round_id")
    head = "📊 Public benchmarks refreshed" + (f" (round `{rid}`)" if rid else "")
    scores = doc.get("scores") if isinstance(doc.get("scores"), dict) else doc
    detail: list[str] = []
    for key, label in _BENCH_SUITE_LABELS:
        for k, v in (scores or {}).items():
            if str(k).replace("-", "_").startswith(key):
                num = _fmt_num(v)
                if num is not None:
                    detail.append(f"{label}: {num}")
                break
    return clamp_content(head + (" — " + " · ".join(detail) if detail else ""))


def _bench_identity(doc: dict) -> str:
    """A stable identity for "is this a new report?" — round_id when present,
    else the as_of stamp, else the whole doc."""
    for key in ("round_id", "as_of"):
        v = doc.get(key)
        if v:
            return f"{key}:{v}"
    return json.dumps(doc, sort_keys=True)[:256]


# ── the diff ─────────────────────────────────────────────────────────────────


def diff_events(
    state: AnnouncerState,
    *,
    index_doc: object = None,
    chain_doc: object = None,
    round_doc: object = None,
    bench_doc: object = None,
) -> tuple[list[Event], AnnouncerState]:
    """Compare feed snapshots against ``state``; return the events to announce
    and the advanced state. A feed that failed to fetch is passed as ``None``
    and leaves its slice of state untouched (no events, nothing forgotten).
    """
    events: list[Event] = []
    new = AnnouncerState(
        seen_rounds=list(state.seen_rounds) if state.seen_rounds is not None else None,
        stage=dict(state.stage) if state.stage is not None else None,
        seen_submissions=[list(p) for p in state.seen_submissions]
        if state.seen_submissions is not None else None,
        bench_mark=state.bench_mark,
    )

    # round settled — new round_id in the receipt index
    rounds = _index_rounds(index_doc)
    if index_doc is not None and isinstance(index_doc, dict):
        best = _best_entry_per_round(rounds)
        if new.seen_rounds is None:
            new.seen_rounds = list(best)  # first sight: prime silently
        else:
            seen = set(new.seen_rounds)
            fresh = [rid for rid in best if rid not in seen]
            fresh.sort(key=lambda rid: int(best[rid].get("epoch_start_block", 0) or 0))
            for rid in fresh:
                events.append(Event("round_settled", format_settled(best[rid])))
                new.seen_rounds.append(rid)
        new.seen_rounds = new.seen_rounds[-MAX_SEEN_ROUNDS:]

    # stage transition — trainer-reported (round_id, stage) changed
    if isinstance(round_doc, dict) and round_doc.get("stage"):
        cur = {"round_id": str(round_doc.get("round_id", "")),
               "stage": str(round_doc.get("stage", ""))}
        if new.stage is None:
            new.stage = cur  # prime silently
        elif (cur["round_id"], cur["stage"]) != (str(new.stage.get("round_id", "")),
                                                 str(new.stage.get("stage", ""))):
            events.append(Event("stage", format_stage(round_doc)))
            new.stage = cur

    # submission revealed — new (hotkey, commit_block) in the chain status feed
    if isinstance(chain_doc, dict) and isinstance(chain_doc.get("submissions"), list):
        subs = [s for s in chain_doc["submissions"] if isinstance(s, dict)]
        keys = [[str(s.get("hotkey", "")), int(s.get("commit_block", 0) or 0)] for s in subs]
        if new.seen_submissions is None:
            new.seen_submissions = keys  # prime silently
        else:
            seen_pairs = {tuple(p) for p in new.seen_submissions}
            fresh_subs = [s for s, k in zip(subs, keys, strict=True) if tuple(k) not in seen_pairs]
            if fresh_subs:
                events.append(Event("submission", format_submissions(fresh_subs, chain_doc)))
            new.seen_submissions.extend(k for k in keys if tuple(k) not in seen_pairs)
        new.seen_submissions = new.seen_submissions[-MAX_SEEN_SUBMISSIONS:]

    # benchmark refreshed — the bench doc's identity moved
    if isinstance(bench_doc, dict) and bench_doc:
        mark = _bench_identity(bench_doc)
        if new.bench_mark is None:
            new.bench_mark = mark  # prime silently
        elif mark != new.bench_mark:
            events.append(Event("bench", format_bench(bench_doc)))
            new.bench_mark = mark

    return events, new
