"""``cascade round`` — a terminal dashboard counting down to the next round.

A round is one ``[round] epoch_blocks`` window on the chain block grid (the
same math the trainer uses in :meth:`~cascade.trainer.loop.TrainerRunner.
run_forever`): the current epoch is ``block // epoch_blocks`` and the next
round starts at the next epoch boundary. Only commitments revealed STRICTLY
BEFORE that boundary enter the next round, so the boundary is also the
submission deadline — the number a miner actually watches.

Beyond the countdown, the dashboard shows where the round roughly is
(``heat ▸ duel ▸ validation ▸ settled``) and a live feed of revealed on-chain
submissions. The stage is *confirmed* when the round's receipt appears in the
public ``receipts/index.json`` (settled), and otherwise *estimated* from the
configured stage budgets — the trainer's internal progress is not public, so
the pre-settle stages are wall-clock estimates, labelled as such. Submissions
come straight from the chain's revealed commitments; in watch mode a commit
that lands while you watch is flagged ``● new`` — the confirmation a miner
looks for right after ``cascade deploy``.

Everything on-chain-exact here is in *blocks*; the wall-clock countdown is an
estimate derived from the configured cadence (``round_hours`` over
``epoch_blocks``, ~12s/block on Bittensor). In watch mode the display ticks
every second by interpolating between chain polls, re-syncs to the real block
height every ``refresh`` seconds, and re-polls commitments + the receipt index
on a slower cadence.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..shared.chain_status import STAGE_OVERHEAD_SECONDS, stage_windows
from ..shared.config import RoundConfig, effective_epoch_blocks

DEFAULT_SECONDS_PER_BLOCK = 12.0
BAR_WIDTH = 28

# Per-stage overhead of the timing estimate — shared with the web dashboard's
# status feed so both estimate off the same numbers (see shared.chain_status).
PHASE_OVERHEAD_SECONDS = STAGE_OVERHEAD_SECONDS

# Max submission rows rendered before collapsing to a "… N more" line.
SUBMISSIONS_SHOWN = 8

# Watch mode re-polls commitments + the receipt index at least this rarely —
# both are heavier than a block-height read (metagraph + storage map / an HTTP
# GET), and the feed only needs to move on human timescales.
FEED_REFRESH_FLOOR_SECONDS = 60.0


def seconds_per_block(round_cfg: RoundConfig) -> float:
    """The configured wall-clock cadence: ``round_hours`` spread over
    ``epoch_blocks``. Falls back to Bittensor's ~12s when the config carries a
    placeholder (non-positive) value."""
    if round_cfg.round_hours > 0 and round_cfg.epoch_blocks > 0:
        return round_cfg.round_hours * 3600.0 / round_cfg.epoch_blocks
    return DEFAULT_SECONDS_PER_BLOCK


@dataclass(frozen=True)
class RoundStatus:
    """A snapshot of where the current block sits on the epoch grid."""

    block: int              # chain block the snapshot was taken at
    epoch_blocks: int       # blocks per round ([round] epoch_blocks)
    spb: float              # estimated seconds per block

    @property
    def epoch(self) -> int:
        return self.block // self.epoch_blocks

    @property
    def epoch_start(self) -> int:
        return self.epoch * self.epoch_blocks

    @property
    def next_epoch_start(self) -> int:
        return self.epoch_start + self.epoch_blocks

    @property
    def blocks_elapsed(self) -> int:
        return self.block - self.epoch_start

    @property
    def blocks_remaining(self) -> int:
        return self.next_epoch_start - self.block

    @property
    def seconds_remaining(self) -> float:
        return self.blocks_remaining * self.spb

    @property
    def progress(self) -> float:
        return self.blocks_elapsed / self.epoch_blocks


def round_status(block: int, round_cfg: RoundConfig) -> RoundStatus:
    # Resolve the grid AT the snapshot block: while a scheduled cadence change
    # ([round] epoch_activation_block) is pending, the raw field is the
    # POST-switch length and would show miners the new boundaries a day early.
    return RoundStatus(
        block=int(block),
        epoch_blocks=max(1, effective_epoch_blocks(round_cfg, int(block))),
        spb=seconds_per_block(round_cfg),
    )


def format_duration(seconds: float) -> str:
    """``93784.0`` → ``"1d 2h 3m 4s"`` (leading zero units dropped)."""
    s = max(0, int(seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts: list[str] = []
    if d:
        parts.append(f"{d}d")
    if h or parts:
        parts.append(f"{h}h")
    if m or parts:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _bar(progress: float, width: int = BAR_WIDTH) -> str:
    filled = min(width, max(0, round(progress * width)))
    return "█" * filled + "░" * (width - filled)


# ── round stage (heat ▸ duel ▸ validation ▸ settled) ─────────────────────────

PHASE_ORDER = ("heat", "duel", "validation", "settled")


@dataclass(frozen=True)
class PhaseEstimate:
    """Where the current round roughly is. ``estimated`` is False only when the
    stage is confirmed by public evidence (the round's receipt in the index)."""

    key: str        # one of PHASE_ORDER
    detail: str     # one-line human explanation shown under the strip
    estimated: bool


@dataclass(frozen=True)
class RoundTimeline:
    """Rough wall-clock stage windows for one round, derived from config.

    The trainer's internal progress is not publicly readable, so the pre-settle
    stages are estimated off the same budgets the trainer enforces: the heat's
    wall-clock cap (``[round] heat_*``) and the final duel's per-size
    ``max_train_seconds`` (summed — sizes train sequentially), each padded with
    :data:`PHASE_OVERHEAD_SECONDS` for fetch/boot/upload. Anything past
    ``heat + duel`` is presumed to be duel validation until the receipt lands.
    """

    heat_seconds: float
    duel_seconds: float

    @classmethod
    def from_chain_config(cls, cfg: object) -> RoundTimeline:
        heat_s, duel_s = stage_windows(cfg)
        return cls(heat_seconds=heat_s, duel_seconds=duel_s)


def phase_from_live(
    doc: object,
    st: RoundStatus,
    *,
    now_s: float | None = None,
) -> PhaseEstimate | None:
    """The trainer-reported stage (``status/round.json``), when trustworthy.

    Returns a confirmed :class:`PhaseEstimate` when the doc is fresh and
    matches the current epoch (see ``live_round_stage``), else None — the
    caller falls back to the wall-clock estimate. Preferred over the estimate
    because the estimate models the heat as ONE competitor's budget and calls
    "duel" hours early on a large field.
    """
    from ..shared.chain_status import live_round_stage

    live = live_round_stage(doc, epoch_start_block=st.epoch_start,
                            now_s=time.time() if now_s is None else now_s)
    if live is None:
        return None
    stage = str(live["stage"])
    if stage == "heat":
        done, total = live.get("heat_done"), live.get("heat_total")
        progress = (f" — screening {int(done)}/{int(total)} challengers"
                    if done is not None and total is not None else "")
        what = f"trainer screening the field at the heat budget{progress}"
    elif stage == "duel":
        n = live.get("finalists")
        who = f"{int(n)} finalist(s)" if n is not None else "finalists"
        what = f"king vs {who} training at the full budget"
    else:
        what = "manifest published; validators scoring (receipt pending)"
    return PhaseEstimate(stage, f"{what} (trainer-reported)", estimated=False)


def phase_for(
    st: RoundStatus,
    timeline: RoundTimeline,
    *,
    drift_seconds: float = 0.0,
    settled_outcome: str | None = None,
) -> PhaseEstimate:
    """The round's current stage: confirmed ``settled`` when an outcome line is
    supplied (the round's receipt is public), else estimated from elapsed
    wall-clock against the configured stage windows."""
    if settled_outcome is not None:
        return PhaseEstimate("settled", settled_outcome, estimated=False)
    elapsed = st.blocks_elapsed * st.spb + max(0.0, drift_seconds)
    if elapsed < timeline.heat_seconds:
        key, what = "heat", "trainer screening challengers at the heat budget"
    elif elapsed < timeline.heat_seconds + timeline.duel_seconds:
        key, what = "duel", "king vs finalists training at the full budget"
    else:
        key, what = "validation", "validators scoring the duel (receipt pending)"
    detail = f"{what} — {format_duration(elapsed)} into the round (est.)"
    return PhaseEstimate(key, detail, estimated=True)


def _phase_strip(current: str) -> str:
    return " ▸ ".join(f"[{k.upper()}]" if k == current else k for k in PHASE_ORDER)


# ── public receipt index (settled-round evidence; no credentials needed) ─────


def fetch_public_json(storage: object, key: str, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET one public-read JSON object from the manifest bucket.

    Receipts, the receipt index and the ``status/`` docs are written public-read
    exactly so third parties can read them with zero credentials (see
    ``cascade.shared.hippius``), so a plain path-style HTTPS GET works without
    boto or the S3 keys. Best-effort: any failure — offline, private backend,
    malformed JSON, a non-object body — returns None and the caller degrades.
    """
    import urllib.request

    endpoint = str(getattr(storage, "s3_endpoint", "") or "").rstrip("/")
    bucket = str(getattr(storage, "manifest_bucket", "") or "")
    if not endpoint.startswith(("http://", "https://")) or not bucket:
        return None
    url = f"{endpoint}/{bucket}/{key}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            doc = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — every public doc is a best-effort enhancement
        return None
    return doc if isinstance(doc, dict) else None


def fetch_public_receipt_index(storage: object, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET the dashboard-facing ``receipts/index.json``. None on any
    failure (the dashboard then simply shows estimates)."""
    from ..shared.hippius import RECEIPT_INDEX_KEY

    doc = fetch_public_json(storage, RECEIPT_INDEX_KEY, timeout=timeout)
    return doc if doc is not None and isinstance(doc.get("rounds"), list) else None


def fetch_public_round_status(storage: object, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET the trainer-reported ``status/round.json``.

    Best-effort: any failure returns None and the dashboard falls back to the
    wall-clock stage estimate. Freshness/round-matching is the CONSUMER's job
    (``phase_from_live``), so a stale doc here is returned as-is.
    """
    from ..shared.chain_status import ROUND_STATUS_KEY

    return fetch_public_json(storage, ROUND_STATUS_KEY, timeout=timeout)


def fetch_public_heat(storage: object, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET the latest published heat standings (``status/heat.json``).

    Written by the trainer the moment the heat settles — hours before the round's
    receipt — so this is a miner's first read on where its submission placed.
    Round-matching is the consumer's job (``cascade.shared.heat_status.
    live_heat``), so another round's standings are returned as-is.
    """
    from ..shared.heat_status import HEAT_STATUS_KEY

    return fetch_public_json(storage, HEAT_STATUS_KEY, timeout=timeout)


def fetch_public_heat_round(
    storage: object, round_id: str, *, timeout: float = 10.0
) -> dict | None:
    """Anonymously GET one round's archived heat standings."""
    from ..shared.heat_status import heat_round_key

    return fetch_public_json(storage, heat_round_key(str(round_id)), timeout=timeout)


def fetch_public_heat_index(storage: object, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET ``heats/index.json`` — the discoverable list of published
    heats (a static reader cannot list the bucket)."""
    from ..shared.heat_status import HEAT_INDEX_KEY

    doc = fetch_public_json(storage, HEAT_INDEX_KEY, timeout=timeout)
    return doc if doc is not None and isinstance(doc.get("heats"), list) else None


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _index_rounds(index_doc: dict | None) -> list[dict]:
    if not isinstance(index_doc, dict):
        return []
    return [r for r in index_doc.get("rounds", []) if isinstance(r, dict)]


def settled_entry_for(index_doc: dict | None, epoch_start: int) -> dict | None:
    """The receipt-index entry that settles the round at ``epoch_start``, or
    None. A scored entry outranks a rejected one for the same round (mirrors
    the scored-precedence rule the index itself applies)."""
    matches = [r for r in _index_rounds(index_doc)
               if _as_int(r.get("epoch_start_block")) == int(epoch_start)]
    if not matches:
        return None
    scored = [r for r in matches if str(r.get("status")) == "scored"]
    return (scored or matches)[-1]


def latest_settled_before(index_doc: dict | None, epoch_start: int) -> dict | None:
    """The most recent settled round STRICTLY BEFORE ``epoch_start`` — shown as
    "last round" context while the current round is still in flight."""
    prior = [r for r in _index_rounds(index_doc)
             if (esb := _as_int(r.get("epoch_start_block"))) is not None
             and esb < int(epoch_start)]
    if not prior:
        return None
    last_esb = max(_as_int(r.get("epoch_start_block")) for r in prior)
    group = [r for r in prior if _as_int(r.get("epoch_start_block")) == last_esb]
    scored = [r for r in group if str(r.get("status")) == "scored"]
    return (scored or group)[-1]


def outcome_line(entry: dict) -> str:
    """One line summarising a settled round from its index entry."""
    if str(entry.get("status")) == "rejected":
        reason = str(entry.get("reject_reason") or "see receipt")
        return f"round settled — rejected ({reason[:72]})"
    if entry.get("dethroned"):
        chal = entry.get("chal_uid")
        who = f"challenger uid {chal}" if chal is not None else "the challenger"
        return f"round settled — DETHRONED: {who} took the throne"
    king = entry.get("post_round_king_uid")
    held = f"king held (uid {king})" if king is not None else "king held"
    return f"round settled — {held}"


# ── heat standings (public heat mirror; no credentials needed) ───────────────
#
# The heat is the cheap screen that ranks every eligible challenger and advances
# the top `finalists`; the losing checkpoints are thrown away. The trainer
# publishes the standings the moment the heat settles (status/heat.json +
# heats/round-<id>.json — see cascade.shared.heat_status), which is hours before
# the round's receipt and, for a round later rejected at a gate, the only place
# they ever appear. This section renders them for the terminal.

# Rows shown in the round dashboard's inline heat block before collapsing (the
# miner's own row is always shown, however far down it placed).
HEAT_ROWS_SHOWN = 6

_HEAT_LABELS = {
    "advanced": "▲ advanced",
    "screened": "screened",
    "failed_train": "did not train",
    "failed_screen": "screen error",
    "duplicate": "duplicate",
}


@dataclass(frozen=True)
class HeatRow:
    """One entrant's heat standing, dashboard-shaped."""

    rank: int | None        # 1-based placement among scored entrants; None if unscored
    uid: int
    hotkey: str
    ref: str
    status: str             # one of manifest.HEAT_STATUSES
    rel_score: float | None  # heat_score / best (1.0 = best); None if unscored
    crps: float | None      # raw CRPS-family error on the round's eval-pool slice
    mase: float | None      # raw geometric-mean MASE on that slice
    mine: bool = False      # this is the hotkey/uid the caller asked about


def _matches_me(uid: int, hotkey: str, me: str | None) -> bool:
    if not me:
        return False
    me = me.strip()
    return me == hotkey or (me.isdigit() and int(me) == uid)


def heat_rows(doc: dict | None, *, me: str | None = None) -> list[HeatRow]:
    """Shape a published heat document into rows, best rank first.

    Unscored entrants (``failed_train`` / ``failed_screen`` / ``duplicate``)
    carry no rank and sort last, by UID. ``me`` is a hotkey (ss58) or UID whose
    row is flagged ``mine``.
    """
    if not isinstance(doc, dict):
        return []
    rows: list[HeatRow] = []
    for e in doc.get("entrants", ()):
        if not isinstance(e, dict):
            continue
        try:
            uid, hotkey = int(e.get("uid", -1)), str(e.get("hotkey", ""))
        except (TypeError, ValueError):
            continue
        rank = e.get("rank")
        rows.append(HeatRow(
            rank=None if rank is None else int(rank),
            uid=uid,
            hotkey=hotkey,
            ref=str(e.get("gen_ref", "")),
            status=str(e.get("status", "")),
            rel_score=_as_float(e.get("rel_score")),
            crps=_as_float(e.get("crps")),
            mase=_as_float(e.get("mase")),
            mine=_matches_me(uid, hotkey, me),
        ))
    rows.sort(key=lambda r: (r.rank is None, r.rank if r.rank is not None else 0, r.uid))
    return rows


def _as_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def heat_headline(doc: dict | None) -> str:
    """One line summarising a published heat: field size, how many advanced, the
    size it screened at — or why no screen ran."""
    if not isinstance(doc, dict):
        return "no heat standings published"
    rows = heat_rows(doc)
    if doc.get("no_screen") or not rows:
        why = str(doc.get("no_screen_reason") or "no screen ran this round")
        return f"no screen — {why}"
    adv = sum(1 for r in rows if r.status == "advanced")
    size = str(doc.get("screen_size") or "")
    parts = [f"{len(rows)} entrants", f"{adv} advanced"]
    if size:
        parts.append(f"screened at {size}")
    return " · ".join(parts)


def _heat_row_line(r: HeatRow, *, show_error: bool = True) -> str:
    """One standings row. ``show_error`` carries the raw CRPS/MASE columns —
    decided per document (a scalar-only screener publishes neither, and empty
    columns on every row would be noise) so the status column stays aligned
    across rows where only SOME entrants were scored."""
    if r.rel_score is not None and r.rel_score > 0:
        gap = "best" if r.rank == 1 else f"+{(r.rel_score - 1.0) * 100:.1f}%"
    else:
        gap = "—"
    err = ""
    if show_error:
        crps = "—" if r.crps is None else f"{r.crps:.4f}"
        mase = "—" if r.mase is None else f"{r.mase:.3f}"
        err = f"  crps {crps:>8}  mase {mase:>7}"
    rank = f"#{r.rank}" if r.rank is not None else "—"
    label = _HEAT_LABELS.get(r.status, r.status or "--")
    return (f"    {rank:>4}  uid {r.uid:>4}  {_short_hotkey(r.hotkey):<11}  "
            f"{_short_ref(r.ref):<32}  {gap:>7}{err}  {label}"
            + ("   ← you" if r.mine else ""))


def heat_block(
    doc: dict | None,
    *,
    me: str | None = None,
    limit: int | None = HEAT_ROWS_SHOWN,
) -> list[str]:
    """The heat section's lines: a headline plus one line per entrant.

    ``limit`` caps the rows rendered (None = all); the caller's own row is
    always included, however far down it placed, and the collapsed remainder is
    reported. Empty list when nothing is published.
    """
    if not isinstance(doc, dict):
        return []
    rows = heat_rows(doc, me=me)
    lines = [f"  heat            {heat_headline(doc)}"]
    if not rows:
        return lines
    shown = rows if limit is None else rows[:limit]
    hidden = rows[len(shown):]
    mine_hidden = [r for r in hidden if r.mine]
    show_error = any(r.crps is not None or r.mase is not None for r in rows)
    lines += [_heat_row_line(r, show_error=show_error) for r in shown]
    lines += [_heat_row_line(r, show_error=show_error) for r in mine_hidden]
    rest = len(hidden) - len(mine_hidden)
    if rest > 0:
        lines.append(f"    … {rest} more (lower placings not shown)")
    return lines


def render_heat(doc: dict | None, *, me: str | None = None) -> str:
    """The standalone ``cascade heat`` view of one published heat document."""
    if not isinstance(doc, dict):
        return ("no heat standings published yet — the trainer writes them when a "
                "round's heat settles (status/heat.json)")
    head = [
        f"cascade heat — round {doc.get('round_id', '--')}"
        f"  ·  epoch start block {doc.get('epoch_start_block', '--')}",
        f"  published       {doc.get('as_of', '--')}",
        f"  field           {heat_headline(doc)}",
    ]
    finalists = doc.get("finalists")
    if finalists is not None:
        head.append(f"  advancing       top {finalists} to the duel against the king")
    lcb, nw, nc = doc.get("leader_lcb"), doc.get("n_windows"), doc.get("n_clusters")
    if lcb is not None:
        decisive = "separated 1st from 2nd" if float(lcb) > 0 else "did NOT separate 1st from 2nd"
        head.append(f"  decisiveness    leader LCB {float(lcb):+.4f} — the screen {decisive}"
                    + (f" (n_windows={nw}, feeds={nc})" if nw is not None else ""))
    body = heat_block(doc, me=me, limit=None)
    return "\n".join(head + ([""] + body[1:] if len(body) > 1 else []))


def render_heat_index(doc: dict | None, *, limit: int = 20) -> str:
    """The ``cascade heat --history`` view of ``heats/index.json``."""
    heats = [h for h in (doc or {}).get("heats", []) if isinstance(h, dict)]
    if not heats:
        return "no published heats found (heats/index.json is absent or empty)"
    lines = [f"cascade heat — {len(heats)} published heat(s), newest last"]
    for h in heats[-limit:]:
        lcb = h.get("leader_lcb")
        lines.append(
            f"  round {str(h.get('round_id', '--')):<14} block {h.get('epoch_start_block', '--'):>11}  "
            f"{h.get('n_entrants', 0):>3} entrants  {h.get('n_advanced', 0):>2} advanced  "
            f"leader uid {str(h.get('leader_uid', '--')):>4}"
            + (f"  lcb {float(lcb):+.4f}" if lcb is not None else "")
            + ("  (no screen)" if h.get("no_screen") else "")
        )
    if len(heats) > limit:
        lines.append(f"  … {len(heats) - limit} older not shown (--limit to widen)")
    return "\n".join(lines)


# ── duel breakdown (`cascade duel` — the verdict behind DETHRONED/held) ──────
#
# The receipt index carries one row per validator per settled round, and a
# scored row already holds the full duel verdict: LCB vs margin, both
# geomeans, win rate, per-domain win rates, bootstrap quantiles. `cascade
# round` compresses all that into one outcome line; this section renders the
# whole thing. Same trust model as the rest of the public docs: anonymous
# GET, no wallet, no credentials.


def duel_round_rows(index_doc: dict | None, round_id: str | None = None) -> list[dict]:
    """Every index row for one round — the latest round with a scored row when
    ``round_id`` is None. Scored rows carry the verdict; rejected rows for the
    same round explain validators that sat the scoring out."""
    rounds = _index_rounds(index_doc)
    if round_id is not None:
        return [r for r in rounds if str(r.get("round_id")) == str(round_id)]
    scored = [r for r in rounds if str(r.get("status")) == "scored"
              and _as_int(r.get("epoch_start_block")) is not None]
    if not scored:
        return []
    latest = max(scored, key=lambda r: _as_int(r.get("epoch_start_block")))
    return [r for r in rounds if str(r.get("round_id")) == str(latest.get("round_id"))]


def _domain_bar(win_rate: float, width: int = 24) -> str:
    """A bar centred on 0.50: right of centre = challenger ahead, left = king.
    Reads at a glance whether a domain flipped against the overall verdict."""
    half = width // 2
    filled = round(abs(win_rate - 0.5) * 2 * half)
    filled = min(filled, half)
    if win_rate >= 0.5:
        return "·" * half + "█" * filled + " " * (half - filled)
    return " " * (half - filled) + "█" * filled + "·" * half


def _duel_outcome(row: dict) -> str:
    if row.get("inconclusive"):
        return "INCONCLUSIVE — validators could not settle the duel"
    chal = row.get("chal_uid")
    who = f"challenger uid {chal}" if chal is not None else "the challenger"
    if row.get("dethroned"):
        return f"DETHRONED — {who} took the throne"
    lcb, margin = _as_float(row.get("lcb")), _as_float(row.get("margin"))
    if lcb is not None and margin is not None:
        return (f"king held — {who} fell short: LCB {lcb:+.4f} vs "
                f"{margin:+.4f} required")
    return "king held"


def _duel_validators_line(rows: list[dict]) -> str | None:
    scored = [r for r in rows if str(r.get("status")) == "scored"]
    rejected = [r for r in rows if str(r.get("status")) == "rejected"]
    parts = []
    if scored:
        lcbs = "/".join(f"{lcb:+.4f}" for r in scored
                        if (lcb := _as_float(r.get("lcb"))) is not None)
        parts.append(f"{len(scored)} scored" + (f" (lcb {lcbs})" if lcbs else ""))
    for r in rejected:
        reason = str(r.get("reject_reason") or "unspecified").split(":")[0]
        parts.append(f"{_short_hotkey(str(r.get('validator_hotkey') or ''))} "
                     f"rejected ({reason})")
    return "  validators     " + " · ".join(parts) if parts else None


def render_duel(rows: list[dict]) -> str:
    """The standalone ``cascade duel`` view of one settled round's index rows."""
    if not rows:
        return ("no settled round found in the public receipt index — receipts land "
                "a few minutes after the duel manifest; try 'cascade round' for the "
                "live stage")
    scored = [r for r in rows if str(r.get("status")) == "scored"]
    row = (scored or rows)[-1]
    lines = [
        f"cascade duel — round {row.get('round_id', '--')}"
        f"  ·  epoch start block {row.get('epoch_start_block', '--')}",
    ]
    if not scored:
        reasons = {str(r.get("reject_reason") or "unspecified") for r in rows}
        lines.append("  outcome        rejected by every reporting validator")
        lines += [f"    {reason[:96]}" for reason in sorted(reasons)]
        return "\n".join(lines)
    lines.append(f"  outcome        {_duel_outcome(row)}")
    kg, cg = _as_float(row.get("king_geomean")), _as_float(row.get("chal_geomean"))
    for role, uid, hotkey, geo, ref in (
        ("king", row.get("king_uid"), row.get("king_hotkey"), kg, row.get("king_gen_ref")),
        ("challenger", row.get("chal_uid"), row.get("chal_hotkey"), cg, row.get("chal_gen_ref")),
    ):
        if uid is None and hotkey is None:
            continue
        line = (f"  {role:<11}    uid {'--' if uid is None else uid:>4}  "
                f"{_short_hotkey(str(hotkey or '')):<11}")
        if geo is not None:
            line += f"  geomean {geo:.5f}"
        if role == "challenger" and kg and cg is not None:
            word = "better than" if cg < kg else "worse than"
            line += f"  ({abs(kg - cg) / kg * 100:.2f}% {word} the king)"
        if ref:
            line += f"  {_short_ref(str(ref))}"
        lines.append(line)
    lcb, margin = _as_float(row.get("lcb")), _as_float(row.get("margin"))
    if lcb is not None:
        bar = "cleared the bar" if margin is not None and lcb > margin else "did not clear the bar"
        lines.append(f"  margin         LCB {lcb:+.4f} vs "
                     + (f"{margin:+.4f} required — challenger {bar}" if margin is not None
                        else "an unpublished margin"))
    win, nw, nc = (_as_float(row.get("win_rate")), _as_int(row.get("n_windows")),
                   _as_int(row.get("n_clusters")))
    if win is not None:
        ev = f"  evidence       challenger won {win * 100:.1f}% of windows"
        if nw:
            ev += f" ({nw} windows, {nc or '--'} feeds)"
        p = _as_float(row.get("wilcoxon_p"))
        if p is not None:
            ev += f" · wilcoxon p={p:.2g}"
        lines.append(ev)
    p50, p95 = _as_float(row.get("boot_p50")), _as_float(row.get("boot_p95"))
    if p50 is not None and p95 is not None:
        lines.append(f"  bootstrap      Δ p50 {p50:+.4f} / p95 {p95:+.4f}")
    gift = row.get("gift_gate_passed")
    if gift is not None:
        lines.append("  gift gate      " + ("passed" if gift else "BLOCKED the dethrone"))
    heat = row.get("heat")
    if isinstance(heat, dict) and heat.get("n_entrants") is not None:
        lines.append(f"  heat           {heat.get('n_entrants')} entrants · "
                     f"{heat.get('n_advanced', '--')} advanced"
                     + (f" · leader p_best {pb:.3f}"
                        if (pb := _as_float(heat.get("leader_p_best"))) is not None else ""))
    uids = row.get("reward_uids")
    if isinstance(uids, list) and uids:
        lines.append(f"  rewards        uids {uids}")
    domains = row.get("per_domain_win_rate")
    if isinstance(domains, dict) and domains:
        lines.append("  per-domain win rate  (right of centre = challenger ahead)")
        ordered = sorted(domains.items(),
                         key=lambda kv: -(_as_float(_domain_pair(kv[1])[0]) or 0.0))
        for name, value in ordered:
            rate, n = _domain_pair(value)
            if rate is None:
                continue
            n_s = f"n={n:>5}" if n is not None else ""
            lines.append(f"    {name:<14} {rate:.2f}  {n_s}  {_domain_bar(rate)}")
    if (validators := _duel_validators_line(rows)) is not None:
        lines.append(validators)
    return "\n".join(lines)


def _domain_pair(value: object) -> tuple[float | None, int | None]:
    """A per-domain entry is ``[win_rate, n_windows]`` in the index; tolerate a
    bare number from older writers."""
    if isinstance(value, (list, tuple)) and value:
        return _as_float(value[0]), _as_int(value[1]) if len(value) > 1 else None
    return _as_float(value), None


def render_duel_index(index_doc: dict | None, *, limit: int = 20) -> str:
    """The ``cascade duel --history`` view: one line per settled round."""
    by_round: dict[str, list[dict]] = {}
    for r in _index_rounds(index_doc):
        by_round.setdefault(str(r.get("round_id")), []).append(r)
    groups = sorted(by_round.values(),
                    key=lambda g: max((_as_int(r.get("epoch_start_block")) or 0) for r in g))
    if not groups:
        return "no settled rounds found (receipts/index.json is absent or empty)"
    lines = [f"cascade duel — {len(groups)} settled round(s), newest last"]
    for g in groups[-limit:]:
        scored = [r for r in g if str(r.get("status")) == "scored"]
        row = (scored or g)[-1]
        if not scored:
            what = "rejected"
        elif row.get("dethroned"):
            what = f"DETHRONED by uid {row.get('chal_uid', '--')}"
        else:
            what = f"king held (uid {row.get('king_uid', '--')})"
        lcb = _as_float(row.get("lcb"))
        lines.append(
            f"  round {str(row.get('round_id', '--')):<22} "
            f"block {row.get('epoch_start_block') or '--':>11}  "
            + (f"lcb {lcb:+.4f}  " if lcb is not None else " " * 13) + what)
    if len(groups) > limit:
        lines.append(f"  … {len(groups) - limit} older not shown (--limit to widen)")
    return "\n".join(lines)


# ── live submissions (revealed on-chain commitments) ─────────────────────────


@dataclass(frozen=True)
class SubmissionRow:
    """One hotkey's latest revealed generator commitment, dashboard-shaped."""

    uid: int
    hotkey: str
    ref: str            # generator repo@digest from the commit payload
    commit_block: int
    next_round: bool    # committed at/after this epoch's start → enters the NEXT round
    new: bool = False   # revealed since this watch session started


def submission_rows(
    commitments: list,
    epoch_start: int,
    *,
    floor_block: int = 0,
    baseline: set[tuple[str, int]] | None = None,
) -> list[SubmissionRow]:
    """Shape chain commitments into dashboard rows, newest first.

    Malformed payloads and pre-``floor_block`` (pre-go-live) commits are
    dropped, mirroring the trainer's eligibility rules. ``epoch_start`` splits
    the field: a commit strictly before it is competing in the CURRENT round,
    at/after it enters the next one. ``baseline`` is the ``(hotkey,
    commit_block)`` set seen at watch start — anything not in it is ``new``.
    """
    from ..interface.validation import parse_commit

    rows: list[SubmissionRow] = []
    for c in commitments:
        if floor_block and c.commit_block < floor_block:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        rows.append(SubmissionRow(
            uid=int(c.uid),
            hotkey=str(c.hotkey),
            ref=parsed.ref,
            commit_block=int(c.commit_block),
            next_round=int(c.commit_block) >= int(epoch_start),
            new=baseline is not None and (str(c.hotkey), int(c.commit_block)) not in baseline,
        ))
    rows.sort(key=lambda r: (-r.commit_block, r.uid))
    return rows


def _short_hotkey(hotkey: str) -> str:
    return hotkey if len(hotkey) <= 13 else f"{hotkey[:6]}…{hotkey[-4:]}"


def _short_ref(ref: str) -> str:
    repo, sep, digest = ref.partition("@")
    if not sep:
        return ref[:32]
    if len(repo) > 20:
        repo = repo[:19] + "…"
    prefix = "hf:" if digest.startswith("hf:") else ""
    digest = digest.removeprefix("sha256:").removeprefix("hf:")
    return f"{repo}@{prefix}{digest[:8]}…"


@dataclass
class LiveFeed:
    """Best-effort live state for the dashboard: chain submissions + the public
    receipt index. Every poll failure keeps the previous snapshot (a chain or
    storage flake dims the feed, it never kills the countdown). The first
    successful commitments poll sets the ``new``-marker baseline, so rows are
    only flagged when they appear DURING the watch session."""

    client: object
    index_fetch: Callable[[], dict | None] | None = None
    status_fetch: Callable[[], dict | None] | None = None
    heat_fetch: Callable[[], dict | None] | None = None
    commitments: list | None = None
    index_doc: dict | None = None
    round_status_doc: dict | None = None
    heat_doc: dict | None = None
    _baseline: set[tuple[str, int]] | None = field(default=None, repr=False)

    def poll(self) -> None:
        poll_fn = getattr(self.client, "poll_commitments", None)
        if poll_fn is not None:
            try:
                cms = list(poll_fn())
            except Exception:  # noqa: BLE001 — keep the previous snapshot
                pass
            else:
                if self._baseline is None:
                    self._baseline = {(str(c.hotkey), int(c.commit_block)) for c in cms}
                self.commitments = cms
        for fetch, attr in ((self.index_fetch, "index_doc"),
                            (self.status_fetch, "round_status_doc"),
                            (self.heat_fetch, "heat_doc")):
            if fetch is None:
                continue
            try:
                doc = fetch()
            except Exception:  # noqa: BLE001 — best-effort; keep the last good doc
                doc = None
            if doc is not None:
                setattr(self, attr, doc)

    def rows(self, epoch_start: int, *, floor_block: int = 0) -> list[SubmissionRow] | None:
        """Current submission rows, or None when the chain feed is unavailable
        (client without ``poll_commitments``, or no successful poll yet)."""
        if self.commitments is None:
            return None
        return submission_rows(self.commitments, epoch_start,
                               floor_block=floor_block, baseline=self._baseline)


# ── frame rendering ──────────────────────────────────────────────────────────


def render(
    st: RoundStatus,
    network: str,
    *,
    drift_seconds: float = 0.0,
    phase: PhaseEstimate | None = None,
    submissions: list[SubmissionRow] | None = None,
    last_outcome: str | None = None,
    heat_lines: list[str] | None = None,
) -> str:
    """The dashboard frame. ``drift_seconds`` is the wall-clock time elapsed
    since ``st.block`` was fetched, so watch mode can tick the countdown every
    second between chain polls. ``phase`` / ``submissions`` / ``last_outcome`` /
    ``heat_lines`` are optional sections (None omits each), so the countdown-only
    frame is unchanged for callers without the live feed."""
    remaining = max(0.0, st.seconds_remaining - drift_seconds)
    eta = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(time.time() + remaining))
    pct = st.progress * 100.0
    lines = [
        f"cascade round — network: {network}",
        f"  current block   {st.block:,}",
        f"  round (epoch)   {st.epoch:,}  ·  started at block {st.epoch_start:,}",
        f"  next round      epoch {st.epoch + 1:,} at block {st.next_epoch_start:,}",
        f"  progress        [{_bar(st.progress)}]  {pct:.1f}%"
        f"  ({st.blocks_elapsed:,} / {st.epoch_blocks:,} blocks)",
        f"  countdown       {format_duration(remaining)} until next round"
        f"  (~{st.spb:.1f}s/block)",
        f"  deadline        commit strictly before block {st.next_epoch_start:,}"
        f" to enter epoch {st.epoch + 1:,}",
        f"  eta             {eta} (estimated)",
    ]
    if phase is not None:
        lines.append(f"  stage           {_phase_strip(phase.key)}")
        lines.append(f"                  {phase.detail}")
    if last_outcome:
        lines.append(f"  last round      {last_outcome}")
    if heat_lines:
        lines += heat_lines
    if submissions is not None:
        n_next = sum(1 for r in submissions if r.next_round)
        n_this = len(submissions) - n_next
        header = (f"{n_this} in this round · {n_next} committed for the next"
                  if submissions else "none revealed yet")
        lines.append(f"  submissions     {header}")
        for r in submissions[:SUBMISSIONS_SHOWN]:
            tag = "→ next round " if r.next_round else "in this round"
            new = "  ● new" if r.new else ""
            lines.append(
                f"    uid {r.uid:>4}  {_short_hotkey(r.hotkey):<11}  "
                f"{_short_ref(r.ref):<32}  block {r.commit_block:>11,}  {tag}{new}"
            )
        if len(submissions) > SUBMISSIONS_SHOWN:
            lines.append(f"    … {len(submissions) - SUBMISSIONS_SHOWN} more (oldest not shown)")
    return "\n".join(lines)


def compose_frame(
    st: RoundStatus,
    network: str,
    round_cfg: RoundConfig,
    feed: LiveFeed,
    timeline: RoundTimeline | None,
    *,
    drift_seconds: float = 0.0,
    me: str | None = None,
) -> str:
    """Assemble one full frame from the chain snapshot + the live feed.

    Stage precedence: a public receipt for THIS round confirms ``settled``;
    then the trainer-reported stage (``status/round.json``) when fresh for
    this round; otherwise the config-timing estimate (when a timeline is
    available). The "last round" context line is shown only while the current
    round is still in flight (it is redundant once this round settles).

    The heat block appears as soon as THIS round's heat is published — the
    trainer writes it when the heat settles, so it lands while the duel is
    still training rather than with the round's receipt. ``me`` (a hotkey or
    UID) flags the caller's own row.
    """
    from ..shared.heat_status import live_heat

    phase: PhaseEstimate | None = None
    last_outcome: str | None = None
    entry = settled_entry_for(feed.index_doc, st.epoch_start)
    if entry is not None:
        phase = phase_for(st, timeline or RoundTimeline(0.0, 0.0),
                          settled_outcome=outcome_line(entry))
    else:
        phase = phase_from_live(feed.round_status_doc, st)
        if phase is None and timeline is not None:
            phase = phase_for(st, timeline, drift_seconds=drift_seconds)
        prior = latest_settled_before(feed.index_doc, st.epoch_start)
        if prior is not None:
            last_outcome = outcome_line(prior).removeprefix("round settled — ")
    heat_doc = live_heat(feed.heat_doc, epoch_start_block=st.epoch_start,
                         now_s=time.time())
    heat_lines = heat_block(heat_doc, me=me) if heat_doc is not None else None
    submissions = feed.rows(st.epoch_start, floor_block=round_cfg.commit_floor_block)
    return render(st, network, drift_seconds=drift_seconds, phase=phase,
                  submissions=submissions, last_outcome=last_outcome,
                  heat_lines=heat_lines)


def run_dashboard(
    client,
    round_cfg: RoundConfig,
    network: str,
    *,
    once: bool = False,
    refresh: float = 30.0,
    out=None,
    timeline: RoundTimeline | None = None,
    index_fetch: Callable[[], dict | None] | None = None,
    status_fetch: Callable[[], dict | None] | None = None,
    heat_fetch: Callable[[], dict | None] | None = None,
    me: str | None = None,
) -> int:
    """Print the round dashboard; in watch mode, keep it live until Ctrl+C.

    Watch mode redraws in place (ANSI cursor-up), ticking the countdown every
    second, re-fetching the real block height every ``refresh`` seconds, and
    re-polling the live feed (commitments + receipt index) on a slower cadence
    (at least :data:`FEED_REFRESH_FLOOR_SECONDS`). A non-TTY ``out`` degrades
    to a single snapshot so piped/scripted runs never emit escape codes.
    ``timeline`` enables the stage estimate; ``index_fetch`` (e.g.
    :func:`fetch_public_receipt_index` bound to the storage config) enables
    settled-round confirmation and last-round context; ``status_fetch`` (e.g.
    :func:`fetch_public_round_status`) enables the trainer-reported live
    stage; ``heat_fetch`` (e.g. :func:`fetch_public_heat`) enables the heat
    standings block once this round's heat settles, with ``me`` (hotkey or UID)
    flagging the caller's own row. A client without ``poll_commitments`` simply
    gets no submissions section.
    """
    out = out if out is not None else sys.stdout
    feed = LiveFeed(client, index_fetch=index_fetch, status_fetch=status_fetch,
                    heat_fetch=heat_fetch)
    st = round_status(client.current_block(), round_cfg)
    feed.poll()
    frame = compose_frame(st, network, round_cfg, feed, timeline, me=me)
    print(frame, file=out)
    if once or not getattr(out, "isatty", lambda: False)():
        return 0

    lines = frame.count("\n") + 1
    fetched_at = feed_at = time.monotonic()
    feed_every = max(float(refresh), FEED_REFRESH_FLOOR_SECONDS)
    try:  # pragma: no cover — interactive loop; the frame logic is tested above
        while True:
            time.sleep(1.0)
            drift = time.monotonic() - fetched_at
            if drift >= refresh:
                st = round_status(client.current_block(), round_cfg)
                fetched_at, drift = time.monotonic(), 0.0
            if time.monotonic() - feed_at >= feed_every:
                feed.poll()
                feed_at = time.monotonic()
            frame = compose_frame(st, network, round_cfg, feed, timeline,
                                  drift_seconds=drift, me=me)
            # move to the top of the previous frame, clear below, redraw
            print(f"\x1b[{lines}F\x1b[J" + frame, file=out)
            lines = frame.count("\n") + 1  # the feed can grow/shrink the frame
    except KeyboardInterrupt:  # pragma: no cover
        return 0
