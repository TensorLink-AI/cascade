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
from ..shared.config import RoundConfig

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
    return RoundStatus(
        block=int(block),
        epoch_blocks=max(1, round_cfg.epoch_blocks),
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


def fetch_public_receipt_index(storage: object, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET the dashboard-facing ``receipts/index.json``.

    Receipts (and the index) are written public-read exactly so third parties
    can read them with zero credentials (see ``cascade.shared.hippius``), so a
    plain path-style HTTPS GET against the manifest bucket works without boto
    or the S3 keys. Best-effort: any failure — offline, private backend,
    malformed JSON — returns None and the dashboard simply shows estimates.
    """
    import urllib.request

    endpoint = str(getattr(storage, "s3_endpoint", "") or "").rstrip("/")
    bucket = str(getattr(storage, "manifest_bucket", "") or "")
    if not endpoint.startswith(("http://", "https://")) or not bucket:
        return None
    from ..shared.hippius import RECEIPT_INDEX_KEY

    url = f"{endpoint}/{bucket}/{RECEIPT_INDEX_KEY}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            doc = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — the index is a best-effort enhancement
        return None
    if isinstance(doc, dict) and isinstance(doc.get("rounds"), list):
        return doc
    return None


def fetch_public_round_status(storage: object, *, timeout: float = 10.0) -> dict | None:
    """Anonymously GET the trainer-reported ``status/round.json``.

    Same zero-credential public-read path as the receipt index. Best-effort:
    any failure returns None and the dashboard falls back to the wall-clock
    stage estimate. Freshness/round-matching is the CONSUMER's job
    (``phase_from_live``), so a stale doc here is returned as-is.
    """
    import urllib.request

    endpoint = str(getattr(storage, "s3_endpoint", "") or "").rstrip("/")
    bucket = str(getattr(storage, "manifest_bucket", "") or "")
    if not endpoint.startswith(("http://", "https://")) or not bucket:
        return None
    from ..shared.chain_status import ROUND_STATUS_KEY

    url = f"{endpoint}/{bucket}/{ROUND_STATUS_KEY}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            doc = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — best-effort enhancement
        return None
    return doc if isinstance(doc, dict) else None


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
    commitments: list | None = None
    index_doc: dict | None = None
    round_status_doc: dict | None = None
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
        if self.index_fetch is not None:
            try:
                doc = self.index_fetch()
            except Exception:  # noqa: BLE001 — best-effort
                doc = None
            if doc is not None:
                self.index_doc = doc
        if self.status_fetch is not None:
            try:
                sdoc = self.status_fetch()
            except Exception:  # noqa: BLE001 — best-effort
                sdoc = None
            if sdoc is not None:
                self.round_status_doc = sdoc

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
) -> str:
    """The dashboard frame. ``drift_seconds`` is the wall-clock time elapsed
    since ``st.block`` was fetched, so watch mode can tick the countdown every
    second between chain polls. ``phase`` / ``submissions`` / ``last_outcome``
    are optional sections (None omits each), so the countdown-only frame is
    unchanged for callers without the live feed."""
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
) -> str:
    """Assemble one full frame from the chain snapshot + the live feed.

    Stage precedence: a public receipt for THIS round confirms ``settled``;
    then the trainer-reported stage (``status/round.json``) when fresh for
    this round; otherwise the config-timing estimate (when a timeline is
    available). The "last round" context line is shown only while the current
    round is still in flight (it is redundant once this round settles).
    """
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
    submissions = feed.rows(st.epoch_start, floor_block=round_cfg.commit_floor_block)
    return render(st, network, drift_seconds=drift_seconds, phase=phase,
                  submissions=submissions, last_outcome=last_outcome)


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
    stage. A client without ``poll_commitments`` simply gets no submissions
    section.
    """
    out = out if out is not None else sys.stdout
    feed = LiveFeed(client, index_fetch=index_fetch, status_fetch=status_fetch)
    st = round_status(client.current_block(), round_cfg)
    feed.poll()
    frame = compose_frame(st, network, round_cfg, feed, timeline)
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
                                  drift_seconds=drift)
            # move to the top of the previous frame, clear below, redraw
            print(f"\x1b[{lines}F\x1b[J" + frame, file=out)
            lines = frame.count("\n") + 1  # the feed can grow/shrink the frame
    except KeyboardInterrupt:  # pragma: no cover
        return 0
