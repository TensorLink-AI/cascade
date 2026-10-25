"""``cascade round`` — a terminal dashboard counting down to the next settlement.

A round is one ``[round] epoch_blocks`` window on the chain block grid (the
same grid the trainer settles on): the current epoch is
``block // epoch_blocks`` and the next settlement is at the next epoch
boundary. Intake is rolling — funded/revealed entries queue for admission and
GPU capacity; completed work can settle at a later boundary — so the boundary is when verdicts land,
not a submission deadline.

Beyond the countdown, the dashboard shows the round's verdict once its receipt
appears in the public ``receipts/index.json`` (and the previous round's while
this one is in flight), the dethrone bar the next verdict is judged at, and a
live feed of revealed on-chain submissions straight from the chain's revealed
commitments; in watch mode a commit that lands while you watch is flagged
``● new`` — the confirmation a miner looks for right after ``cascade deploy``.

Everything on-chain-exact here is in *blocks*; the wall-clock countdown is an
estimate at Bittensor's fixed 12 s/block (never ``round_hours`` over
``epoch_blocks`` — the grid length changes at the rollover, the clock does not). In watch mode the display ticks
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

from ..shared.config import BLOCK_SECONDS, ChainConfig, RoundConfig, effective_epoch_blocks

BAR_WIDTH = 28

# Max submission rows rendered before collapsing to a "… N more" line.
SUBMISSIONS_SHOWN = 8

# Watch mode re-polls commitments + the receipt index at least this rarely —
# both are heavier than a block-height read (metagraph + storage map / an HTTP
# GET), and the feed only needs to move on human timescales.
FEED_REFRESH_FLOOR_SECONDS = 60.0


def seconds_per_block(round_cfg: RoundConfig | None = None) -> float:
    """Bittensor's fixed 12 s block time (:data:`~cascade.shared.config.
    BLOCK_SECONDS`). NOT ``round_hours / epoch_blocks``: the DEC-CA-0045
    rollover shrinks ``epoch_blocks`` 3600 -> 900 while the chain clock stays
    12 s, so that ratio would read 48 s/block and quadruple every ETA.
    ``round_cfg`` is accepted (and ignored) for call-site compatibility."""
    del round_cfg
    return BLOCK_SECONDS


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
        req = urllib.request.Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache"})  # noqa: S310
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


# ── lanes A / B / C (DEC-CA-0050 / 0055 / 0056) from the receipt index ──────


def _pct(x: object) -> str:
    v = _as_float(x)
    return "--" if v is None else f"{v * 100:.0f}%"


def _full_pay(w: dict) -> str:
    """A paid win's gain-tiered full-pay period (owner 2026-10-20): ``, full
    10 eras (tier 0.05)`` — empty when the record carries none."""
    n = w.get("full_eras")
    if n is None:
        return ""
    tier = _as_float(w.get("tier"))
    return (f", full {n} eras" + (f" (tier {tier:g})" if tier is not None else "")
            + (" BELOW the lowest tier" if w.get("below_floor") else ""))


def _paid(wins: object, key: str) -> str:
    rows = [w for w in (wins or []) if isinstance(w, dict)]
    return ", ".join(f"{w.get(key)}→uid {w.get('uid')} ({_pct(w.get('multiplier'))}"
                     f"{_full_pay(w)})" for w in rows) or "none (slots burn)"


def _competition(c: dict) -> str:
    """``mkt: 100% active (from block 7200, no end)`` — one lane-C competition
    of a ``lane_c_pay`` record."""
    fb, ub = _as_int(c.get("from_block")) or 0, _as_int(c.get("until_block")) or 0
    when = (f"from block {fb:,}" if fb else "not started") + (
        f", until block {ub:,}" if ub else ", no end")
    return (f"{c.get('kind')}: {_pct(c.get('frac'))} {str(c.get('state') or '--')}"
            f" ({when})")


def lane_lines(entry: dict | None) -> list[str]:
    """The lane rows of one settled index entry: the lane-A king ladder share
    (DEC-CA-0055 ``ladder``: share, last step, former-king court, key slash)
    and last king check, the lane-B and lane-C verdict counts and paid slots,
    and the emission split. Rows without data are omitted, so a pre-v2 entry
    renders nothing."""
    if not isinstance(entry, dict):
        return []
    out: list[str] = []
    lad = entry.get("ladder") if isinstance(entry.get("ladder"), dict) else None
    kc = entry.get("king_check") if isinstance(entry.get("king_check"), dict) else None
    if lad or kc:
        parts = []
        if lad:
            parts.append(f"king ladder share {_pct(lad.get('king_share'))}"
                         f" ({lad.get('event') or 'none'})")
            if lad.get("crown_units"):
                # DEC-CA-0055 margin-tiered start (owner 2026-10-20)
                lcb, tier = _as_float(lad.get("crown_lcb")), _as_float(lad.get("crown_tier"))
                parts.append("crowned "
                             + (f"at LCB {lcb:.4f}" if lcb is not None else "with no LCB")
                             + (f", tier >= {tier:g}" if tier is not None
                                else f" ({lad['crown_units']} units, fixed start)")
                             + f" -> starts at {_pct(lad.get('king_share'))}")
            if lad.get("court_hotkey"):
                parts.append(f"court {_short_hotkey(str(lad['court_hotkey']))}"
                             f" until block {_as_int(lad.get('court_until_block')) or 0:,}")
            if lad.get("slashed"):
                # DEC-CA-0055 escalating slash: why, and the earliest lift
                until = _as_int(lad.get("slash_until"))
                parts.append(f"key slashed ({lad.get('slash_reason') or 'operator_paid'}"
                             + (f" until block {until:,}" if until
                                else ", rest of reign" if lad.get("slash_reason")
                                == "repeat_lapse" else ", until the key is fixed") + ")")
            else:
                parts.append("no key slash")
        if kc:
            lcb, margin = _as_float(kc.get("lcb")), _as_float(kc.get("margin"))
            nums = (f" (LCB {lcb:.4f} vs margin {margin:.4f})"
                    if lcb is not None and margin is not None else "")
            parts.append(f"king check {kc.get('result') or '--'}{nums}")
        out.append("  lane A          " + " · ".join(parts))
    for lane, rows_key, pay_key, target_key in (("B", "lane_b_verdicts", "lane_b_pay", "domain"),
                                                ("C", "lane_c_verdicts", "lane_c_pay", "target")):
        rows = entry.get(rows_key)
        pay = entry.get(pay_key)
        if rows is None and pay is None:
            continue
        rows = [r for r in (rows or []) if isinstance(r, dict)]
        passed = sum(1 for r in rows if r.get("passed"))
        wins = (pay or {}).get("wins") if isinstance(pay, dict) else None
        out.append(f"  lane {lane}          {len(rows)} judged · {passed} passed · "
                   f"paid: {_paid(wins, target_key)}")
        comps = (pay or {}).get("competitions") if lane == "C" and isinstance(pay, dict) else None
        if comps:
            # concurrent competitions (owner 2026-10-20): weight share + state + dates
            out.append("  lane C comps    " + " · ".join(_competition(c) for c in comps
                                                        if isinstance(c, dict)))
    split = entry.get("emission_split")
    if isinstance(split, dict) and split:
        parts = [f"lane A {_pct(split.get('lane_a'))}", f"lane B {_pct(split.get('lane_b'))}"]
        if "lane_c" in split:
            parts.append(f"lane C {_pct(split.get('lane_c'))}")
        parts.append(f"burn {_pct(split.get('burn'))}")
        out.append("  emission        " + " · ".join(parts))
    return out


# ── current dethrone bar (the tenure-decayed margin, DEC-CA-0016) ────────────
#
# The margin a challenger must clear DECAYS with the king's tenure, so the
# number in the last settled receipt is already one decay step stale for the
# round in flight. The receipt index carries everything needed to derive the
# live bar without validator state: walk the settled rounds newest-first and
# count consecutive holds by the current king (the crowning round — its
# ``dethroned`` row — ends the walk and is not a hold), then apply the same
# affine schedule the validator uses (``eval.koth.margin_for_tenure``,
# replicated here so the dashboard stays free of the eval package).


def current_king_holds(index_doc: dict | None) -> tuple[int, ...] | None:
    """The ``epoch_start_block`` of every settlement the reigning king has
    HELD through (oldest first; the crowning settlement itself excluded), from
    the public receipt index — ``len()`` is the validator's ``tenure_rounds``
    counter. None when no settled round is available."""
    per_round: dict[int, dict] = {}
    for r in _index_rounds(index_doc):
        if str(r.get("status")) != "scored":
            continue
        esb = _as_int(r.get("epoch_start_block"))
        if esb is not None:
            per_round[esb] = r
    if not per_round:
        return None
    ordered = sorted(per_round)
    king = per_round[ordered[-1]].get("post_round_king_hotkey")
    if not king:
        return None
    holds: list[int] = []
    for esb in reversed(ordered):
        row = per_round[esb]
        if row.get("post_round_king_hotkey") != king or row.get("dethroned"):
            break
        holds.append(esb)
    return tuple(reversed(holds))


def tenure_at_block(cfg: ChainConfig, holds: tuple[int, ...], block: int | None) -> int:
    """The tenure a validator judges at the settlement at ``block``, replayed
    from the public hold history with the validator's own functions: before
    ``tenure_blocks_from_block`` the hold counter; after it, blocks reigned ÷
    the grid in force, counted from the anchor ``_anchor_legacy_king`` sets at
    the king's first settlement past the gate (``legacy_king_anchor`` of the
    counter at that settlement — a dethrone clears the anchor, so every king is
    anchored this way)."""
    from ..shared.era import legacy_king_anchor, tenure_blocks_active, tenure_rounds_at

    since: int | None = None
    if block is not None and tenure_blocks_active(cfg.scoring, block):
        gate = int(cfg.scoring.tenure_blocks_from_block)
        post_gate = [b for b in holds if b >= gate]
        first = post_gate[0] if post_gate else int(block)
        pre = sum(1 for b in holds if b < first)
        since = legacy_king_anchor(cfg.round, cfg.scoring, block=first, tenure_rounds=pre)
    return tenure_rounds_at(cfg.round, cfg.scoring, block=block,
                            tenure_rounds=len(holds), king_since_block=since)


def _judged_params(cfg: ChainConfig, block: int | None):
    """The validator's own resolution of the dethrone rule for the settlement
    at epoch boundary ``block``: :meth:`ChainConfig.koth_params` (scheduled
    fresh-king margin, DEC-CA-0045 blocks-denominated warm-up, DEC-CA-0039
    margin mode, DEC-CA-0049 v2 bar)."""
    return cfg.koth_params(block=block)


def dethrone_bar_line(index_doc: dict | None, cfg: ChainConfig | None,
                      block: int | None = None) -> str | None:
    """The forward-looking margin line for the settlement at epoch boundary
    ``block`` (the round in flight), or None when the schedule is flat (no
    decay configured) or the index gives no tenure.

    Tenure is counted the way the validator counts it at ``block``
    (:func:`tenure_at_block`).
    The bar is labelled in its units: under the increment margin (DEC-CA-0039,
    and DEC-CA-0049's v2 bar) it is a share of the per-round increment over the
    shared init, not of the king's absolute score. A round the validator has
    to fall back to level units (no init baseline) is judged at the level bar
    instead — that is decided in-round and cannot be forecast here."""
    if cfg is None:
        return None
    from ..eval.koth import margin_for_tenure

    params = _judged_params(cfg, block)
    if params.win_margin_start == params.win_margin_end:
        return None
    holds = current_king_holds(index_doc)
    if holds is None:
        return None
    tenure = tenure_at_block(cfg, holds, block)
    margin = margin_for_tenure(params, tenure)
    floor_at = params.margin_warmup_rounds
    end = params.win_margin_end
    unit = " of the increment" if params.margin_mode == "increment" else ""
    if tenure >= floor_at:
        tail = f"at the {end * 100:.2f}% floor"
    else:
        tail = f"floor {end * 100:.2f}% at tenure {floor_at}"
    return (f"  dethrone bar    LCB > {margin * 100:.3f}%{unit} this round"
            f"  (king tenure {tenure}; {tail})")


def _as_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


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


def render_duel(rows: list[dict], *, me: str | None = None) -> str:
    """The standalone ``cascade duel`` view of one settled round's index rows.
    ``me`` (a hotkey) appends that miner's own per-domain scores against the
    king — see :func:`my_domain_lines`."""
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
    lines += horizon_lines(row.get("per_horizon"))
    lines += domain_lines(row.get("per_domain"))
    lines += cohort_lines(row)
    gift = row.get("gift_gate_passed")
    if gift is not None:
        lines.append("  gift gate      " + ("passed" if gift else "BLOCKED the dethrone"))
    # Archived boundary-round receipts carry a heat summary; rolling ones don't.
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
    # The win-rate-only block is the pre-per_domain view: once the receipt
    # carries per-domain SCORES the ``by domain`` block above says the same
    # thing with magnitudes, so this one is shown only for older rows.
    if isinstance(domains, dict) and domains and not row.get("per_domain"):
        lines.append("  per-domain win rate  (right of centre = challenger ahead)")
        ordered = sorted(domains.items(),
                         key=lambda kv: -(_as_float(_domain_pair(kv[1])[0]) or 0.0))
        for name, value in ordered:
            rate, n = _domain_pair(value)
            if rate is None:
                continue
            n_s = f"n={n:>5}" if n is not None else ""
            lines.append(f"    {name:<14} {rate:.2f}  {n_s}  {_domain_bar(rate)}")
    if me:
        lines += my_domain_lines(row, me)
    if (validators := _duel_validators_line(rows)) is not None:
        lines.append(validators)
    return "\n".join(lines)


# A domain with fewer windows than this is flagged: a gap on a handful of
# windows is draw noise, not a strength or weakness. Display-only threshold.
DOMAIN_MIN_WINDOWS = 30


def _domain_cells(per_domain: object) -> list[tuple[str, float, float, float, float | None, int | None]]:
    """``[(domain, king, chal, gap%, win_rate, n)]`` from a ``per_domain``
    breakdown, best-for-the-challenger (most negative gap) first. Cells missing
    either geomean are skipped."""
    if not isinstance(per_domain, dict):
        return []
    out = []
    for name, cell in per_domain.items():
        k, c, gap = _horizon_gap(cell if isinstance(cell, dict) else {})
        if k is None or c is None or gap is None:
            continue
        out.append((str(name), k, c, gap, _as_float(cell.get("win_rate")),
                    _as_int(cell.get("n"))))
    out.sort(key=lambda t: t[3])
    return out


def _domain_row(name: str, k: float, c: float, gap: float, wr: float | None,
                n: int | None, *, who: str = "chal") -> str:
    line = f"    {name:<14} king {k:.5f}  {who} {c:.5f}  ({gap:+.2f}%)"
    if wr is not None:
        line += f"  win {wr * 100:.0f}%"
    if n:
        line += f"  n={n}"
        if n < DOMAIN_MIN_WINDOWS:
            line += "  (few windows — noise)"
    return line


def domain_lines(per_domain: object) -> list[str]:
    """The ``by domain`` block of a verdict: one line per pool domain with the
    king's and the decided challenger's domain geomeans, the relative gap and
    the challenger's win rate on it — WHERE the challenger beat the king and by
    how much. Empty when the receipt predates the field or the pool carries no
    domain labels."""
    cells = _domain_cells(per_domain)
    if not cells:
        return []
    lines = ["  by domain      (pooled windows decide; negative gap = challenger better)"]
    lines += [_domain_row(*cell) for cell in cells]
    return lines


def domain_breakdown_for(row: dict, hotkey: str) -> dict | None:
    """One miner's per-domain breakdown on a settled row: their entry in
    ``cohort_per_domain``, or the headline ``per_domain`` when they were the
    decided challenger (the k=1 case — most rolling settlements)."""
    if not hotkey:
        return None
    cpd = row.get("cohort_per_domain")
    if isinstance(cpd, dict) and isinstance(cpd.get(hotkey), dict):
        return cpd[hotkey]
    if str(row.get("chal_hotkey") or "") == hotkey and isinstance(row.get("per_domain"), dict):
        return row["per_domain"]
    return None


def my_domain_lines(row: dict, hotkey: str) -> list[str]:
    """The ``your domains`` block of ``cascade duel --hotkey``: the caller's own
    per-domain scores against the king this round, strongest domain first, or
    one line saying why there are none (king / not judged / pre-field row)."""
    tag = _short_hotkey(hotkey)
    if str(row.get("king_hotkey") or "") == hotkey:
        return [f"  your domains   {tag} was the king this round — see 'by domain' "
                "for how the challenger did against you"]
    pd = domain_breakdown_for(row, hotkey)
    if pd is None:
        judged = row.get("cohort_geomeans")
        if (isinstance(judged, dict) and hotkey in judged) or \
                str(row.get("chal_hotkey") or "") == hotkey:
            return [f"  your domains   {tag} was judged, but this receipt carries no "
                    "per-domain scores (validator predates the field)"]
        return [f"  your domains   {tag} was not judged this round"]
    cells = _domain_cells(pd)
    if not cells:
        return [f"  your domains   {tag}: no per-domain scores on this receipt"]
    better = [c for c in cells if c[3] < 0]
    worse = [c for c in cells if c[3] >= 0]
    lines = [f"  your domains   {tag} vs the king — {len(better)} better, {len(worse)} worse "
             "(negative gap = you beat the king)"]
    lines += [_domain_row(*cell, who="you ") for cell in cells]
    return lines


def render_duel_domain_history(index_doc: dict | None, hotkey: str, *,
                               limit: int = 20) -> str:
    """``cascade duel --hotkey <me> --history``: one line per settled round the
    miner was judged in, their gap vs the king overall and per domain, newest
    last, plus an average per domain over the rounds shown — the "am I getting
    better at web" view."""
    if not hotkey:
        return "pass --hotkey <ss58> to see your per-domain history"
    by_round: dict[str, list[dict]] = {}
    for r in _index_rounds(index_doc):
        by_round.setdefault(str(r.get("round_id")), []).append(r)
    groups = sorted(by_round.values(),
                    key=lambda g: max((_as_int(r.get("epoch_start_block")) or 0) for r in g))
    picked: list[tuple[dict, dict]] = []
    for g in groups:
        scored = [r for r in g if str(r.get("status")) == "scored"]
        if not scored:
            continue
        row = scored[-1]
        pd = domain_breakdown_for(row, hotkey)
        if pd is not None and _domain_cells(pd):
            picked.append((row, pd))
    if not picked:
        return (f"no per-domain scores for {_short_hotkey(hotkey)} in the public receipt "
                "index — either not judged yet, or the validators' receipts predate "
                "the per-domain field")
    shown = picked[-limit:]
    domains = sorted({name for _, pd in shown for name, *_ in _domain_cells(pd)})
    lines = [f"cascade duel — {_short_hotkey(hotkey)} by domain over {len(shown)} judged "
             "round(s), newest last (gap vs the king; negative = you were better)"]
    lines.append("  " + f"{'round':<14}{'overall':>9}"
                 + "".join(f"{d[:11]:>13}" for d in domains))
    sums: dict[str, list[float]] = {d: [] for d in domains}
    for row, pd in shown:
        cells = {name: gap for name, _k, _c, gap, _w, _n in _domain_cells(pd)}
        kg = _as_float(row.get("king_geomean"))
        mine = _as_float((row.get("cohort_geomeans") or {}).get(hotkey)
                         if isinstance(row.get("cohort_geomeans"), dict) else None)
        if mine is None and str(row.get("chal_hotkey") or "") == hotkey:
            mine = _as_float(row.get("chal_geomean"))
        overall = f"{(mine - kg) / kg * 100:+.2f}%" if kg and mine is not None else "--"
        rid = str(row.get("round_id", "--"))
        rid = rid if len(rid) <= 12 else f"{rid[:6]}…{rid[-4:]}"
        won = " ←crown" if row.get("dethroned") and str(row.get("chal_hotkey")) == hotkey else ""
        line = "  " + f"{rid:<14}{overall:>9}"
        for d in domains:
            gap = cells.get(d)
            line += f"{'--':>13}" if gap is None else f"{gap:+.2f}%".rjust(13)
            if gap is not None:
                sums[d].append(gap)
        lines.append(line + won)
    avg = "  " + f"{'average':<14}{'':>9}"
    for d in domains:
        vals = sums[d]
        avg += (f"{sum(vals) / len(vals):+.2f}%".rjust(13) if vals else f"{'--':>13}")
    lines.append(avg)
    strengths = [d for d in domains if sums[d] and sum(sums[d]) / len(sums[d]) < 0]
    weaknesses = [d for d in domains if sums[d] and sum(sums[d]) / len(sums[d]) >= 0]
    lines.append("  better than the king on average: " + (", ".join(strengths) or "none")
                 + "  ·  worse: " + (", ".join(weaknesses) or "none"))
    return "\n".join(lines)


def _horizon_gap(cell: dict) -> tuple[float | None, float | None, float | None]:
    """(king, chal, relative gap %) for one rung; gap negative = challenger better."""
    k, c = _as_float(cell.get("king")), _as_float(cell.get("chal"))
    if k is None or c is None or k == 0:
        return k, c, None
    return k, c, (c - k) / k * 100.0


def horizon_lines(per_horizon: object) -> list[str]:
    """The ``by horizon`` block of a ladder-round verdict: one line per rung
    with the king's and the decided challenger's rung geomeans, the relative
    gap, and the challenger's per-window win rate on that rung. Empty for a
    single-horizon round (the field is absent)."""
    if not isinstance(per_horizon, dict) or not per_horizon:
        return []
    lines = ["  by horizon     (pooled rungs decide; negative gap = challenger better)"]
    for h in sorted(per_horizon, key=lambda x: int(x)):
        cell = per_horizon[h] if isinstance(per_horizon[h], dict) else {}
        k, c, gap = _horizon_gap(cell)
        if k is None or c is None:
            continue
        wr, n = _as_float(cell.get("win_rate")), _as_int(cell.get("n"))
        line = f"    h={int(h):<4} king {k:.5f}  chal {c:.5f}"
        if gap is not None:
            line += f"  ({gap:+.2f}%)"
        if wr is not None:
            line += f"  win {wr * 100:.0f}%"
        if n:
            line += f"  n={n}"
        lines.append(line)
    return lines


def cohort_lines(row: dict) -> list[str]:
    """The ``cohort`` block: every judged challenger with its observed geomean
    vs the king, its LCB against the margin (when published), and its
    per-horizon gaps — the "all miners" view of a duel-only round. Empty when
    the receipt carries no cohort geomeans (pre-ladder receipts)."""
    geos = row.get("cohort_geomeans")
    if not isinstance(geos, dict) or not geos:
        return []
    kg = _as_float(row.get("king_geomean"))
    lcbs = row.get("cohort_lcbs") if isinstance(row.get("cohort_lcbs"), dict) else {}
    phs = (row.get("cohort_per_horizon")
           if isinstance(row.get("cohort_per_horizon"), dict) else {})
    pds = (row.get("cohort_per_domain")
           if isinstance(row.get("cohort_per_domain"), dict) else {})
    margin = _as_float(row.get("margin"))
    winner = row.get("chal_hotkey") if row.get("dethroned") else None
    ranked = sorted(((hk, _as_float(g)) for hk, g in geos.items()),
                    key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0.0))
    k = len(ranked)
    lines = [f"  cohort         {k} challenger(s) judged"
             + (f" at α/{k}" if k > 1 else "") + " — best point estimate first"]
    for i, (hk, g) in enumerate(ranked, 1):
        line = f"    #{i:<2} {_short_hotkey(str(hk)):<11}"
        if g is not None:
            line += f"  geomean {g:.5f}"
            if kg:
                line += f"  ({(g - kg) / kg * 100:+.2f}% vs king)"
        lcb = _as_float(lcbs.get(hk))
        if lcb is not None:
            line += f"  LCB {lcb:+.4f}"
            if margin is not None:
                line += "  ✓ cleared" if lcb > margin else "  ✗"
        ph = phs.get(hk)
        if isinstance(ph, dict) and ph:
            gaps = []
            for h in sorted(ph, key=lambda x: int(x)):
                _, _, gap = _horizon_gap(ph[h] if isinstance(ph[h], dict) else {})
                if gap is not None:
                    gaps.append(f"h{int(h)} {gap:+.1f}%")
            if gaps:
                line += "  [" + " · ".join(gaps) + "]"
        if winner and hk == winner:
            line += "  ← took the throne"
        lines.append(line)
        cells = _domain_cells(pds.get(hk))
        if cells:
            # Per-domain gaps on a continuation line: strongest domain first,
            # so each miner sees where they beat the king without --hotkey.
            lines.append("        by domain: "
                         + " · ".join(f"{name} {gap:+.1f}%" for name, _k, _c, gap, _w, _n in cells))
    return lines


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
    commitments: list | None = None
    index_doc: dict | None = None
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
            except Exception:  # noqa: BLE001 — best-effort; keep the last good doc
                doc = None
            if doc is not None:
                self.index_doc = doc

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
    settled: str | None = None,
    submissions: list[SubmissionRow] | None = None,
    last_outcome: str | None = None,
    bar_line: str | None = None,
    lanes: list[str] | None = None,
) -> str:
    """The dashboard frame. ``drift_seconds`` is the wall-clock time elapsed
    since ``st.block`` was fetched, so watch mode can tick the countdown every
    second between chain polls. ``settled`` / ``submissions`` / ``last_outcome`` /
    ``bar_line`` are optional sections (None omits each), so
    the countdown-only frame is unchanged for callers without the live feed."""
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
        f"  countdown       {format_duration(remaining)} until next settlement"
        f"  (~{st.spb:.1f}s/block)",
        f"  intake          next settlement at block {st.next_epoch_start:,};"
        " funded/revealed entries queue for admission and GPU capacity; a particular settlement is not guaranteed",
        f"  eta             {eta} (estimated)",
    ]
    if settled:
        lines.append(f"  this round      {settled}")
    if last_outcome:
        lines.append(f"  last round      {last_outcome}")
    if bar_line:
        lines.append(bar_line)
    if lanes:
        lines.extend(lanes)
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
    *,
    drift_seconds: float = 0.0,
    chain_cfg: ChainConfig | None = None,
) -> str:
    """Assemble one full frame from the chain snapshot + the live feed.

    A public receipt for THIS round shows its verdict; until then the
    previous settled round's verdict is shown as context, with the dethrone
    bar the next verdict will be judged at.
    """
    settled: str | None = None
    last_outcome: str | None = None
    entry = settled_entry_for(feed.index_doc, st.epoch_start)
    prior = None
    if entry is not None:
        settled = outcome_line(entry).removeprefix("round ")
    else:
        prior = latest_settled_before(feed.index_doc, st.epoch_start)
        if prior is not None:
            last_outcome = outcome_line(prior).removeprefix("round settled — ")
    submissions = feed.rows(st.epoch_start, floor_block=round_cfg.commit_floor_block)
    # Forward-looking bar only while the round is in flight — once this round
    # settles its receipt states the margin it was actually judged at.
    bar_line = (dethrone_bar_line(feed.index_doc, chain_cfg, block=st.epoch_start)
                if entry is None else None)
    # lanes B / C and the lane-A ladder of the settlement on show
    lanes = lane_lines(entry if entry is not None else prior)
    return render(st, network, drift_seconds=drift_seconds, settled=settled,
                  submissions=submissions, last_outcome=last_outcome,
                  bar_line=bar_line, lanes=lanes or None)


def run_dashboard(
    client,
    round_cfg: RoundConfig,
    network: str,
    *,
    once: bool = False,
    refresh: float = 30.0,
    out=None,
    index_fetch: Callable[[], dict | None] | None = None,
    chain_cfg: ChainConfig | None = None,
) -> int:
    """Print the round dashboard; in watch mode, keep it live until Ctrl+C.

    Watch mode redraws in place (ANSI cursor-up), ticking the countdown every
    second, re-fetching the real block height every ``refresh`` seconds, and
    re-polling the live feed (commitments + receipt index) on a slower cadence
    (at least :data:`FEED_REFRESH_FLOOR_SECONDS`). A non-TTY ``out`` degrades
    to a single snapshot so piped/scripted runs never emit escape codes.
    ``index_fetch`` (e.g. :func:`fetch_public_receipt_index` bound to the
    storage config) enables the settled verdict, last-round context and the
    dethrone bar. A client without ``poll_commitments`` simply gets no
    submissions section.
    """
    out = out if out is not None else sys.stdout
    feed = LiveFeed(client, index_fetch=index_fetch)
    st = round_status(client.current_block(), round_cfg)
    feed.poll()
    frame = compose_frame(st, network, round_cfg, feed, chain_cfg=chain_cfg)
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
            frame = compose_frame(st, network, round_cfg, feed,
                                  drift_seconds=drift, chain_cfg=chain_cfg)
            # move to the top of the previous frame, clear below, redraw
            print(f"\x1b[{lines}F\x1b[J" + frame, file=out)
            lines = frame.count("\n") + 1  # the feed can grow/shrink the frame
    except KeyboardInterrupt:  # pragma: no cover
        return 0
