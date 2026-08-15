#!/usr/bin/env python3
"""Replay historic signed round receipts under candidate margin-decay schedules.

WORKSTREAM: tenure-decaying dethrone margin (DEC-CA-0016). The deployed
``chain.toml [scoring]`` is a flat margin (``win_margin_start ==
win_margin_end = 0.02``, ``margin_warmup_rounds = 0``). Setting
``win_margin_end < win_margin_start`` with ``margin_warmup_rounds > 0`` turns
:func:`cascade.eval.koth.margin_for_tenure` into a tenure DECAY: a fresh king
defends the full ``start`` margin, an entrenched one a progressively smaller
one, clamped at ``end`` from ``margin_warmup_rounds`` of tenure on.

This tool answers "which past verdicts would have flipped?" without a GPU and
without re-running any bootstrap: ``margin_for_tenure`` is pure, and every
scored receipt already records the complete inputs — ``verdict.params`` (the
KOTH params that judged the round), ``verdict.king_tenure_rounds`` (tenure at
decision time), the decided ``lcb``, and on cohort rounds (DEC-CA-0012) the
per-challenger ``cohort_lcbs`` at the corrected quantile. Replaying a schedule
is: recompute the margin from recorded tenure, compare recorded LCBs against
it. The bootstrap quantile (and therefore every LCB) is UNTOUCHED by a margin
schedule change, so recorded LCBs remain exactly the right statistic.

Honesty caveat (printed with every report): this is a FIRST-ORDER replay. Each
round is re-judged against the throne history that actually happened; a
flipped verdict at round r would have changed the king, the tenure clock, and
plausibly every subsequent round's field. Flip counts measure how binding the
flat margin was round-by-round — they are not a simulated alternate timeline.

Guardrail (hard): the margin floor (``win_margin_end``) must stay > 0. The
margin is the improvement bar ABOVE the LCB noise gate — at 0 the decision
degenerates to "any statistically nonzero improvement dethrones", which the
bootstrap alpha alone does not price for multiple rounds of attempts.

Usage:
    # local receipts (files or directories of receipts/round-*.json)
    uv run python scripts/replay_margin_decay.py ./receipts/

    # explicit schedule grid (defaults: start=0.02, end=5/8/10bp, warmup=8/12/16)
    uv run python scripts/replay_margin_decay.py ./receipts/ \
        --start 0.02 --ends 0.005,0.008,0.010 --warmups 8,12,16

    # fetch the public receipt trail first (uses [storage]/[manifest] from the
    # chain toml; needs the usual HIPPIUS_S3_* env credentials)
    uv run python scripts/replay_margin_decay.py --fetch ./receipts_cache \
        --chain-toml chain.toml --verify

``--verify`` checks each receipt's validator signature against the pinned
``[manifest] validator_hotkey`` (or ``--validator-hotkey``) and skips any that
fail — replaying an unsigned/tampered receipt would launder a fake flip.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

# Make the repo importable when run as `python scripts/replay_margin_decay.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.eval.koth import KothParams, margin_for_tenure  # noqa: E402
from cascade.shared.receipt import (  # noqa: E402
    RoundReceipt,
    load_receipt,
    verify_receipt_signature,
)

# The candidate grid from the design thread (see the module docstring): keep the
# fresh-king margin where it is, decay to end over warmup rounds of tenure.
# warmup=4 is the owner's steep-ramp candidate (floor by ~2 days at 12h rounds;
# in effect a flat `end` bar with a short fresh-king shield).
DEFAULT_START = 0.02
DEFAULT_ENDS = (0.005, 0.008, 0.010)
DEFAULT_WARMUPS = (4, 8, 12, 16)

# "Near miss" band: a non-flipped loss whose LCB lands within this of the
# candidate margin. These rounds are where one more schedule notch WOULD start
# flipping verdicts — the sensitivity report the owner picks a schedule from.
NEAR_MISS_BAND = 0.005


@dataclasses.dataclass(frozen=True)
class Schedule:
    start: float
    end: float
    warmup: int

    def label(self) -> str:
        return f"start={self.start:g} end={self.end:g} warmup={self.warmup}"

    def margin_at(self, tenure: int, base: KothParams) -> float:
        """The candidate margin at ``tenure``, over the receipt's own params
        (so every non-schedule field rides along unchanged)."""
        return margin_for_tenure(
            dataclasses.replace(
                base,
                win_margin_start=self.start,
                win_margin_end=self.end,
                margin_warmup_rounds=self.warmup,
            ),
            tenure,
        )


def params_from_verdict(vparams: dict) -> KothParams:
    """Rebuild ``KothParams`` from a receipt's recorded ``verdict.params``,
    tolerating fields added after (or absent from) that receipt's era."""
    fields = {f.name for f in dataclasses.fields(KothParams)}
    required = {f.name for f in dataclasses.fields(KothParams)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING}  # type: ignore[misc]
    known = {k: v for k, v in vparams.items() if k in fields}
    missing = required - set(known)
    if missing:
        raise ValueError(f"verdict.params missing required KOTH fields: {sorted(missing)}")
    return KothParams(**known)


def _cohort_clearers(cohort_lcbs: dict, margin: float) -> list[str]:
    return sorted(
        hk for hk, lcb in cohort_lcbs.items()
        if lcb is not None and float(lcb) >= margin
    )


@dataclasses.dataclass
class RoundReplay:
    """One receipt's replay under one schedule."""

    round_id: str
    tenure: int
    lcb: float
    margin_recorded: float
    margin_candidate: float
    won_recorded: bool
    won_candidate: bool
    cohort_k: int = 0
    # cohort members that clear under the candidate but not the recorded margin
    new_clearers: tuple[str, ...] = ()

    @property
    def flipped(self) -> bool:
        return self.won_recorded != self.won_candidate

    @property
    def near_miss(self) -> bool:
        gap = self.margin_candidate - self.lcb
        return (not self.won_candidate) and 0.0 < gap <= NEAR_MISS_BAND


def replay_receipt(receipt: RoundReceipt, schedule: Schedule) -> RoundReplay | None:
    """Re-judge one scored receipt under ``schedule``; ``None`` when the round
    made no margin decision (rejected, inconclusive, or LCB-less)."""
    v = receipt.verdict
    if receipt.status != "scored" or v is None or v.inconclusive or v.lcb is None:
        return None
    base = params_from_verdict(v.params)

    # Consistency gate: the recorded schedule must reproduce the recorded
    # margin from the recorded tenure, or this receipt is not replayable
    # (unknown margin rule — bail loudly rather than fake a verdict).
    recomputed = margin_for_tenure(base, v.king_tenure_rounds)
    if not math.isclose(recomputed, v.margin, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"round {receipt.round_id}: recorded margin {v.margin} != "
            f"margin_for_tenure(recorded params, tenure={v.king_tenure_rounds}) "
            f"= {recomputed}; receipt not replayable"
        )

    margin_new = schedule.margin_at(v.king_tenure_rounds, base)
    if v.cohort_k and v.cohort_lcbs:
        # Cohort round (DEC-CA-0012): each challenger's recorded LCB is already
        # at the alpha/k quantile; the margin is the same flat bar for all.
        old = set(_cohort_clearers(v.cohort_lcbs, v.margin))
        new = set(_cohort_clearers(v.cohort_lcbs, margin_new))
        won_new = bool(new)
        new_clearers = tuple(sorted(new - old))
    else:
        won_new = float(v.lcb) >= margin_new
        new_clearers = ()
    return RoundReplay(
        round_id=receipt.round_id,
        tenure=int(v.king_tenure_rounds),
        lcb=float(v.lcb),
        margin_recorded=float(v.margin),
        margin_candidate=float(margin_new),
        won_recorded=bool(v.challenger_wins_round),
        won_candidate=won_new,
        cohort_k=int(v.cohort_k or 0),
        new_clearers=new_clearers,
    )


def load_receipts(paths: list[Path], *, verify_hotkey: str | None = None
                  ) -> tuple[list[RoundReceipt], list[str]]:
    """Load (and optionally signature-check) every receipt under ``paths``.
    Returns ``(receipts sorted by epoch block, problem strings)``."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.rglob("round-*.json"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    receipts: list[RoundReceipt] = []
    problems: list[str] = []
    seen: set[str] = set()
    for f in files:
        try:
            receipt = load_receipt(f.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as e:
            problems.append(f"{f}: unreadable receipt ({e})")
            continue
        if verify_hotkey:
            try:
                ok = verify_receipt_signature(receipt, verify_hotkey)
            except RuntimeError as e:  # bittensor unavailable
                raise SystemExit(f"--verify needs the [chain] extra: {e}") from e
            if not ok:
                problems.append(f"{f}: signature check FAILED vs {verify_hotkey}; skipped")
                continue
        if receipt.round_id in seen:  # latest.json duplicates the round file
            continue
        seen.add(receipt.round_id)
        receipts.append(receipt)
    receipts.sort(key=lambda r: r.epoch_start_block)
    return receipts, problems


def fetch_receipts(dest: Path, chain_toml: Path | None) -> list[Path]:
    """Pull the public receipt trail into ``dest`` via the configured manifest
    store (receipts index → per-round receipts). Requires storage credentials
    in the environment, exactly like the validator."""
    from cascade.shared.config import load_chain_config
    from cascade.shared.hippius import (
        RECEIPT_INDEX_KEY,
        open_manifest_store,
        receipt_round_key,
    )

    cfg = load_chain_config(chain_toml)
    store = open_manifest_store(cfg.storage)
    hotkey = cfg.manifest.validator_hotkey
    idx = json.loads(store.get_text(RECEIPT_INDEX_KEY))
    rounds = [str(r.get("round_id")) for r in (idx.get("rounds") or [])
              if r.get("round_id")]
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for rid in rounds:
        f = dest / f"round-{rid}.json"
        if not f.is_file():
            text = None
            for key in (receipt_round_key(rid, hotkey), receipt_round_key(rid)):
                try:
                    text = store.get_text(key)
                    break
                except Exception:  # noqa: BLE001 — try the legacy key next
                    continue
            if text is None:
                print(f"  ! round {rid}: receipt unreachable under both keys; skipped")
                continue
            f.write_text(text, encoding="utf-8")
        out.append(f)
    print(f"fetched {len(out)}/{len(rounds)} receipts → {dest}")
    return out


def replay_all(receipts: list[RoundReceipt], schedules: list[Schedule]) -> dict:
    """The full grid: ``{schedule label: {summary, flips, near_misses}}``."""
    report: dict = {}
    for sched in schedules:
        rows: list[RoundReplay] = []
        skipped: list[str] = []
        for receipt in receipts:
            try:
                row = replay_receipt(receipt, sched)
            except ValueError as e:
                skipped.append(str(e))
                continue
            if row is not None:
                rows.append(row)
        flips = [r for r in rows if r.flipped]
        near = [r for r in rows if r.near_miss]
        report[sched.label()] = {
            "schedule": dataclasses.asdict(sched),
            "n_receipts": len(receipts),
            "n_decided": len(rows),
            "n_flips": len(flips),
            "n_flips_loss_to_win": sum(1 for r in flips if r.won_candidate),
            "n_flips_win_to_loss": sum(1 for r in flips if not r.won_candidate),
            "n_near_misses": len(near),
            "min_candidate_margin": min((r.margin_candidate for r in rows), default=None),
            "max_tenure_seen": max((r.tenure for r in rows), default=None),
            "flips": [dataclasses.asdict(r) for r in flips],
            "near_misses": [dataclasses.asdict(r) for r in near],
            "skipped": skipped,
        }
    return report


def print_report(report: dict) -> None:
    print()
    print("=" * 76)
    print("MARGIN-DECAY REPLAY — first-order re-judgement of the signed receipt trail")
    print("=" * 76)
    for label, r in report.items():
        print(f"\n--- {label} ---")
        print(f"  decided rounds: {r['n_decided']}/{r['n_receipts']} receipts"
              f"  (max tenure seen: {r['max_tenure_seen']},"
              f" lowest margin applied: {r['min_candidate_margin']})")
        print(f"  flips: {r['n_flips']}"
              f" (loss→win {r['n_flips_loss_to_win']},"
              f" win→loss {r['n_flips_win_to_loss']})"
              f"   near-misses (≤{NEAR_MISS_BAND:g} short): {r['n_near_misses']}")
        for f in r["flips"]:
            extra = (f"  new clearers: {', '.join(f['new_clearers'])}"
                     if f["new_clearers"] else "")
            print(f"    FLIP round={f['round_id']} tenure={f['tenure']}"
                  f" lcb={f['lcb']:+.4f}"
                  f" margin {f['margin_recorded']:.4f}→{f['margin_candidate']:.4f}"
                  f" win {f['won_recorded']}→{f['won_candidate']}{extra}")
        for f in r["near_misses"]:
            print(f"    near  round={f['round_id']} tenure={f['tenure']}"
                  f" lcb={f['lcb']:+.4f} vs margin {f['margin_candidate']:.4f}"
                  f" (short by {f['margin_candidate'] - f['lcb']:.4f})")
        if r["skipped"]:
            print(f"  ! {len(r['skipped'])} receipt(s) not replayable:")
            for s in r["skipped"]:
                print(f"      {s}")
    print()
    print("CAVEAT: first-order replay. A flip at round r would have changed the")
    print("king, tenure clock, and field for every later round; counts measure")
    print("how binding the flat margin was, not an alternate timeline.")
    print()


def build_schedules(start: float, ends: list[float], warmups: list[int]) -> list[Schedule]:
    for end in ends:
        # HARD GUARDRAIL: the decayed floor must stay strictly positive — the
        # margin is the improvement bar above the LCB noise gate (win = LCB
        # ≥ margin); a 0 floor makes any nonzero LCB a dethrone.
        if end <= 0:
            raise SystemExit(f"guardrail: win_margin_end must stay > 0; got {end}")
        if end > start:
            raise SystemExit(
                f"schedule end={end} > start={start} is a tenure RAMP, not a "
                f"decay — this tool studies decay schedules only")
    for w in warmups:
        if w <= 0:
            raise SystemExit(f"margin_warmup_rounds must be > 0 for a decay; got {w}")
    return [Schedule(start=start, end=e, warmup=w) for e in ends for w in warmups]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Receipt files, or directories containing receipts/round-*.json. "
                         "With --fetch: the single cache directory to download into.")
    ap.add_argument("--start", type=float, default=DEFAULT_START,
                    help=f"win_margin_start (fresh-king margin; default {DEFAULT_START})")
    ap.add_argument("--ends", default=",".join(str(e) for e in DEFAULT_ENDS),
                    help="comma-separated win_margin_end candidates (the decayed floor)")
    ap.add_argument("--warmups", default=",".join(str(w) for w in DEFAULT_WARMUPS),
                    help="comma-separated margin_warmup_rounds candidates")
    ap.add_argument("--fetch", action="store_true",
                    help="download the public receipt trail into PATHS[0] first "
                         "(needs [storage] credentials in the environment)")
    ap.add_argument("--chain-toml", type=Path, default=None,
                    help="chain toml for --fetch / --verify hotkey resolution")
    ap.add_argument("--verify", action="store_true",
                    help="check receipt signatures against the pinned validator hotkey; "
                         "failing receipts are skipped")
    ap.add_argument("--validator-hotkey", default=None,
                    help="explicit ss58 for --verify (default: [manifest] validator_hotkey)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full report as JSON")
    args = ap.parse_args(argv)

    schedules = build_schedules(
        args.start,
        [float(x) for x in str(args.ends).split(",") if x.strip()],
        [int(x) for x in str(args.warmups).split(",") if x.strip()],
    )

    if args.fetch:
        fetch_receipts(args.paths[0], args.chain_toml)

    verify_hotkey = None
    if args.verify:
        verify_hotkey = args.validator_hotkey
        if not verify_hotkey:
            from cascade.shared.config import load_chain_config

            verify_hotkey = load_chain_config(args.chain_toml).manifest.validator_hotkey
        if not verify_hotkey:
            raise SystemExit("--verify: no validator hotkey (flag or [manifest] "
                             "validator_hotkey) to verify against")

    receipts, problems = load_receipts(list(args.paths), verify_hotkey=verify_hotkey)
    for p in problems:
        print(f"  ! {p}")
    if not receipts:
        raise SystemExit("no usable receipts found")
    print(f"loaded {len(receipts)} receipt(s); replaying {len(schedules)} schedule(s)")

    report = replay_all(receipts, schedules)
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True),
                             encoding="utf-8")
        print(f"full report → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
