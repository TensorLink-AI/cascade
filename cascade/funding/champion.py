"""Champion publication: the ONE moment private code goes public.

The sn100/PRISM ``top-model/`` pattern applied to cascade (DEC-CA-0036):
direct submissions live privately in the operator's
:class:`~cascade.funding.store.SubmissionStore`; the throne is what publishes.
``champions/<digest>.zip`` plus a ``champions/index.json`` land public-read on
the manifest store, and ``cascade fetch king`` reads them
anonymously — "beat the visible best" survives, but only ever for the best.

WHEN a king's code reveals is `[round] champion_publish` policy:

* ``"crown"``    — the moment a vault-ref king first appears (public reign);
* ``"delay"``    — after ``champion_publish_delay_rounds`` rounds of reign
                   (a head start, then the open-study contract resumes);
* ``"dethrone"`` — only when the reign ENDS (the code that beat it is judged
                   against a published predecessor, never a live secret).

Under every policy a dethroned vault king publishes at the hand-off if it
has not already — losing the throne is the reveal of last resort, so history
is always fully re-derivable; only the LIVE king's privacy varies. Losers
never publish under any policy: that is the point.

The publisher is trainer-side policy (consensus-inert): it keys off the
resolved king at each round entry and keeps its reign counter in a small
work_root state file. Publication is idempotent by digest and best-effort —
a storage hiccup must never disturb the round; the next round retries.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from .store import SubmissionStore, champion_zip_key, parse_vault_ref

__all__ = ["CHAMPION_INDEX_KEY", "LANE_B_INDEX_KEY", "ChampionPublisher",
           "index_sha256", "lane_b_wins", "rows_slot_id", "should_publish",
           "verify_lane_b_index"]

log = logging.getLogger("cascade.funding.champion")

CHAMPION_INDEX_KEY = "champions/index.json"
# Lane B slot winners (owner 2026-11-27): a lane B leg whose experts were
# ACCEPTED into the king's init publishes its generator under the SAME policy,
# ZIP keys and fetch path as a lane A king; only the index is its own, so the
# lane A index (and ``cascade fetch king``) never sees a lane B entry.
LANE_B_INDEX_KEY = "champions/lane_b/index.json"
LANE_B_INDEX_SCHEMA = "cascade.champions.lane_b/1"

POLICIES = ("off", "crown", "delay", "dethrone")


def should_publish(policy: str, *, reign_rounds: int, delay_rounds: int) -> bool:
    """Whether a LIVE vault-ref king publishes now (dethrone handles itself)."""
    if policy == "crown":
        return True
    if policy == "delay":
        return reign_rounds >= max(0, delay_rounds)
    return False              # "dethrone" (and "off") never publish a live reign


class ChampionPublisher:
    """Tracks the reign across rounds and publishes per policy."""

    def __init__(
        self,
        submission_store: SubmissionStore,
        public_store: object,             # put_bytes/put_text/get_text (S3Store-shaped)
        *,
        policy: str = "dethrone",
        delay_rounds: int = 2,
        state_path: Path | str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"champion_publish={policy!r} invalid; expected {POLICIES}")
        self.submissions = submission_store
        self.public = public_store
        self.policy = policy
        self.delay_rounds = int(delay_rounds)
        self.state_path = Path(state_path)
        self.clock = clock

    # ── reign state (work_root-local, atomic) ────────────────────────────────

    def _load_state(self) -> dict:
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except ValueError:
            return {}

    def _save_state(self, state: dict) -> None:
        import os

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    # ── the round hook ───────────────────────────────────────────────────────

    def note_king(self, king_hotkey: str, king_ref: str, round_id: str) -> list[str]:
        """Called once per round with the resolved king; returns published digests.

        A king change publishes the PREVIOUS vault king if it never published
        (the reveal of last resort, on every policy); the live king then
        publishes per :func:`should_publish`. ``"off"`` does nothing at all —
        not even state tracking, so flipping the policy later starts a fresh
        reign count rather than acting on stale history.
        """
        if self.policy == "off":
            return []
        state = self._load_state()
        published: list[str] = []
        backlog: list[dict] = list(state.get("unpublished", []))
        king_changed = (state.get("king_hotkey") != king_hotkey
                        or state.get("king_ref") != king_ref)
        # Per-round idempotency: the round path can call this more than once
        # for one round_id (a mid-round retry). Only the FIRST call for a
        # round advances the reign counter, or a double-count would fire the
        # "delay" reveal a round early and expose a live king's private code
        # (review 2026-08-29). A king change is always processed (it may carry
        # a new hand-off to publish); publication itself is idempotent.
        same_round = (not king_changed
                      and str(state.get("last_round", "")) == str(round_id))
        if king_changed:
            prev_ref = state.get("king_ref", "")
            prev_digest = parse_vault_ref(prev_ref) if prev_ref else None
            # The reveal of last resort must SURVIVE a failed publish: a
            # deposed king lands on a persistent backlog, not a one-shot
            # attempt — one bucket 500 at hand-off must not erase a reign's
            # audit trail forever (review 2026-08-29).
            unrevealed = prev_digest and not state.get("published", False)
            if unrevealed and not any(b.get("digest") == prev_digest for b in backlog):
                backlog.append({"digest": prev_digest,
                                "hotkey": str(state.get("king_hotkey", ""))})
            # the lane B slot state shares this file and outlives a lane A reign
            state = {"king_hotkey": king_hotkey, "king_ref": king_ref,
                     "reign_rounds": 0, "published": False,
                     **({"lane_b": state["lane_b"]} if "lane_b" in state else {})}
        elif not same_round:
            state["reign_rounds"] = int(state.get("reign_rounds", 0)) + 1
        state["last_round"] = str(round_id)
        still_unpublished: list[dict] = []
        for item in backlog:
            if self._publish(str(item["digest"]), str(item.get("hotkey", "")),
                             round_id, reason="dethroned"):
                published.append(str(item["digest"]))
            else:
                still_unpublished.append(item)
        state["unpublished"] = still_unpublished
        digest = parse_vault_ref(king_ref)
        if digest is None:
            # A Hub-ref king is already public by construction.
            state["published"] = True
        else:
            due = not state.get("published", False) and should_publish(
                self.policy, reign_rounds=int(state["reign_rounds"]),
                delay_rounds=self.delay_rounds)
            if due and self._publish(digest, king_hotkey, round_id, reason="reigning"):
                state["published"] = True
                published.append(digest)
        self._save_state(state)
        return published

    # ── publication ──────────────────────────────────────────────────────────

    def _upload_zip(self, digest: str) -> str | None:
        """Push the vault ZIP of ``digest`` public-read; its key, or ``None``
        on any failure (logged; the caller retries next round).

        The ACL fallback mirrors ``hippius.publish_public_json`` (some
        backends reject canned ACLs).
        """
        from ..shared.hippius import StorageError

        if self.submissions is None:
            log.error("champion publish: no submission vault configured for %s", digest)
            return None
        try:
            data = self.submissions.zip_bytes(digest)
        except StorageError as e:
            log.error("champion publish: vault ZIP for %s unavailable: %s", digest, e)
            return None
        key = champion_zip_key(digest)
        try:
            try:
                self.public.put_bytes(key, data, content_type="application/zip",
                                      acl="public-read")
            except StorageError:
                self.public.put_bytes(key, data, content_type="application/zip")
        except Exception as e:  # noqa: BLE001 — never sink the round on a bucket
            log.error("champion publish of %s failed (will retry next round): %s",
                      digest, e)
            return None
        return key

    def _put_public_json(self, key: str, text: str) -> None:
        """Public-read JSON put with the canned-ACL fallback; raises on failure."""
        from ..shared.hippius import StorageError

        try:
            self.public.put_text(key, text, content_type="application/json",
                                 acl="public-read")
        except StorageError:
            self.public.put_text(key, text, content_type="application/json")

    def _publish(self, digest: str, hotkey: str, round_id: str, *, reason: str) -> bool:
        """Idempotently push the ZIP + index entry public-read; False on failure.

        Best-effort by contract: a throne must never be blocked by a bucket.
        """
        key = self._upload_zip(digest)
        if key is None:
            return False
        from ..shared.hippius import ObjectNotFound

        try:
            index = json.loads(self.public.get_text(CHAMPION_INDEX_KEY))
        except ObjectNotFound:
            index = {"champions": []}          # genuine first publish
        except Exception as e:  # noqa: BLE001 — a TRANSIENT read failure
            # A bucket hiccup reading the existing index must NOT be treated as
            # "first publish" — rewriting it from empty would erase every prior
            # champion from the audit trail. Report not-published so the next
            # round retries the whole step (the ZIP re-upload is idempotent),
            # rather than overwriting a list we could not read (review
            # 2026-08-29).
            log.error("champion index read for %s failed (retrying next round, "
                      "not overwriting): %s", digest, e)
            return False
        entry = {"digest": digest, "hotkey": hotkey, "zip_key": key,
                 "published_round": str(round_id), "reason": reason,
                 "published_at": self.clock()}
        champions = [c for c in index.get("champions", [])
                     if c.get("digest") != digest]
        champions.append(entry)
        index = {"champions": champions, "latest": entry}
        text = json.dumps(index, indent=2, sort_keys=True)
        try:
            self._put_public_json(CHAMPION_INDEX_KEY, text)
        except Exception as e:  # noqa: BLE001 — index is what readers consume
            # The ZIP is public but the index anonymous readers/dashboards use
            # is stale. Report NOT-published so the next round retries the
            # whole step (the ZIP re-upload is idempotent) — returning True
            # here would mark the reign done and never re-write the index
            # (review 2026-08-29).
            log.error("champion index update for %s failed (zip is up; retrying "
                      "next round): %s", digest, e)
            return False
        log.info("champion published: %s (%s, %s)", digest, hotkey, reason)
        return True

    # ── lane B slot winners (owner 2026-11-27) ───────────────────────────────

    def note_lane_b(self, wins: list[dict], round_id: str) -> list[str]:
        """Called once per round with every ACCEPTED lane B slot win
        (:func:`lane_b_wins`); returns the ``target|gen_ref`` keys published.

        The lane A policy, per slot: a slot's holder is a "reign" — a new
        accepted holder on a target starts its counter at 0, every later
        round advances it once (per-round idempotent), and it publishes per
        :func:`should_publish`. A displaced holder that never published lands
        on a persistent backlog and publishes at the hand-off (the reveal of
        last resort, every policy). A leg never accepted into the king — a
        refused, unmerged or merely qualified one — is never in ``wins`` and
        never publishes. ``"off"`` does nothing, not even state tracking.

        After publication the signed-by-hash ``champions/lane_b/index.json``
        is (re)written whenever its body changed or its last write failed.
        """
        if self.policy == "off":
            return []
        full = self._load_state()
        lb = dict(full.get("lane_b") or {})
        holders: dict = dict(lb.get("holders") or {})
        composed: list = list(lb.get("composed") or [])
        backlog: list = list(lb.get("unpublished") or [])
        displaced: list = list(lb.get("displaced") or [])
        same_round = str(lb.get("last_round", "")) == str(round_id)
        seen = {_win_key(c) for c in composed}
        fresh: set[str] = set()
        for w in sorted(wins, key=_win_order):
            k = _win_key(w)
            if k not in seen:
                composed.append(dict(w))
                seen.add(k)
            target = str(w["target"])
            cur = holders.get(target)
            if cur is not None and _win_key(cur) == k:
                continue
            if cur is not None and _win_order(cur) > _win_order(w):
                continue                         # an older merge never retakes a slot
            if cur is not None:
                old = {**cur, "displaced_round": str(round_id)}
                if not cur.get("published"):
                    backlog.append(old)
                else:
                    displaced.append(old)
            holders[target] = {**w, "reign_rounds": 0, "published": False, "zip_key": ""}
            fresh.add(target)
        if not same_round:
            for target, h in holders.items():
                if target not in fresh:
                    h["reign_rounds"] = int(h.get("reign_rounds", 0)) + 1
        published: list[str] = []
        still: list = []
        for item in backlog:
            key = self._publish_lane_b_code(item)
            if key is None:
                still.append(item)
                continue
            displaced.append({**item, "published": True, "zip_key": key,
                              "published_round": str(round_id), "reason": "displaced"})
            published.append(_win_key(item))
        for target in sorted(holders):
            h = holders[target]
            if h.get("published") or not should_publish(
                    self.policy, reign_rounds=int(h.get("reign_rounds", 0)),
                    delay_rounds=self.delay_rounds):
                continue
            key = self._publish_lane_b_code(h)
            if key is None:
                continue
            h.update(published=True, zip_key=key, published_round=str(round_id),
                     reason="holding")
            published.append(_win_key(h))
        index = _lane_b_index(holders, composed, displaced)
        digest = index["sha256"]
        if lb.get("index_sha256") != digest:
            try:
                self._put_public_json(LANE_B_INDEX_KEY,
                                      json.dumps(index, indent=2, sort_keys=True))
                lb["index_sha256"] = digest
            except Exception as e:  # noqa: BLE001 — retried next round
                log.error("lane B champion index update failed (retrying next round): %s", e)
        lb.update(holders=holders, composed=composed, unpublished=still,
                  displaced=displaced, last_round=str(round_id))
        full["lane_b"] = lb
        self._save_state(full)
        for k in published:
            log.info("lane B champion published: %s", k)
        return published

    def _publish_lane_b_code(self, win: dict) -> str | None:
        """Make a slot winner's code public: a vault ref's ZIP goes up at the
        lane A key (``champions/<digest>.zip``, so ``cascade fetch <ref>``
        reads it unchanged); a Hub ref is public by construction (``""``).
        ``None`` = not published (retry next round)."""
        digest = parse_vault_ref(str(win.get("gen_ref", "")))
        if digest is None:
            return ""
        return self._upload_zip(digest)


# ── lane B: the accepted wins and the index ─────────────────────────────────


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def index_sha256(index: dict) -> str:
    """sha256 over the canonical JSON of ``index`` minus its ``sha256`` field."""
    import hashlib

    body = {k: v for k, v in index.items() if k != "sha256"}
    return hashlib.sha256(_canon(body).encode("utf-8")).hexdigest()


def verify_lane_b_index(index: dict) -> bool:
    """The index is well-formed and its ``sha256`` matches its body."""
    return (isinstance(index, dict) and index.get("schema") == LANE_B_INDEX_SCHEMA
            and isinstance(index.get("slots"), dict)
            and index.get("sha256") == index_sha256(index))


def rows_slot_id(rows: dict) -> str:
    """A stable name for a set of expert rows (``layer -> [experts]``):
    ``rows-<12 hex>`` of their canonical JSON — the ``<slot>`` path segment."""
    import hashlib

    canon = {str(k): sorted(int(x) for x in v) for k, v in (rows or {}).items()}
    return "rows-" + hashlib.sha256(_canon(canon).encode("utf-8")).hexdigest()[:12]


def _win_key(w: dict) -> str:
    return f"{w.get('target', '')}|{w.get('rebased_pointer', '')}"


def _win_order(w: dict) -> tuple:
    return (int(w.get("merge_block", 0) or 0), int(w.get("era", 0) or 0),
            str(w.get("round_id", "")), str(w.get("target", "")))


def lane_b_wins(merged_inits: dict) -> list[dict]:
    """Every lane B slot win the trainer MERGED into a king init, merge order.

    ``merged_inits`` is :class:`~cascade.trainer.lane_b_driver.LaneBDriverState`'s
    ``era -> {pointer, base, merges}``: ``pointer`` is the king checkpoint the
    merges were composed into, ``base`` the lane A king checkpoint at the
    time. A merge's winning targets are its acceptance ``wins`` (under
    acceptance, the only path that reaches here today), else its
    ``qualified_wins``; a merge with neither carries no miner identity and is
    skipped. Deferred inits (``pointer == base``, no merges) yield nothing.
    """
    out: list[dict] = []
    for era, rec in (merged_inits or {}).items():
        if not isinstance(rec, dict):
            continue
        merged, base = str(rec.get("pointer") or ""), str(rec.get("base") or "")
        for m in rec.get("merges") or []:
            acc = m.get("acceptance") or {}
            wins = acc.get("wins") or m.get("qualified_wins") or []
            rows = {str(k): sorted(int(x) for x in v)
                    for k, v in (m.get("selection") or {}).items()}
            for w in wins:
                target = str(w.get("domain") or "")
                if not target or not merged:
                    continue
                gen_ref = str(w.get("gen_ref") or "")
                out.append({
                    "target": target,
                    "slot": rows_slot_id(rows),
                    "rows": rows,
                    "hotkey": str(w.get("hotkey") or ""),
                    "uid": int(w.get("uid", -1)),
                    "gen_ref": gen_ref,
                    "generator_digest": gen_ref.split("@", 1)[1] if "@" in gen_ref else "",
                    "commit_block": int(w.get("commit_block", 0) or 0),
                    "round_id": str(m.get("round_id") or w.get("round_id") or ""),
                    "merge_block": int(acc.get("block") or w.get("merge_block") or 0),
                    "era": int(era) if str(era).lstrip("-").isdigit() else 0,
                    "merged_into": merged,
                    "lane_a_king_pointer": base,
                    "rebased_pointer": str(m.get("rebased_pointer") or ""),
                    "rebased_onto": str(m.get("king_pointer") or ""),
                    "targets": list(w.get("targets") or [target]),
                })
    return sorted(out, key=_win_order)


_PUBLIC_FIELDS = ("target", "slot", "rows", "hotkey", "uid", "gen_ref", "generator_digest",
                  "commit_block", "round_id", "merge_block", "era", "merged_into",
                  "lane_a_king_pointer", "rebased_pointer", "rebased_onto", "targets",
                  "published", "zip_key", "published_round", "reason", "displaced_round")


def _public(w: dict) -> dict:
    return {k: w[k] for k in _PUBLIC_FIELDS if k in w}


def _lane_b_index(holders: dict, composed: list, displaced: list) -> dict:
    """The public lane B index: the current holder of every slot, every
    generator composed into the latest king checkpoint (every accepted merge
    is in its lineage), and the displaced holders. Hashed like lane A's
    record: each entry carries its generator's content digest; ``sha256``
    covers the whole body (:func:`verify_lane_b_index`)."""
    latest = max(composed, key=_win_order) if composed else {}
    pub = {_win_key(h): h for h in [*holders.values(), *displaced] if h.get("published")}
    index = {
        "schema": LANE_B_INDEX_SCHEMA,
        "king_checkpoint": str(latest.get("merged_into", "")),
        "slots": {t: _public(h) for t, h in sorted(holders.items())},
        "composed": [_public({**c, "published": _win_key(c) in pub,
                              "zip_key": pub.get(_win_key(c), {}).get("zip_key", "")})
                     for c in sorted(composed, key=_win_order)],
        "archive": [_public(d) for d in sorted(displaced, key=_win_order) if d.get("published")],
    }
    index["sha256"] = index_sha256(index)
    return index
