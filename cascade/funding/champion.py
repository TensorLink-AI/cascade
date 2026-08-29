"""Champion publication: the ONE moment private code goes public.

The sn100/PRISM ``top-model/`` pattern applied to cascade (DEC-CA-0036):
direct submissions live privately in the operator's
:class:`~cascade.funding.store.SubmissionStore`; the throne is what publishes.
``champions/<digest>.zip`` plus a ``champions/index.json`` land public-read on
the same store the heat standings ride, and ``cascade fetch king`` reads them
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

__all__ = ["CHAMPION_INDEX_KEY", "ChampionPublisher", "should_publish"]

log = logging.getLogger("cascade.funding.champion")

CHAMPION_INDEX_KEY = "champions/index.json"

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
            state = {"king_hotkey": king_hotkey, "king_ref": king_ref,
                     "reign_rounds": 0, "published": False}
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

    def _publish(self, digest: str, hotkey: str, round_id: str, *, reason: str) -> bool:
        """Idempotently push the ZIP + index entry public-read; False on failure.

        Best-effort by contract: a throne must never be blocked by a bucket.
        The ACL fallback mirrors ``heat_status._publish_public_json`` (some
        backends reject canned ACLs).
        """
        from ..shared.hippius import StorageError

        try:
            data = self.submissions.zip_bytes(digest)
        except StorageError as e:
            log.error("champion publish: vault ZIP for %s unavailable: %s", digest, e)
            return False
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
            try:
                self.public.put_text(CHAMPION_INDEX_KEY, text,
                                     content_type="application/json", acl="public-read")
            except StorageError:
                self.public.put_text(CHAMPION_INDEX_KEY, text,
                                     content_type="application/json")
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
