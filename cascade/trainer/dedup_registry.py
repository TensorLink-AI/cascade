"""Persistent content-dedup registry for rolling intake (DEC-CA-0043).

The boundary-synchronous trainer screens a whole field at once
(:meth:`TrainerRunner._screen_duplicate_entrants`): every entrant's fetched
tree is fingerprinted and compared pairwise against the king and every
earlier-committed entrant of that round. With rolling intake there is no
field to compare against — entrants arrive one at a time, hours apart — so
the exact-identity digests (DEC-CA-0008's tree / token / rename tiers) are
kept in a registry across the queue, in-flight legs, the era king and every
published entry. Admission compares the newcomer's digests against the
registry; the earliest COMMIT keeps the entry (never the UID, which
recycles). Fail-open on infrastructure faults, exactly like the screen.
"""
from __future__ import annotations

import contextlib
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

TIERS = (("tree_sha256", "tree_identical"), ("token_sha256", "token_identical"),
         ("masked_sha256", "rename_identical"))


class DedupRegistry:
    def __init__(self, runner, path: Path, *, mode: str = "shadow") -> None:
        self.r = runner
        self.path = Path(path)
        self.mode = mode
        self.entries: dict[str, dict] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return {str(k): dict(v) for k, v in (raw.get("refs") or {}).items()}
        except FileNotFoundError:
            return {}
        except Exception as e:  # noqa: BLE001
            log.warning("dedup registry %s unreadable (%s); starting empty", self.path, e)
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"refs": self.entries}, sort_keys=True, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ── fingerprints ─────────────────────────────────────────────────────────

    def fingerprint(self, ref: str) -> dict | None:
        """Fetch + fingerprint ``ref`` (digests only — the token prefix is not
        kept). ``None`` when the fetch or fingerprint fails (fail open)."""
        from ..interface.dedup import fingerprint_dir
        from ..shared.hippius import fetch_from_hub

        rnd = self.r.cfg.round
        root = Path(self.r.work_root) / "_dedup_registry" / ref.replace("/", "_").replace(":", "_")[:120]
        try:
            d = Path(fetch_from_hub(ref, root, hub=self.r.hub()))
            with contextlib.suppress(Exception):   # archive is best-effort
                self.r._archive_generator_tree(ref, d)
            fp = fingerprint_dir(d, max_tokens=rnd.dedup_max_tokens,
                                 max_text_mb=rnd.dedup_max_text_mb)
            return {"tree_sha256": fp.tree_sha256, "token_sha256": fp.token_sha256,
                    "masked_sha256": fp.masked_sha256}
        except Exception as e:  # noqa: BLE001
            log.warning("dedup registry: %s unfetchable/unfingerprintable (%s)", ref[:60], e)
            return None
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ── registry ─────────────────────────────────────────────────────────────

    def register(self, ref: str, *, hotkey: str, commit_block: int, kind: str,
                 digests: dict | None = None) -> None:
        if ref in self.entries:
            cur = self.entries[ref]
            if commit_block and (not cur.get("commit_block") or commit_block < cur["commit_block"]):
                cur["commit_block"], cur["hotkey"] = int(commit_block), hotkey
            cur["kind"] = kind
            self._save()
            return
        if digests is None:
            digests = self.fingerprint(ref)
            if digests is None:
                return
        self.entries[ref] = {**digests, "hotkey": hotkey, "commit_block": int(commit_block),
                             "kind": kind}
        self._save()

    def match(self, digests: dict, *, exclude_hotkey: str) -> tuple[str, str, int] | None:
        """``(hotkey, tier, commit_block)`` of the earliest-committed registered
        entry sharing a digest with ``digests`` (other hotkeys only)."""
        best = None
        for rec in self.entries.values():
            if rec.get("hotkey") == exclude_hotkey:
                continue
            for key, tier in TIERS:
                if digests.get(key) and digests.get(key) == rec.get(key):
                    cand = (rec["hotkey"], tier, int(rec.get("commit_block") or 0))
                    if best is None or cand[2] < best[2]:
                        best = cand
                    break
        return best

    def admit(self, gen, era, history: list) -> tuple[str, str, bool] | None:
        """Screen a newcomer: returns ``(matched_hotkey, tier, enforce)`` when
        it duplicates an EARLIER-committed registered entry, else ``None``
        (and registers it). A later-committed duplicate of the newcomer is
        left alone here — it loses its own admission when it arrives."""
        digests = self.fingerprint(gen.ref)
        if digests is None:
            return None
        my_block = _commit_block(history, gen.hotkey, gen.ref) or int(getattr(gen, "reveal_block", 0) or 0)
        hit = self.match(digests, exclude_hotkey=gen.hotkey)
        if hit is not None and (hit[2] == 0 or my_block == 0 or hit[2] <= my_block):
            return hit[0], hit[1], self.mode == "enforce"
        self.register(gen.ref, hotkey=gen.hotkey, commit_block=my_block, kind="queue",
                      digests=digests)
        return None


def _commit_block(history: list, hotkey: str, ref: str) -> int:
    """Earliest reveal block of ``ref`` by ``hotkey`` in the reveal history
    (the chain deletes the encrypted commit at reveal, so the reveal is the
    earliest block still knowable here)."""
    from ..interface.validation import parse_commit

    best = 0
    for c in history:
        if c.hotkey != hotkey:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None or parsed.ref != ref:
            continue
        if best == 0 or int(c.commit_block) < best:
            best = int(c.commit_block)
    return best
