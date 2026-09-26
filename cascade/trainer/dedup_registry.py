"""Persistent content-dedup registry for rolling intake (DEC-CA-0043).

The boundary-synchronous trainer screens a whole field at once
(:meth:`TrainerRunner._screen_duplicate_entrants`): every entrant's fetched
tree is fingerprinted and compared pairwise against the king and every
earlier-committed entrant of that round. With rolling intake there is no
field to compare against — entrants arrive one at a time, hours apart — so
the exact-identity digests (DEC-CA-0008's tree / token / rename tiers) are
kept in a registry across the queue, in-flight legs, the era king
(``register_king``) and every published champion (``register_champions``). Admission compares the newcomer's digests against the
registry; the earliest COMMIT keeps the entry (never the UID, which
recycles). Fail-open on infrastructure faults, exactly like the screen.
"""
from __future__ import annotations

import contextlib
import json
import logging
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)

TIERS = (("tree_sha256", "tree_identical"), ("token_sha256", "token_identical"),
         ("masked_sha256", "rename_identical"))
PRIVATE_COPY_TIER = "private_copy"
EMBEDDED_TIERS = ("embedded_token_identical", "embedded_rename_identical")
CHAMPION_INDEX_TTL_SECONDS = 300.0


def _is_vault_ref(ref: str) -> bool:
    from ..funding.store import parse_vault_ref

    return parse_vault_ref(ref) is not None


class DedupRegistry:
    def __init__(self, runner, path: Path, *, mode: str = "shadow") -> None:
        self.r = runner
        self.path = Path(path)
        self.mode = mode
        self.entries: dict[str, dict] = self._load()
        self._backfilled = False

    # ── private-copy tier config ─────────────────────────────────────────────
    def _pc_mode(self) -> str:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return str(getattr(rnd, "private_copy_mode", "off") or "off").lower()

    def _pc_threshold(self) -> float:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return float(getattr(rnd, "private_copy_min_containment", 0.8))

    def _pc_min_tokens(self) -> int:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return int(getattr(rnd, "private_copy_min_tokens", 2_000))

    def _emb_mode(self) -> str:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return str(getattr(rnd, "dedup_embedded_mode", "off") or "off").lower()

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
            out = {"tree_sha256": fp.tree_sha256, "token_sha256": fp.token_sha256,
                   "masked_sha256": fp.masked_sha256,
                   "py_sha256": fp.py_sha256, "py_masked_sha256": fp.py_masked_sha256,
                   "components": [{"name": c.name, "token_sha256": c.token_sha256,
                                   "masked_sha256": c.masked_sha256, "n_tokens": c.n_tokens}
                                  for c in fp.components]}
            if self._pc_mode() != "off":
                from ..interface.dedup import module_sketches

                out["modules"] = module_sketches(
                    d, min_tokens=self._pc_min_tokens(), max_text_mb=rnd.dedup_max_text_mb)
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("dedup registry: %s unfetchable/unfingerprintable (%s)", ref[:60], e)
            return None
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ── registry ─────────────────────────────────────────────────────────────

    def register(self, ref: str, *, hotkey: str, commit_block: int, kind: str,
                 digests: dict | None = None, coldkey: str | None = None) -> None:
        if ref in self.entries:
            cur = self.entries[ref]
            if commit_block and (not cur.get("commit_block") or commit_block < cur["commit_block"]):
                cur["commit_block"], cur["hotkey"] = int(commit_block), hotkey
                if coldkey:
                    cur["coldkey"] = coldkey
            cur["kind"] = kind
            self._save()
            return
        if digests is None:
            digests = self.fingerprint(ref)
            if digests is None:
                return
        self.entries[ref] = {**digests, "hotkey": hotkey, "commit_block": int(commit_block),
                             "kind": kind, "private": _is_vault_ref(ref),
                             **({"coldkey": coldkey} if coldkey else {})}
        self._save()

    # ── private-copy tier ────────────────────────────────────────────────────
    def _published(self) -> dict[str, str]:
        """``{vault digest: hotkey}`` of every champion the publish policy has
        made public. Re-read every ``CHAMPION_INDEX_TTL_SECONDS`` (a crown
        publishes mid-process). Unknown (no store / read failure) ⇒ the last
        good read, else empty: a copy of a published king could then read as
        private_copy, which only matters in enforce mode and is logged with
        the digest either way."""
        cached = self.__dict__.get("_published_cache")
        stamp = self.__dict__.get("_published_stamp", 0.0)
        if cached is not None and time.monotonic() - stamp < CHAMPION_INDEX_TTL_SECONDS:
            return cached
        try:
            from ..funding.champion import CHAMPION_INDEX_KEY
            store = self.r.manifest_store()
            index = json.loads(store.get_text(CHAMPION_INDEX_KEY))
            cached = {str(c.get("digest", "")): str(c.get("hotkey", "") or "")
                      for c in index.get("champions", []) if c.get("digest")}
        except Exception as e:  # noqa: BLE001 — best-effort
            log.debug("dedup registry: champion index unavailable (%s)", e)
            cached = cached if cached is not None else {}
        self.__dict__["_published_cache"] = cached
        self.__dict__["_published_stamp"] = time.monotonic()
        return cached

    def _published_digests(self) -> set[str]:
        return set(self._published())

    # ── the king and the published champions are entries too ────────────────

    def register_king(self, era, history: list) -> None:
        """Put the era king's generator on the registry (``kind="king"``) so
        newcomers are screened against it — the boundary screen compares
        against the king explicitly; the registry only ever saw queue
        admissions. Idempotent; fail-open."""
        ref = str(getattr(era, "king_ref", "") or "")
        hotkey = str(getattr(era, "king_hotkey", "") or "")
        if not ref or not hotkey or ref in self.entries:
            return
        try:
            self.register(ref, hotkey=hotkey, kind="king",
                          commit_block=_commit_block(history, hotkey, ref),
                          coldkey=_coldkey_of(history, hotkey))
        except Exception as e:  # noqa: BLE001
            log.warning("dedup registry: could not register king %s (%s)", hotkey[:12], e)

    def register_champions(self) -> None:
        """Register every published champion (``kind="champion"``, commit
        block 0 = earlier than any newcomer). Their vault ZIPs are on this box.
        Once per index read; fail-open per entry."""
        from ..funding.store import vault_ref

        for digest, hotkey in self._published().items():
            try:
                ref = vault_ref(digest)
            except Exception:  # noqa: BLE001 — not a vault digest
                continue
            if ref in self.entries or not hotkey:
                continue
            try:
                self.register(ref, hotkey=hotkey, commit_block=0, kind="champion")
            except Exception as e:  # noqa: BLE001
                log.warning("dedup registry: could not register champion %s (%s)",
                            digest[:12], e)

    def _is_private(self, ref: str, rec: dict) -> bool:
        if not rec.get("private", _is_vault_ref(ref)):
            return False
        from ..funding.store import parse_vault_ref

        digest = parse_vault_ref(ref) or ""
        return digest not in self._published_digests()

    def _public_union(self) -> frozenset[int]:
        """Every shingle of every PUBLIC registered module (public Hub trees,
        published champions): the text nobody owns. A private module's
        "private part" is what is left after removing it — a vault
        submission that concatenates a public lineage into one file must not
        come to own that lineage."""
        u: set[int] = set()
        for ref, rec in self.entries.items():
            if not self._is_private(ref, rec):
                for m in rec.get("modules", []):
                    u.update(m["sketch"])
        return frozenset(u)

    def backfill_modules(self) -> None:
        """Re-fingerprint registered VAULT entries that predate a tier's
        fields (``modules`` for private_copy, ``components`` /
        ``py_masked_sha256`` for the embedded tiers) — their ZIPs are on this
        box, so it is local and cheap. Once per process; every entry fails
        open on its own."""
        if self._backfilled:
            return
        self._backfilled = True
        need = []
        if self._pc_mode() != "off":
            need.append("modules")
        if self._emb_mode() != "off":
            need += ["components", "py_masked_sha256"]
        if not need:
            return
        todo = [ref for ref, rec in self.entries.items()
                if _is_vault_ref(ref) and any(k not in rec for k in need)]
        for ref in todo:
            digests = self.fingerprint(ref)
            if digests is None:
                continue
            for k in ("modules", "components", "py_sha256", "py_masked_sha256"):
                if k in digests:
                    self.entries[ref][k] = digests[k]
            self.entries[ref].setdefault("private", True)
        if todo:
            self._save()
            log.info("dedup registry: re-fingerprinted %d pre-existing entries for %s",
                     len(todo), "/".join(need))

    def match_private_copy(self, modules: list[dict], *, exclude_hotkey: str,
                           exclude_coldkey: str | None = None) -> tuple[str, str, int, str] | None:
        """``(hotkey, tier, commit_block, detail)`` of the earliest-committed
        PRIVATE registered entry (other hotkeys only — and other coldkeys,
        when both are known: an operator re-submitting their own private
        code under a fresh hotkey after a burn is not copying anyone) one of
        whose modules is contained in one of ``modules`` at >= the threshold
        — skipping private modules that are themselves public material (a
        public tree's module carried inside a vault submission is not that
        submitter's to own)."""
        from ..interface.dedup import MIN_PRIVATE_FRACTION, MIN_PRIVATE_SHINGLES, sketch_containment

        if not modules:
            return None
        thr = self._pc_threshold()
        public = self._public_union()
        mine_sets = [(m, frozenset(m["sketch"])) for m in modules]
        hits: list[tuple[int, str, str]] = []      # (commit_block, hotkey, detail)
        for ref, rec in self.entries.items():
            if rec.get("hotkey") == exclude_hotkey or not self._is_private(ref, rec):
                continue
            if exclude_coldkey and rec.get("coldkey") == exclude_coldkey:
                continue
            for theirs in rec.get("modules", []):
                their_set = frozenset(theirs["sketch"])
                private_part = their_set - public
                # mostly public text (a wrapped public lineage) is nobody's to own
                if len(private_part) < max(MIN_PRIVATE_SHINGLES,
                                           MIN_PRIVATE_FRACTION * len(their_set)):
                    continue
                for mine, mine_set in mine_sets:
                    c = sketch_containment(mine_set, private_part)
                    if c >= thr:
                        hits.append((int(rec.get("commit_block") or 0), rec["hotkey"],
                                     f"{mine['name']} carries {c:.0%} of the private text of "
                                     f"{theirs['name']} ({theirs['tokens']} masked tokens, "
                                     f"{ref[:40]}, {rec['hotkey'][:12]})"))
                        break
        if not hits:
            return None
        hits.sort()
        owners = list(dict.fromkeys(h[1] for h in hits))
        detail = hits[0][2]
        if len(owners) > 1:
            detail += f"; also in {len(owners) - 1} other earlier private entr" + (
                "y" if len(owners) == 2 else "ies") + " (" + ", ".join(
                o[:12] for o in owners[1:4]) + ("…" if len(owners) > 4 else "") + ")"
        return hits[0][1], PRIVATE_COPY_TIER, hits[0][0], detail

    def match_embedded(self, digests: dict, *, exclude_hotkey: str) -> tuple[str, str, int, str] | None:
        """``(hotkey, tier, commit_block, detail)`` of the earliest-committed
        registered entry (other hotkeys) whose packed component equals the
        newcomer's whole ``.py`` stream or one of its components — or whose
        whole stream equals one of the newcomer's components — exactly
        (``embedded_token_identical``) or after identifier masking
        (``embedded_rename_identical``). Exact digests only (DEC-CA-0008)."""
        mine = digests.get("components") or []
        my_py, my_pym = digests.get("py_sha256") or "", digests.get("py_masked_sha256") or ""
        if not mine and not my_py:
            return None
        my_tok = {c["token_sha256"]: c["name"] for c in mine}
        my_msk = {c["masked_sha256"]: c["name"] for c in mine}
        best = None
        for ref, rec in self.entries.items():
            if rec.get("hotkey") == exclude_hotkey:
                continue
            theirs = rec.get("components") or []
            their_py, their_pym = rec.get("py_sha256") or "", rec.get("py_masked_sha256") or ""
            hit = None
            if their_py and their_py in my_tok:
                hit = (EMBEDDED_TIERS[0], f"{my_tok[their_py]} is their whole code")
            elif my_py and any(c["token_sha256"] == my_py for c in theirs):
                hit = (EMBEDDED_TIERS[0], "their packed source is this whole code")
            elif any(c["token_sha256"] in my_tok for c in theirs):
                hit = (EMBEDDED_TIERS[0], "same packed source in both")
            elif their_pym and their_pym in my_msk:
                hit = (EMBEDDED_TIERS[1], f"{my_msk[their_pym]} is their whole code, renamed")
            elif my_pym and any(c["masked_sha256"] == my_pym for c in theirs):
                hit = (EMBEDDED_TIERS[1], "their packed source is this whole code, renamed")
            elif any(c["masked_sha256"] in my_msk for c in theirs):
                hit = (EMBEDDED_TIERS[1], "same packed source in both, renamed")
            if hit is None:
                continue
            cand = (int(rec.get("commit_block") or 0), rec["hotkey"], hit[0],
                    f"{hit[1]} ({ref[:40]}, {rec['hotkey'][:12]})")
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            return None
        return best[1], best[2], best[0], best[3]

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
        # The king and the published champions are rivals too (fail-open).
        self.register_king(era, history)
        self.register_champions()
        digests = self.fingerprint(gen.ref)
        if digests is None:
            return None
        my_block = _commit_block(history, gen.hotkey, gen.ref) or int(getattr(gen, "reveal_block", 0) or 0)
        coldkey = _coldkey_of(history, gen.hotkey) or (getattr(gen, "coldkey", None) or None)
        hit = self.match(digests, exclude_hotkey=gen.hotkey)
        if hit is not None and (hit[2] == 0 or my_block == 0 or hit[2] <= my_block):
            return hit[0], hit[1], self.mode == "enforce"
        emb_mode = self._emb_mode()
        if emb_mode in ("shadow", "enforce"):
            self.backfill_modules()
            emb = self.match_embedded(digests, exclude_hotkey=gen.hotkey)
            if emb is not None and (emb[2] == 0 or my_block == 0 or emb[2] <= my_block):
                log.warning("dedup registry: %s %s of %s — %s [%s]",
                            gen.hotkey[:12], emb[1], emb[0][:12], emb[3], emb_mode)
                if emb_mode != "enforce":
                    self.register(gen.ref, hotkey=gen.hotkey, commit_block=my_block,
                                  kind="queue", digests=digests, coldkey=coldkey)
                return emb[0], emb[1], emb_mode == "enforce"
        pc_mode = self._pc_mode()
        if pc_mode in ("shadow", "enforce"):
            self.backfill_modules()
            pc = self.match_private_copy(digests.get("modules", []), exclude_hotkey=gen.hotkey,
                                         exclude_coldkey=coldkey)
            if pc is not None and (pc[2] == 0 or my_block == 0 or pc[2] <= my_block):
                log.warning("dedup registry: %s private_copy of %s — %s [%s]",
                            gen.hotkey[:12], pc[0][:12], pc[3], pc_mode)
                # A flagged-but-admitted entry (shadow) is still registered:
                # what it carries is on record for the next comer.
                if pc_mode != "enforce":
                    self.register(gen.ref, hotkey=gen.hotkey, commit_block=my_block,
                                  kind="queue", digests=digests, coldkey=coldkey)
                return pc[0], pc[1], pc_mode == "enforce"
        self.register(gen.ref, hotkey=gen.hotkey, commit_block=my_block, kind="queue",
                      digests=digests, coldkey=coldkey)
        return None


def _coldkey_of(history: list, hotkey: str) -> str | None:
    """The coldkey the reveal history records for ``hotkey`` (None when unknown)."""
    for c in history:
        if c.hotkey == hotkey and getattr(c, "coldkey", None):
            return str(c.coldkey)
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
