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
NEAR_DUP_TIER = "near_duplicate"
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

    # ── near-duplicate tier config ───────────────────────────────────────────
    def _nd_mode(self) -> str:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return str(getattr(rnd, "near_dup_mode", "off") or "off").lower()

    def _nd_threshold(self) -> float:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return float(getattr(rnd, "near_dup_threshold", 0.97))

    def _nd_min_private(self) -> int:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return int(getattr(rnd, "near_dup_min_private_shingles", 200))

    def _nd_lookback_blocks(self, block: int = 0) -> int | None:
        """The lookback window in BLOCKS at ``block``: ``near_dup_lookback_eras``
        ERAS (``epoch_blocks × era_settlements`` in force there — after the
        900-block grid switch an era is 3600 blocks, not one settlement).
        ``None`` = no limit (``near_dup_lookback_eras = 0``)."""
        cfg = getattr(self.r, "cfg", None)
        rnd = getattr(cfg, "round", None)
        eras = int(getattr(rnd, "near_dup_lookback_eras", 4))
        if eras <= 0:
            return None
        try:
            from ..shared.era import era_length_blocks

            era = int(era_length_blocks(rnd, int(block or 0)))
        except Exception:  # noqa: BLE001 — partial configs (tests): the grid alone
            era = int(getattr(rnd, "epoch_blocks", 3600) or 3600) * max(
                1, int(getattr(rnd, "era_settlements", 0) or 0))
        return eras * era

    def _nd_same_coldkey(self) -> bool:
        rnd = getattr(getattr(self.r, "cfg", None), "round", None)
        return bool(getattr(rnd, "near_dup_same_coldkey", False))

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

    def packed_source_verdict(self, ref: str):
        """Static guard in ``packed_sources="reject"`` mode over ``ref``'s tree
        (fetched to the registry's cache dir, like :meth:`fingerprint`).
        Returns a ``GuardResult``; ``None`` when the fetch fails (fail open)."""
        from ..interface.static_guard import scan_tree
        from ..shared.hippius import fetch_from_hub

        root = Path(self.r.work_root) / "_dedup_registry" / ref.replace("/", "_").replace(":", "_")[:120]
        try:
            try:
                d = Path(fetch_from_hub(ref, root, hub=self.r.hub()))
            except Exception as e:  # noqa: BLE001 — fetch failure is not the miner's fault
                log.warning("dedup registry: packed-source check could not fetch %s (%s)", ref[:40], e)
                return None
            return scan_tree(d, tuple(self.r.cfg.static_guard.blocked), packed_sources="reject")
        finally:
            # a miner-supplied tree never lingers on the orchestrator host —
            # fingerprint() re-fetches for admitted refs and cleans up after itself
            shutil.rmtree(root, ignore_errors=True)

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
            if self._nd_mode() != "off":
                from ..interface.near_dup import signature_of_dir

                self._store_nd_sig(ref, signature_of_dir(d, max_text_mb=rnd.dedup_max_text_mb))
                out["nd_sig"] = True
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("dedup registry: %s unfetchable/unfingerprintable (%s)", ref[:60], e)
            return None
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ── near-duplicate signatures (sidecar files: a signature is ~20k ints,
    #    far too big for the registry JSON that is rewritten on every save) ──

    def _nd_dir(self) -> Path:
        return Path(self.r.work_root) / "_near_dup_sigs"

    def _nd_path(self, ref: str) -> Path:
        import hashlib

        return self._nd_dir() / (hashlib.sha256(ref.encode("utf-8")).hexdigest()[:32] + ".json")

    def _store_nd_sig(self, ref: str, sig: frozenset[int]) -> None:
        p = self._nd_path(ref)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(sig)), encoding="utf-8")
        tmp.replace(p)
        self.__dict__.setdefault("_nd_cache", {})[ref] = sig

    def nd_sig(self, ref: str) -> frozenset[int] | None:
        """The cached signature of ``ref`` (None when never computed)."""
        cache = self.__dict__.setdefault("_nd_cache", {})
        if ref in cache:
            return cache[ref]
        try:
            sig = frozenset(int(x) for x in json.loads(self._nd_path(ref).read_text("utf-8")))
        except Exception:  # noqa: BLE001 — missing/torn ⇒ recompute
            return None
        cache[ref] = sig
        return sig

    def backfill_nd(self) -> None:
        """Compute the signature of every registered entry that lacks one —
        vault ZIPs are on this box, so it is local. Once per process; every
        entry fails open on its own."""
        if self.__dict__.get("_nd_backfilled") or self._nd_mode() == "off":
            return
        self.__dict__["_nd_backfilled"] = True
        todo = [ref for ref in self.entries if self.nd_sig(ref) is None]
        n = 0
        for ref in todo:
            if self.compute_nd_sig(ref) is not None:
                self.entries[ref]["nd_sig"] = True
                n += 1
        if n:
            self._save()
            log.info("dedup registry: near-duplicate signatures backfilled for %d entries", n)

    def compute_nd_sig(self, ref: str) -> frozenset[int] | None:
        """Fetch ``ref`` (vault refs resolve to the on-box ZIP) and store its
        near-duplicate signature only — no digests, no archive. Text only;
        the tree is deleted afterwards. ``None`` on any failure (fail open)."""
        from ..interface.near_dup import signature_of_dir
        from ..shared.hippius import fetch_from_hub

        root = Path(self.r.work_root) / "_dedup_registry" / ("nd_" + ref.replace("/", "_").replace(":", "_")[:117])
        try:
            d = Path(fetch_from_hub(ref, root, hub=self.r.hub()))
            sig = signature_of_dir(d, max_text_mb=self.r.cfg.round.dedup_max_text_mb)
            self._store_nd_sig(ref, sig)
            return sig
        except Exception as e:  # noqa: BLE001
            log.warning("dedup registry: near-duplicate signature for %s failed (%s)", ref[:60], e)
            return None
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _nd_public(self) -> frozenset[int]:
        """Shingles of PUBLIC material — published champions, every king the
        registry has seen, public (non-vault) Hub trees — which no newcomer is
        flagged for sharing: building on the king is the design."""
        u: set[int] = set()
        for ref, rec in self.entries.items():
            if rec.get("kind") in ("king", "champion") or not self._is_private(ref, rec):
                sig = self.nd_sig(ref)
                if sig:
                    u |= sig
        return frozenset(u)

    def match_near_dup(self, ref: str, *, hotkey: str, commit_block: int,
                       coldkey: str | None = None
                       ) -> tuple[str, float, int, str] | None:
        """``(hotkey, J, commit_block, ref)`` of the earliest-committed OTHER
        hotkey's registered queue entry whose private signature is at least
        ``near_dup_threshold`` Jaccard-similar to ``ref``'s, committed within
        the lookback window and STRICTLY earlier by ``(commit_block, hotkey)``
        (a same-block burst keeps exactly one entry). Kings and published
        champions are never rivals here. ``None`` when nothing matches or the
        signature is unavailable (fail open)."""
        from ..interface.near_dup import private_jaccard

        mine = self.nd_sig(ref)
        if mine is None:
            return None
        thr, min_priv = self._nd_threshold(), self._nd_min_private()
        public = self._nd_public()
        span = self._nd_lookback_blocks(int(commit_block or 0))
        lo = int(commit_block) - span if (commit_block and span is not None) else None
        me = (int(commit_block or 0), hotkey)
        my_ck = coldkey or (self.entries.get(ref) or {}).get("coldkey") or None
        same_ck = self._nd_same_coldkey() and bool(my_ck)
        if len(mine - public) < min_priv:
            # Mostly published lineage: the private-only comparison has
            # nothing to compare (fail open) — say so, once per ref, so a
            # silent pass is visible in the log.
            seen = self.__dict__.setdefault("_nd_lineage_logged", set())
            if ref not in seen:
                seen.add(ref)
                log.info("dedup registry: %s is mostly public lineage (%d private shingles "
                         "< %d) — cross-coldkey near-duplicate check not applicable%s",
                         hotkey[:12], len(mine - public), min_priv,
                         "; same-coldkey full check applies" if same_ck else "")
        best = None
        for their_ref, rec in self.entries.items():
            their_hk = str(rec.get("hotkey") or "")
            if their_ref == ref or their_hk == hotkey or rec.get("kind") in ("king", "champion"):
                continue
            cb = int(rec.get("commit_block") or 0)
            if commit_block and (cb == 0 or (lo is not None and cb < lo)):
                continue
            if commit_block and (cb, their_hk) >= me:
                continue
            theirs = self.nd_sig(their_ref)
            if not theirs or min(len(mine), len(theirs)) < thr * max(len(mine), len(theirs)):
                continue      # Jaccard <= size ratio: cannot reach the threshold
            if same_ck and rec.get("coldkey") == my_ck:
                # One operator, two hotkeys: the FULL trees are compared —
                # sharing the public lineage is no excuse within a coldkey.
                union = len(mine | theirs)
                j = (len(mine & theirs) / union) if union else None
            else:
                j = private_jaccard(mine, theirs, public, min_private=min_priv)
            if j is None or j < thr:
                continue
            cand = (cb, their_hk, j, their_ref)
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is None:
            return None
        return best[1], best[2], best[0], best[3]

    def near_dup_verdict(self, ref: str, *, hotkey: str, commit_block: int,
                         coldkey: str | None = None
                         ) -> tuple[str, float, str] | None:
        """Rent-time re-check for a leg admitted earlier (or re-attached after
        a restart): ``(matched hotkey, J, mode)`` when ``ref`` near-duplicates
        an earlier entry and the tier is on, else ``None``. Fail-open."""
        mode = self._nd_mode()
        if mode not in ("shadow", "enforce"):
            return None
        self.backfill_nd()
        if self.nd_sig(ref) is None:
            self.compute_nd_sig(ref)
        hit = self.match_near_dup(ref, hotkey=hotkey, commit_block=commit_block,
                                  coldkey=coldkey)
        if hit is None:
            return None
        return hit[0], hit[1], mode

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
        nd_mode = self._nd_mode()
        if nd_mode in ("shadow", "enforce"):
            self.backfill_nd()
            nd = self.match_near_dup(gen.ref, hotkey=gen.hotkey, commit_block=my_block,
                                     coldkey=coldkey)
            if nd is not None:
                log.warning("rolling: near-duplicate %s — %s of %s (J=%.3f, earlier commit "
                            "%d) [%s]", "refused" if nd_mode == "enforce" else "shadow",
                            gen.hotkey[:12], nd[0][:12], nd[1], nd[2], nd_mode)
                if nd_mode != "enforce":
                    self.register(gen.ref, hotkey=gen.hotkey, commit_block=my_block,
                                  kind="queue", digests=digests, coldkey=coldkey)
                return nd[0], NEAR_DUP_TIER, nd_mode == "enforce"
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
