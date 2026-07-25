"""Content-level duplicate screening for submitted generator repos.

The on-chain same-ref dedup (``plan_round``) only catches byte-identical
``repo@digest`` pointers; re-uploading the same tree mints a fresh OCI digest
and walks straight past it into a heat GPU slot. This module compares what the
digests point AT:

* **tree digest** — sha256 over the sorted ``(path, bytes)`` of every file:
  catches re-uploads of an identical tree.
* **token digest** — sha256 over the comment/whitespace-normalized Python token
  stream: catches comment shuffles and reformatting done purely to change the
  digest.
* **masked-token digest** — the same stream with identifier NAMEs masked:
  catches rename-only copies.
* **similarity** — difflib ratio over the normalized token streams, for the
  near-copy tier (observed abuse sits at 0.99+; honest template-sharing sits
  well below).

Enforcement is strictly **pairwise against a specific earlier submission**
(king first, then kept challengers in UID order — the same lowest-UID-wins
convention as ``plan_round``'s same-ref dedup). Never transitive: chained
similarity clusters merge honest template users and must not gate anything.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tokenize
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# Token types dropped from the normalized stream: pure formatting, comments,
# and the encoding pseudo-token — everything a lazy re-upload shuffles.
_DROP_TOKEN_TYPES = frozenset({
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
})

_NAME_MASK = "\x00N"  # placeholder for identifier tokens in the masked stream


def normalized_tokens(source: str) -> tuple[str, ...]:
    """The comment/whitespace-insensitive token stream of one Python source.

    Falls back to whitespace-split words when the source does not tokenize
    (a submission with a syntax error still deserves a stable fingerprint —
    the static guard rejects it separately).
    """
    try:
        toks = tokenize.generate_tokens(io.StringIO(source).readline)
        out: list[str] = []
        for tok in toks:
            if tok.type in _DROP_TOKEN_TYPES:
                continue
            # Docstrings are STRING tokens and stay in: replacing a docstring
            # is a real (if tiny) edit and the similarity tier absorbs it.
            out.append(tok.string)
        return tuple(out)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return tuple(source.split())


def _mask_names(tokens: tuple[str, ...], maskable: tuple[bool, ...]) -> tuple[str, ...]:
    """Replace identifier-shaped tokens so rename-only copies collapse.

    Only tokens flagged ``maskable`` (Python-source tokens) are masked — config
    values and requirement pins are DATA, and two repos that differ only in
    those are a parameter sweep, not a rename. Python keywords are
    identifier-shaped too; masking them as well is fine — two sources that
    differ only in NAME-shaped tokens are the same program skeleton, which is
    exactly what this digest is for.
    """
    return tuple(
        _NAME_MASK if m and (t[:1].isalpha() or t[:1] == "_") else t
        for t, m in zip(tokens, maskable, strict=True)
    )


_LINE_CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".toml", ".cfg", ".ini"})
_JSON_PUNCT = re.compile(r"([{}\[\],:])")


def _config_tokens(path: Path, text: str) -> tuple[str, ...] | None:
    """Tokens for a functional non-Python file, or ``None`` for files that
    don't shape the generative process (docs, licenses — tree-hash only).

    JSON is parsed and re-serialized (sorted keys, canonical separators) so
    reformatting is cosmetic but any value change is a real token delta —
    a ``config.json`` sweep reads as a near-duplicate, never as identical.
    """
    if path.suffix.lower() == ".json":
        try:
            norm = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
        except ValueError:
            norm = " ".join(text.split())
        return tuple(t for t in _JSON_PUNCT.split(norm) if t)
    if path.name.lower() == "requirements.txt":
        # Sorted + comment-stripped: dependency ORDER is cosmetic, the pin
        # set is functional (deps can carry generator logic).
        return tuple(sorted(
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")))
    if path.suffix.lower() in _LINE_CONFIG_SUFFIXES:
        return tuple(ln.strip() for ln in text.splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))
    return None


@dataclass(frozen=True)
class RepoFingerprint:
    """Content identity of one submitted generator repo."""

    tree_sha256: str
    token_sha256: str
    masked_sha256: str
    py_sha256: str            # digest of the .py-only token stream (config_only tier)
    tokens: tuple[str, ...] = field(repr=False)


def _sha256_tokens(tokens: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


# Path components / names that never shape the generative process and change
# between uploads of the same content (hub download caches embed content hashes
# and timestamps): excluded from BOTH the tree digest and the token stream, so
# re-uploading with fresh cache junk cannot mint a "different" tree.
_JUNK_DIR_PARTS = frozenset({".cache", ".git", "__pycache__"})
_JUNK_NAMES = frozenset({".gitattributes"})
_JUNK_SUFFIXES = frozenset({".metadata"})


def _is_junk(rel_parts: tuple[str, ...]) -> bool:
    if any(part in _JUNK_DIR_PARTS for part in rel_parts[:-1]):
        return True
    name = rel_parts[-1]
    return name in _JUNK_NAMES or Path(name).suffix in _JUNK_SUFFIXES


def fingerprint_dir(repo_dir: Path | str) -> RepoFingerprint:
    """Fingerprint a fetched repo tree.

    The tree digest covers every regular file (sorted by relative path),
    minus cache/VCS junk (``.cache/``, ``.git/``, ``__pycache__/``,
    ``*.metadata``, ``.gitattributes``) whose bytes churn on every upload.
    The token stream concatenates, in the same order, the normalized tokens of
    every ``.py`` file AND of every functional config file (``*.json``,
    ``requirements.txt``, yaml/toml-style configs) — parameters live in
    configs as much as in code, so a config-only delta must read as a real
    delta, not collapse into "identical code". A path-independent file
    separator keeps split/merged modules from accidentally colliding. Docs
    and other non-functional files enter the tree digest only.
    """
    root = Path(repo_dir)
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(root)),
    )
    tree = hashlib.sha256()
    tokens: list[str] = []
    maskable: list[bool] = []
    py_tokens: list[str] = []
    for p in files:
        rel = p.relative_to(root)
        if _is_junk(rel.parts):
            continue
        data = p.read_bytes()
        tree.update(str(rel).encode("utf-8", "replace"))
        tree.update(b"\x00")
        tree.update(data)
        tree.update(b"\x00")
        if p.suffix == ".py":
            file_toks = normalized_tokens(data.decode("utf-8", "replace"))
            mask = True
            py_tokens.append("\x00FILE")
            py_tokens.extend(file_toks)
        else:
            cfg_toks = _config_tokens(p, data.decode("utf-8", "replace"))
            if cfg_toks is None:
                continue
            file_toks, mask = cfg_toks, False
        tokens.append("\x00FILE")
        maskable.append(False)
        tokens.extend(file_toks)
        maskable.extend([mask] * len(file_toks))
    toks = tuple(tokens)
    return RepoFingerprint(
        tree_sha256=tree.hexdigest(),
        token_sha256=_sha256_tokens(toks),
        masked_sha256=_sha256_tokens(_mask_names(toks, tuple(maskable))),
        py_sha256=_sha256_tokens(tuple(py_tokens)),
        tokens=toks,
    )


def similarity(a: RepoFingerprint, b: RepoFingerprint) -> float:
    """difflib ratio over the normalized token streams (identical ⇒ 1.0).

    ``quick_ratio`` is a documented upper bound on ``ratio``, so using it as a
    cheap gate can only skip pairs whose true ratio is below the caller's
    floor — the caller passes that floor via :func:`screen_duplicates`.
    """
    if a.token_sha256 == b.token_sha256:
        return 1.0
    return SequenceMatcher(None, a.tokens, b.tokens, autojunk=False).ratio()


def _bounded_similarity(a: RepoFingerprint, b: RepoFingerprint, floor: float) -> float:
    sm = SequenceMatcher(None, a.tokens, b.tokens, autojunk=False)
    if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
        return 0.0
    return sm.ratio()


def _abs_token_delta(a: RepoFingerprint, b: RepoFingerprint) -> int:
    """Absolute changed-token count between two streams: over the non-equal
    SequenceMatcher opcodes, the max of the two sides' spans, summed. A pure
    ratio dilutes with repo size (7–11k-token repos tolerate ~90–110 changed
    tokens at 0.99); the absolute count is what separates a rename from a
    research edit."""
    sm = SequenceMatcher(None, a.tokens, b.tokens, autojunk=False)
    return sum(max(i2 - i1, j2 - j1)
               for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


@dataclass(frozen=True)
class DedupVerdict:
    """One pairwise judgement, kept for the audit log whether or not it drops.

    Tiers: ``tree_identical`` | ``token_identical`` | ``rename_identical`` |
    ``config_only`` | ``near_duplicate`` | ``near_duplicate_large_delta``
    (shadow-only) | ``shadow`` (similarity band) | ``behavior_identical``.
    """

    hotkey: str
    uid: int
    matched_hotkey: str
    matched_uid: int          # -2 marks the king (any sentinel outside uid space)
    tier: str
    score: float
    abs_delta: int | None = None   # changed-token count (similarity tiers only)


@dataclass(frozen=True)
class DedupResult:
    kept_hotkeys: tuple[str, ...]
    dropped: tuple[DedupVerdict, ...]
    shadow: tuple[DedupVerdict, ...]   # threshold > score ≥ shadow_floor (never drops)


KING_UID = -2


def screen_duplicates(
    entries: list[tuple[str, int, RepoFingerprint]],
    king: RepoFingerprint | None,
    *,
    threshold: float = 0.99,
    shadow_floor: float = 0.90,
    max_abs_delta: int = 0,
    config_only_enforce: bool = False,
    enforce: bool = True,
) -> DedupResult:
    """Pairwise duplicate screen over ``(hotkey, uid, fingerprint)`` entries.

    Entries are processed in ascending UID order; each is compared against the
    king and every previously KEPT entry. The first match at or above
    ``threshold`` (or any identical digest) drops it — lowest UID keeps the
    slot, so copying an existing submission can never displace it. Matches in
    ``[shadow_floor, threshold)`` are recorded but never drop.

    ``max_abs_delta`` (0 = disabled): a ``near_duplicate`` drop additionally
    requires the absolute changed-token count to be at most this — pairs over
    the ratio bar but past the cap land in the shadow list as
    ``near_duplicate_large_delta`` (logged, never dropped), so a substantive
    edit inside mostly-shared scaffold survives size dilution of the ratio.

    ``config_only_enforce`` (default False): a pair whose ``.py`` token
    streams are identical but whose functional config files differ gets a
    ``config_only`` verdict. With the flag on it drops as its own tier; with
    the flag off it is shadow-logged as a LABEL and the pair still faces the
    similarity tier — a tiny config sweep keeps dropping as
    ``near_duplicate``, while a config rewrite large enough to fall under the
    ratio bar survives with the label recorded ("identical code, different
    weights" is both the ticket-spam pattern and the legitimate fork path;
    the log decides enforcement, it never exempts).

    With ``enforce=False`` (shadow mode) would-be drops are logged as
    verdicts but every entry is kept.
    """
    ordered = sorted(entries, key=lambda e: e[1])
    kept: list[tuple[str, int, RepoFingerprint]] = []
    dropped: list[DedupVerdict] = []
    shadow: list[DedupVerdict] = []

    for hotkey, uid, fp in ordered:
        rivals: list[tuple[str, int, RepoFingerprint]] = []
        if king is not None:
            rivals.append(("king", KING_UID, king))
        rivals.extend(kept)

        verdict: DedupVerdict | None = None
        best_shadow: DedupVerdict | None = None
        for r_hotkey, r_uid, r_fp in rivals:
            delta: int | None = None
            if fp.tree_sha256 == r_fp.tree_sha256:
                tier, score = "tree_identical", 1.0
            elif fp.token_sha256 == r_fp.token_sha256:
                tier, score = "token_identical", 1.0
            elif fp.masked_sha256 == r_fp.masked_sha256:
                tier, score = "rename_identical", 1.0
            elif fp.py_sha256 == r_fp.py_sha256 and config_only_enforce:
                # Identical code, different functional configs — enforced as
                # its own tier.
                delta = _abs_token_delta(fp, r_fp)
                tier, score = "config_only", 1.0
            else:
                if fp.py_sha256 == r_fp.py_sha256:
                    # Identical code, different configs with enforcement off:
                    # shadow-log the label, then FALL THROUGH to the
                    # similarity tier — a tiny config sweep must still drop as
                    # near_duplicate (a byte-identical-code A/B ticket is the
                    # clearest spam there is); the label only measures how
                    # often the pattern occurs so enforcement can be decided
                    # from data, it never exempts the pair.
                    shadow.append(DedupVerdict(hotkey, uid, r_hotkey, r_uid,
                                               "config_only", 1.0,
                                               _abs_token_delta(fp, r_fp)))
                score = _bounded_similarity(fp, r_fp, shadow_floor)
                if score >= threshold:
                    delta = _abs_token_delta(fp, r_fp)
                    if max_abs_delta > 0 and delta > max_abs_delta:
                        # Over the ratio bar but a real edit by absolute size:
                        # logged for threshold calibration, never dropped.
                        shadow.append(DedupVerdict(
                            hotkey, uid, r_hotkey, r_uid,
                            "near_duplicate_large_delta", round(score, 4), delta))
                        continue
                    tier = "near_duplicate"
                elif score >= shadow_floor:
                    cand = DedupVerdict(hotkey, uid, r_hotkey, r_uid, "shadow", round(score, 4))
                    if best_shadow is None or cand.score > best_shadow.score:
                        best_shadow = cand
                    continue
                else:
                    continue
            verdict = DedupVerdict(hotkey, uid, r_hotkey, r_uid, tier,
                                   round(score, 4), delta)
            break

        if verdict is not None:
            dropped.append(verdict)
            if not enforce:
                kept.append((hotkey, uid, fp))
        else:
            if best_shadow is not None:
                shadow.append(best_shadow)
            kept.append((hotkey, uid, fp))

    return DedupResult(
        kept_hotkeys=tuple(h for h, _, _ in kept),
        dropped=tuple(dropped),
        shadow=tuple(shadow),
    )


def collapse_identical_behavior(
    entries: list[tuple[str, int, str]],
    king_digest: str | None,
) -> tuple[tuple[str, ...], tuple[DedupVerdict, ...]]:
    """Collapse entrants whose PROBE OUTPUT is byte-identical under the shared
    round seed: ``entries`` is ``(hotkey, uid, behavior_digest)``.

    Two repos that produce identical bytes from the same seed are the same
    generative process no matter how different the code looks — this is the
    backstop for obfuscated forks and logic hidden in dependencies. Exact
    equality is transitive-safe (unlike similarity), so a plain first-owner
    map is equivalent to the pairwise rule: king first, then lowest UID.
    """
    seen: dict[str, tuple[str, int]] = {}
    if king_digest:
        seen[king_digest] = ("king", KING_UID)
    kept: list[str] = []
    dropped: list[DedupVerdict] = []
    for hotkey, uid, digest in sorted(entries, key=lambda e: e[1]):
        owner = seen.get(digest)
        if owner is not None:
            dropped.append(DedupVerdict(
                hotkey, uid, owner[0], owner[1], "behavior_identical", 1.0))
        else:
            seen[digest] = (hotkey, uid)
            kept.append(hotkey)
    return tuple(kept), tuple(dropped)
