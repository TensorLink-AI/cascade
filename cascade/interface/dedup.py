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


def _mask_names(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Replace identifier-shaped tokens so rename-only copies collapse.

    Python keywords are identifier-shaped too; masking them as well is fine —
    two sources that differ only in NAME-shaped tokens are the same program
    skeleton, which is exactly what this digest is for.
    """
    return tuple(
        _NAME_MASK if t[:1].isalpha() or t[:1] == "_" else t for t in tokens
    )


@dataclass(frozen=True)
class RepoFingerprint:
    """Content identity of one submitted generator repo."""

    tree_sha256: str
    token_sha256: str
    masked_sha256: str
    tokens: tuple[str, ...] = field(repr=False)


def _sha256_tokens(tokens: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def fingerprint_dir(repo_dir: Path | str) -> RepoFingerprint:
    """Fingerprint a fetched repo tree.

    The tree digest covers every regular file (sorted by relative path); the
    token stream concatenates the normalized tokens of every ``.py`` file in
    the same order, with a path-independent file separator so splitting one
    module into two files does not accidentally collide.
    """
    root = Path(repo_dir)
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(root)),
    )
    tree = hashlib.sha256()
    tokens: list[str] = []
    for p in files:
        rel = str(p.relative_to(root))
        data = p.read_bytes()
        tree.update(rel.encode("utf-8", "replace"))
        tree.update(b"\x00")
        tree.update(data)
        tree.update(b"\x00")
        if p.suffix == ".py":
            tokens.append("\x00FILE")
            tokens.extend(normalized_tokens(data.decode("utf-8", "replace")))
    toks = tuple(tokens)
    return RepoFingerprint(
        tree_sha256=tree.hexdigest(),
        token_sha256=_sha256_tokens(toks),
        masked_sha256=_sha256_tokens(_mask_names(toks)),
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


@dataclass(frozen=True)
class DedupVerdict:
    """One pairwise judgement, kept for the audit log whether or not it drops."""

    hotkey: str
    uid: int
    matched_hotkey: str
    matched_uid: int          # -2 marks the king (any sentinel outside uid space)
    tier: str                 # tree_identical | token_identical | rename_identical | near_duplicate | shadow
    score: float


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
    enforce: bool = True,
) -> DedupResult:
    """Pairwise duplicate screen over ``(hotkey, uid, fingerprint)`` entries.

    Entries are processed in ascending UID order; each is compared against the
    king and every previously KEPT entry. The first match at or above
    ``threshold`` (or any identical digest) drops it — lowest UID keeps the
    slot, so copying an existing submission can never displace it. Matches in
    ``[shadow_floor, threshold)`` are recorded but never drop. With
    ``enforce=False`` (shadow mode) would-be drops are logged as verdicts but
    every entry is kept.
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
            if fp.tree_sha256 == r_fp.tree_sha256:
                tier, score = "tree_identical", 1.0
            elif fp.token_sha256 == r_fp.token_sha256:
                tier, score = "token_identical", 1.0
            elif fp.masked_sha256 == r_fp.masked_sha256:
                tier, score = "rename_identical", 1.0
            else:
                score = _bounded_similarity(fp, r_fp, shadow_floor)
                if score >= threshold:
                    tier = "near_duplicate"
                elif score >= shadow_floor:
                    cand = DedupVerdict(hotkey, uid, r_hotkey, r_uid, "shadow", round(score, 4))
                    if best_shadow is None or cand.score > best_shadow.score:
                        best_shadow = cand
                    continue
                else:
                    continue
            verdict = DedupVerdict(hotkey, uid, r_hotkey, r_uid, tier, round(score, 4))
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
