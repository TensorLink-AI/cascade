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
Enforcement rests on EXACT identity only — of code (the digests above) or,
via the trainer's behavioral probe, of output. A similarity threshold was
shipped briefly (drop at difflib ratio ≥ 0.99, with a linear sketch tier for
oversize streams) and deliberately REMOVED: a ratio bar is gameable by
construction — operators were already spacing variants at 0.90–0.986, so it
caught only the lazy tail while teaching spacing — and it produced the one
confirmed false positive (a round's eventual finalist, a 46-token documented
mechanism change at 0.995; a false drop permanently burns a hotkey). The
zero-delta re-rolls it targeted produce byte-identical probe output and are
caught by the ``behavior_identical`` tier instead, ungameably.

Judgement is strictly **pairwise against a specific earlier submission**
(king first, then previously kept entries, oldest submission first). Never
transitive.

COST (why the caps exist): tokenizing is linear but not cheap (~2 s/MB), and
the input size is attacker-chosen up to ``[generator] max_repo_mb``. Digests
are fed incrementally (never from a retained buffer), the decode budget
bounds per-repo CPU, and the retained token prefix — used only to measure the
absolute config delta on ``config_only`` labels — is capped by
``max_tokens``; past the cap a fingerprint keeps its exact digests but is not
``scoreable`` and the label simply carries no delta.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tokenize
from collections.abc import Mapping
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


def iter_normalized_tokens(source: str):
    """Yield the comment/whitespace-insensitive token stream of one Python
    source, one token at a time (never materialising the whole stream — a
    submission may be far larger than anything worth holding in memory).

    Falls back to whitespace-split words when the source does not tokenize
    (a submission with a syntax error still deserves a stable fingerprint —
    the static guard rejects it separately). The fallback restarts the stream,
    so a source that fails PART-WAY through tokenizing yields its already-
    emitted tokens and then the whole word split; that is deterministic, which
    is all the digests require.
    """
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in _DROP_TOKEN_TYPES:
                continue
            # Docstrings are STRING tokens and stay in: replacing a docstring
            # is a real (if tiny) edit and the similarity tier absorbs it.
            yield tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        yield from source.split()


def normalized_tokens(source: str) -> tuple[str, ...]:
    """:func:`iter_normalized_tokens` as a tuple (tests and small callers)."""
    return tuple(iter_normalized_tokens(source))


def _masked(token: str, maskable: bool) -> str:
    """One token as it enters the name-masked stream, so rename-only copies
    collapse.

    Only tokens flagged ``maskable`` (Python-source tokens) are masked — config
    values and requirement pins are DATA, and two repos that differ only in
    those are a parameter sweep, not a rename. Python keywords are
    identifier-shaped too; masking them as well is fine — two sources that
    differ only in NAME-shaped tokens are the same program skeleton, which is
    exactly what this digest is for.
    """
    if maskable and (token[:1].isalpha() or token[:1] == "_"):
        return _NAME_MASK
    return token


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


# Retained-stream cap: bounds the O(n²) SequenceMatcher pass that measures the
# absolute delta on config_only labels (the only quadratic consumer left).
DEFAULT_MAX_TOKENS = 50_000    # field repos run 7–11k; nothing honest is near this


@dataclass(frozen=True)
class RepoFingerprint:
    """Content identity of one submitted generator repo.

    ``tokens`` is the retained prefix of the normalized stream and is EMPTY
    when the stream ran past the cap (``scoreable`` False) — the digests cover
    the whole stream either way.
    """

    tree_sha256: str
    token_sha256: str
    masked_sha256: str
    py_sha256: str            # digest of the .py-only token stream ("" = no .py)
    tokens: tuple[str, ...] = field(repr=False)
    n_tokens: int = 0
    scoreable: bool = True    # False ⇒ over the cap, no quadratic pass may run
    truncated: bool = False   # some file was too large to decode in full


class _Accum:
    """Streaming fingerprint state: exact digests + a bounded token prefix,
    all fed one token at a time. Peak footprint is bounded by the caller's
    decode budget and ``max_tokens``, not by repo size."""

    def __init__(self, max_tokens: int) -> None:
        self.tree = hashlib.sha256()
        self._tok = hashlib.sha256()
        self._masked = hashlib.sha256()
        self._py = hashlib.sha256()
        self._max_tokens = max_tokens
        self.tokens: list[str] = []
        self.n_tokens = 0
        self.n_py_tokens = 0
        self.overflow = False
        self.truncated = False

    def push(self, token: str, *, maskable: bool, py: bool) -> None:
        blob = token.encode("utf-8", "replace") + b"\x00"
        self._tok.update(blob)
        self._masked.update(_masked(token, maskable).encode("utf-8", "replace") + b"\x00")
        if py:
            self._py.update(blob)
            self.n_py_tokens += 1
        self.n_tokens += 1
        if self._max_tokens <= 0 or len(self.tokens) < self._max_tokens:
            self.tokens.append(token)
        else:
            self.overflow = True

    def finish(self) -> RepoFingerprint:
        return RepoFingerprint(
            tree_sha256=self.tree.hexdigest(),
            token_sha256=self._tok.hexdigest(),
            masked_sha256=self._masked.hexdigest(),
            py_sha256=self._py.hexdigest() if self.n_py_tokens else "",
            tokens=tuple(self.tokens) if not self.overflow else (),
            n_tokens=self.n_tokens,
            scoreable=not self.overflow,
            truncated=self.truncated,
        )


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


_CHUNK = 1 << 20              # tree-digest read size
# Cumulative decode budget per repo. Tokenizing is linear but not cheap
# (~2 s/MB measured), and the input size is attacker-chosen up to
# ``[generator] max_repo_mb``; a whole-repo budget bounds one entrant's
# fingerprint cost at ~10 s. Field submissions are code-only and run ~100 KB
# of text, so this is ~40× headroom over anything honest.
DEFAULT_MAX_TEXT_MB = 4


def fingerprint_dir(
    repo_dir: Path | str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_text_mb: float = DEFAULT_MAX_TEXT_MB,
) -> RepoFingerprint:
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

    Bounded by construction (see the module docstring): files are hashed in
    chunks, tokens are streamed rather than materialised per file, only the
    first ``max_tokens`` are retained (``max_tokens=0`` disables the cap), and
    once ``max_text_mb`` of text has been decoded every remaining file enters
    the token streams as an OPAQUE content digest instead — so two repos
    differing only inside skipped files still differ in every digest, while
    both peak memory and CPU stay ``O(max_text_mb + max_tokens)`` however
    large the repo is.
    """
    root = Path(repo_dir)
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(root)),
    )
    acc = _Accum(max_tokens)
    text_cap = int(max(0.0, float(max_text_mb)) * 1024 * 1024)  # float: tests use fractions
    text_used = 0
    for p in files:
        rel = p.relative_to(root)
        if _is_junk(rel.parts):
            continue
        acc.tree.update(str(rel).encode("utf-8", "replace"))
        acc.tree.update(b"\x00")
        file_h = hashlib.sha256()
        size = 0
        with p.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                acc.tree.update(chunk)
                file_h.update(chunk)
                size += len(chunk)
        acc.tree.update(b"\x00")

        is_py = p.suffix == ".py"
        over_budget = bool(text_cap) and text_used + size > text_cap
        if over_budget and not is_py and _config_tokens(p, "") is None:
            continue

        # Past the decode budget, tokenize the PREFIX that still fits and
        # append the whole file's content digest — the digest keeps the token
        # tiers exact, so two repos differing only inside the skipped tail
        # cannot read as identical. Beyond this a submission is mostly padding,
        # which `truncated` surfaces in the report rather than hiding.
        take = size if not over_budget else max(0, text_cap - text_used)
        with p.open("rb") as fh:
            text = fh.read(take).decode("utf-8", "replace")
        text_used += take
        if is_py:
            stream, mask = iter_normalized_tokens(text), True
        else:
            cfg_toks = _config_tokens(p, text)
            if cfg_toks is None:
                continue
            stream, mask = iter(cfg_toks), False
        acc.push("\x00FILE", maskable=False, py=is_py)
        for t in stream:
            acc.push(t, maskable=mask, py=is_py)
        if over_budget:
            acc.push(f"\x00OPAQUE:{file_h.hexdigest()}", maskable=False, py=is_py)
            acc.overflow = True
            acc.truncated = True
    return acc.finish()


def _same_code(a: RepoFingerprint, b: RepoFingerprint) -> bool:
    """Identical normalized ``.py`` streams — the ``config_only`` precondition.

    An empty ``py_sha256`` (a repo with no Python at all) never matches: two
    such repos are not "the same code", they are two repos with no code, and
    collapsing them on that basis would be a false positive.
    """
    return bool(a.py_sha256) and a.py_sha256 == b.py_sha256


def _abs_token_delta(a: RepoFingerprint, b: RepoFingerprint) -> int:
    """Absolute changed-token count between two streams: over the non-equal
    SequenceMatcher opcodes, the max of the two sides' spans, summed. A pure
    ratio dilutes with repo size (7–11k-token repos tolerate ~90–110 changed
    tokens at 0.99); the absolute count is what separates a rename from a
    research edit."""
    sm = SequenceMatcher(None, a.tokens, b.tokens, autojunk=False)
    return sum(max(i2 - i1, j2 - j1)
               for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def _safe_abs_token_delta(a: RepoFingerprint, b: RepoFingerprint) -> int | None:
    """:func:`_abs_token_delta` when both streams are scoreable, else ``None``.

    This is a SECOND quadratic pass, so it inherits the same cap — and unlike
    the similarity tier it has no ``quick_ratio`` gate in front of it.
    """
    if not (a.scoreable and b.scoreable):
        return None
    return _abs_token_delta(a, b)


@dataclass(frozen=True)
class DedupVerdict:
    """One pairwise judgement, kept for the audit log whether or not it drops.

    Tiers: ``tree_identical`` | ``token_identical`` | ``rename_identical`` |
    ``config_only`` | ``behavior_identical``.
    """

    hotkey: str
    uid: int
    matched_hotkey: str
    matched_uid: int          # -2 marks the king (any sentinel outside uid space)
    tier: str
    score: float
    abs_delta: int | None = None   # changed-token count (config_only labels)


@dataclass(frozen=True)
class DedupResult:
    kept_hotkeys: tuple[str, ...]
    dropped: tuple[DedupVerdict, ...]
    shadow: tuple[DedupVerdict, ...]   # labels that never drop (config_only)


KING_UID = -2

# Sort position for an entry with no known submission block: after every entry
# that has one. Only reachable when the caller passes a partial ``priority``;
# the trainer always supplies a block for every entrant.
_LAST = float("inf")


def screen_duplicates(
    entries: list[tuple[str, int, RepoFingerprint]],
    king: RepoFingerprint | None,
    *,
    config_only_enforce: bool = False,
    priority: Mapping[str, int] | None = None,
    enforce: bool = True,
) -> DedupResult:
    """Pairwise duplicate screen over ``(hotkey, uid, fingerprint)`` entries.

    Entries are processed OLDEST SUBMISSION FIRST; each is compared against
    the king and every previously KEPT entry. Any identical digest drops it —
    the earlier submission keeps the slot, so copying an existing submission
    can never displace it. There is NO similarity threshold: drops ride on
    exact identity only (a ratio bar is gameable by spacing edits under it;
    see the module docstring).

    ``priority`` maps hotkey → the block that submission was made at (the
    witnessed commit block, falling back to the reveal block); entries without
    one sort last, and UID breaks any remaining tie. UID ALONE is not a
    submission order: Bittensor recycles the UIDs of deregistered neurons to
    new registrants, so on a saturated subnet a low UID means "inherited a
    pruned slot", not "was here first" — and ordering a copy contest by it
    would let a newcomer holding a recycled slot expropriate the miner they
    copied (who then also burns their one lifetime submission). The block a
    submission was committed at is the actual evidence, and a copyist cannot
    have committed before the thing they copied was visible.

    ``config_only_enforce`` (default False): a pair whose ``.py`` token
    streams are identical but whose functional config files differ gets a
    ``config_only`` verdict. With the flag on it drops as its own tier; off,
    it is shadow-logged as a LABEL (with the absolute config delta recorded
    when both streams are under the token cap) and the entry is kept —
    "identical code, different weights" is both the ticket-spam pattern and
    the legitimate fork path, so the log decides enforcement.

    With ``enforce=False`` (shadow mode) every entry is kept, but the RIVAL
    set still follows enforce semantics — a would-be-dropped entry does not
    become a rival for later entries. Shadow is therefore a true counterfactual
    of enforce (same verdicts, no drops).
    """
    prio = priority or {}
    ordered = sorted(entries, key=lambda e: (prio.get(e[0], _LAST), e[1]))
    kept: list[tuple[str, int, RepoFingerprint]] = []
    dropped: list[DedupVerdict] = []
    shadow: list[DedupVerdict] = []

    for hotkey, uid, fp in ordered:
        rivals: list[tuple[str, int, RepoFingerprint]] = []
        if king is not None:
            rivals.append(("king", KING_UID, king))
        rivals.extend(kept)

        verdict: DedupVerdict | None = None
        for r_hotkey, r_uid, r_fp in rivals:
            if fp.tree_sha256 == r_fp.tree_sha256:
                tier = "tree_identical"
            elif not (fp.n_tokens and r_fp.n_tokens):
                # At least one side has NO functional content (no .py, no
                # config — docs only). Every such repo shares the empty-stream
                # digest, so the token tiers would collapse two unrelated junk
                # repos into a copy verdict, and the loser burns its one
                # submission for it. No content, no content identity: only the
                # tree tier above (real bytes) may speak for these.
                continue
            elif fp.token_sha256 == r_fp.token_sha256:
                tier = "token_identical"
            elif fp.masked_sha256 == r_fp.masked_sha256:
                tier = "rename_identical"
            elif _same_code(fp, r_fp):
                # Identical code, different functional configs.
                delta = _safe_abs_token_delta(fp, r_fp)
                if not config_only_enforce:
                    shadow.append(DedupVerdict(hotkey, uid, r_hotkey, r_uid,
                                               "config_only", 1.0, delta))
                    continue
                verdict = DedupVerdict(hotkey, uid, r_hotkey, r_uid,
                                       "config_only", 1.0, delta)
                break
            else:
                continue
            verdict = DedupVerdict(hotkey, uid, r_hotkey, r_uid, tier, 1.0)
            break

        if verdict is not None:
            dropped.append(verdict)
        else:
            kept.append((hotkey, uid, fp))

    return DedupResult(
        kept_hotkeys=(tuple(h for h, _, _ in kept) if enforce
                      else tuple(h for h, _, _ in ordered)),
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
