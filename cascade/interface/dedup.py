"""Content fingerprints for submitted generator repos (DEC-CA-0008).

Two ``repo@digest`` pointers can name the same content: re-uploading a tree
mints a fresh OCI digest. These fingerprints compare what the digests point
AT, and the rolling admission registry (:mod:`cascade.trainer.dedup_registry`)
keeps them across the queue, in-flight legs, the king and published champions:

* **tree digest** — sha256 over the sorted ``(path, bytes)`` of every file:
  catches re-uploads of an identical tree.
* **token digest** — sha256 over the comment/whitespace-normalized Python token
  stream: catches comment shuffles and reformatting done purely to change the
  digest.
* **masked-token digest** — the same stream with identifier NAMEs masked:
  catches rename-only copies.
* **components** — the same digests over every Python source packed into a
  string constant, so a wrapper carrying someone's generator as a blob and
  that generator submitted plain share a digest (the ``embedded_*`` tiers).
* **module sketches** — subsampled shingles of each module's masked stream,
  for the private-copy tier (containment, not identity).

Enforcement on the digest tiers rests on EXACT identity only. A similarity
threshold was shipped briefly (drop at difflib ratio ≥ 0.99) and deliberately
REMOVED: a ratio bar is gameable by construction — operators were already
spacing variants at 0.90–0.986, so it caught only the lazy tail while teaching
spacing — and it produced the one confirmed false positive (a 46-token
documented mechanism change at 0.995; a false drop permanently burns a hotkey).

COST: tokenizing is linear but not cheap (~2 s/MB), and the input size is
attacker-chosen up to ``[generator] max_repo_mb``. Digests are fed
incrementally (never from a retained buffer) and the decode budget
(``max_text_mb``) bounds per-repo CPU and memory.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tokenize
from dataclasses import dataclass
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
            # is a real (if tiny) edit.
            yield tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        yield from source.split()


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
    a ``config.json`` sweep reads as a different repo, never as identical.
    """
    if path.suffix.lower() == ".json":
        try:
            norm = json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
        except (ValueError, RecursionError):
            # RecursionError: json.loads on attacker-chosen bytes — a few KB of
            # nested brackets blows the parser stack, and it is NOT a ValueError.
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
    """Content identity of one submitted generator repo (digests only)."""

    tree_sha256: str
    token_sha256: str
    masked_sha256: str
    py_sha256: str            # digest of the .py-only token stream ("" = no .py)
    n_tokens: int = 0
    truncated: bool = False   # some file was too large to decode in full
    py_masked_sha256: str = ""    # name-masked .py-only stream ("" = no .py)
    # Every Python source packed into a string constant of a .py file (plain,
    # base64, zlib, hex — see `_packed_sources`), fingerprinted exactly as a
    # one-file repo's .py stream would be, so a wrapper carrying someone's
    # generator as a blob and that generator submitted plain share a digest.
    components: tuple[ComponentFingerprint, ...] = ()


@dataclass(frozen=True)
class ComponentFingerprint:
    """One string-packed source inside a repo, as an exact-identity digest pair.

    ``token_sha256`` / ``masked_sha256`` equal the ``py_sha256`` /
    ``py_masked_sha256`` a repo whose ONLY ``.py`` file is this source would
    carry (same stream, same file separator), which is what makes the
    embedded tiers a plain digest comparison in both directions.
    """

    name: str                 # "<rel path>#packed<i>"
    token_sha256: str
    masked_sha256: str
    n_tokens: int
    n_bytes: int


# A packed source shorter than this is a snippet (a template, a docstring
# that happens to parse), not a generator anyone could have copied whole.
MIN_COMPONENT_TOKENS = 200


class _Accum:
    """Streaming fingerprint state: exact digests fed one token at a time.
    Peak footprint is bounded by the caller's decode budget, not by repo
    size."""

    def __init__(self) -> None:
        self.tree = hashlib.sha256()
        self._tok = hashlib.sha256()
        self._masked = hashlib.sha256()
        self._py = hashlib.sha256()
        self._pym = hashlib.sha256()
        self.n_tokens = 0
        self.n_py_tokens = 0
        self.truncated = False

    def push(self, token: str, *, maskable: bool, py: bool) -> None:
        blob = token.encode("utf-8", "replace") + b"\x00"
        mblob = _masked(token, maskable).encode("utf-8", "replace") + b"\x00"
        self._tok.update(blob)
        self._masked.update(mblob)
        if py:
            self._py.update(blob)
            self._pym.update(mblob)
            self.n_py_tokens += 1
        self.n_tokens += 1

    def finish(self, components: tuple[ComponentFingerprint, ...] = ()) -> RepoFingerprint:
        return RepoFingerprint(
            tree_sha256=self.tree.hexdigest(),
            token_sha256=self._tok.hexdigest(),
            masked_sha256=self._masked.hexdigest(),
            py_sha256=self._py.hexdigest() if self.n_py_tokens else "",
            n_tokens=self.n_tokens,
            truncated=self.truncated,
            py_masked_sha256=self._pym.hexdigest() if self.n_py_tokens else "",
            components=components,
        )


def component_fingerprint(name: str, source: str) -> ComponentFingerprint:
    """Digest one packed source exactly like a one-file repo's ``.py`` stream."""
    acc = _Accum()
    acc.push("\x00FILE", maskable=False, py=True)
    for t in iter_normalized_tokens(source):
        acc.push(t, maskable=True, py=True)
    return ComponentFingerprint(
        name=name, token_sha256=acc._py.hexdigest(), masked_sha256=acc._pym.hexdigest(),
        n_tokens=acc.n_py_tokens - 1, n_bytes=len(source.encode("utf-8", "replace")))


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
    chunks, tokens are streamed rather than materialised, and once
    ``max_text_mb`` of text has been decoded every remaining file enters the
    token streams as an OPAQUE content digest instead — so two repos differing
    only inside skipped files still differ in every digest, while both peak
    memory and CPU stay ``O(max_text_mb)`` however large the repo is.
    """
    root = Path(repo_dir)
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: str(p.relative_to(root)),
    )
    acc = _Accum()
    text_cap = int(max(0.0, float(max_text_mb)) * 1024 * 1024)  # float: tests use fractions
    text_used = 0
    # Packed sources are unpacked and digested on their OWN copy of the decode
    # budget: the three whole-repo digests must not move because a repo near
    # the cap grew a blob, and a blob past the budget is simply not a component.
    comp_used = 0
    comp_over = False
    components: list[ComponentFingerprint] = []
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
            acc.truncated = True
        elif is_py and not comp_over:
            for i, src in enumerate(_packed_sources(text), 1):
                n_bytes = len(src.encode("utf-8", "replace"))
                if text_cap and comp_used + n_bytes > text_cap:
                    comp_over = True
                    acc.truncated = True
                    break
                comp_used += n_bytes
                comp = component_fingerprint(f"{rel}#packed{i}", src)
                if comp.n_tokens >= MIN_COMPONENT_TOKENS:
                    components.append(comp)
    return acc.finish(tuple(components))


# ── per-module sketches: the private-copy tier ───────────────────────────────
#
# The exact tiers above catch re-uploads, comment shuffles and rename-only
# copies of a WHOLE tree. They cannot see a submission that wraps another
# miner's private module inside a different outer tree (e.g. a never-published
# vault submission's packed module, renamed identifier-for-identifier, glued
# to a public lineage). Copying public code
# is the design; copying code that was never public can only mean it left
# the operator's store — so that, and only that, is what this tier screens.
#
# Each Python module (a ``.py`` file, or a source packed into a string
# constant — plain, base64, zlib, hex — nested up to three deep) becomes a
# sketch: the set of 8-gram shingles over its NAME-MASKED token stream,
# subsampled by hash (``h % SKETCH_SAMPLE_MOD == 0``), so containment
# ``|A∩B| / |B|`` is an unbiased estimate of how much of module B's masked
# text appears in module A — rename-proof, and robust to the small edits a
# copier makes. Sketch hashes are BLAKE2b, never ``hash()`` (which is
# salted per process — the sketches persist across restarts).

SKETCH_SHINGLE = 8
SKETCH_SAMPLE_MOD = 8
MIN_MODULE_TOKENS = 2_000         # helpers below this size are not "a module" worth owning
# A module's PRIVATE part (its shingles minus every public module's) must be
# at least this many sampled shingles AND this fraction of the module before
# anyone can be said to own it — a vault file that is a public lineage with a
# few edits owns the edits, not the lineage, and the edits alone are too
# little to build a copy claim on.
MIN_PRIVATE_SHINGLES = 64
MIN_PRIVATE_FRACTION = 0.25
MIN_PACKED_CHARS = 200
MAX_PACK_DEPTH = 3


def _packed_sources(source: str, _depth: int = 0) -> list[str]:
    """Every string constant in ``source`` that decodes to Python source
    (plain, base64, zlib, base64+zlib, hex), recursively."""
    import ast
    import base64
    import binascii
    import zlib

    if _depth >= MAX_PACK_DEPTH:
        return []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))
                and len(node.value) >= MIN_PACKED_CHARS):
            continue
        raw = node.value.encode("utf-8", "surrogatepass") if isinstance(node.value, str) else node.value
        stripped = b"".join(raw.split())
        cands = [raw]
        for fn in (lambda b: base64.b64decode(b, validate=True),
                   lambda b: zlib.decompress(base64.b64decode(b, validate=True)),
                   zlib.decompress, binascii.unhexlify):
            try:
                dec = fn(stripped)
            except Exception:  # noqa: BLE001 — not that encoding
                continue
            if dec and dec not in cands:
                cands.append(dec)
        for dec in cands:
            head = dec[:262_144]
            if not (b"import " in head or b"def " in head or b"class " in head):
                continue
            text = dec.decode("utf-8", "replace")
            try:
                if not ast.parse(text).body:
                    continue
            except (SyntaxError, ValueError):
                continue
            out.append(text)
            out.extend(_packed_sources(text, _depth + 1))
    return out


def sketch_source(source: str, *, min_tokens: int = MIN_MODULE_TOKENS) -> tuple[int, list[int]]:
    """``(token_count, sorted sampled shingle hashes)`` of one Python source's
    name-masked token stream; ``(n, [])`` when it is shorter than ``min_tokens``."""
    toks = [_masked(t, True) for t in iter_normalized_tokens(source)]
    n = len(toks)
    if n < min_tokens:
        return n, []
    sample: set[int] = set()
    for i in range(0, n - SKETCH_SHINGLE + 1):
        h = int.from_bytes(hashlib.blake2b("\x1f".join(toks[i:i + SKETCH_SHINGLE]).encode(
            "utf-8", "surrogatepass"), digest_size=8).digest(), "big")
        if h % SKETCH_SAMPLE_MOD == 0:
            sample.add(h)
    return n, sorted(sample)


def module_sketches(
    repo_dir: Path | str,
    *,
    min_tokens: int = MIN_MODULE_TOKENS,
    max_text_mb: float = DEFAULT_MAX_TEXT_MB,
) -> list[dict]:
    """Sketch every Python module of a fetched tree — each ``.py`` file and
    each source packed into one of its string constants — as
    ``{"name", "tokens", "sketch"}``. Modules under ``min_tokens`` are left
    out; decoding stops after ``max_text_mb`` of source (bounded like
    :func:`fingerprint_dir`)."""
    root = Path(repo_dir)
    budget = int(max(0.0, float(max_text_mb)) * 1024 * 1024)
    out: list[dict] = []
    for p in sorted((q for q in root.rglob("*.py") if q.is_file()),
                    key=lambda q: str(q.relative_to(root))):
        rel = p.relative_to(root)
        if _is_junk(rel.parts):
            continue
        size = p.stat().st_size
        if budget - size < 0:
            break
        budget -= size
        text = p.read_bytes().decode("utf-8", "replace")
        sources = [(str(rel), text)]
        sources += [(f"{rel}#packed{i}", s) for i, s in enumerate(_packed_sources(text), 1)]
        for name, src in sources:
            n, sk = sketch_source(src, min_tokens=min_tokens)
            if sk:
                out.append({"name": name, "tokens": n, "sketch": sk})
    return out


def sketch_containment(a: list[int] | set[int], b: list[int] | set[int]) -> float:
    """How much of sketch ``b`` appears in sketch ``a`` (0.0 when ``b`` is empty).
    Pass sets when comparing many pairs — lists are converted per call."""
    sb = b if isinstance(b, (set, frozenset)) else set(b)
    if not sb:
        return 0.0
    sa = a if isinstance(a, (set, frozenset)) else set(a)
    return len(sa & sb) / len(sb)
