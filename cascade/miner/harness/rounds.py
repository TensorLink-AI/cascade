"""The round window: which past rounds the gauntlet replays, refreshed daily.

A round is REPLAYABLE when its signed receipt is scored (king per-window scores
present) and its pool snapshot has been revealed and copied locally. The window
is the newest ``n_a + n_b`` replayable rounds, split by time:

* pool **B** = the newest ``n_b`` rounds (confirm stage, pass/fail only);
* pool **A** = the ``n_a`` rounds before those (search; each G2 screen draws one).

The window only moves when new rounds become replayable — the pool is published
once a day and revealed ~48h later, so in steady state it slides once a day.
Every slide changes the :func:`fingerprint` and starts a new gauntlet epoch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..replay import SHA_MARKER, ReplayError, _norm_digest, find_snapshot_dir

log = logging.getLogger("cascade.miner.harness.rounds")

_BLOCK_RE = re.compile(r"block-(\d+)")


@dataclass(frozen=True)
class RoundRef:
    round_id: str
    epoch_start_block: int
    receipt_path: Path
    pool_sha256: str
    snapshot_dir: Path

    def to_json(self) -> dict:
        return {"round_id": self.round_id, "epoch_start_block": self.epoch_start_block,
                "receipt_path": str(self.receipt_path), "pool_sha256": self.pool_sha256,
                "snapshot_dir": str(self.snapshot_dir)}

    @classmethod
    def from_json(cls, d: dict) -> RoundRef:
        return cls(str(d["round_id"]), int(d["epoch_start_block"]), Path(d["receipt_path"]),
                   str(d["pool_sha256"]), Path(d["snapshot_dir"]))


@dataclass(frozen=True)
class RoundWindow:
    a: tuple[RoundRef, ...]          # search rounds, oldest first
    b: tuple[RoundRef, ...]          # confirm rounds (newest), oldest first

    @property
    def all(self) -> tuple[RoundRef, ...]:
        return self.a + self.b

    def newest(self, n: int) -> tuple[RoundRef, ...]:
        return self.all[-n:] if n > 0 else ()

    def to_json(self) -> dict:
        return {"a": [r.to_json() for r in self.a], "b": [r.to_json() for r in self.b]}

    @classmethod
    def from_json(cls, d: dict) -> RoundWindow:
        return cls(tuple(RoundRef.from_json(r) for r in d.get("a", [])),
                   tuple(RoundRef.from_json(r) for r in d.get("b", [])))


# --------------------------------------------------------------------------- #
# receipts                                                                     #
# --------------------------------------------------------------------------- #

IndexFetch = Callable[[], dict | None]
TextFetch = Callable[[str], str]


def _default_index_fetch(chain_cfg) -> IndexFetch:
    from ..dashboard import fetch_public_receipt_index
    return lambda: fetch_public_receipt_index(chain_cfg.storage)


def _default_text_fetch(chain_cfg) -> TextFetch:
    from ...audit.main import _fetch_text
    return lambda key: _fetch_text(chain_cfg, key)


def refresh_receipts(chain_cfg, cache_dir: Path, *, validator: str = "", scan: int = 60,
                     index_fetch: IndexFetch | None = None,
                     text_fetch: TextFetch | None = None) -> int:
    """Cache the newest scored receipts under ``cache_dir/<round_id>.json``.

    Network failures are logged, never raised: a dead index just means the
    window does not move today. Returns the number of receipts newly cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_fetch = index_fetch or _default_index_fetch(chain_cfg)
    text_fetch = text_fetch or _default_text_fetch(chain_cfg)
    try:
        doc = index_fetch()
    except Exception as e:  # noqa: BLE001
        log.warning("receipt index unavailable: %s", e)
        return 0
    rows = [r for r in (doc or {}).get("rounds", []) if isinstance(r, dict)
            and r.get("status") == "scored" and r.get("receipt_key")
            and (not validator or r.get("validator_hotkey") == validator)]
    rows.sort(key=lambda r: (int(r.get("epoch_start_block") or 0), str(r.get("round_id"))))
    added = 0
    seen: set[str] = set()
    for r in reversed(rows[-scan * 4:]):
        rid = str(r["round_id"])
        if rid in seen:
            continue
        seen.add(rid)
        if len(seen) > scan:
            break
        dest = cache_dir / f"{rid}.json"
        if dest.is_file():
            continue
        try:
            text = text_fetch(str(r["receipt_key"]))
            json.loads(text)
        except Exception as e:  # noqa: BLE001
            log.warning("receipt %s unavailable: %s", rid, e)
            continue
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(dest)
        added += 1
    return added


def receipt_pool(receipt) -> tuple[str, int | None]:
    """``(pool sha256, snapshot effective block)`` a receipt was scored on."""
    m = receipt.manifest if isinstance(receipt.manifest, dict) else {}
    sha = _norm_digest(m.get("eval_pool_sha256") or (
        receipt.eval_context.pool_digest if receipt.eval_context else ""))
    key = str(m.get("eval_pool_key") or (receipt.eval_context.pool_ref
                                          if receipt.eval_context else ""))
    hit = _BLOCK_RE.search(key)
    return sha, int(hit.group(1)) if hit else None


# --------------------------------------------------------------------------- #
# revealed snapshots                                                           #
# --------------------------------------------------------------------------- #

ListFolders = Callable[[], list[str]]
Download = Callable[[str, Path], None]


def _hf_list_folders(repo: str) -> ListFolders:
    def run() -> list[str]:
        from huggingface_hub import HfApi
        tree = HfApi().list_repo_tree(repo, path_in_repo="snapshots", repo_type="dataset")
        return [t.path for t in tree if not getattr(t, "size", None)]
    return run


def _hf_download(repo: str) -> Download:
    def run(folder: str, root: Path) -> None:
        from huggingface_hub import snapshot_download
        # The unpacked .npy + metadata are what the draw reads; the tar copy is
        # redundant bytes. POOL_SHA256 is the match key.
        snapshot_download(repo, repo_type="dataset", local_dir=str(root),
                          allow_patterns=[f"{folder}/*"], ignore_patterns=["*.tar"])
    return run


def sync_snapshots(blocks: set[int], root: Path, *, repo: str,
                   list_folders: ListFolders | None = None,
                   download: Download | None = None) -> dict:
    """Download the revealed folders for snapshot ``blocks`` not yet under ``root``.
    Returns ``{"fetched": [folders], "unrevealed": [blocks]}``; failures are
    logged (the round just waits)."""
    want = sorted(blocks - local_snapshot_blocks(root))
    out: dict = {"fetched": [], "unrevealed": []}
    if not want:
        return out
    try:
        folders = (list_folders or _hf_list_folders(repo))()
    except Exception as e:  # noqa: BLE001
        log.warning("cannot list revealed snapshots in %s: %s", repo, e)
        return out
    by_block = {int(m.group(1)): f for f in folders if (m := _BLOCK_RE.search(f))}
    for b in want:
        folder = by_block.get(b)
        if folder is None:
            out["unrevealed"].append(b)           # not revealed yet (~48h lag)
            continue
        try:
            (download or _hf_download(repo))(folder, root)
            out["fetched"].append(folder)
        except Exception as e:  # noqa: BLE001
            log.warning("download of %s failed: %s", folder, e)
    return out


# --------------------------------------------------------------------------- #
# the window                                                                   #
# --------------------------------------------------------------------------- #

def replayable_rounds(receipt_dir: Path, snapshot_root: Path) -> list[RoundRef]:
    """Every cached scored receipt whose snapshot is present locally, oldest first."""
    from ...shared.receipt import load_receipt

    out: list[RoundRef] = []
    for p in sorted(receipt_dir.glob("*.json")):
        try:
            r = load_receipt(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("unreadable receipt %s: %s", p.name, e)
            continue
        if r.status != "scored" or r.verdict is None or not any(
                e.role == "king" for e in r.entry_scores):
            continue
        sha, _ = receipt_pool(r)
        try:
            snap = find_snapshot_dir(snapshot_root, sha)
        except ReplayError:
            continue
        out.append(RoundRef(str(r.round_id), int(r.epoch_start_block), p, sha, snap))
    out.sort(key=lambda x: (x.epoch_start_block, x.round_id))
    return out


def needed_snapshot_blocks(receipt_dir: Path, *, newest: int = 0) -> set[int]:
    """Snapshot blocks the cached receipts were scored on; ``newest > 0`` keeps
    only the newest that many distinct blocks (a window never needs older)."""
    from ...shared.receipt import load_receipt

    by_round: list[tuple[int, int]] = []
    for p in receipt_dir.glob("*.json"):
        try:
            r = load_receipt(p.read_text(encoding="utf-8"))
            _, b = receipt_pool(r)
        except Exception:  # noqa: BLE001
            continue
        if b is not None:
            by_round.append((int(r.epoch_start_block), b))
    blocks: list[int] = []
    for _, b in sorted(by_round, reverse=True):
        if b not in blocks:
            blocks.append(b)
    return set(blocks[:newest] if newest > 0 else blocks)


def local_snapshot_blocks(root: Path) -> set[int]:
    return {int(m.group(1)) for d in Path(root).glob("snapshots/*")
            if (d / SHA_MARKER).is_file() and (m := _BLOCK_RE.search(d.name))}


def build_window(refs: list[RoundRef], *, n_a: int, n_b: int) -> RoundWindow | None:
    """The newest ``n_a + n_b`` rounds split A/B; None until at least
    ``n_b + 1`` rounds are replayable (a window needs both pools)."""
    if len(refs) < n_b + 1:
        return None
    tail = refs[-(n_a + n_b):]
    return RoundWindow(a=tuple(tail[:-n_b]), b=tuple(tail[-n_b:]))


def fingerprint(window: RoundWindow, **extra: str) -> str:
    """Everything a score depends on besides the candidate: the rounds (ids +
    snapshot digests) plus ``extra`` (king digest, chain.toml digest, image)."""
    doc = {"rounds": [[r.round_id, r.pool_sha256] for r in window.all],
           **{k: str(v) for k, v in sorted(extra.items())}}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def tree_digest(d: Path) -> str:
    """Content digest of a generator tree (dedup + fingerprint)."""
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."):
            h.update(str(p.relative_to(d)).encode() + b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
