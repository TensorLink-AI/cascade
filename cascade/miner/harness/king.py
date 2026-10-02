"""The live king: the generator the gauntlet measures against, kept current.

``king_source = "live"`` resolves the king the way the TRAINER does
(``TrainerRunner._receipt_king``, mirrored by ``scripts/champions_archive.
king_from_receipt``): the anchor validator's latest SIGNED receipt
(``[manifest] validator_hotkey``), status ``scored``, ``verdict.king_hotkey``
(a dethrone's winner the moment it is scored), never a forfeited hotkey; the
code is that hotkey's entry in the receipt's signed manifest.

Like the trainer's sticky last-known king, anything unreadable or unverifiable
keeps the current king instead of guessing. A new king is fetched once into
``<workdir>/king/<digest>/`` and changes the epoch fingerprint, so the next
refresh (at most hourly) re-baselines against it.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("cascade.miner.harness.king")


def king_from_receipt(doc: dict, *, forfeited: frozenset = frozenset()) -> dict | None:
    """``{hotkey, gen_ref, digest, round_id}`` of the king one receipt names."""
    if str(doc.get("status")) != "scored":
        return None
    v = doc.get("verdict") or {}
    hk = str(v.get("king_hotkey") or "")
    if not hk or hk in forfeited:
        return None
    entries = [e for e in ((doc.get("manifest") or {}).get("entries") or [])
               if str(e.get("miner_hotkey")) == hk]
    dethroned = bool(v.get("dethroned"))
    entries.sort(key=lambda e: e.get("role") != ("challenger" if dethroned else "king"))
    ref = str((entries[0] if entries else {}).get("gen_ref") or "")
    if "@sha256:" not in ref:
        return None
    return {"hotkey": hk, "gen_ref": ref, "digest": ref.split("@sha256:")[-1][:64],
            "round_id": str(doc.get("round_id") or "")}


def _anchor_receipt_text(chain_cfg) -> str:
    from ...audit.main import _fetch_text
    from ...shared.hippius import receipt_latest_key

    anchor = str(chain_cfg.manifest.validator_hotkey or "")
    if not anchor:
        raise ValueError("[manifest] validator_hotkey is unset: no receipt anchor")
    return _fetch_text(chain_cfg, receipt_latest_key(anchor))


def _verify(chain_cfg, text: str) -> bool:
    from ...shared.receipt import load_receipt, verify_receipt_signature

    return verify_receipt_signature(load_receipt(text),
                                    str(chain_cfg.manifest.validator_hotkey))


def _fetch(chain_cfg, ref: str, out: Path) -> Path:
    from ..cli import fetch_generator
    return fetch_generator(ref, out, chain_cfg)


def refresh_live_king(chain_cfg, root: Path, current: dict | None, *,
                      receipt_text: Callable[[], str] | None = None,
                      verify: Callable[[str], bool] | None = None,
                      fetch: Callable[[str, Path], Path] | None = None) -> dict | None:
    """The live king record ``{…, dir}``; ``current`` when nothing usable is
    found (sticky). Never raises."""
    try:
        text = (receipt_text or (lambda: _anchor_receipt_text(chain_cfg)))()
        if not (verify or (lambda t: _verify(chain_cfg, t)))(text):
            log.warning("anchor receipt signature does not verify; keeping the current king")
            return current
        forfeited = frozenset(getattr(chain_cfg.scoring, "forfeit_hotkeys", ()) or ())
        king = king_from_receipt(json.loads(text), forfeited=forfeited)
    except Exception as e:  # noqa: BLE001
        log.warning("live king unavailable (%s); keeping the current king", e)
        return current
    if king is None:
        return current
    if current and current.get("digest") == king["digest"] and Path(current["dir"]).is_dir():
        return current
    dest = Path(root) / king["digest"]
    if not (dest / "generator.py").is_file():
        tmp = Path(root) / f".{king['digest']}.part"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            got = Path((fetch or (lambda r, o: _fetch(chain_cfg, r, o)))(king["gen_ref"], tmp))
        except Exception as e:  # noqa: BLE001
            log.warning("fetching king %s failed (%s); keeping the current king",
                        king["gen_ref"], e)
            return current
        shutil.rmtree(dest, ignore_errors=True)
        got.rename(dest)
    log.info("live king: %s (round %s)", king["gen_ref"], king["round_id"])
    return {**king, "dir": str(dest)}
