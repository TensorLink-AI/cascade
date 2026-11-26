"""W-IMG (plan rev6.24 (e), owner 2026-11-24): the trainer picks the worker
image per v2 SEED EPOCH.

Owner decision (verbatim): "can we relax need for image to be pinned"; "we
need a system so that we roll all these lanes out, validators only need to
update for major upgrades, not if we tweak training image or comp rules";
"we just log what image pin was used, it's auditable".

This generalises DEC-CA-0067's automatic repin (``[v2] *_after`` pins at
``image_decoupled_from_block``): with ``[v2] image_epoch_from_block`` armed,
a configured ``*_after`` release binds at the next seed-epoch START the
trainer can see far enough ahead (:func:`cascade.shared.era.next_image_epoch_block`)
— never mid-epoch — and the bound block is persisted
(``image_epoch_state.json`` beside the trainer state), so a restart keeps it.
From that block every leg (king and challengers) runs the after release;
earlier eras keep the current pins. Validators check only that one seed
epoch carries one image (:mod:`cascade.validator.era_gate`), never a pin.

Nothing here runs while ``image_epoch_from_block`` is 0 or no ``*_after``
release is configured: the config is returned unchanged (same object).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from ..shared.config import image_digest
from ..shared.era import image_after_armed, image_epoch_from, next_image_epoch_block

log = logging.getLogger(__name__)

STATE_NAME = "image_epoch_state.json"


def load_binding(path: Path | str | None) -> dict | None:
    """The persisted binding ``{"image", "block"}`` or ``None`` (absent).
    A file that exists but does not parse raises (never silently unbound)."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("block"), int):
        raise ValueError(f"image epoch state {p} is malformed")
    return {"image": str(obj.get("image", "")), "block": int(obj["block"])}


def save_binding(path: Path | str | None, binding: dict) -> None:
    if path is None:
        return
    from ..shared.atomic import write_text_atomic

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(p, json.dumps({"image": str(binding["image"]),
                                     "block": int(binding["block"])}, sort_keys=True) + "\n")


def resolve_binding(cfg, warm_eras, now_block: int, stored: dict | None,
                    *, typed_block: int = 0) -> dict | None:
    """The binding in force: ``None`` when W-IMG is off, no after release is
    configured, or no seed epoch is scheduled far enough ahead yet. A typed
    ``[v2] image_after_epoch_block`` (``typed_block``) is the owner override;
    a stored binding for the SAME after image is kept (one-way); a stored
    binding for another image is superseded by a fresh one."""
    if not image_epoch_from(cfg) or not image_after_armed(cfg):
        return None
    after = image_digest(str(cfg.v2.train_image_digest_after))
    if typed_block:
        return {"image": after, "block": int(typed_block)}
    if stored and stored.get("image") == after and int(stored.get("block", 0) or 0):
        return {"image": after, "block": int(stored["block"])}
    blk = next_image_epoch_block(cfg, warm_eras, int(now_block))
    if blk is None:
        return None
    return {"image": after, "block": int(blk)}


def apply_binding(cfg, binding: dict | None):
    """``cfg`` with ``[v2] image_after_epoch_block`` set to the binding (and
    the after image re-bound onto the v2 contract), or ``cfg`` itself."""
    if binding is None:
        return cfg
    block = int(binding["block"])
    if int(getattr(cfg.v2, "image_after_epoch_block", 0) or 0) == block:
        return cfg
    from ..shared.config.chain_config import bind_image_after

    return bind_image_after(replace(cfg, v2=replace(cfg.v2, image_after_epoch_block=block)))


class ImageEpochMixin:
    """The trainer tick: resolve, persist, apply (never raises)."""

    def _image_epoch_state_path(self):
        base = getattr(getattr(self, "activation_store", None), "path", None)
        return None if base is None else Path(base).with_name(STATE_NAME)

    def _image_epoch_tick(self, block: int) -> None:
        try:
            base = self._base_cfg()
            if not image_epoch_from(base) or not image_after_armed(base):
                return
            if "_image_epoch_typed" not in self.__dict__:
                # the owner's typed override, captured before any apply
                self.__dict__["_image_epoch_typed"] = int(
                    getattr(base.v2, "image_after_epoch_block", 0) or 0)
            path = self._image_epoch_state_path()
            stored = load_binding(path)
            warm: list[int] = []
            sched = self.__dict__.get("_rolling_sched")
            if sched is not None:
                warm = list(sched._warm_eras())
            binding = resolve_binding(base, warm, int(block), stored,
                                      typed_block=self.__dict__["_image_epoch_typed"])
            if binding is None:
                return
            if binding != stored:
                save_binding(path, binding)
                log.warning("W-IMG: after image %s… BOUND at seed-epoch start block %d — "
                            "every leg of epochs from it runs that image; earlier eras keep "
                            "the current pins", binding["image"][:19], binding["block"])
            new_cfg = apply_binding(base, binding)
            if new_cfg is not base:
                self._set_base_cfg(new_cfg)
                if self._cfg_base is None:
                    self._point_cfg_consumers(new_cfg)
        except Exception as e:  # noqa: BLE001 — a binding step must never sink a tick
            log.warning("W-IMG image epoch step failed (%s); retrying next poll", e)
