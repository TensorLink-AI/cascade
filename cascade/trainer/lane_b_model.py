"""Lane B checkpoint model work — rebase, merge, stage — over a ``fetch`` and
an ``upload``, the SAME code wherever it runs.

The lane B driver never touches checkpoint bytes itself: it asks a *model*
object. Two implementations exist and both run the functions below:

* :class:`LaneBModel` over the box's ``ops.fetch`` / ``ops.upload`` /
  ``ops.bench_rows`` (tests, dev boxes with ``CASCADE_LANE_B_LOCAL_EVAL=1``,
  and every driver ops object that does not offer ``lane_b_model``);
* :class:`cascade.trainer.lane_b_offload.PodLaneBModel` — the production
  path: each call is one eval-pod step whose worker builds a
  :class:`LaneBModel` over the pod's own Hub pull and Hub push, so the bytes
  a rebase / merge writes are produced by this module's code on the pod's CPU
  (float32 CPU arithmetic, exactly what the box did) and go to the Hub from
  the pod. Nothing but pointers, digests and small JSON comes back.

Torch is imported inside the methods only (never on the box's pod path).
:class:`MergeRefused` is the torch-free face of ``moe_model.MergeError``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("cascade.trainer.lane_b_model")

CODE_FILES = ("config.json", "model.py", "forecast_wrapper.py")


class MergeRefused(ValueError):
    """A rebase / merge refused by ``moe_model.MergeError`` (the leg touched
    other rows, a non-finite delta under the guard, overlapping merges).
    Torch-free, so the box can catch a pod's refusal."""


@dataclass(frozen=True)
class RebaseResult:
    """One rebased leg: ``actuals`` (the limits leg's ``lane_b_leg`` record,
    ``{}`` when not read / absent), ``pointer`` (the uploaded rebased
    checkpoint; None ⇒ not rebased: a limits leg without its record) and the
    ``bench`` handle :meth:`LaneBModel.bench` takes."""

    actuals: dict
    pointer: str | None
    bench: object = None


def _merge_error():
    from .moe_model import MergeError

    return MergeError


class LaneBModel:
    """Model work over ``fetch(pointer) -> dir`` and ``upload(dir, repo) ->
    pointer``; ``bench_rows(dir)`` (optional) gives a checkpoint's public rows.
    One instance per settlement / era init: loaded states are cached on it."""

    def __init__(self, fetch: Callable[[str], Path], upload: Callable[[Path, str], str],
                 bench_rows: Callable[[Path], dict | None] | None = None):
        self._fetch, self._upload, self._bench_rows = fetch, upload, bench_rows
        self._states: dict[str, dict] = {}

    # ── bench rows ───────────────────────────────────────────────────────────

    def rows(self, pointer: str) -> dict | None:
        """The public rows of ``pointer`` (None = not benched yet)."""
        if self._bench_rows is None:
            return None
        return self._bench_rows(self._fetch(pointer))

    def bench(self, handle) -> dict | None:
        """The public rows of a rebased / staged checkpoint (its handle)."""
        if self._bench_rows is None or handle is None:
            return None
        return self._bench_rows(Path(handle))

    # ── settlement ───────────────────────────────────────────────────────────

    def rebase(self, *, king_pointer: str, base_pointer: str, leg_pointer: str,
               selection: dict, out_dir: Path, repo: str, refuse_nonfinite: bool,
               limits: bool) -> RebaseResult:
        """The leg carried onto ``king_pointer`` (``settle_leg``) and uploaded
        as ``repo``. ``limits`` reads the leg's ``lane_b_leg`` record first; a
        limits leg without it is not rebased (``pointer`` None). Raises
        :class:`MergeRefused` on a refused rebase."""
        # through the driver module's names (``lane_b_driver.settle_leg`` is
        # what the settlement always called; tests stub it there)
        from . import lane_b_driver as drv

        king_dir = self._fetch(king_pointer)
        leg_dir = self._fetch(leg_pointer)
        actual = drv.leg_actuals(leg_dir) if limits else {}
        if limits and not actual:
            return RebaseResult({}, None, None)
        try:
            rebased = drv.settle_leg(king_dir, self._fetch(base_pointer), leg_dir, selection,
                                 Path(out_dir), refuse_nonfinite=refuse_nonfinite)
        except _merge_error() as e:
            raise MergeRefused(str(e)) from e
        return RebaseResult(actual, self._upload(rebased, repo), rebased)

    # ── era init ─────────────────────────────────────────────────────────────

    def _load(self, p: str) -> dict:
        if p not in self._states:
            from safetensors.torch import load_file

            self._states[p] = load_file(str(self._fetch(p) / "weights.safetensors"))
        return self._states[p]

    def _names(self, init_pointer: str):
        from .expert_lane import _moe_names

        return _moe_names(self._fetch(init_pointer))

    def finite(self, init_pointer: str, merges: list[dict]) -> list[bool]:
        """Per merge: every delta (``rebased − king`` on its rows) finite."""
        import torch

        from .expert_lane import merge_deltas

        names = self._names(init_pointer)
        return [not any(not bool(torch.isfinite(d).all())
                        for _, _, d in merge_deltas([m], self._load, names)) for m in merges]

    def stage(self, init_pointer: str, merges: list[dict], *, out_dir: Path, repo: str,
              strict: bool) -> tuple[str, object]:
        """An acceptance stage: ``init + merges`` (``apply_merges``) on the
        scored weights with the init's code files, uploaded as ``repo``.
        ``(pointer, bench handle)``; :class:`MergeRefused` when refused."""
        from .ckpt_digest import save_tensors_hashed
        from .expert_lane import apply_merges

        init_dir = self._fetch(init_pointer)
        names = self._names(init_pointer)
        stage = Path(out_dir)
        stage.mkdir(parents=True, exist_ok=True)
        for file in CODE_FILES:
            shutil.copyfile(init_dir / file, stage / file)
        try:
            state = apply_merges(self._load(init_pointer), merges, self._load, names,
                                 strict=strict)
        except _merge_error() as e:
            raise MergeRefused(str(e)) from e
        save_tensors_hashed(state, stage / "weights.safetensors")
        return self._upload(stage, repo), stage

    def build(self, init_pointer: str, chosen: list[dict], *, out_dir: Path, repo: str,
              strict: bool, era_index: int) -> str:
        """The merged era init (DEC-CA-0050 "state at merge"): the scored
        weights get ``apply_merges``'s bytes, the same deltas go onto the
        stable weights and every power-EMA shadow, the merged rows' optimizer
        moments are reset; uploaded as ``repo``. A refusing ``apply_merges``
        raises its ``MergeError`` (the pod path: :class:`MergeRefused`)."""
        from safetensors.torch import load_file

        from .ckpt_digest import save_tensors_hashed
        from .expert_lane import (
            apply_merge_deltas,
            apply_merges,
            merge_deltas,
            merge_power_ema,
            reset_optimizer_rows,
        )
        from .moe_trainer import POWER_EMA_FILE
        from .toto2_trainer import OPTIM_STATE_FILE, STABLE_WEIGHTS_FILE

        init_dir = self._fetch(init_pointer)
        names = self._names(init_pointer)
        state = apply_merges(load_file(str(init_dir / "weights.safetensors")), chosen,
                             self._load, names, strict=strict)
        deltas = merge_deltas(chosen, self._load, names)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in CODE_FILES:
            shutil.copyfile(init_dir / f, out / f)
        save_tensors_hashed(state, out / "weights.safetensors")
        # the lineage files follow the same merge (see expert_lane "state at merge")
        if (init_dir / STABLE_WEIGHTS_FILE).is_file():
            save_tensors_hashed(apply_merge_deltas(
                load_file(str(init_dir / STABLE_WEIGHTS_FILE)), deltas),
                out / STABLE_WEIGHTS_FILE)
        if (init_dir / OPTIM_STATE_FILE).is_file():
            flat, n = reset_optimizer_rows(load_file(str(init_dir / OPTIM_STATE_FILE)),
                                           chosen, names)
            save_tensors_hashed(flat, out / OPTIM_STATE_FILE)
            log.info("lane B: era %d init — optimizer moments reset on %d merged expert "
                     "tensor(s)", era_index, n)
        if (init_dir / POWER_EMA_FILE).is_file():
            save_tensors_hashed(merge_power_ema(load_file(str(init_dir / POWER_EMA_FILE)),
                                                deltas), out / POWER_EMA_FILE)
        return self._upload(out, repo)
