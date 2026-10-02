"""``cascade gauntlet selftest`` — one full gauntlet cycle on synthetic rounds.

No GPU, no network, no LLM: rounds are fabricated (snapshot folders + signed-
shape receipts whose windows really are the verdict ladder draw), compute is
faked from each tree's ``config.json`` ``quality``, and workers are scripted.
Everything else — the window, epochs, the king baseline, σ and margins, every
stage's pairing and bootstrap, the population, G4, G4.5 and G5 in approval
mode — is the real code path. The image smoke test runs it; so do the unit
tests (``tests/unit/test_harness.py``).
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from .workers import Outcome

BLOCK_HASH = "0x" + "ab" * 32


def selftest_chain(cfg):
    """``cfg`` with a small, early-gated ladder so a handful of series fills it."""
    ev = replace(cfg.eval, scored_horizons=(64, 256, 720), scored_from_block=1000,
                 n_windows=30)
    return replace(cfg, eval=ev)


def make_generator(d: Path, quality: float = 1.0, speed: float = 1.0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "generator.py").write_text("# selftest generator\n")
    (d / "config.json").write_text(json.dumps({"quality": quality, "speed": speed}))
    (d / "requirements.txt").write_text("")
    return d


def make_round(cfg, root: Path, i: int):
    """One replayable round: a revealed snapshot folder + its scored receipt.
    Returns ``(receipt, king_scores)``."""
    from ...eval.koth import evaluate_round
    from ...eval.scoring import WindowScore
    from ...shared.manifest import TrainedEntry, TrainingManifest, contract_digest
    from ...shared.manifest import format_trained_pointer as ptr
    from ...shared.receipt import (
        EntryScores,
        EvalContext,
        VerdictRecord,
        WindowScoreRecord,
        build_receipt,
    )
    from ...trainer.contract import RoundSeeds
    from ...validator import state as state_mod
    from ..replay import SHA_MARKER, verdict_windows

    block = 5000 + 100 * i
    sha = f"{i:02x}" * 32
    snap = root / "snapshots" / f"2026-09-{10 + i:02d}-block-{block}"
    snap.mkdir(parents=True)
    rng = np.random.default_rng(i)
    md = {}
    for dom in ("energy", "nature", "web"):
        for j in range(8):
            sid = f"{dom}{j:02d}"
            np.save(snap / f"{sid}.npy", rng.normal(size=4200).cumsum())
            md[sid] = {"domain": dom, "source": f"{dom}-feed"}
    (snap / "metadata.json").write_text(json.dumps(md))
    (snap / SHA_MARKER).write_text(sha + "\n")
    base_seed = 1000 + i
    windows = verdict_windows(cfg, snap, base_seed, block)
    king = [WindowScore(w.series_id, float(rng.uniform(0.5, 1.5)), rng.uniform(0.1, 1.0, 9),
                        float(rng.uniform(5, 10)), source=f"src{k % 12}")
            for k, w in enumerate(windows)]
    params = replace(cfg.koth_params(), min_windows=10, min_clusters=0)
    size = cfg.training.arch_preset
    entries = [
        TrainedEntry("king_hk", 0, "king", "a/gen@sha256:" + "a" * 64,
                     ptr("c/king@sha256:" + "c" * 64), "3" * 64, block + 50, size=size),
        TrainedEntry("chal_hk", 1, "challenger", "b/gen@sha256:" + "b" * 64,
                     ptr("c/chal@sha256:" + "d" * 64), "4" * 64, block + 50, size=size),
    ]
    manifest = TrainingManifest(
        round_id=str(base_seed), created_block=block + 50,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest, eval_dataset=cfg.eval.eval_dataset,
        entries=entries, signature="00ff" * 16, eval_pool_sha256=sha,
        eval_pool_key=f"pool/snapshots/block-{block}.tar")
    chal = [WindowScore(s.series_id, s.mase * 1.01, s.qloss_per_q * 1.01, s.abs_target,
                        source=s.source) for s in king]
    res = evaluate_round(king, chal, params, seed=base_seed, king_tenure_rounds=0)
    tr = state_mod.apply_round(state_mod.genesis("king_hk", 0), challenger_hotkey="chal_hk",
                               challenger_uid=1, result=res, dethrone_cp=params.dethrone_cp,
                               keep_former_kings=1)
    receipt = build_receipt(
        round_id=str(base_seed), status="scored", epoch_start_block=block,
        epoch_block_hash=BLOCK_HASH, base_seed=base_seed,
        seeds=RoundSeeds.derive(base_seed, cfg.training), manifest=manifest,
        eval_context=EvalContext(pool_ref="x", pool_digest=sha,
                                 window_ids=tuple(w.series_id for w in windows),
                                 n_windows=len(windows), num_samples=cfg.eval.num_samples),
        entry_scores=(
            EntryScores("king", size, "king_hk", 0,
                        tuple(WindowScoreRecord.from_score(s) for s in king)),
            EntryScores("challenger", size, "chal_hk", 1,
                        tuple(WindowScoreRecord.from_score(s) for s in chal))),
        verdict=VerdictRecord.from_round(res, tr, params=params, bootstrap_seed=base_seed,
                                         king_tenure_rounds=0))
    return receipt, king


class FakeCompute:
    """Scores derive from the tree's config.json ``quality`` (higher = better)
    scaled onto each round's real receipt king scores, so the gauntlet's
    arithmetic, pairing and bootstraps all run for real."""

    capacity = 4

    def __init__(self, king_by_receipt: dict[str, list]):
        from .executor import SpendLedger

        self.king = king_by_receipt
        self.jobs: list[dict] = []
        self.ledger = SpendLedger(Path(tempfile.gettempdir()) / "selftest-never-written.json")

    @staticmethod
    def _q(gen: str) -> tuple[float, float]:
        c = json.loads((Path(gen) / "config.json").read_text())
        return float(c.get("quality", 1.0)), float(c.get("speed", 1.0))

    @staticmethod
    def _scaled(base: list, f: float) -> list:
        from ...eval.scoring import WindowScore
        return [WindowScore(s.series_id, s.mase * f, s.qloss_per_q * f, s.abs_target,
                            source=s.source) for s in base]

    def run(self, spec, *, job_id=None, est_hours=1.0):
        from ...eval.scoring import global_geomean
        from .jobs import scores_to_json

        self.jobs.append(spec)
        kind, inp, par = spec["kind"], spec["inputs"], spec.get("params", {})
        if kind == "throughput":
            _, speed = self._q(inp["gen"])
            return {"verify_ok": True, "gen": {"points_per_sec": 1000 * speed},
                    "king": {"points_per_sec": 1000.0}}
        if kind == "replay":
            base = self.king[inp["receipt"]]
            q, _ = self._q(inp["gen"])
            f = (1.0 / q) * (1.0 + 0.001 * int(par.get("seed_salt", 0)))
            scores = self._scaled(base, f)
            out = {"geomean": global_geomean(scores), "king_geomean": global_geomean(base),
                   "scores": scores_to_json(scores), "full_budget": par["train_hours"] is None}
            if par["train_hours"] is None:
                out["verdict"] = {"wins": q > 1.05, "lcb": 1 - f, "margin": 0.01}
                if spec.get("outputs", {}).get("checkpoint"):
                    Path(spec["outputs"]["checkpoint"]).mkdir(parents=True, exist_ok=True)
            return out
        if kind == "eval_pool":
            base = next(iter(self.king.values()))
            return {"scores": {"king": scores_to_json(base),
                               "cand": scores_to_json(self._scaled(base, 1 / 1.6))}}
        raise ValueError(kind)

    def reap(self):
        pass

    def close(self):
        pass


class ScriptedWorkers:
    """Each proposal sets ``(quality, speed)`` to the next scripted value."""

    def __init__(self, script):
        self.script = list(script)

    def run(self, proposals):
        out = []
        for p in proposals:
            q, speed = self.script.pop(0) if self.script else (0.9, 1.0)
            (p.tree / "config.json").write_text(json.dumps({"quality": q, "speed": speed}))
            out.append(Outcome(p.id, ok=True, note=f"q={q} speed={speed}"))
        return out


def build_selftest(root: Path, cfg, *, script, submit_mode="approval", n_rounds=5,
                   submit_runner=None, **sections):
    """A wired :class:`Gauntlet` on ``n_rounds`` synthetic rounds under ``root``.
    Returns ``(gauntlet, compute, submitter)``."""
    from ...shared.receipt import dump_receipt
    from . import config as hconfig
    from .gauntlet import Gauntlet
    from .submit import Submitter

    made = [make_round(cfg, root / "pool", i) for i in range(n_rounds)]
    rows = [{"round_id": r.round_id, "status": "scored", "epoch_start_block":
             r.epoch_start_block, "receipt_key": f"k/{r.round_id}"} for r, _ in made]
    texts = {f"k/{r.round_id}": dump_receipt(r) for r, _ in made}
    king = make_generator(root / "king")
    h = hconfig.HarnessConfig(
        workdir=root / "gw", start_dir=king, king_dir=king,
        rounds=hconfig.RoundsConfig(snapshot_root=root / "pool", sync_reveals=False,
                                    n_a=3, n_b=2))
    base = {"search": {"proposals_per_cycle": len(script), "population_k": 2},
            "stages": {"g4_every_cycles": 1, "g4_rounds": 3, "g4_min_wins": 2,
                       "g2_explore_frac": 0.0},
            "submit": {"mode": submit_mode, "intake": "https://x", "wallet_name": "w",
                       "hotkeys": ("hk1",)}}
    for name, kv in sections.items():
        base[name] = {**base.get(name, {}), **kv}
    h = hconfig.with_overrides(h, **base)
    gw = Path(h.workdir)
    compute = FakeCompute({str(gw / "receipts" / f"{r.round_id}.json"): k for r, k in made})
    sub = Submitter(h.submit, gw, runner=submit_runner or (lambda argv: 0))

    def build_pool(out, day, sources):
        out.mkdir(parents=True)
        (out / "metadata.json").write_text("{}")

    g = Gauntlet(h, chain_cfg=cfg, executor=compute, workers=ScriptedWorkers(script),
                 submitter=sub, index_fetch=lambda: {"rounds": rows},
                 text_fetch=texts.__getitem__, pool_builder=build_pool,
                 verify_fn=lambda tree, chain: (True, ""))
    return g, compute, sub


def run_selftest() -> int:
    from ...shared.config import load_chain_config

    cfg = selftest_chain(load_chain_config())
    with tempfile.TemporaryDirectory(prefix="gauntlet-selftest-") as td:
        g, compute, sub = build_selftest(Path(td), cfg, script=[
            (1.6, 1.0), (1.6, 0.5), (0.8, 1.0), (1.0005, 1.0)])
        outcome = g.cycle()
        metas = {m["id"]: m for m in g.all_metas()}
        checks = {
            "cycle ran": outcome == "ran",
            "epoch + baseline": g.epoch == 1 and g.state.get("king_ready"),
            "slow candidate died at G1": metas["c00002"].get("reason", "").startswith("G1"),
            "worse candidate died at G2": metas["c00003"].get("reason", "").startswith("G2"),
            "noise-level candidate died at G2":
                metas["c00004"].get("reason", "").startswith("G2"),
            "winner passed G3, G4, G4.5":
                all(metas["c00001"]["stages"].get(s, {}).get("pass")
                    for s in ("G3", "G4", "G4.5")),
            "winner pending approval": [p["id"] for p in sub.pending()] == ["c00001"],
        }
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    ok = all(checks.values())
    print(f"gauntlet selftest {'OK' if ok else 'FAILED'} ({len(compute.jobs)} fake jobs)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run_selftest())
