"""Init-baseline: heat SHADOW row + validator duel-side floor.

The 2026-08-26 blind spot: KOTH compares entrants to each other and to the
king, never to "do nothing" — and the raw warm-start init measured BETTER
than every trained entrant on the post-mix windows. Two mechanisms, split
deliberately (owner 2026-08-26):

- Heat ([round] init_gate_mode = "shadow"): the round's init scored on the
  heat slice, published on the standings. NEVER filters the heat.
- Duel ([scoring] init_gate_mode): the authoritative floor — a challenger
  that beat the king but is worse than the init cannot be CROWNED. The
  gift-gate shape: off → shadow → enforce; can only block a dethrone, never
  grant one; king retention untouched. Consensus-relevant.

Off (both defaults) must be byte-identical to the pre-gate code — the first
tests of each half pin that, and the receipt tests pin signature stability.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.eval.koth import KothParams, evaluate_round
from cascade.eval.scoring import WindowScore
from cascade.shared.manifest import HeatResult, _heat_from_json, heat_to_json
from cascade.trainer.contract import RoundSeeds
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner
from tests.unit.test_trainer_round import _FakeBaseTrainer

WS_REF = "metro-v1:trained:hippius:cascade/ckpt-init@sha256:" + "a" * 64

# heat entrant scores: b beats the init (0.30), c and d are worse
SCORES = {"b": 0.25, "c": 0.35, "d": 0.40}
INIT_SCORE = 0.30


# ── heat shadow row ───────────────────────────────────────────────────────────


def _gens():
    return [ResolvedGenerator(hotkey=h, uid=u, ref=f"{h}/gen@sha256:{'1' * 64}",
                              reveal_block=10 + u)
            for u, h in enumerate(["b", "c", "d"])]


def _screen(ckpt_dir, gen, base_seed, block=None):
    if gen is None:                      # the init-baseline probe
        return INIT_SCORE
    return SCORES[gen.hotkey]


def _runner(cfg, tmp_path, monkeypatch, *, mode="off", screen=_screen):
    cfg = replace(cfg, round=replace(
        cfg.round, max_finalists=1, finalists=1, init_gate_mode=mode,
    ))
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(),
                           work_root=tmp_path, use_sandbox=False,
                           screen_fn=screen)
    monkeypatch.setattr(
        runner, "_heat_train",
        lambda challengers, seeds, block, contract, tokens, warm_start_ref=None: [
            (c, tmp_path / c.hotkey, f"digest-{c.hotkey}") for c in challengers
        ],
    )
    monkeypatch.setattr(runner, "_fetch_checkpoint_dir",
                        lambda ref: tmp_path / "init")
    return runner, _gens()


def _heat(runner, gens, *, warm_start=True):
    seeds = RoundSeeds(base_seed=1, generation_seed=2, training_seed=3)
    ws = ((WS_REF, runner.cfg.training.arch_preset) if warm_start else None)
    return runner._run_heat(gens, seeds, block=100, screen_block=100,
                            warm_start=ws)


def test_heat_shadow_off_is_inert(cfg, tmp_path, monkeypatch):
    runner, gens = _runner(cfg, tmp_path, monkeypatch, mode="off")
    winners, heat = _heat(runner, gens)
    assert [c.hotkey for c in winners] == ["b"]
    assert heat.init_baseline is None
    assert "init_baseline" not in heat_to_json(heat)   # frozen JSON shape


def test_default_config_ships_shadow_off(cfg):
    assert cfg.round.init_gate_mode == "off"
    assert cfg.scoring.init_gate_mode == "off"
    assert cfg.scoring.init_gate_tolerance == 0.0


def test_heat_shadow_publishes_baseline_and_never_filters(cfg, tmp_path, monkeypatch):
    runner, gens = _runner(cfg, tmp_path, monkeypatch, mode="shadow")
    winners, heat = _heat(runner, gens)
    assert [c.hotkey for c in winners] == ["b"]        # same advance as off
    assert heat.init_baseline == pytest.approx(INIT_SCORE)
    assert heat_to_json(heat)["init_baseline"] == pytest.approx(INIT_SCORE)
    # c and d are worse than the init — and still merely "screened"
    assert {e.status for e in heat.entrants} == {"advanced", "screened"}


def test_heat_enforce_is_not_a_thing(cfg, tmp_path, monkeypatch):
    # An operator setting "enforce" on [round] gets shadow-off + a warning,
    # never a filtered heat — enforcement lives in [scoring].
    runner, gens = _runner(cfg, tmp_path, monkeypatch, mode="enforce")
    winners, heat = _heat(runner, gens)
    assert [c.hotkey for c in winners] == ["b"]
    assert heat.init_baseline is None


def test_heat_shadow_fails_open(cfg, tmp_path, monkeypatch):
    def screen(ckpt_dir, gen, base_seed, block=None):
        if gen is None:
            raise RuntimeError("pool hiccup")
        return SCORES[gen.hotkey]

    runner, gens = _runner(cfg, tmp_path, monkeypatch, mode="shadow", screen=screen)
    winners, heat = _heat(runner, gens)
    assert [c.hotkey for c in winners] == ["b"]
    assert heat.init_baseline is None


def test_heat_json_round_trips_init_baseline():
    heat = HeatResult(screen_size="toto2-4m", finalists=1, init_baseline=0.2086)
    assert _heat_from_json(heat_to_json(heat)).init_baseline == pytest.approx(0.2086)
    bare = _heat_from_json(heat_to_json(HeatResult(screen_size="x", finalists=1)))
    assert bare.init_baseline is None


# ── duel-side floor ───────────────────────────────────────────────────────────


def _pw(scale: float, n: int = 80) -> list[WindowScore]:
    """Paired per-window scores at a controlled quality ``scale`` (lower is
    better): abs_target fixed per window index, losses scaled — every list is
    paired with every other by construction."""
    win = np.random.default_rng(9000)
    rng = np.random.default_rng(7)
    return [
        WindowScore(
            series_id=f"w{i}",
            mase=float(rng.uniform(0.8, 1.2)) * scale,
            qloss_per_q=rng.uniform(0.3, 0.7, size=9) * scale,
            abs_target=float(win.uniform(5.0, 10.0)),
        )
        for i in range(n)
    ]


def _params(**kw) -> KothParams:
    base = dict(win_margin_start=0.02, win_margin_end=0.02,
                margin_warmup_rounds=0, min_windows=10, bootstrap_B=200,
                bootstrap_alpha=0.05, dethrone_cp=1, min_clusters=0)
    base.update(kw)
    return KothParams(**base)


KING = _pw(1.00)
GOOD_CHAL = _pw(0.80)     # beats the king decisively AND beats a 0.90 init
BAD_CHAL = _pw(0.85)      # beats the king decisively but is worse than 0.82 init


def test_floor_off_rejects_stray_baseline():
    with pytest.raises(ValueError, match="init-baseline gate is off"):
        evaluate_round(KING, GOOD_CHAL, _params(), seed=1,
                       baseline_scores=_pw(0.9))


def test_floor_off_verdict_unchanged():
    r = evaluate_round(KING, GOOD_CHAL, _params(), seed=1)
    assert r.challenger_wins_round is True
    assert r.init_floor_passed is None and r.baseline_geomean is None


def test_floor_shadow_records_but_never_gates():
    r = evaluate_round(KING, BAD_CHAL, _params(init_gate_mode="shadow"),
                       seed=1, baseline_scores=_pw(0.82))
    assert r.challenger_wins_round is True     # shadow: win stands
    assert r.init_floor_passed is False        # …but the floor verdict is recorded
    assert r.baseline_geomean is not None


def test_floor_enforce_blocks_worse_than_init():
    r = evaluate_round(KING, BAD_CHAL, _params(init_gate_mode="enforce"),
                       seed=1, baseline_scores=_pw(0.82))
    assert r.init_floor_passed is False
    assert r.challenger_wins_round is False    # blocked: worse than doing nothing
    # same challenger, gate off: it wins — the floor is what blocked it
    assert evaluate_round(KING, BAD_CHAL, _params(), seed=1).challenger_wins_round


def test_floor_enforce_passes_better_than_init():
    r = evaluate_round(KING, GOOD_CHAL, _params(init_gate_mode="enforce"),
                       seed=1, baseline_scores=_pw(0.90))
    assert r.init_floor_passed is True
    assert r.challenger_wins_round is True


def test_floor_tolerance_widens_the_bar():
    # BAD_CHAL (~0.85) vs init ~0.82: fails strict, passes at 10% slack.
    strict = evaluate_round(KING, BAD_CHAL, _params(init_gate_mode="enforce"),
                            seed=1, baseline_scores=_pw(0.82))
    slack = evaluate_round(
        KING, BAD_CHAL,
        _params(init_gate_mode="enforce", init_gate_tolerance=0.10),
        seed=1, baseline_scores=_pw(0.82))
    assert strict.init_floor_passed is False
    assert slack.init_floor_passed is True and slack.challenger_wins_round


def test_floor_never_grants_a_win():
    # A challenger that does NOT clear the margin stays a loser even when it
    # beats the init soundly — the floor can only block, never grant.
    near_king = _pw(0.995)
    r = evaluate_round(KING, near_king, _params(init_gate_mode="enforce"),
                       seed=1, baseline_scores=_pw(1.5))
    assert r.init_floor_passed is True
    assert r.challenger_wins_round is False


def test_floor_without_baseline_cannot_run():
    r = evaluate_round(KING, GOOD_CHAL, _params(init_gate_mode="enforce"), seed=1)
    assert r.init_floor_passed is None
    assert r.challenger_wins_round is True     # judged as if the gate were off


# ── receipt signature stability ───────────────────────────────────────────────


def test_receipt_params_drop_gate_defaults():
    from cascade.shared.receipt import _PARAMS_DROP_WHEN_DEFAULT

    assert _PARAMS_DROP_WHEN_DEFAULT["init_gate_mode"] == "off"
    assert _PARAMS_DROP_WHEN_DEFAULT["init_gate_tolerance"] == 0.0


def test_verdict_body_drops_absent_floor_fields():
    from cascade.shared.receipt import VerdictRecord, _verdict_body

    base = dict(params={}, bootstrap_seed="1", king_tenure_rounds=0, lcb=0.1,
                margin=0.02, challenger_wins_round=True, inconclusive=False,
                n_windows=10, king_geomean=1.0, chal_geomean=0.9,
                gift_lcb=None, gift_gate_passed=None, dethroned=True,
                note="", king_hotkey="k", king_uid=1)
    off = _verdict_body(VerdictRecord(**base))
    assert "init_baseline_geomean" not in off and "init_floor_passed" not in off
    on = _verdict_body(VerdictRecord(**base, init_baseline_geomean=0.82,
                                     init_floor_passed=False))
    assert on["init_baseline_geomean"] == 0.82
    assert on["init_floor_passed"] is False


# ── loader round-trip (the DEC-CA-0035 sweep's regression class) ─────────────


def test_loader_round_trips_init_gate_fields(tmp_path):
    """Regression: like the DEC-CA-0033 [training] knobs, these fields landed
    with dataclass defaults but no load_chain_config parsing — arming the
    shadow gate in chain.toml silently no-oped."""
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "chain.toml").read_text()
    assert "\nwin_margin_start" in text and "\nheat_train_hours" in text
    patched = text.replace(
        "\nwin_margin_start",
        '\ninit_gate_mode = "shadow"\ninit_gate_tolerance = 0.05\nwin_margin_start',
        1,
    ).replace("\nheat_train_hours",
              '\ninit_gate_mode = "shadow"\nheat_train_hours', 1)
    p = tmp_path / "chain.toml"
    p.write_text(patched)
    c = load_chain_config(p)
    assert c.scoring.init_gate_mode == "shadow"
    assert c.scoring.init_gate_tolerance == 0.05
    assert c.round.init_gate_mode == "shadow"
    # and the koth params carry them through
    kp = c.koth_params()
    assert kp.init_gate_mode == "shadow"
    assert kp.init_gate_tolerance == 0.05


def test_loader_rejects_scoring_init_gate_typo(tmp_path):
    """[scoring] is strict (gift-gate rule: a typo must not silently
    un-enforce the floor); [round] stays warn-and-off at the use site."""
    from pathlib import Path

    import pytest as _pytest

    from cascade.shared.config import load_chain_config

    repo_root = Path(__file__).resolve().parents[2]
    text = (repo_root / "chain.toml").read_text()
    patched = text.replace(
        "\nwin_margin_start", '\ninit_gate_mode = "enforcee"\nwin_margin_start', 1)
    p = tmp_path / "chain.toml"
    p.write_text(patched)
    with _pytest.raises(ValueError, match=r"init_gate_mode='enforcee' invalid"):
        load_chain_config(p)
