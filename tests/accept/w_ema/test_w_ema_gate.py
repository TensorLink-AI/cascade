"""W-EMA on the box (no torch): the scored-file gate ``power_ema_active``
and the image constants, read from pyramid_trainer.py's source.
Contract: README.md."""

from __future__ import annotations

import ast
import tomllib

import pytest

from . import helpers as h

SELECTION = {0: [1], 1: [0, 2]}


@pytest.fixture
def gate():
    ns = h.namespace()
    assert "power_ema_active" in ns, "pyramid_trainer.power_ema_active must stay a module-level def"
    return ns["power_ema_active"]


@pytest.fixture
def kings(tmp_path):
    return {"raw": h.write_king(tmp_path / "raw_king", marked=False),
            "marked": h.write_king(tmp_path / "marked_king", marked=True)}


# ── lane A: king and challenger legs from a warm-start init ─────────────────


def test_warm_start_leg_from_a_raw_init_scores_the_raw_endpoint(gate, kings):
    """A king retrain / challenger (heat or full) of a warm-start seed epoch:
    the promoted init is passed as warm_start_dir, no expert selection."""
    assert gate(kings["raw"], None) is False


def test_warm_start_leg_from_an_ema_marked_init_scores_the_raw_endpoint(gate, kings):
    assert gate(kings["marked"], None) is False


def test_warm_start_leg_given_the_init_as_a_str_path_scores_raw(gate, kings):
    assert gate(str(kings["marked"]), None) is False


# ── lane B: expert legs fine-tune FROM a king ───────────────────────────────


def test_lane_b_leg_from_an_ema_marked_king_still_scores_raw(gate, kings):
    """§8 Tests: "a lane B leg from an EMA-marked king still scores raw"."""
    assert gate(kings["marked"], SELECTION) is False


def test_lane_b_leg_from_a_raw_king_scores_raw(gate, kings):
    assert gate(kings["raw"], SELECTION) is False


# ── random init: today's behaviour, unchanged ───────────────────────────────


def test_random_init_leg_scores_the_raw_endpoint(gate):
    assert gate(None, None) is False


def test_every_leg_kind_scores_raw(gate, kings, tmp_path):
    """No leg, of any lane or init, scores an average: one rule, never
    "best of both" per model (§8)."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{", encoding="utf-8")
    cases = [(None, None)]
    for d in (kings["raw"], kings["marked"], broken, tmp_path / "missing"):
        cases += [(d, None), (d, SELECTION)]
    assert [gate(w, s) for w, s in cases] == [False] * len(cases)


def test_gate_is_a_pure_function_of_the_init(gate, kings):
    """The decision derives only from the leg's init (no clock, no state):
    the same inputs give the same answer, so a continuation re-derives it."""
    before = (kings["marked"] / "config.json").read_bytes()
    first = [gate(kings["marked"], SELECTION), gate(kings["raw"], None)]
    second = [gate(kings["marked"], SELECTION), gate(kings["raw"], None)]
    assert first == second
    assert (kings["marked"] / "config.json").read_bytes() == before


# ── the unscored averages keep being written, under the same names ──────────


def test_the_two_averages_keep_their_file_names():
    ns = h.namespace()
    assert ns["ema_weights_file"](0.05) == h.EMA005
    assert ns["ema_weights_file"](0.10) == h.EMA010


def test_image_ema_widths_are_unchanged():
    """The averages written alongside are the image's two widths, still
    image-only constants (§8: "power EMA isn't a config knob")."""
    assert tuple(h.constants()["POWER_EMA_SIGMA_RELS"]) == (0.05, 0.10)


def test_validators_score_only_the_four_checkpoint_files():
    """Off the critical path: the scored checkpoint (what eval reads and the
    receipt hashes) never includes an EMA file."""
    from cascade.validator.private_worker import CHECKPOINT_FILES

    assert tuple(CHECKPOINT_FILES) == h.SCORED_FILES
    for name in (h.EMA005, h.EMA010, h.RAW):
        assert name not in CHECKPOINT_FILES


# ── the contract is untouched (image-only change) ───────────────────────────


def _recipe_expected() -> dict:
    """The literal ``expected`` dict train_pyramid checks the contract
    against (the recipe pins)."""
    for node in h.trainer_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == "train_pyramid":
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Name) and sub.targets[0].id == "expected"
                        and isinstance(sub.value, ast.Call)):
                    return {kw.arg: ast.literal_eval(kw.value) for kw in sub.value.keywords}
    raise AssertionError("train_pyramid has no literal `expected = dict(...)` recipe check")


def test_the_contract_still_pins_no_power_ema():
    """The recipe check still requires the contract's own
    power_ema_sigma_rels == () : the average can't come from the contract."""
    exp = _recipe_expected()
    assert exp["power_ema_sigma_rels"] == ()
    assert exp["ema_decay"] == 0.0


@pytest.mark.parametrize("name", ["chain.toml", "chain.testnet.toml"])
def test_shipped_contracts_carry_no_power_ema_term(name):
    """contract_digest is untouched: the shipped configs parse to no power
    EMA widths and the hashed payload has no width / eval-index field (the
    existing ``power_ema_scope`` term is the moe_trainer's, not this WP's)."""
    from cascade.shared.config import parse_chain_config
    from cascade.shared.manifest import contract_payload

    cfg = parse_chain_config(tomllib.loads((h.REPO / name).read_text(encoding="utf-8"))).training
    for c in (cfg, cfg.v2_contract):
        assert tuple(c.power_ema_sigma_rels) == () and c.power_ema_eval == -1
        payload = contract_payload(c)
        assert "power_ema_sigma_rels" not in payload and "power_ema_eval" not in payload
