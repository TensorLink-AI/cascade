"""WB14: ``cascade.trainer.declared_rows.parse_declared_rows`` (LB2).

TB13 (index out of range; key outside MoE ordinals 0-6), TB15 (count outside
the per-layer clamp, only on layers with >= 1 row), TB15b (rows on 2 of 7
layers accepted), the total >= 1 rule, shape refusals, the output format and
its round-trip through the worker's own ``--selection-json`` parse."""

from __future__ import annotations

import ast
import json

import pytest

from . import helpers as h
from .helpers import BAD_LAYER, COUNT, MALFORMED, OUT_OF_RANGE


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


# ── the model geometry and the clamps the tests rely on (pass on main) ───────


def test_model_geometry_is_seven_moe_ordinals():
    epl, blocks = h.experts_per_layer(), h.moe_blocks()
    assert epl == (128, 256, 256, 512, 512, 512, 512)
    assert blocks == (1, 2, 3, 5, 6, 7, 8)
    assert len(epl) == len(blocks) == 7


def test_live_fracs_give_the_plan_clamps():
    lo, hi = h.live_fracs()
    assert (lo, hi) == (h.PLAN_MIN_FRAC, h.PLAN_MAX_FRAC)
    for e, want in h.PLAN_CLAMPS.items():
        assert h.clamp(e, lo, hi) == want


# ── module surface ───────────────────────────────────────────────────────────


def test_module_imports_without_torch_and_names_its_surface():
    mod = h.dr()
    assert callable(getattr(mod, "parse_declared_rows", None))
    assert callable(getattr(mod, "model_experts_per_layer", None))
    err = getattr(mod, "DeclaredRowsError", None)
    assert isinstance(err, type) and issubclass(err, ValueError)


def test_model_experts_per_layer_is_read_from_the_model_source():
    mod = h.dr()
    assert tuple(mod.model_experts_per_layer()) == h.experts_per_layer()


def test_module_does_not_hardcode_the_geometry():
    """experts_per_layer comes from pyramid_model.py, not a literal copy."""
    path = h.REPO / "cascade" / "trainer" / "declared_rows.py"
    assert path.is_file(), "cascade/trainer/declared_rows.py is missing"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    epl = h.experts_per_layer()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List)):
            try:
                v = ast.literal_eval(node)
            except ValueError:
                continue
            assert tuple(v) != epl, "experts_per_layer is hardcoded in declared_rows.py"


# ── TB13: out of range / bad layer ───────────────────────────────────────────


def test_tb13_index_out_of_range_on_the_128_expert_ordinal_0():
    e0 = h.experts_per_layer()[0]
    assert e0 == 128
    ok = h.parse({"0": [0, 1, 2, e0 - 1]})
    assert h.nonempty(ok) == {0: [0, 1, 2, e0 - 1]}
    assert h.refusal({"0": [0, 1, 2, e0]}) == OUT_OF_RANGE
    assert h.refusal({"0": [0, 1, 2, 300]}) == OUT_OF_RANGE


@pytest.mark.parametrize("ordinal", range(7))
def test_tb13_range_is_each_ordinals_own_expert_count(ordinal):
    e = h.experts_per_layer()[ordinal]
    lo, _ = h.clamp(e, *h.live_fracs())
    top = h.rows(lo - 1) + [e - 1]
    assert h.nonempty(h.parse({str(ordinal): top})) == {ordinal: sorted(top)}
    assert h.refusal({str(ordinal): h.rows(lo - 1) + [e]}) == OUT_OF_RANGE


def test_tb13_ordinals_not_block_indices():
    """Ordinal 4 is block 6 (512 experts); block 4 is a variate block. An
    index valid only at 512 experts is accepted on ordinal 4 and refused on
    ordinal 1 (256 experts)."""
    assert h.nonempty(h.parse({"4": h.rows(15) + [400]})) == {4: h.rows(15) + [400]}
    assert h.refusal({"1": h.rows(7) + [400]}) == OUT_OF_RANGE


@pytest.mark.parametrize("key", ["7", "8", "-1", "9", "a", "", "1.0", "0x1"])
def test_tb13_key_outside_moe_ordinals_is_bad_layer(key):
    assert h.refusal({key: h.rows(16)}) == BAD_LAYER


def test_tb13_block_index_8_is_bad_layer_even_with_valid_ordinals_alongside():
    assert h.refusal({"0": h.rows(4), "8": h.rows(16)}) == BAD_LAYER


@pytest.mark.parametrize("bad", [[-1, 1, 2, 3], [0, 1, 2, "3"], [0, 1, 2, 3.0],
                                 [0, 2, 3, True], [0, 1, 2, None]])
def test_tb13_non_int_or_negative_index_is_out_of_range(bad):
    assert h.refusal({"0": bad}) == OUT_OF_RANGE


def test_layer_range_follows_the_given_experts_per_layer():
    """Validated against ``experts_per_layer``: three layers ⇒ ordinals 0-2."""
    epl = (64, 64, 64)
    assert h.nonempty(h.parse({"2": h.rows(2)}, epl=epl)) == {2: h.rows(2)}
    assert h.refusal({"3": h.rows(2)}, epl=epl) == BAD_LAYER
    assert h.refusal({"2": [1, 64]}, epl=epl) == OUT_OF_RANGE


# ── TB15: per-layer count clamp, on layers with >= 1 row ─────────────────────


@pytest.mark.parametrize("ordinal", range(7))
def test_tb15_count_clamp_at_each_layer_size(ordinal):
    e = h.experts_per_layer()[ordinal]
    lo, hi = h.clamp(e, *h.live_fracs())
    assert (lo, hi) == h.PLAN_CLAMPS[e]
    for n in (lo, hi):
        assert h.nonempty(h.parse({str(ordinal): h.rows(n)})) == {ordinal: h.rows(n)}
    for n in (lo - 1, hi + 1):
        assert h.refusal({str(ordinal): h.rows(n)}) == COUNT


def test_tb15_one_bad_layer_refuses_the_whole_declaration():
    assert h.refusal({"0": h.rows(4), "3": h.rows(77)}) == COUNT
    assert h.refusal({"0": h.rows(3), "3": h.rows(16)}) == COUNT


def test_tb15_duplicates_are_refused_as_count():
    assert h.refusal({"0": [1, 1, 2, 3, 4]}) == COUNT


def test_clamp_reads_the_fractions_it_is_given():
    """ceil / floor of the configured fractions, not a fixed table:
    0.05 / 0.1 at 128 experts ⇒ 7..12."""
    fr = (0.05, 0.1)
    assert h.clamp(128, *fr) == (7, 12)
    for n in (7, 12):
        assert h.nonempty(h.parse({"0": h.rows(n)}, fracs=fr)) == {0: h.rows(n)}
    for n in (6, 13):
        assert h.refusal({"0": h.rows(n)}, fracs=fr) == COUNT
    # the same 4 rows the live config accepts at 128 are refused here
    assert h.refusal({"0": h.rows(4)}, fracs=fr) == COUNT


# ── TB15b: rows on 2 of 7 layers ─────────────────────────────────────────────


def test_tb15b_two_layers_within_clamp_others_empty_explicitly():
    raw = {"0": h.rows(4), "5": h.rows(76, start=100)}
    raw.update({str(i): [] for i in (1, 2, 3, 4, 6)})
    got = h.parse(raw)
    assert h.nonempty(got) == {0: h.rows(4), 5: h.rows(76, start=100)}


def test_tb15b_two_layers_within_clamp_others_absent():
    got = h.parse({"1": h.rows(38, start=200), "6": h.rows(16, start=496)})
    assert h.nonempty(got) == {1: h.rows(38, start=200), 6: h.rows(16, start=496)}


# ── at least one row in total (TB21: none declared ⇒ ESFT, not this parser) ──


@pytest.mark.parametrize("raw", [{}, {"0": []}, {str(i): [] for i in range(7)}, "{}"])
def test_no_rows_in_total_is_refused_as_count(raw):
    assert h.refusal(raw) == COUNT


# ── shape ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", "5", "null", '"0"',
                                 {"0": 5}, {"0": "0,1,2,3"}, {"0": {"1": 1}},
                                 {"0": None}])
def test_malformed_declarations_are_refused(raw):
    assert h.refusal(raw) == MALFORMED


# ── input and output forms ───────────────────────────────────────────────────


def test_json_text_str_keys_and_int_keys_parse_alike():
    sel = {"0": [7, 3, 1, 2], "3": h.rows(16, start=10)}
    want = {0: [1, 2, 3, 7], 3: h.rows(16, start=10)}
    assert h.nonempty(h.parse(json.dumps(sel))) == want
    assert h.nonempty(h.parse(sel)) == want
    assert h.nonempty(h.parse({0: [7, 3, 1, 2], 3: h.rows(16, start=10)})) == want


def test_output_is_sorted_int_rows_keyed_by_int_ordinal():
    got = h.parse({"2": [250, 0, 9, 8, 7, 6, 5, 4]})
    assert isinstance(got, dict)
    assert h.nonempty(got) == {2: [0, 4, 5, 6, 7, 8, 9, 250]}
    assert all(type(x) is int for v in got.values() for x in v)


def test_output_round_trips_through_the_worker_selection_parse():
    """The parsed rows, sent as ``--selection-json`` (JSON object), come back
    from the worker's own parse (worker.py:226-227) unchanged: the keys are
    MoE ordinals exactly as the worker reads them."""
    raw = {"0": h.rows(4, start=124), "4": h.rows(16, start=300)}
    got = h.parse(raw)
    back = h.worker_parse_selection(json.dumps(got))
    assert back == got
    assert h.nonempty(back) == {0: h.rows(4, start=124), 4: h.rows(16, start=300)}


def test_input_is_not_mutated():
    raw = {"0": [3, 2, 1, 0]}
    h.parse(raw)
    assert raw == {"0": [3, 2, 1, 0]}
