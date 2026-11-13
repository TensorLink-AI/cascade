"""The pre-decision helper is the EXACT inverse of the typed decided blocks.

Scenario fixtures (and the session ``cfg``) run on ``predecision_text()``; if
a later release types another decided block without adding it to
``UNTYPED``, those fixtures would silently run on a partly typed file. This
pins that the helper differs from the shipped file on exactly ``UNTYPED`` and
on nothing else.
"""
from __future__ import annotations

import tomllib

from tests.unit.predecision_main import REPO, UNTYPED, predecision_text


def _flat(doc, prefix=""):
    out = {}
    for k, v in doc.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def test_predecision_main_is_the_exact_inverse():
    shipped = _flat(tomllib.loads((REPO / "chain.toml").read_text()))
    before = _flat(tomllib.loads(predecision_text()))
    assert shipped.keys() == before.keys()
    differ = {k for k in shipped if shipped[k] != before[k]}
    want = {f"{section}.{key}" for section, key, _ in UNTYPED}
    # every untyped line really is typed in the release, and nothing else moved
    assert differ == want
    for section, key, value in UNTYPED:
        assert before[f"{section}.{key}"] == tomllib.loads(f"x = {value}")["x"]
    # the decided values themselves (a bad edit to either side fails here)
    assert {k: shipped[k] for k in want} == {
        "round.epoch_blocks": 900, "round.epoch_blocks_prev": 3600,
        "round.epoch_activation_block": 9136800, "round.rolling_from_block": 9136800,
        "scoring.cohort_maxt_increment_from_block": 9136800,
        "scoring.era_king_from_block": 9136800, "scoring.tenure_blocks_from_block": 9136800,
        "v2.commit_from_block": 9395601, "v2.from_block": 9399600,
    }
