"""R6x / F6 (plan rev6.18 B9 main side, B12): the release chain.toml types the
S0a main <- deployed values and both owner keys; the deployed-file edits in
docs/notes/S0A_DEPLOYED_EDITS.md are exactly what turns `cascade-audit
decided` green; the DEC-CA-0034 init gate is retired behind a typed
`[scoring] init_gate_retired_from_block` and an archived shadow-era receipt
still round-trips byte-identically and verifies.

Nothing heavy: no torch, no network, no chain. The deployed host file is read
only when LANED_S0A_DEPLOYED names it.
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NOTES = REPO / "docs" / "notes" / "S0A_DEPLOYED_EDITS.md"
RECEIPT = HERE / "fixtures" / "receipt_shadow_init_gate.json"
# The fixture: mainnet round 9550553995117568107 (epoch 9560700, init gate in
# shadow), entry rows trimmed, re-signed by the //Alice dev key with the
# pre-F6 code.
ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
SHADOW_EPOCH = 9_560_700

# B9 (owner 2026-11-23): main <- deployed live values, and both owner keys.
B9_MAIN = {
    ("round", "funded_sku_wall_measured_image"):
        "sha256:5d41d94bc0c0d3ae3dfd042b0cc7e4d900c7d1f48173808436c12bf764a67641",
    ("training", "budget_denomination_after"): "points+mv20",
    ("training", "budget_denomination_after_block"): 9_165_600,
    ("static_guard", "packed_sources"): "reject",
    ("static_guard", "packed_sources_from_block"): 9_169_200,
}
B9_WALLS = {"8xH100": 13_600, "8xH200": 12_270}
DECIDED_VOTES = {("v2", "prelaunch_burn_from_block"): 9_486_900,
                 ("v2", "image_decoupled_from_block"): 9_493_200}


def _main_text() -> str:
    return (REPO / "chain.toml").read_text(encoding="utf-8")


def _load(text: str, tmp_path: Path, name: str = "chain.toml"):
    from cascade.shared.config import load_chain_config

    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return load_chain_config(p)


def _decided_fails(cfg) -> list[str]:
    from cascade.shared import decided as D

    record = D.load_decided(D.default_decided_path())
    return [f"{f.as_dict()}" for f in D.drift_report(cfg, record) if f.fails]


def _edits() -> list[tuple[str, str, object]]:
    """(table, key, value) of the notes file's ```toml block."""
    assert NOTES.is_file(), f"{NOTES.relative_to(REPO)} is missing"
    text = NOTES.read_text(encoding="utf-8")
    blocks = re.findall(r"```toml\n(.*?)```", text, flags=re.S)
    assert len(blocks) == 1, f"{NOTES.name}: want exactly one ```toml block, got {len(blocks)}"
    parsed = tomllib.loads(blocks[0])
    return [(t, k, v) for t, kv in parsed.items() for k, v in kv.items()]


def _set_in_table(text: str, table: str, key: str, value: object) -> str:
    """Replace ``key = …`` inside ``[table]`` or insert it under the header."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"[{table}]")
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r"^\s*\[", lines[i])), len(lines))
    rendered = f"{key} = {value!r}\n" if isinstance(value, int) else f'{key} = "{value}"\n'
    for i in range(start + 1, end):
        if re.match(rf"^{re.escape(key)}\s*=", lines[i]):
            lines[i] = rendered
            return "".join(lines)
    lines.insert(start + 1, rendered)
    return "".join(lines)


def _drop_in_table(text: str, table: str, key: str) -> str:
    out, cur = [], ""
    for ln in text.splitlines(keepends=True):
        m = re.match(r"^\s*\[([^\]]+)\]", ln)
        if m:
            cur = m.group(1).strip()
        if cur == table and re.match(rf"^{re.escape(key)}\s*=", ln):
            continue
        out.append(ln)
    return "".join(out)


# ── B9 main side ─────────────────────────────────────────────────────────────


def test_main_types_the_b9_live_values():
    raw = tomllib.loads(_main_text())
    wrong = {f"{t}.{k}": (raw.get(t, {}).get(k), v)
             for (t, k), v in B9_MAIN.items() if raw.get(t, {}).get(k) != v}
    walls = raw["round"]["funded_sku_wall_seconds"]
    wrong |= {f"round.funded_sku_wall_seconds.{s}": (walls.get(s), v)
              for s, v in B9_WALLS.items() if walls.get(s) != v}
    assert not wrong, f"main does not carry the B9 live values (main, want): {wrong}"


def test_main_b9_values_take_effect_and_decided_is_green(tmp_path):
    from cascade.shared.config import funded_sku_wall_for

    cfg = _load(_main_text(), tmp_path)
    assert cfg.static_guard.packed_sources_at(9_169_199) == "scan"
    assert cfg.static_guard.packed_sources_at(9_169_200) == "reject"
    assert cfg.training.budget_denomination_after == "points+mv20"
    assert cfg.training.budget_denomination_after_block == 9_165_600
    for sku, wall in B9_WALLS.items():
        assert funded_sku_wall_for(cfg.round.funded_sku_wall_seconds, sku, 1) == float(wall)
    assert cfg.round.funded_sku_wall_measured_image == B9_MAIN[
        ("round", "funded_sku_wall_measured_image")]
    # `cascade-audit decided` against the release file: no failing finding
    assert not _decided_fails(cfg), _decided_fails(cfg)


# ── the deployed-file edits (docs/notes/S0A_DEPLOYED_EDITS.md) ───────────────


def test_notes_list_exactly_the_decided_votes_and_the_retire_block():
    edits = {(t, k): v for t, k, v in _edits()}
    main = tomllib.loads(_main_text())
    want = dict(DECIDED_VOTES)
    want[("scoring", "init_gate_retired_from_block")] = (
        main["scoring"].get("init_gate_retired_from_block"))
    assert edits == want, f"notes edits {edits} != {want}"


def test_notes_edits_turn_a_stale_deployed_file_green(tmp_path):
    """Main with the two votes untyped (the deployed state) and no retire
    block fails `decided`; the notes' lines make it pass and give the same
    init-gate window as main."""
    edits = _edits()
    stale = _set_in_table(_main_text(), "v2", "image_decoupled_from_block", 0)
    stale = _drop_in_table(stale, "v2", "prelaunch_burn_from_block")
    stale = _drop_in_table(stale, "scoring", "init_gate_retired_from_block")
    assert len(_decided_fails(_load(stale, tmp_path, "stale.toml"))) == 2
    fixed = stale
    for t, k, v in edits:
        fixed = _set_in_table(fixed, t, k, v)
    cfg = _load(fixed, tmp_path, "fixed.toml")
    assert not _decided_fails(cfg), _decided_fails(cfg)
    main = _load(_main_text(), tmp_path)
    assert cfg.scoring.init_gate_retired_from_block == main.scoring.init_gate_retired_from_block


def test_notes_edits_on_the_deployed_file(tmp_path):
    """On the box (LANED_S0A_DEPLOYED=<deployed chain.toml>): the deployed
    file plus the notes' lines passes `decided`."""
    edits = _edits()
    path = os.environ.get("LANED_S0A_DEPLOYED")
    if not path:
        pytest.skip("LANED_S0A_DEPLOYED not set (the deployed file lives on the box)")
    text = Path(path).read_text(encoding="utf-8")
    for t, k, v in edits:
        text = _set_in_table(text, t, k, v)
    cfg = _load(text, tmp_path, "deployed.toml")
    assert not _decided_fails(cfg), _decided_fails(cfg)


# ── B12: the init gate retired behind a typed activation block ──────────────


def test_init_gate_config_keys_are_retired(tmp_path):
    import dataclasses

    from cascade.eval.koth import KothParams
    from cascade.shared.config import RETIRED_KEYS, ScoringConfig

    scoring = {f.name for f in dataclasses.fields(ScoringConfig)}
    assert "init_gate_mode" not in scoring and "init_gate_tolerance" not in scoring
    assert {"init_gate_mode", "init_gate_tolerance"} <= RETIRED_KEYS["scoring"]
    assert "init_gate_retired_from_block" in scoring
    assert "init_gate_tolerance" not in {f.name for f in dataclasses.fields(KothParams)}
    raw = tomllib.loads(_main_text())
    assert "init_gate_mode" not in raw["scoring"]
    # a deployed file still carrying the retired keys loads, and they change nothing
    old = _set_in_table(_main_text(), "scoring", "init_gate_mode", "enforce")
    old = _set_in_table(old, "scoring", "init_gate_tolerance", "0.5")
    assert _load(old, tmp_path, "old.toml").scoring == _load(_main_text(), tmp_path).scoring


def test_init_gate_runs_as_before_the_block_and_stops_from_it(tmp_path):
    from cascade.shared.receipt import _PARAMS_DROP_WHEN_DEFAULT

    text = _set_in_table(_main_text(), "scoring", "init_gate_retired_from_block", 9_600_300)
    cfg = _load(text, tmp_path)
    assert cfg.koth_params(block=SHADOW_EPOCH).init_gate_mode == "shadow"
    assert cfg.koth_params(block=9_600_299).init_gate_mode == "shadow"
    after = cfg.koth_params(block=9_600_300)
    assert after.init_gate_mode == "off"
    # off is the drop-when-default value: post-block receipts hash as pre-gate ones
    assert after.init_gate_mode == _PARAMS_DROP_WHEN_DEFAULT["init_gate_mode"]
    # unset (0): off at every block
    unset = _load(_set_in_table(_main_text(), "scoring", "init_gate_retired_from_block", 0),
                  tmp_path, "unset.toml")
    assert unset.koth_params(block=SHADOW_EPOCH).init_gate_mode == "off"


def test_archived_shadow_receipt_round_trips_and_verifies(tmp_path):
    # the release file carries the typed retire block the replay resolves by
    assert "init_gate_retired_from_block" in tomllib.loads(_main_text())["scoring"]
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.checks import check_koth_params
    from cascade.shared.receipt import (
        dump_receipt,
        load_receipt,
        receipt_has_unknown_fields,
        verify_receipt_signature,
    )

    text = RECEIPT.read_text(encoding="utf-8")
    r = load_receipt(text)
    assert r.verdict.params["init_gate_mode"] == "shadow"
    assert r.verdict.init_baseline_geomean is not None
    assert r.verdict.init_floor_passed is not None
    assert not receipt_has_unknown_fields(r), "the parse dropped a field"
    assert dump_receipt(r) == text, "archived receipt does not round-trip byte-identically"
    assert verify_receipt_signature(r, ALICE)
    # rebuilt from the parse alone (no received body), it still verifies
    object.__setattr__(r, "_received_body", None)
    assert verify_receipt_signature(r, ALICE)
    # its recorded params replay against the release file (shadow before the block)
    assert check_koth_params(r, _load(_main_text(), tmp_path)).status == "PASS"
