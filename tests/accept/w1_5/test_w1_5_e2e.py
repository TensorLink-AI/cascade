"""W1.5 walking skeleton, end to end (plan rev6 §3.0 W1.5 row, T13, §3.2
invariants 1, 2 and 9).

fixture ``pool_d`` (with ``end_ts``) -> tiny checkpoint -> intake (filesystem
adapter behind the real storage interface) -> ingest guards -> scorer ->
signed report -> ``lane_d_state.json`` -> emission -> ``check_lane_d``.

The chain (built from each test body by ``_chain``) comes in two kinds:

* ``model`` (marked ``torch``): the real chain. Two tiny 2m pyramid
  checkpoints are built with the repo's model code (the house is F32 with its
  median shifted by +3 scale units, the challenger is BF16). The challenger goes
  through intake and ingest, and both are forecast by the lane D scorer.
  The targets are the challenger's median plus small noise, so it must win.
* ``synthetic`` (no torch): the same box-side pipeline, with the eval host's
  quantile forecasts given directly (the eval host is a pod, an external
  service on this path). The challenger is centred on the targets and the
  house is 3 sd off.

Every box-side test runs on both kinds. On main they fail at the first import
of a module named in PATHS.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .helpers import (
    BOUNDARY_1,
    BOUNDARY_2,
    BURN_UID,
    CHALLENGER_UID,
    CONTEXT,
    HISTORY,
    HORIZON,
    HOUSE_T,
    LANE_A_COURT,
    N_UIDS,
    READY_BLOCK,
    REVEAL_BLOCK,
    T0,
    as_bytes,
    file_digest,
    fixture_hotkey,
    gate_inputs,
    lane_d_cfg,
    make_series,
    make_signer,
    parts_with_lane_d,
    pyramid_contract,
    quantiles_around,
    text_digest,
    tiny_checkpoint,
    today_parts,
    today_vector,
    vector_with_lane_d,
    window_id,
    write_pool_d,
)

torch_mark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)

SNAPSHOTS = (1, 2)
SEEDS = {1: 101, 2: 202}


# ── fixtures ────────────────────────────────────────────────────────────────


def _synthetic_chain(tmp: Path) -> SimpleNamespace:
    hotkey, coldkey = fixture_hotkey("challenger"), fixture_hotkey("challenger-coldkey")
    chal, house = text_digest("w1_5 synthetic challenger"), text_digest("w1_5 synthetic house")
    rng = np.random.default_rng(7)
    pool_dirs, forecasts = {}, {}
    for snap in SNAPSHOTS:
        series = make_series(snap, SEEDS[snap])
        targets, fc = {}, {chal: {}, house: {}}
        for s in series:
            sd = float(np.std(s["history"]))
            tail = s["history"][-24:]
            target = np.resize(tail, HORIZON) + rng.normal(0, 0.3, HORIZON)
            targets[s["series_id"]] = target
            wid = window_id(s["series_id"], s["target_ts"])
            fc[chal][wid] = quantiles_around(target + rng.normal(0, 0.05 * sd, HORIZON), 0.3 * sd)
            fc[house][wid] = quantiles_around(target + 3.0 * sd, 0.3 * sd)
        pool_dirs[snap] = write_pool_d(tmp / f"pool_d_{snap}", snap, series, targets)
        forecasts[snap] = fc
    entry = {"hotkey": hotkey, "digest": chal, "context": CONTEXT, "coldkey_at_ready": coldkey,
             "ready_block": READY_BLOCK, "reveal_block": REVEAL_BLOCK, "reveal_ts": T0}
    return SimpleNamespace(kind="synthetic", pool_dirs=pool_dirs, forecasts=forecasts,
                           entries=[entry], house={"digest": house, "reveal_ts": HOUSE_T},
                           uid_of={hotkey: CHALLENGER_UID}, challenger_hotkey=hotkey,
                           challenger_digest=chal, house_digest=house, stages={})


def _model_chain(tmp: Path) -> SimpleNamespace:
    # The W1.5 modules before any model work: on main the test stops here.
    import torch  # noqa: I001
    from cascade.eval.lane_d_ingest import ingest_entry
    from cascade.eval.lane_d_scorer import SCORING_LEVELS, forecast_quantiles
    from cascade.lane_d.intake import (
        checkpoint_manifest,
        manifest_digest,
        ready,
        upload_checkpoint,
    )
    from cascade.lane_d.storage import FilesystemStorage

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        contract = pyramid_contract()
        hotkey, coldkey = fixture_hotkey("challenger"), fixture_hotkey("challenger-coldkey")
        storage = FilesystemStorage(tmp / "vault")
        chal_src = tiny_checkpoint(tmp / "challenger_src", contract, seed=11,
                                   median_shift=0.0, bf16=True)
        house_dir = tiny_checkpoint(tmp / "house", contract, seed=12,
                                    median_shift=3.0, bf16=False)
        house = manifest_digest(checkpoint_manifest(house_dir))
        chal = upload_checkpoint(storage, hotkey, chal_src)
        shutil.rmtree(chal_src)          # from here on only the storage has it
        record = ready(storage, hotkey=hotkey, digest=chal, context=CONTEXT, coldkey=coldkey,
                       ready_block=READY_BLOCK, cfg=lane_d_cfg())
        entry = {**record, "reveal_block": REVEAL_BLOCK, "reveal_ts": T0}
        ingested = Path(ingest_entry(storage, entry, tmp / "ingest", contract=contract))
        median = list(SCORING_LEVELS).index(0.5)
        rng = np.random.default_rng(7)
        pool_dirs, forecasts, shapes = {}, {}, []
        for snap in SNAPSHOTS:
            series = make_series(snap, SEEDS[snap])
            hists = [s["history"] for s in series]
            q_chal = np.asarray(forecast_quantiles(ingested, hists, context=CONTEXT,
                                                   horizon=HORIZON))
            q_house = np.asarray(forecast_quantiles(house_dir, hists, context=CONTEXT,
                                                    horizon=HORIZON))
            shapes.append((q_chal.shape, q_house.shape,
                           bool(np.isfinite(q_chal).all() and np.isfinite(q_house).all())))
            targets, fc = {}, {chal: {}, house: {}}
            for i, s in enumerate(series):
                sd = float(np.std(s["history"]))
                targets[s["series_id"]] = q_chal[i, :, median] + rng.normal(0, 0.05 * sd, HORIZON)
                wid = window_id(s["series_id"], s["target_ts"])
                fc[chal][wid] = q_chal[i]
                fc[house][wid] = q_house[i]
            pool_dirs[snap] = write_pool_d(tmp / f"pool_d_{snap}", snap, series, targets)
            forecasts[snap] = fc
    finally:
        torch.set_num_threads(threads)
    stages = {"record": record, "ingested": ingested, "shapes": shapes,
              "n_windows": len(series), "scoring_levels": tuple(SCORING_LEVELS)}
    return SimpleNamespace(kind="model", pool_dirs=pool_dirs, forecasts=forecasts,
                           entries=[entry], house={"digest": house, "reveal_ts": HOUSE_T},
                           uid_of={hotkey: CHALLENGER_UID}, challenger_hotkey=hotkey,
                           challenger_digest=chal, house_digest=house, stages=stages)


@pytest.fixture(scope="module", params=["synthetic", pytest.param("model", marks=torch_mark)])
def chain_kind(request, tmp_path_factory):
    """Only the kind and a scratch dir. Fixtures import nothing from PATHS and
    build nothing, so on main every failure happens inside a test body."""
    return SimpleNamespace(kind=request.param,
                           tmp=tmp_path_factory.mktemp(f"w1_5_{request.param}"))


_BUILT: dict[str, tuple[SimpleNamespace, object]] = {}


def _chain(spec: SimpleNamespace) -> tuple[SimpleNamespace, object]:
    """Called from each test body: the box-side W1.5 modules are imported
    first (on main the test stops here), then the chain and the test signer
    are built once per kind and reused."""
    import cascade.audit.lane_d  # noqa: F401, I001
    import cascade.validator.lane_d  # noqa: F401
    import cascade.validator.lane_d_report  # noqa: F401

    from cascade.shared.emission import lane_d_vector  # noqa: F401

    key = str(spec.tmp)
    if key not in _BUILT:
        built = (_synthetic_chain(spec.tmp / "chain") if spec.kind == "synthetic"
                 else _model_chain(spec.tmp / "chain"))
        _BUILT[key] = (built, make_signer(spec.tmp / "key"))
    return _BUILT[key]


LANE_A_FILES = {"cascade_state.json": b'{"king": "lane-a-king", "king_since_block": 9000000}',
                "validator_state.db": b"SQLite format 3\x00 lane A verdict rows"}


def _state_dir(tmp_path: Path) -> Path:
    """A validator state dir that already holds lane A's state files."""
    d = tmp_path / "state"
    d.mkdir(parents=True)
    for name, data in LANE_A_FILES.items():
        (d / name).write_bytes(data)
    return d


def _settle(chain, cfg, state_path: Path, signer, boundary: int, snap: int):
    """`settle_lane_d`, with the W5e gate inputs when it accepts them
    (`helpers.gate_inputs`): the skeleton crowns at BOUNDARY_2."""
    from cascade.validator.lane_d import settle_lane_d

    return settle_lane_d(boundary_ts=2_000_000_000 + 12 * int(boundary), cfg=cfg, boundary=boundary, pool_dir=chain.pool_dirs[snap],
                         entries=[dict(e) for e in chain.entries], house=dict(chain.house),
                         forecasts=chain.forecasts[snap], uid_of=dict(chain.uid_of),
                         state_path=state_path, signer=signer,
                         **gate_inputs(boundary, chain.house["digest"]))


def _settle_both(chain, cfg, state_path: Path, signer):
    """Two settlements on disjoint forward window sets (snapshots 1 and 2), so
    a confirmation duel (W5e) still crowns inside the skeleton."""
    return (_settle(chain, cfg, state_path, signer, BOUNDARY_1, 1),
            _settle(chain, cfg, state_path, signer, BOUNDARY_2, 2))


def _assert_valid_report(report, cfg, signer, boundary):
    from cascade.audit.lane_d import check_lane_d  # noqa: I001
    from cascade.validator.lane_d_report import REPORT_SCHEMA, verify_report

    from cascade.audit.checks import PASS

    assert report is not None, "an armed settlement produced no lane D report"
    assert REPORT_SCHEMA == "lane-d-report/1"
    assert json.loads(json.dumps(report, allow_nan=False)) == report, "report is not plain JSON"
    assert report["schema"] == "lane-d-report/1"
    assert report["boundary"] == boundary
    assert report["params"]["share"] == cfg.share
    sha = report["rows_sha256"]
    assert isinstance(sha, str) and len(sha) == 64 and int(sha, 16) >= 0
    assert verify_report(report, signer.ss58_address), "report signature does not verify"
    res = check_lane_d(report, cfg, signer_hotkey=signer.ss58_address)
    assert res.status == PASS, f"check_lane_d: {res.status} {res.detail}"


def _assert_judged_rows(report, chain):
    rows = report["rows"]
    assert rows, "the first armed settlement scored no rows"
    for row in rows:
        assert set(row) >= {"window_id", "digest", "crps", "mase"}, row
        assert math.isfinite(float(row["crps"])) and math.isfinite(float(row["mase"]))
    assert {r["digest"] for r in rows} >= {chain.challenger_digest, chain.house_digest}


# ── the W1.5 row ────────────────────────────────────────────────────────────


def test_e2e_challenger_gets_nonzero_lane_d_weight(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg = lane_d_cfg()
    state_path = _state_dir(tmp_path) / "lane_d_state.json"
    _, s2 = _settle_both(chain, cfg, state_path, signer)
    if chain.kind == "model":
        st = chain.stages
        rec = st["record"]
        assert rec["hotkey"] == chain.challenger_hotkey
        assert rec["digest"] == chain.challenger_digest
        assert rec["context"] == CONTEXT and rec["ready_block"] == READY_BLOCK
        assert (st["ingested"] / "weights.safetensors").is_file()
        assert st["scoring_levels"] == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        want = (st["n_windows"], HORIZON, 9)
        for q_chal, q_house, finite in st["shapes"]:
            assert q_chal == want and q_house == want and finite
    assert s2.pay is not None, "an armed settlement produced no lane D pay record"
    assert s2.pay["share"] == cfg.share
    vec = vector_with_lane_d(s2.pay, cfg)
    assert len(vec) == N_UIDS and math.isclose(sum(vec), 1.0, rel_tol=1e-9)
    assert today_vector()[CHALLENGER_UID] == 0.0
    assert vec[CHALLENGER_UID] > 0.0, f"challenger lane D weight is {vec[CHALLENGER_UID]}"
    assert state_path.is_file(), "no lane_d_state.json written"
    state = json.loads(state_path.read_text())
    assert state["king"]["digest"] == chain.challenger_digest
    assert state["house"]["digest"] == chain.house_digest


def test_e2e_audit_passes(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg = lane_d_cfg()
    s1, s2 = _settle_both(chain, cfg, _state_dir(tmp_path) / "lane_d_state.json", signer)
    _assert_valid_report(s1.report, cfg, signer, BOUNDARY_1)
    _assert_judged_rows(s1.report, chain)
    _assert_valid_report(s2.report, cfg, signer, BOUNDARY_2)


def test_e2e_tampered_row_fails(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    from cascade.audit.lane_d import check_lane_d  # noqa: I001
    from cascade.validator.lane_d_report import sign_report, verify_report

    from cascade.audit.checks import FAIL

    cfg = lane_d_cfg()
    s1, _ = _settle_both(chain, cfg, _state_dir(tmp_path) / "lane_d_state.json", signer)
    _assert_valid_report(s1.report, cfg, signer, BOUNDARY_1)
    tampered = copy.deepcopy(s1.report)
    tampered["rows"][0]["crps"] = float(tampered["rows"][0]["crps"]) + 1.0
    # As published (signature now stale) ...
    assert not verify_report(tampered, signer.ss58_address)
    res = check_lane_d(tampered, cfg, signer_hotkey=signer.ss58_address)
    assert res.status == FAIL, f"a tampered row passed check_lane_d: {res}"
    # ... and re-signed by the same key: rows_sha256 no longer matches the rows.
    resigned = sign_report(tampered, signer)
    # W7b: verify_report validates the report first; a rows_sha256 mismatch
    # makes it schema-invalid, so a fresh signature does not rescue it.
    assert not verify_report(resigned, signer.ss58_address)
    res = check_lane_d(resigned, cfg, signer_hotkey=signer.ss58_address)
    assert res.status == FAIL, f"a re-signed tampered row passed check_lane_d: {res}"


# ── invariant 1: unarmed => byte-identical (paired, invariant 10) ───────────


def test_inv1_weights_unchanged_when_unarmed(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg = lane_d_cfg(from_block=0)       # everything else as armed, share 0.25 included
    assert not cfg.active(BOUNDARY_1) and not cfg.active(BOUNDARY_2)
    state_dir = _state_dir(tmp_path)
    before = sorted(p.name for p in state_dir.iterdir())
    for s in _settle_both(chain, cfg, state_dir / "lane_d_state.json", signer):
        assert s.report is None, "an unarmed settlement produced a lane D report"
        assert s.pay is None, "an unarmed settlement produced a lane D pay record"
        assert as_bytes(vector_with_lane_d(s.pay, cfg)) == as_bytes(today_vector())
    assert sorted(p.name for p in state_dir.iterdir()) == before, "unarmed lane D wrote state"


def test_inv1_weights_armed_nonempty_lane_d_effect(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg = lane_d_cfg()
    _, s2 = _settle_both(chain, cfg, _state_dir(tmp_path) / "lane_d_state.json", signer)
    assert s2.report is not None and s2.pay is not None
    vec = vector_with_lane_d(s2.pay, cfg)
    assert as_bytes(vec) != as_bytes(today_vector())
    assert vec[CHALLENGER_UID] > 0.0


# ── invariant 2 / T13: lane A verdicts and court identical (paired) ─────────


def _lane_a_files_unchanged(state_dir: Path) -> None:
    for name, data in LANE_A_FILES.items():
        assert (state_dir / name).read_bytes() == data, f"lane D settlement changed {name}"


def test_inv2_t13_lane_a_court_unchanged_when_unarmed(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg = lane_d_cfg(from_block=0)
    state_dir = _state_dir(tmp_path)
    a_today, _ = today_parts()
    for s in _settle_both(chain, cfg, state_dir / "lane_d_state.json", signer):
        a, _ = parts_with_lane_d(s.pay, cfg)
        assert as_bytes(a) == as_bytes(a_today)
    _lane_a_files_unchanged(state_dir)


def test_inv2_t13_lane_a_court_armed_nonempty_lane_d_effect(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg = lane_d_cfg()
    state_dir = _state_dir(tmp_path)
    a_today, _ = today_parts()
    court_today = [u for u, w in enumerate(a_today) if w > 0.0]
    assert court_today == sorted(LANE_A_COURT)
    _, s2 = _settle_both(chain, cfg, state_dir / "lane_d_state.json", signer)
    a, b = parts_with_lane_d(s2.pay, cfg)
    # The court (who lane A pays) and its relative split are unchanged ...
    assert [u for u, w in enumerate(a) if w > 0.0] == court_today
    for u in court_today:
        assert math.isclose(a[u] / sum(a), a_today[u] / sum(a_today), rel_tol=1e-12)
    # ... lane A's share-of-total changes by design (D1/D48: D takes 0.25 from A) ...
    assert math.isclose(sum(a), 1.0 - cfg.share, rel_tol=1e-12)
    assert a[CHALLENGER_UID] == 0.0 and a[BURN_UID] == a_today[BURN_UID]
    # ... and lane D's effect lands outside lane A.
    assert b[CHALLENGER_UID] > 0.0
    _lane_a_files_unchanged(state_dir)


# ── invariant 9: share = 0 => vector identical, report still valid (paired) ─


def test_inv9_share_zero_vector_unchanged_when_unarmed(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg0 = lane_d_cfg(share=0.0)
    unarmed = lane_d_cfg(from_block=0)
    s_unarmed = _settle_both(chain, unarmed, _state_dir(tmp_path / "u") / "lane_d_state.json",
                             signer)
    s_zero = _settle_both(chain, cfg0, _state_dir(tmp_path / "z") / "lane_d_state.json", signer)
    for su, sz in zip(s_unarmed, s_zero, strict=True):
        assert sz.pay is not None and sz.pay["share"] == 0.0
        assert as_bytes(vector_with_lane_d(sz.pay, cfg0)) == as_bytes(
            vector_with_lane_d(su.pay, unarmed))
        assert as_bytes(vector_with_lane_d(sz.pay, cfg0)) == as_bytes(today_vector())


def test_inv9_share_zero_vector_armed_nonempty_lane_d_effect(chain_kind, tmp_path):
    chain, signer = _chain(chain_kind)
    cfg0 = lane_d_cfg(share=0.0)
    s1, s2 = _settle_both(chain, cfg0, _state_dir(tmp_path) / "lane_d_state.json", signer)
    _assert_valid_report(s1.report, cfg0, signer, BOUNDARY_1)
    _assert_judged_rows(s1.report, chain)
    _assert_valid_report(s2.report, cfg0, signer, BOUNDARY_2)


def test_fixture_pool_d_carries_end_ts(chain_kind):
    chain, _ = _chain(chain_kind)
    """The fixture itself: every series has ``end_ts`` and both clocks."""
    meta = json.loads((chain.pool_dirs[1] / "metadata.json").read_text())
    assert meta["series"]
    for s in meta["series"]:
        assert s["end_ts"] == s["target_ts"][-1] and len(s["target_ts"]) == HORIZON
        assert len(s["first_seen_ts"]) == HORIZON
        # RV3: forward-only fails closed, so every series records its S0 and
        # names a feed listed in the snapshot's ``feeds`` block.
        assert isinstance(s["frontier"], list) and s["frontier"]
        assert all(end < s["target_ts"][0] for _as_of, end in s["frontier"])
        assert s["feed"] in meta["feeds"]
        assert meta["feeds"][s["feed"]]["mirror_lag_hours"] is not None
    with np.load(chain.pool_dirs[1] / "hist.npz") as hist:
        assert all(hist[k].shape == (HISTORY,) for k in hist.files)
    assert file_digest(chain.pool_dirs[1] / "tgt.npz") != file_digest(chain.pool_dirs[2] / "tgt.npz")
