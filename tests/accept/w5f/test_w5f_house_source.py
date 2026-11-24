"""W5f house source producer (plan §1.4 "The house", D46 (a), D47 (a)).

``python -P -m cascade.lane_d.house_source build`` (pod / builder side) casts
lane A's king to the BF16 house entry (the four lane D entry files, the lane
D manifest digest), dates it by its lineage (D46: the chain timestamp of the
latest lineage commit block) and writes ``laned_house_source/1`` plus its
``laned_house_provenance/1``. ``ops_jobs house-source`` (box side, stdlib)
re-checks the provenance and installs the source where ``house-pointer``
reads it.

The cast rule is pinned here: round-to-nearest-even on the F32 bit pattern,
F16 widened exactly first, BF16 copied, anything else / non-finite refused,
tensors in sorted-name order with ``{"format": "pt"}`` metadata, so one king
always casts to one set of bytes. Fixtures are tiny numpy safetensors (no
torch); keys are deterministic TEST keys. Every PATHS import is inside a test
body.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tests.accept.w3.helpers import entry_config

from .helpers import fixture_key

REPO = Path(__file__).resolve().parents[3]
KING = "metro-v1:trained:hippius:cascade/ckpt-r7-king-120m@sha256:" + "7" * 64
ENTRY_FILES = ["config.json", "forecast_wrapper.py", "model.py", "weights.safetensors"]


def hs():
    return importlib.import_module("cascade.lane_d.house_source")


def jobs():
    return importlib.import_module("cascade.lane_d.ops_jobs")


def _main(argv: list[str]) -> int:
    try:
        return jobs().main(argv)
    except SystemExit as e:              # argparse usage error (a job this base lacks)
        return int(e.code or 0)


def _st(tensors: dict) -> bytes:
    from safetensors.numpy import save

    return save(tensors)


def _bf16_bits(blob: bytes) -> dict[str, list[int]]:
    import safetensors

    out = {}
    for name, t in safetensors.deserialize(blob):
        assert t["dtype"] == "bfloat16" or t["dtype"] == "BF16", t["dtype"]
        out[name] = np.frombuffer(bytes(t["data"]), "<u2").tolist()
    return out


def _king_dir(tmp: Path, *, weights: bytes | None = None, config: dict | None = None) -> Path:
    d = tmp / "king"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config or entry_config()))
    (d / "weights.safetensors").write_bytes(weights if weights is not None else _st(
        {"w.b": np.array([1.0, 1 + 2 ** -8, 1 + 3 * 2 ** -8, -2.5], dtype=np.float32),
         "w.a": np.ones((2, 3), dtype=np.float16)}))
    (d / "trainer_state.pt").write_bytes(b"not part of a lane D entry")
    return d


def _build(tmp: Path, king_dir: Path, *extra: str) -> int:
    return hs().main(["build", "--king-pointer", KING, "--king-dir", str(king_dir),
                      "--lineage-block", "100", "--lineage-block", "250",
                      "--block-ts", "250:1790000123", "--block-ts", "100:1789990000",
                      "--entry-dir", str(tmp / "out" / "entry"),
                      "--out", str(tmp / "out" / "house_source.json"), *extra])


# ── the cast rule ────────────────────────────────────────────────────────────


def test_cast_rounds_to_nearest_even_and_is_deterministic():
    src = _st({"b": np.array([1.0, 1 + 2 ** -8, 1 + 3 * 2 ** -8, -2.5, 2 ** -130],
                             dtype=np.float32),
               "a": np.array([0.5, -0.0], dtype=np.float16)})
    out = hs().cast_safetensors_bf16(src)
    bits = _bf16_bits(out)
    # 1.0; the tie 1+2^-8 goes to the even 1.0; the tie 1+3*2^-8 to the even
    # 1+2^-6 (0x3F82); -2.5 exact; a subnormal rounds on its bit pattern.
    sub = (np.array([2 ** -130], dtype=np.float32).view("<u4")[0])
    want_sub = int((int(sub) + 0x7FFF + ((int(sub) >> 16) & 1)) >> 16)
    assert bits["b"] == [0x3F80, 0x3F80, 0x3F82, 0xC020, want_sub]
    assert bits["a"] == [0x3F00, 0x8000]                    # F16 widened exactly
    assert hs().cast_safetensors_bf16(src) == out, "one king casts to one byte string"
    assert hs().cast_safetensors_bf16(out) == out, "a BF16 file is copied unchanged"
    n = struct.unpack("<Q", out[:8])[0]
    header = json.loads(out[8:8 + n])
    assert n % 8 == 0 and header["__metadata__"] == {"format": "pt"}
    assert [k for k in header if k != "__metadata__"] == sorted(k for k in header
                                                                if k != "__metadata__")
    assert header["a"]["data_offsets"] == [0, 4] and header["b"]["data_offsets"] == [4, 14]


@pytest.mark.parametrize("tensors", [
    {"x": np.array([1.0], dtype=np.float64)},
    {"x": np.array([3], dtype=np.int64)},
    {"x": np.array([np.nan], dtype=np.float32)},
    {"x": np.array([np.inf], dtype=np.float32)},
    {"x": np.array([3.4e38], dtype=np.float32)},             # rounds past the BF16 max
], ids=["f64", "i64", "nan", "inf", "overflow"])
def test_cast_refuses_what_a_lane_d_entry_cannot_hold(tensors):
    with pytest.raises(hs().HouseSourceError):
        hs().cast_safetensors_bf16(_st(tensors))


# ── build (pod side) ─────────────────────────────────────────────────────────


def test_build_writes_the_bf16_house_entry_and_the_source(tmp_path):
    from cascade.lane_d.intake import checkpoint_manifest, manifest_digest

    king = _king_dir(tmp_path)
    assert _build(tmp_path, king) == 0
    entry = tmp_path / "out" / "entry"
    assert sorted(p.name for p in entry.iterdir()) == ENTRY_FILES
    assert (entry / "config.json").read_bytes() == (king / "config.json").read_bytes()
    assert (entry / "model.py").read_bytes() == (REPO / "cascade/trainer/pyramid_model.py"
                                                ).read_bytes()
    assert (entry / "forecast_wrapper.py").read_bytes() == (
        REPO / "cascade/trainer/pyramid_wrapper.py").read_bytes()
    assert hs().weights_dtypes((entry / "weights.safetensors").read_bytes()) == {"BF16": 2}
    src = json.loads((tmp_path / "out" / "house_source.json").read_text())
    assert sorted(src) == ["digest", "king_pointer", "reveal_ts", "schema"]
    assert src["schema"] == "laned_house_source/1" and src["king_pointer"] == KING
    assert src["digest"] == manifest_digest(checkpoint_manifest(entry))
    assert src["reveal_ts"] == 1790000123, "D46: the latest lineage block's timestamp"
    prov = json.loads((tmp_path / "out" / "house_source.json.provenance.json").read_text())
    assert prov["schema"] == "laned_house_provenance/1"
    assert prov["lineage_blocks"] == [100, 250] and prov["reveal_block"] == 250
    assert prov["cast"]["rule"] == "laned_bf16_cast/1"
    assert prov["cast"]["input_weights_sha256"] == hashlib.sha256(
        (king / "weights.safetensors").read_bytes()).hexdigest()
    assert prov["king_ref"] == KING.split("hippius:", 1)[1]


def test_build_is_reproducible(tmp_path):
    king = _king_dir(tmp_path)
    assert _build(tmp_path, king) == 0
    first = json.loads((tmp_path / "out" / "house_source.json").read_text())
    other = tmp_path / "again"
    other.mkdir()
    assert hs().main(["build", "--king-pointer", KING, "--king-dir", str(king),
                      "--lineage-block", "250", "--block-ts", "250:1790000123",
                      "--entry-dir", str(other / "entry"),
                      "--out", str(other / "src.json")]) == 0
    assert json.loads((other / "src.json").read_text()) == first


def test_build_refuses_a_king_that_is_not_a_lane_d_checkpoint(tmp_path):
    bad = entry_config()
    bad["moe"]["not_a_field"] = 1
    king = _king_dir(tmp_path, config=bad)
    assert _build(tmp_path, king) == 1
    assert not (tmp_path / "out" / "house_source.json").exists()
    assert not (tmp_path / "out" / "entry").exists()


def test_build_fetches_the_pointer_ref_when_no_dir_is_given(tmp_path):
    king = _king_dir(tmp_path)
    got = []

    def fetch(ref, dest):
        got.append(ref)
        Path(dest).mkdir(parents=True)
        for p in king.iterdir():
            (Path(dest) / p.name).write_bytes(p.read_bytes())

    args = hs().build_arg_parser().parse_args(
        ["build", "--king-pointer", KING, "--work-dir", str(tmp_path / "work"),
         "--lineage-block", "9", "--block-ts", "9:1700000000",
         "--entry-dir", str(tmp_path / "e"), "--out", str(tmp_path / "s.json")])
    out = hs().build(args, fetch=fetch)
    assert got == [KING.split("hippius:", 1)[1]]
    assert out["reveal_ts"] == 1700000000


def test_build_admits_the_house_into_the_eval_host_cache(tmp_path):
    from cascade.eval.lane_d_cache import PrefetchCache

    king = _king_dir(tmp_path)
    cache = tmp_path / "cache"
    assert _build(tmp_path, king, "--cache-dir", str(cache), "--budget-bytes", "100000000") == 0
    digest = json.loads((tmp_path / "out" / "house_source.json").read_text())["digest"]
    assert PrefetchCache(cache, budget_bytes=100000000).get(digest) is not None


# ── install (box side) ───────────────────────────────────────────────────────


def _install(tmp_path, *extra: str) -> int:
    return _main(["house-source", "--source", str(tmp_path / "out" / "house_source.json"),
                        "--dest", str(tmp_path / "lane-d" / "house_source.json"), *extra])


def _last(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_install_checks_then_installs_once(tmp_path, capsys):
    assert _build(tmp_path, _king_dir(tmp_path)) == 0
    capsys.readouterr()
    dest = tmp_path / "lane-d" / "house_source.json"
    assert _install(tmp_path, "--dry-run") == 0
    assert _last(capsys)["action"] == "planned" and not dest.exists()
    assert _install(tmp_path) == 0
    assert _last(capsys)["action"] == "installed"
    assert json.loads(dest.read_text()) == json.loads(
        (tmp_path / "out" / "house_source.json").read_text())
    assert (tmp_path / "lane-d" / "house_source.json.provenance.json").is_file()
    assert _install(tmp_path) == 0
    assert _last(capsys)["action"] == "current"


@pytest.mark.parametrize("tamper", ["reveal_ts", "manifest", "code", "lineage"])
def test_install_refuses_a_source_its_provenance_does_not_prove(tmp_path, capsys, tamper):
    assert _build(tmp_path, _king_dir(tmp_path)) == 0
    src_p = tmp_path / "out" / "house_source.json"
    prov_p = tmp_path / "out" / "house_source.json.provenance.json"
    src, prov = json.loads(src_p.read_text()), json.loads(prov_p.read_text())
    if tamper == "reveal_ts":
        src["reveal_ts"] += 1
    elif tamper == "manifest":
        prov["manifest"]["files"]["weights.safetensors"]["size"] += 1
    elif tamper == "code":
        f = prov["manifest"]["files"]["model.py"]
        f["sha256"] = "0" * 64
        src["digest"] = prov["digest"] = hashlib.sha256(json.dumps(
            prov["manifest"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    else:
        prov["reveal_block"] = 100
    src_p.write_text(json.dumps(src))
    prov_p.write_text(json.dumps(prov))
    capsys.readouterr()
    assert _install(tmp_path) == 1
    assert _last(capsys)["action"] == "failed"
    assert not (tmp_path / "lane-d" / "house_source.json").exists()


def test_the_installed_source_lets_the_house_pointer_job_sign(tmp_path, monkeypatch, capsys):
    import cascade.shared.chain as chain
    from cascade.validator.lane_d_house import load_house_pointer

    signer = fixture_key("validator")
    monkeypatch.setattr(chain, "open_wallet", lambda *a, **k: signer)
    assert _build(tmp_path, _king_dir(tmp_path)) == 0
    assert _install(tmp_path) == 0
    state = tmp_path / "validator_state.json"
    state.write_text(json.dumps({"king_pointer": KING}))
    pointer = tmp_path / "lane-d" / "house_pointer.json"
    assert _main(["house-pointer", "--king-source", str(state), "--king-field",
                        "king_pointer", "--pointer", str(pointer), "--signer-ss58",
                        signer.ss58_address, "--house-source",
                        str(tmp_path / "lane-d" / "house_source.json")]) == 0
    src = json.loads((tmp_path / "lane-d" / "house_source.json").read_text())
    assert load_house_pointer(pointer, signer_ss58=signer.ss58_address) == {
        "digest": src["digest"], "reveal_ts": src["reveal_ts"]}
