"""W3 published-weights registry sources (plan §1.2 dedup, D47):
``python -P -m cascade.lane_d.published_sources``.

* ``mirror-index`` fetches the signed ``champions/lane_d/index.json`` and
  every king's ``manifest.json`` from public URLs, verifies the signature and
  each manifest's digest BEFORE writing, and lays them out as the
  ``--mirror-dir`` the registry job reads;
* ``checkpoint-shas`` (pod side) fetches each published lane A checkpoint and
  records its weights sha256 AND the sha256 of its D47 BF16 cast, so a byte
  copy or a cast copy dedups (the house is the cast of KING_A);
* ``published_weights --shas-file`` adds both shas to the registry.

The HTTP GET and the Hub fetch are fakes; keys are deterministic TEST keys.
Every PATHS import is inside a test body.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

KING_REF = "cascade/ckpt-r7-king-120m@sha256:" + "7" * 64
LEG_REF = "cascade/ckpt-r7-challenger-120m@sha256:" + "8" * 64


def src():
    return importlib.import_module("cascade.lane_d.published_sources")


def _key(name: str):
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair.create_from_uri(f"//w3-sources-{name}")


def _king_files(seed: int) -> dict[str, bytes]:
    return {"config.json": b"{}", "forecast_wrapper.py": b"# w\n", "model.py": b"# m\n",
            "weights.safetensors": hashlib.sha256(f"w3 {seed}".encode()).digest() * 4}


def _manifest(files: dict[str, bytes]) -> bytes:
    from cascade.lane_d.intake import canonical_manifest

    return canonical_manifest({"files": {n: {"sha256": hashlib.sha256(b).hexdigest(),
                                             "size": len(b)} for n, b in files.items()}})


def _published(signer, n: int = 2) -> tuple[dict[str, bytes], list[dict]]:
    """``(url -> bytes, kings)``: a signed index and n king manifests."""
    from cascade.lane_d.intake import manifest_digest
    from cascade.lane_d.publish import build_index

    urls, kings = {}, []
    for i in range(n):
        files = _king_files(i)
        raw = _manifest(files)
        digest = manifest_digest(json.loads(raw))
        urls[f"https://mirror.example/kings/lane-d-kings/{digest}/manifest.json"] = raw
        kings.append({"hotkey": f"5Hk{i}", "context": 4096, "digest": digest,
                      "crown_block": 100 + i, "lcb": 0.01, "gate": {}, "sentinel": {"suite": "none"},
                      "license": "MIT", "weights": hashlib.sha256(files["weights.safetensors"])
                      .hexdigest()})
    index = build_index([{k: v for k, v in king.items() if k != "weights"} for king in kings],
                        signer)
    urls["https://store.example/champions/lane_d/index.json"] = json.dumps(index).encode()
    return urls, kings


def _getter(urls):
    def get(url, *, max_size):
        if url not in urls:
            raise src().SourceFetchError(f"404 {url}")
        assert len(urls[url]) <= max_size
        return urls[url]

    return get


def test_mirror_index_writes_the_registry_job_inputs(tmp_path):
    import cascade.lane_d.published_weights as pw

    signer = _key("validator")
    urls, kings = _published(signer)
    out = src().mirror_index(index_url="https://store.example/champions/lane_d/index.json",
                             mirror_url="https://mirror.example/kings/",
                             signer_ss58=signer.ss58_address, dest=tmp_path / "m",
                             get=_getter(urls))
    assert out["kings"] == 2
    reg = tmp_path / "published_weights.json"
    assert pw.main(["--out", str(reg), "--lane-d-index", out["index"], "--index-signer",
                    signer.ss58_address, "--mirror-dir", str(tmp_path / "m")]) == 0
    assert json.loads(reg.read_text())["weights_sha256"] == sorted(k["weights"] for k in kings)


@pytest.mark.parametrize("break_it", ["signer", "manifest"])
def test_mirror_index_verifies_everything_before_writing(tmp_path, break_it):
    signer = _key("validator")
    urls, kings = _published(signer)
    if break_it == "manifest":
        key = next(u for u in urls if kings[1]["digest"] in u)
        urls[key] = _manifest(_king_files(99))
    signer_ss58 = _key("other").ss58_address if break_it == "signer" else signer.ss58_address
    with pytest.raises(src().SourceFetchError):
        src().mirror_index(index_url="https://store.example/champions/lane_d/index.json",
                           mirror_url="https://mirror.example/kings",
                           signer_ss58=signer_ss58, dest=tmp_path / "m", get=_getter(urls))
    assert not (tmp_path / "m").exists(), "nothing is written before everything verifies"


def test_trained_refs_are_found_anywhere_in_a_manifest():
    doc = {"round": 7, "king": {"pointer": f"metro-v1:trained:hippius:{KING_REF}"},
           "legs": [{"trained_pointer": f"metro-v1:trained:hippius:{LEG_REF}"},
                    {"note": "metro-v1:gen:hippius:miner/gen@sha256:" + "9" * 64}]}
    assert src().trained_refs([doc, doc]) == sorted([KING_REF, LEG_REF])


def _st(seed: int) -> bytes:
    from safetensors.numpy import save

    rng = np.random.default_rng(seed)
    return save({"w": rng.standard_normal(16).astype(np.float32)})


def test_checkpoint_shas_record_the_raw_and_the_bf16_cast_sha(tmp_path):
    from cascade.lane_d.house_source import cast_safetensors_bf16

    blobs = {KING_REF: _st(1), LEG_REF: _st(2)}
    fetched = []

    def fetch(ref, dest):
        fetched.append(ref)
        Path(dest).mkdir(parents=True)
        (Path(dest) / "weights.safetensors").write_bytes(blobs[ref])

    doc = src().checkpoint_shas([LEG_REF, KING_REF, KING_REF], work_dir=tmp_path / "w",
                                fetch=fetch)
    assert doc["schema"] == "laned_published_shas/1" and sorted(fetched) == sorted(blobs)
    by_ref = {s["ref"]: s for s in doc["sources"]}
    for ref, blob in blobs.items():
        assert by_ref[ref]["weights_sha256"] == hashlib.sha256(blob).hexdigest()
        assert by_ref[ref]["bf16_weights_sha256"] == hashlib.sha256(
            cast_safetensors_bf16(blob)).hexdigest()
    assert not list((tmp_path / "w").iterdir()), "each checkpoint is removed after hashing"


def _pw(argv: list[str]) -> int:
    import cascade.lane_d.published_weights as pw

    try:
        return pw.main(argv)
    except SystemExit as e:              # argparse usage error (an option this base lacks)
        return int(e.code or 0)


def test_the_registry_takes_the_shas_file(tmp_path):

    shas = {"schema": "laned_published_shas/1",
            "sources": [{"ref": KING_REF, "weights_sha256": "1" * 64,
                         "bf16_weights_sha256": "2" * 64}]}
    f = tmp_path / "shas.json"
    f.write_text(json.dumps(shas))
    reg = tmp_path / "published_weights.json"
    assert _pw(["--out", str(reg), "--shas-file", str(f)]) == 0
    assert json.loads(reg.read_text())["weights_sha256"] == ["1" * 64, "2" * 64]
    f.write_text(json.dumps({**shas, "sources": [{"ref": KING_REF, "weights_sha256": "x"}]}))
    before = reg.read_bytes()
    assert _pw(["--out", str(reg), "--shas-file", str(f)]) == 1
    assert reg.read_bytes() == before
