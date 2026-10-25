"""Release tooling must never leave contracts and paid-worker pins mismatched."""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import pytest

from cascade.provision.codeprint import canonical_fingerprint
from scripts import repin_worker_config as repin

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/tensorlink-ai/cascade-worker:qualified-test@sha256:" + "a" * 64


def fingerprint():
    return canonical_fingerprint((str(p.relative_to(ROOT)), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted((ROOT / "cascade").rglob("*.py")))


def test_repin_updates_all_four_pins_together(tmp_path, monkeypatch):
    config = tmp_path / "chain.toml"
    config.write_text((ROOT / "chain.toml").read_text())
    fp = fingerprint()
    monkeypatch.setattr(repin, "registry_code_fingerprint", lambda image: (fp, 300))
    assert repin.main([str(config), IMAGE]) == 0
    data = tomllib.loads(config.read_text())
    assert data["training"]["train_image_digest"] == IMAGE.rsplit("@", 1)[1]
    assert data["v2"]["training"]["train_image_digest"] == IMAGE.rsplit("@", 1)[1]
    assert data["round"]["funded_pod_image"] == IMAGE
    assert data["round"]["worker_code_fingerprint"] == fp


@pytest.mark.parametrize("fault", ["stale_image", "missing_pin", "unavailable_registry"])
def test_failed_repin_preserves_original_file(tmp_path, monkeypatch, fault):
    config = tmp_path / "chain.toml"
    original = (ROOT / "chain.toml").read_text()
    if fault == "missing_pin":
        original = original.replace("worker_code_fingerprint =", "missing_worker_pin =")
    config.write_text(original)

    def registry(image):
        if fault == "unavailable_registry":
            raise OSError("registry unavailable")
        return ("0" * 64 if fault == "stale_image" else fingerprint()), 300

    monkeypatch.setattr(repin, "registry_code_fingerprint", registry)
    with pytest.raises((ValueError, OSError)):
        repin.main([str(config), IMAGE])
    assert config.read_text() == original
