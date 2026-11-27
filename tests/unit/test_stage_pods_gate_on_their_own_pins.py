"""A provisioner stage whose pods boot their own image is health-gated on that image.

2026-11-27 (#964): MarketCast pods boot ``[marketcast] pod_image`` but the
provisioner gated them on the fleet's ``train_image_digest`` /
``worker_code_fingerprint``, so every mkt pod failed its gate (replace →
terminate → retry every 600 s) and lane C never got a verdict.

The class: a new section-level ``pod_image`` pin without a matching
``stage_gate_pins`` entry. This test finds every config dataclass that
declares its own ``pod_image`` and requires (a) a stage for it in
``STAGE_OF`` and (b) that ``stage_gate_pins`` returns THAT image's digest for
that stage, not the fleet's. Torch-free (the provisioner imports none).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

# config dataclass with its own pod image -> (chain.toml section, provisioner stage)
STAGE_OF = {"MarketCastConfig": ("marketcast", "mkt")}
FLEET_DIGEST = "sha256:" + "1" * 64
STAGE_DIGEST = "sha256:" + "2" * 64


def own_pod_image_sections() -> set[str]:
    found = set()
    for path in sorted((ROOT / "cascade/shared/config").glob("*.py")):
        for cls in ast.parse(path.read_text()).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            for st in cls.body:
                if (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                        and st.target.id == "pod_image"):
                    found.add(cls.name)
    return found


def test_every_own_image_section_has_a_gate_stage():
    found = own_pod_image_sections()
    assert "MarketCastConfig" in found, found
    missing = found - set(STAGE_OF)
    assert not missing, (f"config sections pin their own pod image but no provisioner stage gates "
                         f"on it (add it to stage_gate_pins and STAGE_OF): {sorted(missing)}")


def test_each_stage_gates_on_its_own_image_not_the_fleets():
    from cascade.provision.main import stage_gate_pins

    for cls, (section, stage) in STAGE_OF.items():
        cfg = SimpleNamespace(
            training=SimpleNamespace(train_image_digest=FLEET_DIGEST),
            round=SimpleNamespace(worker_code_fingerprint="f" * 64,
                                  funded_pod_image="reg/w@" + FLEET_DIGEST))
        setattr(cfg, section, SimpleNamespace(pod_image="reg/stage:rc@" + STAGE_DIGEST,
                                              worker_code_fingerprint="e" * 64))
        pins = stage_gate_pins(cfg)
        assert pins.get(stage) == (STAGE_DIGEST, "e" * 64), (cls, pins)
