"""Production adapter contracts tested without real marketplace calls or payouts."""

import copy
import io

import pytest
from bittensor.sp_core import Keypair

from cascade.direct.evaluation import PublicSidecar
from cascade.direct.intake import DirectFundingIntake
from cascade.direct.signing import sign, verify
from cascade.direct.storage import FileObjects, publish
from cascade.shared.direct_moe import SUITES
from tests.unit import test_direct_moe as helpers

environment = helpers.environment
guard = helpers.guard
admit = helpers.admit


def test_persistent_file_objects_reject_corruption(tmp_path):
    objects = FileObjects(tmp_path)
    ref = publish(objects, b"checkpoint")
    objects = FileObjects(tmp_path)
    assert objects.get(ref["uri"]) == b"checkpoint"
    (tmp_path / ref["sha256"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        objects.get(ref["uri"])


def test_signed_intake_binds_payer_and_persists_retry(environment, guard):
    ctl, _, _, _, policy, path = environment
    terms = admit(environment, guard)
    # Admit challenger under a real sr25519 fixture key into the same cohort.
    key = Keypair.create_from_uri("//Alice")
    terms = {
        **terms,
        "role": "challenger",
        "reward": 0,
        "payer": key.ss58_address,
        "funding_ref": "vault:signed",
    }
    request = sign(
        {"job": "signed", "nonce": "signed", "expires_at": 20, "terms": terms},
        key,
        scope="fund-job",
    )
    intake = DirectFundingIntake(
        path,
        policy,
        registered=lambda _: True,
        revealed=lambda *_: True,
        funding_authorized=lambda *_: True,
    )
    assert intake.fund(request, now=2)[0] == 202
    assert intake.fund(request, now=100)[0] == 200
    forged = copy.deepcopy(request)
    forged["body"]["terms"]["spend_cap"] += 1
    assert intake.fund(forged, now=3)[0] == 400
    journal = sign(ctl.ledger.export(), key, scope="journal")
    from cascade.validator.direct_moe import verify_direct_receipt

    assert verify_direct_receipt(journal, expected_trainer=key.ss58_address) == ctl.ledger.state
    with pytest.raises(ValueError, match="trust anchor"):
        verify_direct_receipt(journal)
    assert verify(journal, expected_signer=key.ss58_address, scope="journal") == ctl.ledger.export()


def test_all_public_adapters_refuse_partial_revision_or_case_drift(guard):
    from dataclasses import replace

    guard = replace(guard, cases=tuple(replace(c, metric="mase") for c in guard.cases))
    names = {"GIFT": "gift-eval", "FEV": "fev", "TIME": "time", "BOOM": "boom"}
    adapter = PublicSidecar(
        "python", data_dir="pinned-data", revisions={s: "revision1" for s in SUITES}, timeout=10
    )
    report = {
        "data_revisions": {s: "revision1" for s in names.values()},
        "suites": [
            {
                "suite": name,
                "status": "ok",
                "n_expected": 12,
                "skipped": [],
                "rows": [{"full": str(i), "CRPS": 0.5, "MASE": 0.7} for i in range(12)],
            }
            for name in names.values()
        ],
    }
    assert len(adapter.rows_from_report(report, guard)) == 48
    for mutation in ("skip", "revision", "case", "nan"):
        bad = copy.deepcopy(report)
        if mutation == "skip":
            bad["suites"][1]["skipped"] = [{"full": "missing"}]
        if mutation == "revision":
            bad["data_revisions"]["fev"] = "other"
        if mutation == "case":
            bad["suites"][1]["rows"][0]["full"] = "other"
        if mutation == "nan":
            bad["suites"][1]["rows"][0]["MASE"] = float("nan")
        with pytest.raises(ValueError):
            adapter.rows_from_report(bad, guard)


def test_combined_checkpoint_resets_only_edited_optimizer_rows(tmp_path):
    import torch
    from safetensors.torch import load, save

    from cascade.direct.experts import merge_checkpoint
    from cascade.trainer.direct_model import build_direct_model
    from cascade.trainer.moe_model import moe_param_names
    from tests.integration.test_direct_torch import training_spec

    spec, objects = training_spec(tmp_path)
    model = build_direct_model(spec["architecture"], device="cpu")
    weights = model.state_dict()
    names = moe_param_names(model)
    optimizer = {
        f"muon.{name}.{suffix}": torch.ones_like(weights[name])
        for pair in names
        for name in pair
        for suffix in ("momentum", "row_v")
    }
    base = {
        "ranks": {
            "0": {
                "weights": publish(objects, save(weights)),
                "optimizer": publish(objects, save(optimizer)),
            }
        }
    }
    edited = {k: v.clone() for k, v in weights.items()}
    for name in names[0]:
        edited[name][0] += 0.1
    candidate = {"ranks": {"0": {"weights": publish(objects, save(edited))}}}
    merged = merge_checkpoint(
        objects, base, [(candidate, {0: [0]})], names, optimizer_policy="reset_edited_rows"
    )
    result = load(objects.get(merged["ranks"]["0"]["optimizer"]["uri"]))
    for name in names[0]:
        for suffix in ("momentum", "row_v"):
            value = result[f"muon.{name}.{suffix}"]
            assert torch.all(value[0] == 0) and torch.all(value[1:] == 1)
    edited["patch_embed.weight"] = torch.ones(1)  # frozen topology cannot be injected
    candidate["ranks"]["0"]["weights"] = publish(objects, save(edited))
    with pytest.raises((ValueError, KeyError)):
        merge_checkpoint(
            objects, base, [(candidate, {0: [0]})], names, optimizer_policy="reset_edited_rows"
        )


def test_specialist_selection_provenance_replays_and_refuses_changed_ownership():
    from cascade.direct.experts import verify_selection

    evidence = dict(
        anchor="accepted",
        domain="energy",
        panel="a" * 64,
        counts=[[9, 3, 1, 1]],
        fraction=0.5,
        minimum=1,
        maximum=2,
        excluded={},
        selection={"0": [0]},
    )
    verify_selection(
        evidence, anchor="accepted", domain="energy", selection={0: [0]}, layers=1, experts=4
    )
    with pytest.raises(ValueError, match="does not replay"):
        verify_selection(
            evidence, anchor="accepted", domain="energy", selection={0: [1]}, layers=1, experts=4
        )
    with pytest.raises(ValueError, match="provenance"):
        verify_selection(
            evidence, anchor="other", domain="energy", selection={0: [0]}, layers=1, experts=4
        )


def test_generator_archive_is_content_pinned_and_safely_delivered(tmp_path):
    import zipfile

    from cascade.direct.data import SandboxSource, tree_digest

    source = tmp_path / "source"
    source.mkdir()
    (source / "generator.py").write_text("def generate(seed):\n    return iter(())\n")
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr("generator.py", (source / "generator.py").read_bytes())
    objects = FileObjects(tmp_path / "objects")
    spec = dict(
        kind="sandbox",
        archive=publish(objects, body.getvalue()),
        source_sha256=tree_digest(source),
        generator_config=dict(
            corpus_n_series=1,
            min_length=8,
            max_length=8,
            max_total_points=16,
            max_generate_seconds=10,
            max_memory_mb=512,
            max_channels=2,
            sandbox_strict=True,
        ),
        shard_examples=1,
        seed=3,
        blocked_imports=[],
    )
    loaded = SandboxSource(spec, objects=objects)
    assert tree_digest(loaded.root) == spec["source_sha256"]
    with pytest.raises(ValueError, match="source changed"):
        SandboxSource({**spec, "source_sha256": "0" * 64}, objects=objects)


