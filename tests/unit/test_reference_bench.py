"""References retain FEV losses and cannot relabel an incomplete Timestral run."""
from __future__ import annotations

import pytest

from scripts.publish_reference_bench import reference_scores

SOURCE = "mistralai/Timestral-209M@" + "a" * 40


def report(fev=True):
    suites = ["gift-eval", "boom", "time"] + (["fev"] if fev else [])
    return {"suites": [{"suite": name, "status": "ok", "metrics": {"crps": .4, "mase": .6}}
                       for name in suites]}


def test_timestral_reference_preserves_fev():
    scores = reference_scores(report(), preset="timestral-209m", source=SOURCE)
    assert len(scores) == 8
    assert scores["fev_crps"] == .4 and scores["fev_mase"] == .6


def test_timestral_requires_a_complete_four_suite_report():
    with pytest.raises(ValueError, match="FEV"):
        reference_scores(report(False), preset="timestral-209m", source=SOURCE)


@pytest.mark.parametrize("source", ["mistralai/Timestral-209M@main", "Datadog/Toto-2.0@" + "a" * 40])
def test_timestral_requires_its_exact_checkpoint_revision(source):
    with pytest.raises(ValueError, match="revision"):
        reference_scores(report(), preset="timestral-209m", source=source)


def test_archived_six_score_reference_still_works():
    assert len(reference_scores(report(False), preset="toto2-4m", source="old")) == 6


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), 0, -1])
def test_invalid_loss_is_refused(loss):
    doc = report()
    doc["suites"][-1]["metrics"]["crps"] = loss
    with pytest.raises(ValueError, match="finite and positive"):
        reference_scores(doc, preset="timestral-209m", source=SOURCE)


def test_dry_run_exports_local_document_without_storage(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from scripts import publish_reference_bench as publish

    path, out = tmp_path / 'report.json', tmp_path / 'reference.json'
    path.write_text(json.dumps({**report(), 'data_revisions': {'fev': 'f1'}}))
    monkeypatch.setattr(sys, 'argv', ['publish_reference_bench', str(path),
                                    '--preset', 'timestral-209m', '--source', SOURCE,
                                    '--dry-run', '--out', str(out)])
    assert publish.main() == 0
    doc = json.loads(out.read_text())
    assert doc['scores']['fev_mase'] == .6
    assert doc['data_revisions'] == {'fev': 'f1'}
    assert '[dry-run]' in capsys.readouterr().out
