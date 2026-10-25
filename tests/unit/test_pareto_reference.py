"""Catalog ingestion preserves suite-level provenance and refuses invented losses."""
from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import refresh_pareto_reference as frontier

SOURCE = 'mistralai/Timestral-209M@' + 'a' * 40


@pytest.fixture
def metadata(monkeypatch):
    calls = []

    def fetch(repo, revision):
        calls.append((repo, revision))
        return {'parameters': 209312256, 'model_revision': revision, 'size_source': 'https://example.com'}

    monkeypatch.setattr(frontier, 'model_metadata', fetch)
    return calls


def test_merge_keeps_provenance_specific_to_measured_suites(metadata):
    gift = {'mase': .8, 'crps': .5}
    catalog = {'models': [{'id': 'timestral-209m', 'basis': 'published',
                           'scores': {'gifteval': deepcopy(gift)}}]}
    frontier.add_measured_reference(catalog, {
        'kind': 'reference_bench', 'preset': 'timestral-209m', 'source': SOURCE,
        'scores': {'boom_mase': .6, 'boom_crps': .4}, 'data_revisions': {'boom': 'b1'}})
    model = catalog['models'][0]
    assert model['scores']['gifteval'] == gift
    assert model['basis'] == 'published'
    assert model['scores']['boom']['basis'] == 'same-pipeline reference'
    assert model['scores']['boom']['data_revisions'] == {'boom': 'b1'}
    assert model['parameters'] == 209312256
    assert metadata == [('mistralai/Timestral-209M', 'a' * 40)]


@pytest.mark.parametrize('scores', [
    {}, {'fev_skill': .9}, {'fev_mase': .6}, {'fev_mase': True, 'fev_crps': .4},
    {'fev_mase': .6, 'fev_crps': float('nan')}, {'fev_mase': -.1, 'fev_crps': .4},
])
def test_refuses_missing_invalid_or_skill_scores(metadata, scores):
    with pytest.raises(ValueError):
        frontier.add_measured_reference({'models': []}, {
            'kind': 'reference_bench', 'preset': 'timestral-209m', 'source': SOURCE, 'scores': scores})
    assert not metadata


def test_refuses_unpinned_source(metadata):
    with pytest.raises(ValueError, match='pinned'):
        frontier.add_measured_reference({'models': []}, {
            'kind': 'reference_bench', 'source': 'mistralai/Timestral-209M@main'})
    assert not metadata


def test_packaged_preset_sizes_match_current_architectures():
    import json

    catalog = json.loads(frontier.OUT.read_text())
    assert catalog['cascade_presets'] == frontier.cascade_presets()
