"""Exercise the actual browser projection, frontier and king-history functions."""
from __future__ import annotations

import json

import pytest

from .test_website_render import _bucket, _html, _render


def run_js(tmp_path, *calls, fixtures=None):
    return _render("index.html", fixtures or {}, "localhost", list(calls), tmp_path)


def test_strict_dominance_ties_and_axis_projections(tmp_path):
    run_js(tmp_path, """
      var pts=[{id:'a',x:1,y:3},{id:'b',x:2,y:2},{id:'c',x:3,y:3},
               {id:'tie',x:2,y:2},{id:'d',x:4,y:1}];
      if(paretoFrontier(pts).map(p=>p.id).join(',')!=='a,b,tie,d') throw Error('wrong dominance');
      var p={parameters:191496256,active:27951168,mase:.6,crps:.4};
      _paretoAxes='size-mase';
      if(paretoCoordinates(p).x!==191496256||paretoCoordinates(p).y!==.6) throw Error('wrong total size');
      _paretoAxes='size-crps';
      if(paretoCoordinates(p).y!==.4) throw Error('wrong CRPS');
      _paretoAxes='mase-crps';
      if(paretoCoordinates(p).x!==.6||paretoCoordinates(p).y!==.4) throw Error('wrong loss axes');
      if(paretoCoordinates({mase:NaN,crps:.4})||paretoCoordinates({mase:.6,crps:null})) throw Error('invalid losses');
      _paretoAxes='size-mase';
      if(paretoCoordinates({parameters:0,mase:.6})) throw Error('invalid size');
      if(checkpointSize({size:'moe-220m'}).total!==null) throw Error('guessed preset size');
    """)


def test_reigns_follow_promotions_not_best_public_scores(tmp_path):
    run_js(tmp_path, """
      _index={rounds:[{epoch_start_block:3,post_round_king_hotkey:'b'}]}; _v2={king:{pointer:'b2'}};
      function entry(hk,ptr,mase,crowned){return {hotkey:hk,pointer:ptr,size:'s',six:{gifteval_mase:mase},crowned:crowned};}
      var a=entry('a','a1',.4), a2=entry('a','a2',.8), b=entry('b','b1',.7,true), b2=entry('b','b2',.9);
      var loser=entry('loser','l1',.1,false);
      var rows=[{id:'1',king:a,entries:[a,loser]}, {id:'2',king:a2,entries:[a2,b]}, {id:'3',king:b2,entries:[b2]}];
      var result=kingTrajectory(rows);
      if(result.history.map(x=>x.entry.pointer).join(',')!=='a2,b2') throw Error('cherry-picked or included loser');
      if(result.current.entry!==b2) throw Error('wrong current king');
      _v2.king.pointer='not-benched';
      if(!kingTrajectory(rows).pending||kingTrajectory(rows).current) throw Error('stale checkpoint presented as current');
    """)


def test_never_connects_across_missing_measurements_and_escapes_labels(tmp_path):
    run_js(tmp_path, """
      _paretoAxes='mase-crps';
      var points=[{id:'1',label:'<img onerror=bad>',kind:'king',x:.8,y:.5,order:0},
                  {id:'3',label:'king',kind:'current',x:.7,y:.4,order:2}];
      var html=paretoChartSVG(points,1000,420);
      if(html.includes('class="king-trajectory"')||html.includes('<img')) throw Error('bridged gap or unescaped label');
      points[1].order=1;
      if(!paretoChartSVG(points,1000,420).includes('class="king-trajectory"')) throw Error('missing chronological edge');
    """)


@pytest.mark.parametrize('suite', ['gifteval', 'boom', 'time', 'fev'])
def test_suite_selection_and_known_revision_mismatches(tmp_path, suite):
    run_js(tmp_path, f"_paretoBenchmark={json.dumps(suite)}", """
      _paretoCatalog={models:[
        {id:'gift-only',label:'gift only',parameters:10,scores:{gifteval:{mase:.8,crps:.5}}},
        {id:'measured',label:'measured',parameters:20,scores:{
          gifteval:{mase:.7,crps:.4,data_revisions:{'gift-eval':'old'}},
          boom:{mase:.7,crps:.4,data_revisions:{boom:'old'}},
          time:{mase:.7,crps:.4,data_revisions:{time:'old'}},
          fev:{mase:.7,crps:.4,data_revisions:{fev:'old'}}}}]};
      _refBench={}; _index={rounds:[]}; _v2={king:{pointer:'k'}};
      var e={hotkey:'hk',pointer:'k',size:'s',params:30,revisions:{'gift-eval':'new',boom:'new',time:'new',fev:'new'},
        six:{gifteval_mase:.6,gifteval_crps:.3,boom_mase:.6,boom_crps:.3,time_mase:.6,time_crps:.3,fev_mase:.6,fev_crps:.3}};
      var data=modelPoints([{id:'r',entries:[e],king:e}]);
      if(data.references.length!==(_paretoBenchmark==='gifteval'?2:1)) throw Error('borrowed another suite');
      if(paretoCoordinates(data.references.find(p=>p.id==='ref:measured'))) throw Error('compared mismatched revisions');
      if(!paretoCoordinates(data.kings[0])) throw Error('lost current king');
    """)


def test_packaged_frontier_renders_and_missing_sizes_are_not_guessed(tmp_path):
    fx = _bucket(v2_from=1000, block=50000, lane_b=True)
    els = run_js(tmp_path, "setPage('train')", "renderTrain()", fixtures=fx)
    assert 'Timestral-5M' in _html(els, 'pareto-rows')
    assert 'TiRex' not in _html(els, 'pareto-rows')
    assert 'missing size/score' in _html(els, 'pareto-status')
    els = run_js(tmp_path, "setPage('train')", "_paretoAxes='mase-crps';renderTrain()", fixtures=fx)
    assert 'TiRex' in _html(els, 'pareto-rows')


@pytest.mark.parametrize('published', [None, {'schema': 99, 'models': []}])
def test_static_catalog_fallback_on_missing_or_invalid_bucket_object(tmp_path, published):
    fx = _bucket(v2_from=1000, block=50000, lane_b=True)
    catalog = fx.pop('benchmarks/model-frontier.json')
    fx['data/benchmark-models.json'] = catalog
    if published is not None:
        fx['benchmarks/model-frontier.json'] = published
    els = run_js(tmp_path, "setPage('train')", "renderTrain()", fixtures=fx)
    assert 'Timestral-5M' in _html(els, 'pareto-rows')


def test_matrix_follows_newly_crowned_king_and_its_actual_training(tmp_path):
    fx = _bucket(v2_from=1000, block=50000, lane_b=True)
    row = fx['receipts/index.json']['rounds'][0]
    row.update(dethroned=True, chal_hotkey='new', post_round_king_hotkey='new')
    bench = fx[f"benchmarks/round-{row['round_id']}.json"]
    challenger = {**bench['entries'][0], 'role': 'challenger', 'miner_hotkey': 'new',
                  'trained_pointer': 'new-pointer', 'scores': {**bench['entries'][0]['scores'],
                                                             'gifteval_mase': .1234}}
    bench['entries'].append(challenger)
    fx['status/v2.json']['king']['pointer'] = 'new-pointer'
    fx[f"training/round-{row['round_id']}.json"] = {
        'legs': [{'role': 'king', 'trained_pointer': bench['entries'][0]['trained_pointer'],
                  'steps': 100, 'tokens_seen': 1000, 'channel_tokens': 1000, 'measured': True},
                 {'role': 'challenger', 'trained_pointer': 'new-pointer',
                  'steps': 200, 'tokens_seen': 2000, 'channel_tokens': 2000, 'measured': True}]}
    els = run_js(tmp_path, "setPage('train')", "renderTrain()", fixtures=fx)
    assert '0.1234' in _html(els, 'train-stats')
    assert '>2k<' in _html(els, 'train-stats')
