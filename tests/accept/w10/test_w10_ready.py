"""W10: `cascade lane-d ready` (plan §1.2 Flow steps 3-4, B13, D43): the
signed `POST /v1/lane-d/ready` with the licence attestation, then the commit
of the returned `commit_payload`, revealed IMMEDIATELY
(`blocks_until_reveal = min_blocks`), never at boundary − margin. Contract:
README.md "ready" and "The commit (D43)"."""

from __future__ import annotations

import ast
import inspect

import pytest

from .helpers import (
    CONTEXT,
    LANE_D_PY,
    LICENCES,
    PATH_IDS,
    READY,
    SUBMIT,
    FakeChain,
    assert_no_secrets,
    block_external_network,
    cli_env,
    digest_of,
    fixture_key,
    install_chain,
    intake,
    require_chosen_path,
    run_cli,
    wallet_args,
    write_checkpoint,
)

COLD = fixture_key("cold").ss58_address


def _min_blocks() -> int:
    """`min_blocks`: the reveal-now floor of lane A's timing helper (its
    default), which D43 says the lane D commit uses. Read once, before the
    tripwire below replaces the helper."""
    from cascade.shared.chain import blocks_until_boundary_reveal

    return int(inspect.signature(blocks_until_boundary_reveal).parameters["min_blocks"].default)


_MIN_BLOCKS = _min_blocks()


def min_blocks() -> int:
    return _MIN_BLOCKS


@pytest.fixture
def env(tmp_path, monkeypatch):
    cli_env(monkeypatch, tmp_path)
    key = fixture_key("miner")
    chain = install_chain(monkeypatch, FakeChain(key))
    blocked = block_external_network(monkeypatch)
    calls = []
    import cascade.miner.cli as cli_mod
    import cascade.shared.chain as chain_mod

    def tripwire(name):
        def _f(*a, **k):
            calls.append(name)
            raise AssertionError(f"the lane D commit path called lane A's {name}")
        return _f

    monkeypatch.setattr(chain_mod, "blocks_until_boundary_reveal",
                        tripwire("blocks_until_boundary_reveal"))
    monkeypatch.setattr(cli_mod, "_resolve_blocks_until_reveal",
                        tripwire("_resolve_blocks_until_reveal"))
    return {"tmp": tmp_path, "key": key, "chain": chain, "blocked": blocked, "lane_a": calls,
            "monkeypatch": monkeypatch}


def _seed(fake, env, capsys, caplog):
    """A submitted, fully uploaded entry (the chosen path)."""
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    res = run_cli(capsys, caplog, "lane-d", "submit", fake.base, str(ckpt),
                  "--context", str(CONTEXT), *wallet_args())
    assert res.rc == 0, f"fixture: submit failed: {res}"
    return digest_of(ckpt)


def _ready(capsys, caplog, fake, digest, *extra, licence="MIT", attest=True):
    args = ["lane-d", "ready", fake.base, "--digest", digest, "--context", str(CONTEXT),
            "--licence", licence, *wallet_args(), *extra]
    if attest:
        args.append("--attest")
    return run_cli(capsys, caplog, *args)


def test_the_attestation_statement_names_both_licences_and_the_data_rule():
    from cascade.miner import lane_d

    text = lane_d.LICENCE_ATTESTATION
    assert isinstance(text, str) and len(text) >= 80
    for word in (*LICENCES, "training data", "teacher"):
        assert word.lower() in text.lower(), f"the attestation does not name {word!r}"
    assert tuple(lane_d.LICENCES) == LICENCES


def test_the_attestation_is_the_intakes_statement():
    """Once W3 is on main its `service.LICENCE_ATTESTATION` is the statement
    the intake accepts; the CLI sends exactly it."""
    from cascade.miner import lane_d

    try:
        from cascade.lane_d import service
    except ImportError:
        pytest.skip("W3's cascade.lane_d.service is not on main yet (W10 depends on W3)")
    assert lane_d.LICENCE_ATTESTATION == service.LICENCE_ATTESTATION
    assert tuple(lane_d.LICENCES) == tuple(service.LICENCES)


@pytest.mark.parametrize("block", [1_000_000, 1_000_000 - 1_000_000 % 900 + 890, 7_201])
def test_d43_ready_commits_the_payload_with_an_immediate_reveal(env, capsys, caplog, block):
    """D43: whatever the block (deep inside a settlement, inside lane A's
    reveal margin, anywhere), `blocks_until_reveal == min_blocks` and lane A's
    boundary timing is never consulted."""
    path = require_chosen_path()
    env["chain"].block = block
    with intake(path, coldkey=COLD) as fake:
        dg = _seed(fake, env, capsys, caplog)
        res = _ready(capsys, caplog, fake, dg)
        (req,) = fake.routes(READY)
    assert res.rc == 0, res
    want = f"metro-v1:weights:c{CONTEXT}:hippius:lane-d/{env['key'].ss58_address}/{dg}"
    assert env["chain"].commits == [(want, min_blocks())], env["chain"].commits
    assert env["lane_a"] == [], f"lane A timing used: {env['lane_a']}"
    assert req["headers"]["X-Content-Digest"] == dg
    assert int(req["headers"]["X-Context-Length"]) == CONTEXT


def test_ready_sends_the_licence_and_the_exact_attestation(env, capsys, caplog):
    import json

    from cascade.miner import lane_d
    path = require_chosen_path()
    with intake(path, coldkey=COLD) as fake:
        dg = _seed(fake, env, capsys, caplog)
        res = _ready(capsys, caplog, fake, dg, licence="MIT")
        (req,) = fake.routes(READY)
    assert res.rc == 0, res
    body = json.loads(req["body"])
    assert body == {"licence": "MIT", "attestation": lane_d.LICENCE_ATTESTATION}


def test_ready_result_carries_the_ready_record_and_the_commit(env, capsys, caplog):
    path = require_chosen_path()
    with intake(path, coldkey=COLD, ready_block=5000, ttl=7200) as fake:
        dg = _seed(fake, env, capsys, caplog)
        res = _ready(capsys, caplog, fake, dg)
        sigs = [s for s in fake.signatures if s]
    assert res.rc == 0, res
    d = res.data
    assert d["digest"] == dg
    assert d["coldkey_at_ready"] == COLD
    assert d["ready_block"] == 5000 and d["ready_expires_block"] == 12200
    assert d["commit_payload"].startswith(f"metro-v1:weights:c{CONTEXT}:hippius:")
    assert d["blocks_until_reveal"] == min_blocks()
    assert d["commit_block"] == env["chain"].block
    assert_no_secrets(res, key=env["key"], extra=tuple(sigs))
    assert not env["chain"].coldkey_touched


def test_ready_without_the_attestation_flag_sends_nothing(env, capsys, caplog):
    with intake(PATH_IDS[0], coldkey=COLD) as fake:
        res = _ready(capsys, caplog, fake, "ab" * 32, attest=False)
        assert fake.requests == []
    assert res.rc != 0 and res.code == "licence_attestation_missing", res
    assert env["chain"].commits == []


def test_ready_refuses_a_licence_outside_the_open_pair(env, capsys, caplog):
    with intake(PATH_IDS[0], coldkey=COLD) as fake:
        res = _ready(capsys, caplog, fake, "ab" * 32, licence="CC-BY-NC-4.0")
        assert fake.requests == []
    assert res.rc != 0, res
    assert env["chain"].commits == []


@pytest.mark.parametrize("edit", [
    lambda p: p.replace("metro-v1:weights:", "metro-v1:gen:"),
    lambda p: p.replace(f"c{CONTEXT}", "c8192"),
    lambda p: p.rsplit("/", 1)[0] + "/" + "0" * 64,
], ids=["wrong_grammar", "wrong_context", "wrong_digest"])
def test_a_payload_that_is_not_this_entry_is_never_committed(env, capsys, caplog, edit):
    path = require_chosen_path()
    with intake(path, coldkey=COLD) as fake:
        dg = _seed(fake, env, capsys, caplog)
        fake.payload_edit = edit
        res = _ready(capsys, caplog, fake, dg)
    assert res.rc == 1 and res.code == "bad_commit_payload", res
    assert env["chain"].commits == []


@pytest.mark.parametrize("status,code,rc", [(409, "coldkey_cap_open", 1), (409, "ready_expired", 1),
                                            (400, "licence_attestation_missing", 1),
                                            (503, "chain_unavailable", 3)])
def test_an_intake_refusal_at_ready_commits_nothing(env, capsys, caplog, status, code, rc):
    path = require_chosen_path()
    with intake(path, coldkey=COLD) as fake:
        dg = _seed(fake, env, capsys, caplog)
        fake.refuse[READY] = (status, code)
        res = _ready(capsys, caplog, fake, dg)
    assert res.rc == rc, res
    if rc == 1:
        assert res.code == code, res
    assert env["chain"].commits == []


def test_a_chain_refusal_of_the_commit_is_a_network_failure(env, capsys, caplog):
    path = require_chosen_path()
    with intake(path, coldkey=COLD) as fake:
        dg = _seed(fake, env, capsys, caplog)
        env["chain"].fail_commit = True
        res = _ready(capsys, caplog, fake, dg)
    assert res.rc == 3, res


def test_the_wallet_flags_reach_the_chain_client(env, capsys, caplog):
    path = require_chosen_path()
    with intake(path, coldkey=COLD) as fake:
        dg = _seed(fake, env, capsys, caplog)
        _ready(capsys, caplog, fake, dg)
    for kw in env["chain"].from_config_calls:
        assert kw.get("wallet_name") == "w10" and kw.get("wallet_hotkey") == "lane-d", kw
    assert fake.routes(SUBMIT) and fake.routes(READY)


def test_the_lane_d_module_never_names_lane_a_reveal_timing():
    """D43, statically: the lane D CLI never references lane A's boundary
    timing helpers or the reveal margin."""
    assert LANE_D_PY.is_file(), "cascade/miner/lane_d.py does not exist"
    tree = ast.parse(LANE_D_PY.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    names |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
              and isinstance(n.value, str)}
    banned = {"blocks_until_boundary_reveal", "_resolve_blocks_until_reveal",
              "reveal_margin_blocks", "next_epoch"}
    assert not names & banned, f"lane_d.py names lane A timing: {sorted(names & banned)}"
