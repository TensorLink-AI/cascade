"""W10: `cascade lane-d submit` (plan §1.2 Flow steps 1-2): a signed
`POST /v1/lane-d/submit`, then the direct upload through the grants of the
W3a-chosen path only; nothing transits the box; no grant is ever printed.
Contract: README.md "submit"."""

from __future__ import annotations

import json

import pytest

from .helpers import (
    ALLOWED_FILES,
    CONTEXT,
    PATH_IDS,
    SUBMIT,
    FakeChain,
    assert_no_secrets,
    block_external_network,
    canonical,
    cli_env,
    digest_of,
    fixture_key,
    install_chain,
    intake,
    manifest_of,
    require_chosen_path,
    run_cli,
    wallet_args,
    write_checkpoint,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cli_env(monkeypatch, tmp_path)
    key = fixture_key("miner")
    chain = install_chain(monkeypatch, FakeChain(key))
    blocked = block_external_network(monkeypatch)
    return {"tmp": tmp_path, "key": key, "chain": chain, "blocked": blocked}


def _submit(capsys, caplog, fake, ckpt, *extra):
    return run_cli(capsys, caplog, "lane-d", "submit", fake.base, str(ckpt),
                   "--context", str(CONTEXT), *wallet_args(), *extra)


def test_submit_uploads_every_file_through_the_chosen_paths_grants(env, capsys, caplog):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(path, coldkey=fixture_key("cold").ss58_address) as fake:
        res = _submit(capsys, caplog, fake, ckpt)
        assert res.rc == 0, res
        for name in ALLOWED_FILES:
            assert fake.assembled(name) == (ckpt / name).read_bytes(), \
                f"{name}: the uploaded bytes differ from the file"
        (req,) = fake.routes(SUBMIT)
    assert req["headers"].get("X-Upload-Path") == path
    assert env["blocked"] == [], f"a non-loopback connect was attempted: {env['blocked']}"


def test_submit_is_signed_over_w3s_canonical_message(env, capsys, caplog):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(path, coldkey="c") as fake:
        res = _submit(capsys, caplog, fake, ckpt)
        assert res.rc == 0, res
        (req,) = fake.routes(SUBMIT)
    h = req["headers"]
    assert h["X-Miner-Hotkey"] == env["key"].ss58_address
    assert h["X-Content-Digest"] == digest_of(ckpt)
    assert int(h["X-Context-Length"]) == CONTEXT
    assert req["body"] == canonical(manifest_of(ckpt)), \
        "the body is not the canonical manifest (intake.canonical_manifest)"


def test_submit_result_names_the_entry_and_never_a_grant(env, capsys, caplog):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(path, coldkey="c") as fake:
        res = _submit(capsys, caplog, fake, ckpt)
        sigs = [s for s in fake.signatures if s]
    assert res.rc == 0, res
    d = res.data
    assert d["digest"] == digest_of(ckpt)
    assert d["hotkey"] == env["key"].ss58_address
    assert d["context"] == CONTEXT
    assert d["upload_path"] == path
    assert d["uploaded_bytes"] == sum((ckpt / n).stat().st_size for n in ALLOWED_FILES)
    assert "grants" not in d
    assert_no_secrets(res, key=env["key"], extra=tuple(sigs))
    assert not env["chain"].coldkey_touched
    assert env["chain"].commits == [], "submit never commits (the commit follows ready)"


@pytest.mark.parametrize("bad", ["extra_py", "pickle", "missing_weights"])
def test_submit_refuses_a_wrong_file_set_before_any_request(env, capsys, caplog, bad):
    extra = {"extra_py": {"evil.py": b"raise SystemExit"},
             "pickle": {"model.pt": b"\x80\x04K\x01."}}.get(bad, {})
    drop = ("weights.safetensors",) if bad == "missing_weights" else ()
    ckpt = write_checkpoint(env["tmp"] / "ckpt", extra=extra, drop=drop)
    with intake(PATH_IDS[0], coldkey="c") as fake:
        res = _submit(capsys, caplog, fake, ckpt)
        assert fake.requests == [], "a request was sent for a refused checkpoint"
    assert res.rc != 0 and res.code == "file_not_allowed", res


def test_submit_refuses_an_unarmed_context_before_any_request(env, capsys, caplog):
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(PATH_IDS[0], coldkey="c") as fake:
        res = run_cli(capsys, caplog, "lane-d", "submit", fake.base, str(ckpt),
                      "--context", "1234", *wallet_args())
        assert fake.requests == []
    assert res.rc != 0 and res.code == "context_not_armed", res


@pytest.mark.parametrize("status,code", [(409, "duplicate_weights"), (409, "hotkey_entry_open"),
                                         (403, "not_registered")])
def test_an_intake_refusal_is_rejected_with_its_code_and_nothing_uploaded(
        env, capsys, caplog, status, code):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(path, coldkey="c") as fake:
        fake.refuse[SUBMIT] = (status, code)
        res = _submit(capsys, caplog, fake, ckpt)
        assert fake.puts == {}
    assert res.rc == 1 and res.code == code, res


def test_an_intake_503_is_a_network_failure(env, capsys, caplog):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(path, coldkey="c") as fake:
        fake.refuse[SUBMIT] = (503, "storage_unavailable")
        res = _submit(capsys, caplog, fake, ckpt)
    assert res.rc == 3, res


def test_an_unreachable_intake_is_a_network_failure(env, capsys, caplog):
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    res = run_cli(capsys, caplog, "lane-d", "submit", "http://127.0.0.1:9", str(ckpt),
                  "--context", str(CONTEXT), *wallet_args())
    assert res.rc == 3, res


def test_plain_http_to_a_non_local_intake_is_refused(env, capsys, caplog):
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    res = run_cli(capsys, caplog, "lane-d", "submit", "http://intake.example.org", str(ckpt),
                  "--context", str(CONTEXT), *wallet_args())
    assert res.rc != 0 and env["blocked"] == [], res


def test_an_answer_for_another_upload_path_is_refused_without_uploading(env, capsys, caplog):
    path = require_chosen_path()
    other = next(p for p in PATH_IDS if p != path)
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(other, coldkey="c") as fake:
        res = _submit(capsys, caplog, fake, ckpt)
        assert fake.puts == {} and not [r for r in fake.requests if r["method"] == "PUT"]
    assert res.rc == 1 and res.code == "upload_path_not_offered", res


def _drop_last_request(grants):
    g = json.loads(json.dumps(grants))
    g["files"]["weights.safetensors"]["requests"] = \
        g["files"]["weights.safetensors"]["requests"][:-1] or []
    return g


def _overlap(grants):
    g = json.loads(json.dumps(grants))
    reqs = g["files"]["config.json"]["requests"]
    reqs[0]["length"] = reqs[0]["length"] + 1
    return g


def _missing_file(grants):
    g = json.loads(json.dumps(grants))
    del g["files"]["model.py"]
    return g


def _extra_file(grants):
    g = json.loads(json.dumps(grants))
    g["files"]["evil.py"] = g["files"]["config.json"]
    return g


@pytest.mark.parametrize("edit", [_drop_last_request, _overlap, _missing_file, _extra_file])
def test_grants_that_do_not_cover_the_manifest_exactly_are_refused(env, capsys, caplog, edit):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt", tag="cover" * 400)
    with intake(path, coldkey="c") as fake:
        fake.grants_edit = edit
        res = _submit(capsys, caplog, fake, ckpt)
        assert not [r for r in fake.requests if r["method"] == "PUT"], \
            "a PUT was sent for grants that do not cover the manifest"
    assert res.rc == 1 and res.code == "bad_grants", res


def test_a_grant_url_that_is_not_https_or_loopback_is_refused(env, capsys, caplog):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")

    def to_plain_http(grants):
        g = json.loads(json.dumps(grants))
        for f in g["files"].values():
            for r in f["requests"]:
                r["url"] = r["url"].replace("http://127.0.0.1", "http://198.51.100.7")
        return g

    with intake(path, coldkey="c") as fake:
        fake.grants_edit = to_plain_http
        res = _submit(capsys, caplog, fake, ckpt)
    assert env["blocked"] == [], "the CLI connected to a plain-http grant URL"
    assert res.rc == 1 and res.code == "insecure_grant", res
    assert_no_secrets(res, extra=("198.51.100.7",))


def test_a_failed_upload_is_a_registry_failure(env, capsys, caplog):
    path = require_chosen_path()
    ckpt = write_checkpoint(env["tmp"] / "ckpt")
    with intake(path, coldkey="c") as fake:
        fake.fail_put = True
        res = _submit(capsys, caplog, fake, ckpt)
    assert res.rc == 4 and res.code == "upload_failed", res
    assert_no_secrets(res, key=env["key"])
