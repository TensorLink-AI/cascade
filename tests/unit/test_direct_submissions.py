"""Direct-to-gateway submissions: store, vault refs, intake, champion policy."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from types import SimpleNamespace

import pytest

from cascade.funding.champion import CHAMPION_INDEX_KEY, ChampionPublisher, should_publish
from cascade.funding.intake import FundingIntake, canonical_fund_message
from cascade.funding.queue import FundedQueue
from cascade.funding.store import (
    SubmissionStore,
    extract_zip_safely,
    is_vault_ref,
    parse_vault_ref,
    vault_ref,
)
from cascade.funding.vault import PayerKeyVault
from cascade.interface.validation import parse_commit
from cascade.miner.cli import zip_repo_bytes
from cascade.shared.hippius import StorageError, fetch_from_hub

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
HK2 = "5FakeHotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


GOOD_ZIP = make_zip({"generator.py": b"def generate():\n    yield [1.0]\n"})
GOOD_DIGEST = hashlib.sha256(GOOD_ZIP).hexdigest()


# ── vault refs ride the existing grammar ─────────────────────────────────────


def test_vault_ref_parses_as_ordinary_commit():
    ref = vault_ref(GOOD_DIGEST)
    assert is_vault_ref(ref)
    assert parse_vault_ref(ref) == GOOD_DIGEST
    # The consensus-critical property: a vault commit is a byte-ordinary
    # hippius payload every deployed validator already parses — participant
    # sets cannot fork on the scheme.
    parsed = parse_commit(f"metro-v1:gen:hippius:{ref}")
    assert parsed is not None and parsed.ref == ref
    assert parse_vault_ref("somebody/repo@sha256:" + "a" * 64) is None


# ── the private store ────────────────────────────────────────────────────────


def test_store_put_owner_and_roundtrip(tmp_path):
    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(GOOD_ZIP, HK)
    assert digest == GOOD_DIGEST
    assert store.owner(digest) == HK
    assert store.put(GOOD_ZIP, HK) == digest          # own re-upload: idempotent
    with pytest.raises(StorageError, match="digest_owned"):
        store.put(GOOD_ZIP, HK2)                      # earliest upload owns bytes
    out = store.extract(digest, tmp_path / "out")
    assert (out / "generator.py").read_bytes().startswith(b"def generate")


def test_store_rejects_oversize_and_hostile_zips(tmp_path):
    store = SubmissionStore(tmp_path / "vault", max_bytes=64)
    with pytest.raises(StorageError, match="zip_too_large"):
        store.put(GOOD_ZIP, HK)
    store2 = SubmissionStore(tmp_path / "vault2")
    with pytest.raises(StorageError, match="not a valid zip"):
        store2.put(b"PK\x03\x04 garbage", HK)
    evil = make_zip({"../escape.py": b"x"})
    with pytest.raises(StorageError, match="escapes"):
        store2.put(evil, HK)
    with pytest.raises(StorageError, match="escapes"):
        extract_zip_safely(make_zip({"/abs.py": b"x"}), tmp_path / "x")


def test_store_stage_for_dispatch_ships_one_zip_only(tmp_path):
    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(GOOD_ZIP, HK)
    other = store.put(make_zip({"b.py": b"2"}), HK2)
    staged = store.stage_for_dispatch(digest, tmp_path / "pod")
    names = sorted(p.name for p in (tmp_path / "pod").iterdir())
    assert names == [f"{digest}.zip"]                 # never the sibling, never meta
    assert staged.read_bytes() == GOOD_ZIP
    assert other != digest


def test_fetch_from_hub_resolves_vault_refs_from_env_dir(tmp_path, monkeypatch):
    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(GOOD_ZIP, HK)
    monkeypatch.setenv("CASCADE_VAULT_DIR", str(tmp_path / "vault"))
    dest = fetch_from_hub(vault_ref(digest), tmp_path / "fetched")
    assert (dest / "generator.py").is_file()
    # Marker honoured: a second fetch reuses the completed snapshot.
    assert fetch_from_hub(vault_ref(digest), tmp_path / "fetched") == dest
    monkeypatch.delenv("CASCADE_VAULT_DIR")
    with pytest.raises(StorageError, match="generator_artifact_unreachable"):
        fetch_from_hub(vault_ref("f" * 64), tmp_path / "missing")


# ── one-request submit-and-fund at the intake ────────────────────────────────


def _submit_headers(hotkey: str, body: bytes, *, ts: float, key: str = "",
                    sign=lambda m: b"sig") -> dict:
    digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
    h = {
        "X-Miner-Hotkey": hotkey,
        "X-Content-Digest": digest,
        "X-Timestamp": str(int(ts)),
        "X-Signature": sign(canonical_fund_message(
            "submit", hotkey, digest, str(int(ts)))).hex()
            if callable(sign) else "",
    }
    if key:
        h["X-Lium-Api-Key"] = key
    return h


def make_intake(tmp_path, *, resolve=lambda hk, ref: None, require_signature=False):
    return FundingIntake(
        FundedQueue(tmp_path / "queue.json"),
        PayerKeyVault(dir=None),
        resolve_reveal=resolve,
        require_signature=require_signature,
        store=SubmissionStore(tmp_path / "subs"),
    )


def test_submit_stores_code_and_parks_funding_until_reveal(tmp_path):
    revealed: dict = {}
    intake = make_intake(tmp_path, resolve=lambda hk, ref: revealed.get((hk, ref)))
    headers = _submit_headers(HK, GOOD_ZIP, ts=time.time(), key="sk-live")
    status, body = intake.submit(headers, GOOD_ZIP)
    assert (status, body["status"], body["funding"]) == (201, "stored", "pending_reveal")
    ref = body["ref"]
    assert parse_commit(body["commit_payload"]).ref == ref
    assert intake.store.owner(GOOD_DIGEST) == HK
    assert intake.vault.get(HK) == "sk-live"
    assert intake.queue.get(HK).status == "pending_reveal"
    # Reveal lands on chain → the next queue read promotes with the REAL block.
    revealed[(HK, ref)] = 4242
    _, view = intake.queue_view()
    entry = intake.queue.get(HK)
    assert (entry.status, entry.reveal_block) == ("queued", 4242)
    assert view["queued_depth"] == 1


def test_submit_without_key_stores_only(tmp_path):
    intake = make_intake(tmp_path)
    status, body = intake.submit(_submit_headers(HK, GOOD_ZIP, ts=time.time()),
                                 GOOD_ZIP)
    assert (status, body["funding"]) == (201, "none")
    assert intake.queue.get(HK) is None
    assert intake.vault.get(HK) is None


def test_submit_digest_binding_and_fail_closed_codes(tmp_path):
    intake = make_intake(tmp_path)
    # Body tampered after signing/declaring → refused before storing.
    headers = _submit_headers(HK, GOOD_ZIP, ts=time.time())
    status, body = intake.submit(headers, GOOD_ZIP + b"tamper")
    assert (status, body["code"]) == (400, "digest_mismatch")
    assert not intake.store.has(GOOD_DIGEST)
    status, body = intake.submit({"X-Miner-Hotkey": HK}, GOOD_ZIP)
    assert (status, body["code"]) == (400, "missing_digest")
    status, body = intake.submit(_submit_headers("", GOOD_ZIP, ts=time.time()),
                                 GOOD_ZIP)
    assert (status, body["code"]) == (400, "missing_hotkey")
    # Second hotkey claiming the same bytes → 409.
    intake.submit(_submit_headers(HK, GOOD_ZIP, ts=time.time()), GOOD_ZIP)
    status, body = intake.submit(_submit_headers(HK2, GOOD_ZIP, ts=time.time()),
                                 GOOD_ZIP)
    assert (status, body["code"]) == (409, "digest_owned")


def test_submit_signature_binds_content(tmp_path):
    def verify(hotkey, msg, sig):
        return sig == hashlib.sha256(hotkey.encode() + msg).hexdigest()

    def sign(msg):
        return hashlib.sha256(HK.encode() + msg)

    intake = FundingIntake(
        FundedQueue(tmp_path / "queue.json"), PayerKeyVault(dir=None),
        resolve_reveal=lambda hk, ref: None, require_signature=True,
        verify=verify, store=SubmissionStore(tmp_path / "subs"),
    )
    headers = _submit_headers(HK, GOOD_ZIP, ts=time.time(),
                              sign=lambda m: sign(m).digest())
    assert intake.submit(headers, GOOD_ZIP)[0] == 201
    # A signature over DIFFERENT bytes cannot be replayed for this body.
    other = make_zip({"other.py": b"x"})
    headers_other = _submit_headers(HK, other, ts=time.time(),
                                    sign=lambda m: sign(m).digest())
    headers_other["X-Content-Digest"] = f"sha256:{GOOD_DIGEST}"
    status, body = intake.submit(headers_other, GOOD_ZIP)
    assert (status, body["code"]) == (400, "bad_signature")


def test_submit_disabled_without_store(tmp_path):
    intake = FundingIntake(
        FundedQueue(tmp_path / "queue.json"), PayerKeyVault(dir=None),
        resolve_reveal=lambda hk, ref: None, require_signature=False,
    )
    status, body = intake.submit(_submit_headers(HK, GOOD_ZIP, ts=time.time()),
                                 GOOD_ZIP)
    assert (status, body["code"]) == (503, "submissions_disabled")


# ── champion publication policy ──────────────────────────────────────────────


class FakePublicStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_bytes(self, key, data, *, content_type="", acl=""):
        self.objects[key] = bytes(data)

    def put_text(self, key, text, *, content_type="", acl=""):
        self.objects[key] = text.encode()

    def get_text(self, key):
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key].decode()


def _publisher(tmp_path, policy, **kw):
    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(GOOD_ZIP, HK)
    public = FakePublicStore()
    pub = ChampionPublisher(store, public, policy=policy,
                            state_path=tmp_path / "state.json", **kw)
    return pub, public, digest


def test_should_publish_matrix():
    assert should_publish("crown", reign_rounds=0, delay_rounds=5)
    assert not should_publish("delay", reign_rounds=1, delay_rounds=2)
    assert should_publish("delay", reign_rounds=2, delay_rounds=2)
    assert not should_publish("dethrone", reign_rounds=99, delay_rounds=0)


def test_crown_policy_publishes_immediately(tmp_path):
    pub, public, digest = _publisher(tmp_path, "crown")
    assert pub.note_king(HK, vault_ref(digest), "r1") == [digest]
    assert public.objects[f"champions/{digest}.zip"] == GOOD_ZIP
    index = json.loads(public.objects[CHAMPION_INDEX_KEY])
    assert index["latest"]["digest"] == digest
    assert pub.note_king(HK, vault_ref(digest), "r2") == []   # idempotent reign


def test_delay_policy_publishes_after_reign(tmp_path):
    pub, public, digest = _publisher(tmp_path, "delay", delay_rounds=2)
    assert pub.note_king(HK, vault_ref(digest), "r1") == []   # reign_rounds=0
    assert pub.note_king(HK, vault_ref(digest), "r2") == []   # 1
    assert pub.note_king(HK, vault_ref(digest), "r3") == [digest]  # 2 ≥ delay


def test_dethrone_policy_reveals_only_at_handoff(tmp_path):
    pub, public, digest = _publisher(tmp_path, "dethrone")
    for r in ("r1", "r2", "r3"):
        assert pub.note_king(HK, vault_ref(digest), r) == []
    assert not public.objects                                  # live reign: private
    # A new king appears → the deposed vault king reveals.
    assert pub.note_king(HK2, "somebody/newking@sha256:" + "b" * 64, "r4") == [digest]
    assert public.objects[f"champions/{digest}.zip"] == GOOD_ZIP


def test_hub_ref_king_is_a_noop(tmp_path):
    pub, public, _ = _publisher(tmp_path, "crown")
    assert pub.note_king(HK, "ns/genesis@sha256:" + "c" * 64, "r1") == []
    assert not public.objects


# ── trainer hooks ────────────────────────────────────────────────────────────


def _hook_runner(tmp_path, **round_kw):
    from cascade.shared.config import RoundConfig
    from cascade.trainer.loop import TrainerRunner

    rnd = RoundConfig(**round_kw)
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)
    for name in ("_submission_store", "_verify_vault_ownership", "_maybe_publish_champion"):
        setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
    return fake


def test_vault_ownership_guard(tmp_path):
    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(GOOD_ZIP, HK)
    runner = _hook_runner(tmp_path, submission_vault_dir="vault")
    mine = SimpleNamespace(hotkey=HK, ref=vault_ref(digest))
    thief = SimpleNamespace(hotkey=HK2, ref=vault_ref(digest))       # copied digest
    ghost = SimpleNamespace(hotkey=HK2, ref=vault_ref("d" * 64))     # never uploaded
    hub = SimpleNamespace(hotkey=HK2, ref="ns/x@sha256:" + "e" * 64)  # hub ref passes
    kept = runner._verify_vault_ownership([mine, thief, ghost, hub])
    assert kept == [mine, hub]
    # No store configured ⇒ vault refs cannot resolve and are dropped.
    runner2 = _hook_runner(tmp_path, submission_vault_dir="")
    assert runner2._verify_vault_ownership([mine, hub]) == [hub]


def test_maybe_publish_champion_wires_policy(tmp_path):
    store = SubmissionStore(tmp_path / "vault")
    digest = store.put(GOOD_ZIP, HK)
    public = FakePublicStore()
    runner = _hook_runner(tmp_path, submission_vault_dir="vault",
                          champion_publish="crown")
    runner.manifest_store = lambda: public
    king = SimpleNamespace(hotkey=HK, ref=vault_ref(digest))
    runner._maybe_publish_champion(king, "r1")
    assert f"champions/{digest}.zip" in public.objects


# ── deterministic client-side packaging ──────────────────────────────────────


def test_zip_repo_bytes_is_deterministic(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / "generator.py").write_bytes(b"code")
    (repo / "sub" / "util.py").write_bytes(b"more")
    a = zip_repo_bytes(repo)
    time.sleep(0.01)
    (repo / "generator.py").touch()                    # mtime must not matter
    b = zip_repo_bytes(repo)
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    (repo / "generator.py").write_bytes(b"code2")
    assert zip_repo_bytes(repo) != a
    # …and the archive round-trips through the store's safe extraction.
    out = extract_zip_safely(a, tmp_path / "out")
    assert (out / "sub" / "util.py").read_bytes() == b"more"
