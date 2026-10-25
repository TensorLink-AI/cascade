"""ChainClient.poll_commitments robustness (bittensor 11.x, DEC-CA-0051).

The whole netuid's ``RevealedCommitments`` is read in ONE ``query_map`` and
decoded per entry by the repo's tolerant decoder (both substrate renderings),
so one miner's malformed reveal costs exactly that entry — never the field
(10.5's SDK batch decoder raised for the whole netuid: a single-commit
field-wide DoS). If the map read itself fails, the poll degrades to one read
per registered hotkey, skipping only the unreadable ones."""

from __future__ import annotations

from types import SimpleNamespace

from cascade.shared.chain import ChainClient

HK = [f"5Hotkey{i}" for i in range(4)]
CK = [f"5Cold{i}" for i in range(4)]


def _payload(name: str) -> str:
    return f"metro-v1:gen:hippius:acct/{name}@sha256:" + "ab" * 32


def _scale_wrap(payload: str, *, as_hex: bool) -> str:
    scale = ((len(payload) << 2) | 1).to_bytes(2, "little")
    raw = scale + payload.encode()
    return "0x" + raw.hex() if as_hex else raw.decode("utf-8", errors="ignore")


# 91 chars: the SCALE prefix (0x6d 0x01 = 'm\x01') is valid UTF-8, so substrate
# really does return this one as a raw string — the payload-length lottery.
RAW_PAYLOAD = "metro-v1:gen:hippius:acct/gen1@hf:" + "9" * 57
assert len(RAW_PAYLOAD) == 91


def _meta():
    """The 11.x typed metagraph shape ChainClient reads: uid-ordered neurons."""
    return SimpleNamespace(block=500, neurons=[
        SimpleNamespace(hotkey=hk, coldkey=ck) for hk, ck in zip(HK, CK, strict=True)])


# per-hotkey RevealedCommitments values: [(scale-bytes, reveal_block)]; uid 2
# has none at all. HK1 carries an older reveal + a garbage entry + its latest
# (raw-rendered) reveal; a foreign hotkey not on the metagraph is ignored.
_STORE = {
    HK[0]: [(_scale_wrap(_payload("gen0"), as_hex=True), 100)],
    HK[1]: [(_scale_wrap(_payload("gen1old"), as_hex=True), 90),
            ("not-scale-garbage", 95),
            (_scale_wrap(RAW_PAYLOAD, as_hex=False), 110)],
    HK[3]: [(_scale_wrap(_payload("gen3"), as_hex=True), 130)],
    "5ForeignHotkeyNotOnMetagraph": [(_scale_wrap(_payload("x"), as_hex=True), 1)],
}


class _Sub:
    """Fake 11.x subtensor. ``map_raises`` makes the one-shot ``query_map``
    fail (the fallback path); ``bad_hotkeys`` make that hotkey's per-key
    ``query`` fail too."""

    def __init__(self, *, map_raises=False, bad_hotkeys=()):
        self.map_raises = map_raises
        self.bad_hotkeys = set(bad_hotkeys)
        self.map_calls = 0
        self.query_calls = 0
        self.blocks = []

    def query_map(self, item, params=None, *, block=None):
        assert tuple(item)[:2] == ("Commitments", "RevealedCommitments")
        assert params == [259]
        self.map_calls += 1
        self.blocks.append(block)
        if self.map_raises:
            raise ConnectionError("websocket closed mid-map")
        return list(_STORE.items())

    def query(self, item, params=None, *, block=None):
        assert tuple(item)[:2] == ("Commitments", "RevealedCommitments")
        self.query_calls += 1
        self.blocks.append(block)
        netuid, hotkey = params
        if hotkey in self.bad_hotkeys:
            raise ValueError("unreadable entry")
        return _STORE.get(hotkey)


def _client(sub):
    c = ChainClient(netuid=259, _subtensor=sub)
    c._metagraph = lambda block=None: _meta()  # type: ignore[method-assign]
    return c


def test_one_shot_map_decodes_whole_field():
    """Live 2026-07-14: per-UID reads took ~13 min for the field, long enough to
    blow the provisioner's rental window. ONE query_map with tolerant per-entry
    decode: full field back, latest reveal per hotkey, garbage entries and
    off-metagraph hotkeys skipped, per-hotkey reads never touched — and the
    map is pinned to the metagraph's block (one consistent snapshot)."""
    sub = _Sub()
    out = _client(sub).poll_commitments()
    assert sorted(c.uid for c in out) == [0, 1, 3]
    by_uid = {c.uid: c for c in out}
    assert {u: c.commit_block for u, c in by_uid.items()} == {0: 100, 1: 110, 3: 130}
    assert by_uid[1].payload == RAW_PAYLOAD               # raw rendering decoded
    assert by_uid[0].payload == _payload("gen0")          # hex rendering decoded
    assert by_uid[0].coldkey == CK[0]
    assert (sub.map_calls, sub.query_calls) == (1, 0)
    assert sub.blocks == [500]


def test_failed_map_falls_back_and_skips_only_bad_hotkey():
    sub = _Sub(map_raises=True, bad_hotkeys=[HK[1]])
    out = _client(sub).poll_commitments()
    # every good UID survives; only the unreadable one is dropped — not the field.
    assert sorted(c.uid for c in out) == [0, 3]
    assert all(c.payload.startswith("metro-v1:gen:hippius:") for c in out)
    assert sub.query_calls == len(HK)


def test_failed_map_recovers_full_field_per_hotkey():
    out = _client(_Sub(map_raises=True)).poll_commitments()
    assert sorted(c.uid for c in out) == [0, 1, 3]
    assert {c.uid: c.commit_block for c in out} == {0: 100, 1: 110, 3: 130}


# ── include_history: the eligibility-cutoff read ──────────────────────────────
# Callers that apply a cutoff (trainer resolve, receipt participants, audit)
# need EVERY retained reveal: latest-only reads erase a miner whose newest
# commit landed after the boundary, even though the pre-cutoff reveal that
# fielded the round is still on chain (observed live: 9 of 30 entrants missing
# from a round receipt after re-committing for the next round).


def test_map_history_returns_every_decodable_reveal():
    out = _client(_Sub()).poll_commitments(include_history=True)
    # all of HK1's entries come back (the tolerant decoder yields SOME payload
    # for the garbage one — parse_commit rejects it downstream, same as any
    # malformed commitment); the real reveals are intact
    by_block = {c.commit_block: c for c in out if c.uid == 1}
    assert sorted(by_block) == [90, 95, 110]
    assert by_block[110].payload == RAW_PAYLOAD
    assert sorted({c.uid for c in out}) == [0, 1, 3]
    # the latest-only default is unchanged
    latest = _client(_Sub()).poll_commitments()
    assert [c.commit_block for c in latest if c.uid == 1] == [110]


def test_per_hotkey_history_returns_every_reveal():
    out = _client(_Sub(map_raises=True)).poll_commitments(include_history=True)
    assert sorted(c.commit_block for c in out if c.uid == 1) == [90, 95, 110]
