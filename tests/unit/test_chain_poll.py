"""ChainClient.poll_commitments robustness: one miner's malformed (non-hex)
revealed commitment makes bittensor's BATCH decoder raise for the whole netuid.
The poll must degrade to a per-UID decode that skips only the bad entry rather
than going blind to the entire field (a single-commit field-wide DoS)."""

from __future__ import annotations

from cascade.shared.chain import ChainClient

HK = [f"5Hotkey{i}" for i in range(4)]
CK = [f"5Cold{i}" for i in range(4)]


def _payload(name: str) -> str:
    return f"metro-v1:gen:hippius:acct/{name}@sha256:" + "ab" * 32


class _Meta:
    def __init__(self):
        self.hotkeys = list(HK)
        self.coldkeys = list(CK)
        self.n = len(HK)


class _Sub:
    """Fake subtensor. ``bulk_raises`` simulates the batch-decode DoS; ``bad_uids``
    are the UIDs whose per-UID revealed decode also raises (the malformed ones)."""

    def __init__(self, *, bulk_raises=False, bad_uids=()):
        self.bulk_raises = bulk_raises
        self.bad_uids = set(bad_uids)
        # per-UID revealed data: (block, payload) tuples; uid 2 has none at all
        self._reveals = {
            0: ((100, _payload("gen0")),),
            1: ((110, _payload("gen1")),),
            3: ((130, _payload("gen3")),),
        }

    def metagraph(self, netuid, lite=True):
        return _Meta()

    def get_all_revealed_commitments(self, netuid, block=None):
        if self.bulk_raises:
            raise ValueError("non-hexadecimal number found in fromhex() arg at position 1")
        return {HK[u]: r for u, r in self._reveals.items()}

    def get_revealed_commitment(self, netuid, uid, block=None):
        if uid in self.bad_uids:
            raise ValueError("non-hexadecimal number found in fromhex() arg at position 1")
        return self._reveals.get(uid)


class _SubNoBulk(_Sub):
    """Older bittensor: no batch API at all — must use the per-UID path."""
    get_all_revealed_commitments = None


def _client(sub):
    return ChainClient(netuid=259, _subtensor=sub)


def test_bulk_success_decodes_whole_field():
    out = _client(_Sub()).poll_commitments()
    assert sorted(c.uid for c in out) == [0, 1, 3]
    assert {c.uid: c.reveal_block for c in out} == {0: 100, 1: 110, 3: 130}


def test_malformed_batch_falls_back_and_skips_only_bad_uid():
    # batch decode raises (as on the live netuid); uid 1 is the malformed one.
    out = _client(_Sub(bulk_raises=True, bad_uids=[1])).poll_commitments()
    # every good UID survives; only the undecodable one is dropped — not the field.
    assert sorted(c.uid for c in out) == [0, 3]
    assert all(c.payload.startswith("metro-v1:gen:hippius:") for c in out)


def test_malformed_batch_recovers_full_field_when_bad_uid_has_no_commit():
    # batch raises but every UID with a commitment decodes per-UID → full recovery.
    out = _client(_Sub(bulk_raises=True)).poll_commitments()
    assert sorted(c.uid for c in out) == [0, 1, 3]


def test_no_bulk_api_uses_per_uid():
    out = _client(_SubNoBulk()).poll_commitments()
    assert sorted(c.uid for c in out) == [0, 1, 3]
