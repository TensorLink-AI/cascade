"""Bittensor chain client — commitment polling and weight setting.

Wraps the minimal subtensor surface cascade needs:

* :meth:`ChainClient.poll_commitments` — read every miner's revealed generator
  pointer for the netuid.
* :meth:`ChainClient.commit_submission` — miner-side ``set_reveal_commitment``
  with the generator pointer string.
* :meth:`ChainClient.set_winner_take_all_weights` — push the KOTH weight vector
  (1.0 on the champion, 0.0 elsewhere).
* :meth:`ChainClient.set_equal_share_weights` — split weight equally across the
  current king plus registered prior kings (teutonic-style payout), burning to
  ``burn_uid`` when none are registered.
* :meth:`ChainClient.current_block`.

This module is the single gating point for ``import bittensor`` so the rest of
the package stays importable in environments without it (unit tests, CI). The
static_guard blocklist pins this module name so a submitted generator can never
reach it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import ChainConfig

log = logging.getLogger("cascade.chain")


class ChainError(RuntimeError):
    """Wraps any bittensor exception so callers don't import bittensor."""


def decayed_share_vector(
    reward_uids: list[int], n_uids: int, *, decay: float = 1.0, burn_uid: int = 0
) -> list[float]:
    """Length-``n_uids`` weight vector with **geometric decay** across
    ``reward_uids`` in order.

    ``reward_uids`` is ordered *current king first, then former kings by
    recency*. Share is ``decay**i`` for the i-th entry, normalised to sum 1: the
    king gets the largest slice, each older king progressively less. ``decay =
    1.0`` reproduces the equal split; ``0 < decay < 1`` skews toward the current
    king (so it is unambiguously the highest-incentive UID, which is how the
    trainer identifies the king). With no valid reward UID all weight burns to
    ``burn_uid``.

    Order is preserved (unlike a set): the first occurrence of each in-range UID
    wins its slot, so the current king keeps the top share even if it also
    appears later as a former king. Pure (no chain I/O) — unit-testable.
    """
    if n_uids <= 0:
        raise ChainError(f"n_uids must be positive, got {n_uids}")
    if not (0.0 < decay <= 1.0):
        raise ChainError(f"decay must be in (0, 1], got {decay}")
    seen: set[int] = set()
    ordered: list[int] = []
    for u in reward_uids:
        if 0 <= u < n_uids and u not in seen:
            seen.add(u)
            ordered.append(u)
    weights = [0.0] * n_uids
    if not ordered:
        if not (0 <= burn_uid < n_uids):
            raise ChainError(f"burn_uid {burn_uid} out of range [0,{n_uids})")
        weights[burn_uid] = 1.0
        return weights
    raw = [decay ** i for i in range(len(ordered))]
    total = sum(raw)
    for u, w in zip(ordered, raw, strict=True):
        weights[u] = w / total
    return weights


def equal_share_vector(
    reward_uids: list[int], n_uids: int, *, burn_uid: int = 0
) -> list[float]:
    """Equal split across the distinct, in-range ``reward_uids`` (burns to
    ``burn_uid`` if none). The ``decay = 1.0`` case of
    :func:`decayed_share_vector`, kept as a named helper for the winner-take-all
    path and back-compat."""
    return decayed_share_vector(reward_uids, n_uids, decay=1.0, burn_uid=burn_uid)


def blocks_until_boundary_reveal(
    current_block: int,
    epoch_blocks: int,
    margin_blocks: int,
    *,
    next_epoch: bool = False,
    min_blocks: int = 1,
) -> int:
    """The ``blocks_until_reveal`` delay that lands a timelock reveal just
    before the next epoch boundary.

    Eligibility gates on the REVEAL block strictly before the boundary
    (:func:`cascade.trainer.loop.resolve_commitments`), so a reveal that lands
    at/after the boundary silently costs the miner the round. The target is
    therefore ``boundary − margin_blocks``: hidden for (almost) the whole
    submission window, public only for the last ``margin_blocks`` — long enough
    to absorb commit-inclusion and drand reveal jitter, short enough that a
    copier cannot fetch + re-commit + land their own reveal before the same
    boundary.

    When the current block is already inside the margin, floors to
    ``min_blocks`` (reveal now): the residual exposure is below the margin
    anyway, and revealing beats silently slipping past the deadline and losing
    a full epoch. Pass ``next_epoch=True`` to target the following boundary
    instead (for miners who prefer a guaranteed-hidden window over entering the
    imminent round). Pure — unit-testable without a chain.
    """
    if epoch_blocks < 1:
        raise ValueError(f"epoch_blocks must be >= 1, got {epoch_blocks}")
    if not (0 <= margin_blocks < epoch_blocks):
        raise ValueError(
            f"margin_blocks must be in [0, epoch_blocks={epoch_blocks}), got {margin_blocks}"
        )
    if min_blocks < 1:
        raise ValueError(f"min_blocks must be >= 1, got {min_blocks}")
    boundary = (current_block // epoch_blocks + 1) * epoch_blocks
    if next_epoch:
        boundary += epoch_blocks
    delay = boundary - margin_blocks - current_block
    return max(delay, min_blocks)


def seed_from_block_hash(block_hash: str) -> int:
    """A round base seed from a chain block hash: blake2b of the hex digest as a
    64-bit int. Pure — the audit CLI recomputes a receipt's ``base_seed`` from
    its recorded ``epoch_block_hash`` without a chain connection; the live
    :meth:`ChainClient.block_seed` goes through the same function so the two can
    never diverge.
    """
    import hashlib

    digest = str(block_hash).lower().removeprefix("0x")
    return int.from_bytes(
        hashlib.blake2b(digest.encode(), digest_size=8).digest(), "big", signed=False
    )


def _defuse_substrate_destructor() -> None:
    """Neuter async_substrate_interface's hanging ``__del__``.

    Its destructor closes the websocket, whose close handshake ``join()``s a
    thread with NO timeout — on a dead connection that join never returns, and
    because ``__del__`` runs wherever garbage collection happens to fire, it
    hangs the MAIN thread of whatever service triggered it. Observed live
    2026-07-14: the validator froze mid-poll for 5.5h (rounds went unscored)
    with the main thread parked in ``__del__ → close → join``. A leaked socket
    on GC is harmless (the process's daemon threads die at exit); a hang is
    fatal. Idempotent, best-effort, version-tolerant.
    """
    try:
        from async_substrate_interface import sync_substrate

        for cls_name in ("SubstrateInterface",):
            cls = getattr(sync_substrate, cls_name, None)
            if cls is not None and getattr(cls, "__del__", None) is not None:
                cls.__del__ = lambda self: None  # type: ignore[assignment]
    except Exception:  # noqa: BLE001 — defusal must never break client construction
        pass


def _import_bittensor():
    try:
        import bittensor  # type: ignore
    except ImportError as e:
        raise ChainError(
            "bittensor not installed; install the [chain] extra to run "
            "validator / trainer / miner against a live network"
        ) from e
    return bittensor


@dataclass(frozen=True)
class Commitment:
    """One miner's revealed generator pointer string.

    ``commit_block`` is the block the timelock payload became publicly
    readable at (the REVEAL block — what bittensor's ``RevealedCommitments``
    store records), NOT the block the encrypted commit landed. Round
    eligibility gates on it (reveal strictly before the epoch boundary), so a
    timed reveal must target the boundary minus a safety margin
    (:func:`blocks_until_boundary_reveal`).
    """

    uid: int
    hotkey: str
    coldkey: str | None
    payload: str
    commit_block: int


@dataclass
class ChainClient:
    """Thin facade over a subtensor + wallet pair. The connection is opened
    lazily on first use; reuse a single client across rounds."""

    netuid: int
    network: str = "finney"
    wallet_name: str | None = None
    wallet_hotkey: str | None = None
    wallet_path: str | None = None
    _subtensor: Any = None
    _wallet: Any = None

    @classmethod
    def from_config(
        cls,
        cfg: ChainConfig,
        *,
        network: str = "finney",
        wallet_name: str | None = None,
        wallet_hotkey: str | None = None,
        wallet_path: str | None = None,
    ) -> ChainClient:
        if cfg.netuid <= 0:
            raise ChainError(
                f"chain.toml [subnet] netuid={cfg.netuid} is a placeholder; "
                "set the live netuid before launching"
            )
        return cls(
            netuid=cfg.netuid,
            network=network,
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
            wallet_path=wallet_path,
        )

    def subtensor(self):
        if self._subtensor is None:
            bt = _import_bittensor()
            _defuse_substrate_destructor()
            # bittensor <9 exposed a lowercase ``subtensor`` factory; 9+/10 only
            # ship the ``Subtensor`` class. Support both so the client works
            # across the ``bittensor>=8`` range pyproject allows.
            subtensor_factory = getattr(bt, "subtensor", None) or bt.Subtensor
            self._subtensor = subtensor_factory(network=self.network)
        return self._subtensor

    def wallet(self):
        if self._wallet is None:
            if self.wallet_name is None or self.wallet_hotkey is None:
                raise ChainError("wallet_name and wallet_hotkey are required")
            bt = _import_bittensor()
            kwargs: dict[str, Any] = {"name": self.wallet_name, "hotkey": self.wallet_hotkey}
            if self.wallet_path is not None:
                kwargs["path"] = self.wallet_path
            # bittensor <9 exposed a lowercase ``wallet`` factory; 9+/10 only ship
            # the ``Wallet`` class. Support both (matches the subtensor shim above).
            wallet_factory = getattr(bt, "wallet", None) or bt.Wallet
            self._wallet = wallet_factory(**kwargs)
        return self._wallet

    def reconnect(self) -> None:
        """Drop the cached subtensor so the next call re-opens the websocket.

        A long-lived bittensor websocket can go quietly stale — serving a
        ~20-minute-old block or hanging without erroring — and the only
        reliable recovery is a fresh connection. Mirrors the provisioner's
        chain_client_factory rebuild (cascade.provision.loop._current_block)."""
        self._subtensor = None

    def current_block(self) -> int:
        try:
            return int(self.subtensor().get_current_block())
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"get_current_block_failed: {e}") from e

    def block_hash(self, block: int | None = None) -> str:
        """The chain block hash at ``block`` (current block when None)."""
        sub = self.subtensor()
        try:
            blk = int(self.current_block()) if block is None else int(block)
            return str(sub.get_block_hash(blk))
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"get_block_hash_failed: {e}") from e

    def block_seed(self, block: int | None = None) -> int:
        """The round base seed: the chain block hash as a 64-bit int.

        Both the trainer and every validator derive their per-round seeds from
        this, so a re-derived run reproduces byte-for-byte. Uses the current
        block when ``block`` is None. The hash→seed mapping is the pure
        :func:`seed_from_block_hash`, which auditors reuse offline.
        """
        return seed_from_block_hash(self.block_hash(block))

    def highest_incentive_hotkey(self) -> str | None:
        """The reigning king's hotkey: the UID with the highest incentive on the
        metagraph (validators set this via weights). None on an empty metagraph."""
        sub = self.subtensor()
        try:
            meta = sub.metagraph(netuid=self.netuid, lite=True)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"metagraph_failed: {e}") from e
        n = int(meta.n)
        if n == 0:
            return None
        incentive = list(meta.incentive)
        best_uid = max(range(n), key=lambda u: float(incentive[u]))
        if float(incentive[best_uid]) <= 0.0:
            return None  # vacant throne — let the trainer pick an interim king
        return str(meta.hotkeys[best_uid])

    def n_uids(self) -> int:
        """Number of UIDs registered on the netuid (for the weight vector)."""
        sub = self.subtensor()
        try:
            return int(sub.metagraph(netuid=self.netuid, lite=True).n)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"metagraph_failed: {e}") from e

    def uid_for_hotkey(self, hotkey: str) -> int | None:
        """Resolve a hotkey to its UID on the netuid, or None if absent."""
        sub = self.subtensor()
        try:
            meta = sub.metagraph(netuid=self.netuid, lite=True)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"metagraph_failed: {e}") from e
        for uid in range(int(meta.n)):
            if str(meta.hotkeys[uid]) == hotkey:
                return uid
        return None

    def weights_for_hotkey(self, hotkey: str) -> list[float] | None:
        """The weight row a validator hotkey currently has on chain, or None.

        Reads the full (non-lite) metagraph and returns ``hotkey``'s row of the
        weight matrix as floats (chain-normalised; callers comparing against a
        receipt should compare the *support* — which UIDs carry weight — not
        magnitudes). None when the hotkey is not registered. Used by
        ``cascade-audit`` to cross-check a receipt's recorded weight vector.
        """
        sub = self.subtensor()
        try:
            meta = sub.metagraph(netuid=self.netuid, lite=False)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"metagraph_failed: {e}") from e
        uid = next(
            (u for u in range(int(meta.n)) if str(meta.hotkeys[u]) == hotkey), None
        )
        if uid is None:
            return None
        # bittensor 9/10 expose the weight matrix as ``W`` (property) with the
        # raw attribute ``weights``; accept either for version tolerance.
        matrix = getattr(meta, "W", None)
        if matrix is None:
            matrix = getattr(meta, "weights", None)
        if matrix is None:
            raise ChainError("metagraph carries no weight matrix (lite node?)")
        return [float(w) for w in matrix[uid]]

    def poll_commitments(self, include_history: bool = False) -> list[Commitment]:
        """Return the revealed generator pointer for every UID on the netuid.
        UIDs without a commitment are omitted.

        Miners write via ``set_reveal_commitment`` (timelock commit-reveal), so the
        payload lands in the *revealed*-commitment store — NOT the plain commitment
        store that ``get_commitment`` reads. We therefore read
        ``get_all_revealed_commitments`` (one call for the whole netuid) and map
        each hotkey back to its UID, taking the latest reveal per hotkey. Falls
        back to the per-UID ``get_commitment`` path on older bittensor builds.

        ``include_history=True`` returns EVERY retained reveal per hotkey (one
        ``Commitment`` each) instead of only the latest. Any caller applying an
        eligibility cutoff (trainer resolve, validator receipt participants,
        audit cross-checks) needs the history: collapsing to the latest reveal
        FIRST erases a miner whose newest commit landed after the cutoff, even
        though their eligible pre-cutoff reveal is still on chain.
        """
        sub = self.subtensor()
        try:
            meta = sub.metagraph(netuid=self.netuid, lite=True)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"metagraph_failed: {e}") from e

        uid_by_hotkey = {str(meta.hotkeys[u]): u for u in range(int(meta.n))}
        coldkeys = list(meta.coldkeys) if hasattr(meta, "coldkeys") else None

        get_revealed = getattr(sub, "get_all_revealed_commitments", None)
        out: list[Commitment] = []
        if get_revealed is not None:
            try:
                revealed = get_revealed(self.netuid) or {}
            except Exception as e:  # noqa: BLE001
                # One miner's malformed (non-hex) revealed commitment makes
                # bittensor's BATCH decoder raise for the WHOLE netuid — which
                # would blind the trainer to every other miner's submission (a
                # field-wide DoS from a single bad commit). Read the raw store
                # ourselves: ONE query_map for the netuid, tolerant per-entry
                # decode (~1s), instead of N per-UID queries (~13 min live —
                # long enough to blow the provisioner's rental window).
                log.warning("bulk revealed-commitment decode failed (%s); "
                            "reading the raw store map", e)
                try:
                    return self._revealed_raw_map(sub, uid_by_hotkey, coldkeys,
                                                  include_history=include_history)
                except Exception as e2:  # noqa: BLE001
                    log.warning("raw store map failed (%s); "
                                "falling back to per-UID decode", e2)
                    return self._revealed_per_uid(sub, meta, coldkeys,
                                                  include_history=include_history)
            for hotkey, reveals in revealed.items():
                uid = uid_by_hotkey.get(str(hotkey))
                if uid is None or not reveals:
                    continue
                # ``reveals`` is a sequence of (block, payload).
                picked = reveals if include_history else \
                    [max(reveals, key=lambda r: int(r[0]))]
                for block, payload in picked:
                    if not isinstance(payload, str) or not payload:
                        continue
                    out.append(
                        Commitment(
                            uid=uid,
                            hotkey=str(hotkey),
                            coldkey=str(coldkeys[uid]) if coldkeys else None,
                            payload=payload,
                            commit_block=int(block),
                        )
                    )
            return out

        # No bulk API (older bittensor): decode per-UID.
        return self._revealed_per_uid(sub, meta, coldkeys,
                                      include_history=include_history)

    def _raw_revealed_entries(self, sub: Any, hotkey: str) -> list[tuple[int, str]]:
        """Read ``Commitments::RevealedCommitments`` directly and decode BOTH
        substrate renderings — the escape hatch for bittensor's decoder bug.

        py-substrate-interface returns storage bytes as a ``0x…`` hex string
        UNLESS the bytes happen to be valid UTF-8, in which case it returns the
        decoded string — and whether a reveal's bytes are UTF-8-valid depends
        on its SCALE length prefix, i.e. on the PAYLOAD LENGTH
        (``(4·len+1) mod 256 < 128`` ⇒ raw). bittensor's
        ``decode_revealed_commitment`` assumes hex and raises ``fromhex`` on the
        raw rendering, silently costing that miner every round (observed live:
        7 UIDs skipped purely by pointer-length lottery — ``@hf:`` pointers at
        91 chars raw, ``@sha256:`` at 109 hex). Returns ``[(block, payload)]``.
        """
        q = sub.substrate.query(module="Commitments",
                                storage_function="RevealedCommitments",
                                params=[self.netuid, hotkey])
        return self._decode_reveal_entries(getattr(q, "value", None))

    @staticmethod
    def _decode_reveal_entries(v: Any) -> list[tuple[int, str]]:
        """Decode a RevealedCommitments storage value (either rendering) into
        ``[(block, payload)]``, skipping undecodable entries."""
        out: list[tuple[int, str]] = []
        for entry in (v or []):
            try:
                com, block = entry
                if not isinstance(com, str) or not com:
                    continue
                if com.startswith("0x"):  # noqa: SIM108 — explicit branch reads clearer here
                    raw = bytes.fromhex(com[2:])
                else:
                    raw = com.encode("utf-8")
                # strip the SCALE compact length prefix (mode in the low 2 bits)
                mode = raw[0] & 0b11
                offset = 1 if mode == 0 else 2 if mode == 1 else 4
                payload = raw[offset:].decode("utf-8", errors="ignore")
                if payload:
                    out.append((int(block), payload))
            except Exception:  # noqa: BLE001 — one bad entry is not the field
                continue
        return out

    def _revealed_raw_map(self, sub: Any, uid_by_hotkey: dict[str, int],
                          coldkeys: list | None,
                          include_history: bool = False) -> list[Commitment]:
        """All revealed commitments for the netuid in ONE ``query_map``, with the
        tolerant both-renderings decode per entry.

        Verified live on testnet: 38 hotkeys in ~1.1s where the per-UID path
        takes ~13 minutes — the difference between making and missing the
        provisioner's pre-boundary rental window."""
        qm = sub.substrate.query_map(module="Commitments",
                                     storage_function="RevealedCommitments",
                                     params=[self.netuid], page_size=200)
        out: list[Commitment] = []
        for key, value in qm:
            hotkey = str(getattr(key, "value", key))
            uid = uid_by_hotkey.get(hotkey)
            if uid is None:
                continue
            entries = self._decode_reveal_entries(getattr(value, "value", value))
            if not entries:
                continue
            picked = entries if include_history else \
                [max(entries, key=lambda r: int(r[0]))]
            for block, payload in picked:
                out.append(Commitment(uid=uid, hotkey=hotkey,
                                      coldkey=str(coldkeys[uid]) if coldkeys else None,
                                      payload=payload, commit_block=int(block)))
        return out

    def _revealed_per_uid(self, sub: Any, meta: Any, coldkeys: list | None,
                          include_history: bool = False) -> list[Commitment]:
        """Decode each UID's revealed commitment individually — robust to a single
        malformed entry that would poison bittensor's batch decoder.

        Prefers the per-UID *revealed* store (``get_revealed_commitment``, the
        timelock reveal path miners actually write to); falls back to the plain
        commitment store (``get_commitment``) on older builds. An entry that won't
        decode is skipped with a warning, never fatal — so one garbage commitment
        costs exactly that one UID, not the whole field."""
        per_uid = getattr(sub, "get_revealed_commitment", None)
        out: list[Commitment] = []
        for uid in range(int(meta.n)):
            hotkey = str(meta.hotkeys[uid])
            coldkey = str(coldkeys[uid]) if coldkeys else None
            if per_uid is not None:
                try:
                    reveals = per_uid(self.netuid, uid)
                except Exception as e:  # noqa: BLE001 — try the raw store before skipping
                    # bittensor's decoder chokes on the raw substrate rendering
                    # (see _raw_revealed_entries) — read the store ourselves so a
                    # miner is never skipped for their pointer's LENGTH.
                    try:
                        reveals = self._raw_revealed_entries(sub, hotkey)
                    except Exception:  # noqa: BLE001
                        reveals = None
                    if not reveals:
                        log.warning("skipping uid %d: revealed-commitment decode failed: %s",
                                    uid, e)
                        continue
                    log.info("uid %d: recovered reveal via raw store (bittensor "
                             "decoder bug — payload-length lottery)", uid)
                if reveals:
                    picked = reveals if include_history else \
                        [max(reveals, key=lambda r: int(r[0]))]
                    for block, payload in picked:
                        if isinstance(payload, str) and payload:
                            out.append(Commitment(uid=uid, hotkey=hotkey, coldkey=coldkey,
                                                  payload=payload, commit_block=int(block)))
                    continue
            # Last resort: the plain commitment store (older bittensor).
            try:
                rec = sub.get_commitment(netuid=self.netuid, uid=uid)
            except Exception:  # noqa: BLE001
                rec = None
            if not rec:
                continue
            payload, reveal_block = _split_commitment(rec)
            if payload is None:
                continue
            out.append(Commitment(uid=uid, hotkey=hotkey, coldkey=coldkey,
                                  payload=payload, commit_block=int(reveal_block)))
        return out

    def commit_submission(self, payload: str, blocks_until_reveal: int = 1) -> None:
        """Miner-side: write the generator pointer via ``set_reveal_commitment``."""
        sub = self.subtensor()
        w = self.wallet()
        try:
            sub.set_reveal_commitment(
                wallet=w,
                netuid=self.netuid,
                data=payload,
                blocks_until_reveal=blocks_until_reveal,
            )
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"set_reveal_commitment_failed: {e}") from e

    def set_winner_take_all_weights(self, champion_uid: int, n_uids: int) -> None:
        """Push a weight vector with 1.0 on ``champion_uid`` and 0.0 elsewhere."""
        if not (0 <= champion_uid < n_uids):
            raise ChainError(f"champion_uid {champion_uid} out of range [0,{n_uids})")
        self._set_weights(equal_share_vector([champion_uid], n_uids))

    def set_equal_share_weights(
        self, reward_uids: list[int], n_uids: int, *, decay: float = 1.0, burn_uid: int = 0
    ) -> None:
        """Push a (geometrically decayed) share vector across ``reward_uids``.

        ``reward_uids`` is the current king first, then any registered prior
        kings by recency. With ``decay < 1`` the king gets the largest slice and
        each older king progressively less (so the king is unambiguously the
        highest-incentive UID); ``decay = 1.0`` is the flat equal split.
        Duplicates and out-of-range UIDs are dropped; when none survive, all
        weight burns to ``burn_uid`` so emission still leaves the network.
        """
        self._set_weights(decayed_share_vector(reward_uids, n_uids, decay=decay, burn_uid=burn_uid))

    def _set_weights(self, weights: list[float]) -> None:
        sub = self.subtensor()
        w = self.wallet()
        uids = list(range(len(weights)))
        try:
            sub.set_weights(
                wallet=w,
                netuid=self.netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"set_weights_failed: {e}") from e


def _split_commitment(rec: Any) -> tuple[str | None, int]:
    """Best-effort extraction of ``(payload, reveal_block)`` across bittensor
    versions: a plain string, a 2-tuple, a dict, or an object with attrs."""
    if rec is None:
        return None, 0
    if isinstance(rec, str):
        return rec, 0
    if isinstance(rec, tuple) and len(rec) >= 2 and isinstance(rec[0], str):
        return rec[0], int(rec[1])
    if isinstance(rec, dict):
        data = rec.get("data") or rec.get("payload")
        block = rec.get("block") or rec.get("commit_block") or 0
        if isinstance(data, str):
            return data, int(block)
        return None, 0
    data = getattr(rec, "data", None) or getattr(rec, "payload", None)
    block = getattr(rec, "block", None) or getattr(rec, "commit_block", None) or 0
    if isinstance(data, str):
        return data, int(block)
    return None, 0
