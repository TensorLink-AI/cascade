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

from dataclasses import dataclass
from typing import Any

from .config import ChainConfig


class ChainError(RuntimeError):
    """Wraps any bittensor exception so callers don't import bittensor."""


def equal_share_vector(
    reward_uids: list[int], n_uids: int, *, burn_uid: int = 0
) -> list[float]:
    """Build a length-``n_uids`` weight vector that splits 1.0 equally across the
    distinct, in-range ``reward_uids``. With no valid reward UID, all weight goes
    to ``burn_uid``. Pure (no chain I/O) so the routing math is unit-testable.
    """
    if n_uids <= 0:
        raise ChainError(f"n_uids must be positive, got {n_uids}")
    uniq = sorted({u for u in reward_uids if 0 <= u < n_uids})
    weights = [0.0] * n_uids
    if uniq:
        share = 1.0 / len(uniq)
        for u in uniq:
            weights[u] = share
    else:
        if not (0 <= burn_uid < n_uids):
            raise ChainError(f"burn_uid {burn_uid} out of range [0,{n_uids})")
        weights[burn_uid] = 1.0
    return weights


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
    """One miner's revealed generator pointer string."""

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

    def current_block(self) -> int:
        try:
            return int(self.subtensor().get_current_block())
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"get_current_block_failed: {e}") from e

    def block_seed(self, block: int | None = None) -> int:
        """The round base seed: the chain block hash as a 64-bit int.

        Both the trainer and every validator derive their per-round seeds from
        this, so a re-derived run reproduces byte-for-byte. Uses the current
        block when ``block`` is None.
        """
        sub = self.subtensor()
        try:
            blk = int(self.current_block()) if block is None else int(block)
            h = sub.get_block_hash(blk)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"get_block_hash_failed: {e}") from e
        digest = str(h).lower().removeprefix("0x")
        import hashlib

        return int.from_bytes(
            hashlib.blake2b(digest.encode(), digest_size=8).digest(), "big", signed=False
        )

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

    def poll_commitments(self) -> list[Commitment]:
        """Return the revealed generator pointer for every UID on the netuid.
        UIDs without a commitment are omitted.

        Miners write via ``set_reveal_commitment`` (timelock commit-reveal), so the
        payload lands in the *revealed*-commitment store — NOT the plain commitment
        store that ``get_commitment`` reads. We therefore read
        ``get_all_revealed_commitments`` (one call for the whole netuid) and map
        each hotkey back to its UID, taking the latest reveal per hotkey. Falls
        back to the per-UID ``get_commitment`` path on older bittensor builds.
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
                raise ChainError(f"get_all_revealed_commitments_failed: {e}") from e
            for hotkey, reveals in revealed.items():
                uid = uid_by_hotkey.get(str(hotkey))
                if uid is None or not reveals:
                    continue
                # ``reveals`` is a sequence of (block, payload); take the latest.
                block, payload = max(reveals, key=lambda r: int(r[0]))
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

        # Fallback: plain per-UID commitment store (older bittensor).
        for uid in range(int(meta.n)):
            hotkey = str(meta.hotkeys[uid])
            coldkey = str(coldkeys[uid]) if coldkeys else None
            try:
                rec = sub.get_commitment(netuid=self.netuid, uid=uid)
            except Exception:  # noqa: BLE001
                rec = None
            if not rec:
                continue
            payload, commit_block = _split_commitment(rec)
            if payload is None:
                continue
            out.append(
                Commitment(
                    uid=uid,
                    hotkey=hotkey,
                    coldkey=coldkey,
                    payload=payload,
                    commit_block=int(commit_block),
                )
            )
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
        self, reward_uids: list[int], n_uids: int, *, burn_uid: int = 0
    ) -> None:
        """Push an equal-share weight vector across ``reward_uids``.

        ``reward_uids`` is the current king plus any registered prior kings.
        Duplicates and out-of-range UIDs are dropped; each survivor gets
        ``1/k``. When none survive, all weight burns to ``burn_uid`` so emission
        still leaves the network rather than reverting. Mirrors teutonic's
        equal-share-across-recent-kings payout.
        """
        self._set_weights(equal_share_vector(reward_uids, n_uids, burn_uid=burn_uid))

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
    """Best-effort extraction of ``(payload, commit_block)`` across bittensor
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
