"""Direct-MoE validation: check the trainer's scores, never evaluate.

Validators verify the trainer's sr25519 signature over the journal and replay
it: every event re-runs the ledger reducer, and every decision is re-derived
from the per-case scores the trainer recorded (private verdict, paired
bootstrap guard, cohort settlement, payout split). A replay that does not
reproduce the trainer's state is refused. No checkpoint is loaded and no
model is run here.

Whether the recorded scores are true is settled by reproduction, not here:
``cascade-audit direct-moe --reproduce`` re-scores each decision's bound,
public checkpoints (DEC-CA-0052 trust model).

Under an owner-signed network config (``cascade.direct.network``) the trainer
hotkey, journal location, burn uid and allowed scoring rules all come from the
owner's main validator; a journal that used rules the config does not list is
refused. Validators then need no update when those rules change. They also
check each round stays auditable (DEC-CA-0052 §9): bound checkpoints publicly
fetchable, panels pre-listed, and retired panels revealed in time, or the
boundary burns.

Raw (unsigned) journals remain supported for historical offline replay.
"""
import logging

from ..eval.direct_guard import replay_receipt
from ..funding.direct_ledger import replay_ledger

log = logging.getLogger("cascade.validator.direct")


def verify_direct_receipt(receipt, *, expected_trainer=None):
    if receipt.get("protocol") == "cascade-direct-signed/1":
        from ..direct.signing import verify
        if not expected_trainer:
            raise ValueError("configured trainer trust anchor required")
        receipt = verify(receipt, expected_signer=expected_trainer, scope="journal")
    elif expected_trainer:
        raise ValueError("unsigned direct journal refused")
    if receipt.get("protocol") == "direct-decision/1":
        from ..direct.evaluation import replay_decision
        return replay_decision(receipt)
    if receipt.get("protocol") == "direct-moe-ledger/1":
        return replay_ledger(receipt)
    return replay_receipt(receipt)


def verify_under_network(signed, network, *, now_block=None):
    """Replay a journal signed by the config's trainer and check it used only
    the rules the owner config lists."""
    from ..direct.network import check_conformance

    state = verify_direct_receipt(signed, expected_trainer=network["trainer"])
    check_conformance(signed["body"], state, network, now_block=now_block)
    return state


def public_probes(storage, network):
    """Credential-free ``(available(uri), revealed(panel_digest))`` probes."""
    from ..direct.journal import reveal_key
    from ..direct.storage import public_exists

    journal = {**storage, **network["journal"]}

    def available(uri):
        if not uri.startswith("hippius://"):
            return False
        bucket, _, key = uri.removeprefix("hippius://").partition("/")
        return public_exists(bucket, key, endpoint=storage["endpoint"], region=storage["region"])

    def revealed(panel_digest):
        return all(public_exists(journal["bucket"], reveal_key(panel_digest, suffix),
                                 endpoint=journal["endpoint"], region=journal["region"])
                   for suffix in ("json", "npz"))

    return available, revealed


def direct_weights(signed, *, uid_of, n_uids, expected_trainer=None, burn_uid=None,
                   network=None, available=None, revealed=None, now_block=None):
    """The weight vector a validator sets from a signed journal.

    With ``network`` (a verified owner config) the trainer, burn uid and the
    allowed rules come from it and the explicit arguments are ignored, and the
    round must stay auditable: ``available``/``revealed`` (see
    ``public_probes``) are required, an unfetchable bound checkpoint refuses
    the journal, and an overdue panel reveal burns the boundary.
    ``None`` when the journal has no settled payout boundary yet (the caller
    keeps its existing weights).
    """
    from ..direct.weights import boundary_units, latest_settled_boundary, weight_vector

    if network is None:
        state = verify_direct_receipt(signed, expected_trainer=expected_trainer)
    elif network["burn_all"]:
        return burn_vector(n_uids, network["burn_uid"])
    else:
        state = verify_under_network(signed, network, now_block=now_block)
        burn_uid = network["burn_uid"]
    boundary = latest_settled_boundary(state)
    if boundary is None:
        return None
    if network is not None:
        from ..direct.network import audit_obligations

        if available is None or revealed is None:
            raise ValueError("auditability probes required under a network config")
        overdue = audit_obligations(signed["body"], state, network, available=available,
                                    revealed=revealed, boundary=int(boundary))
        if overdue:
            log.warning("private panels overdue for reveal (%s); boundary %s burns",
                        ", ".join(p[:12] for p in overdue), boundary)
            return burn_vector(n_uids, burn_uid)
    return weight_vector(boundary_units(state, boundary), uid_of=uid_of, n_uids=n_uids,
                         burn_uid=burn_uid)


def burn_vector(n_uids, burn_uid):
    if not 0 <= burn_uid < n_uids:
        raise ValueError("invalid metagraph size or burn uid")
    return [1.0 if uid == burn_uid else 0.0 for uid in range(n_uids)]
