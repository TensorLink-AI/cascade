"""Signed, durable direct-job admission; plugs into the existing funding HTTP server."""

from __future__ import annotations

from ..funding.direct_ledger import DirectLedger
from ..shared.direct_moe import digest
from .signing import verify


class DirectFundingIntake:
    def __init__(
        self, ledger_path, policy, *, registered, revealed, funding_authorized, bootstrap_payer=None
    ):
        self.path, self.policy = ledger_path, policy
        self.bootstrap_payer = bootstrap_payer
        self.registered, self.revealed, self.funding_authorized = (
            registered,
            revealed,
            funding_authorized,
        )

    def fund(self, envelope, *, now):
        """The credential lookup is server-side; requests contain opaque funding refs.

        Signature binds job, immutable terms, expiry and replay nonce. The current
        king (or explicitly declared bootstrap payer) funds the incumbent. Already
        admitted commands return without spending again even after request expiry.
        """
        try:
            signer = envelope["signer"]
            body = verify(envelope, expected_signer=signer, scope="fund-job")
            job, terms = body["job"], body["terms"]
            if signer != terms["payer"]:
                raise ValueError("payer signature mismatch")
            ledger = DirectLedger(self.path, self.policy)
            try:
                old = ledger.state["jobs"].get(job)
                if old is not None:
                    if old["terms"] != terms:
                        raise ValueError("immutable admission terms changed")
                    return 200, {"status": "admitted", "job": job, "terms": digest(terms)}
                if old is None:
                    if not now <= body["expires_at"] or body["nonce"] != job:
                        raise ValueError("expired/invalid funding authorization")
                    if (
                        self.registered(signer) is not True
                        or self.revealed(signer, terms["generator"]) is not True
                    ):
                        return 403, {"code": "not_registered_or_revealed"}
                    # The payer's key for every marketplace the terms name must be
                    # on file; keys never enter the envelope or the journal.
                    if self.funding_authorized(signer, terms["providers"]) is not True:
                        return 403, {"code": "funding_authorization_required"}
                    if terms["role"] == "incumbent" and signer != (
                        ledger.state["king"] or self.bootstrap_payer
                    ):
                        return 403, {"code": "incumbent_payer_must_be_king"}
                # Admission and signature evidence commit in the same event.
                ledger.apply(
                    f"admit:{job}",
                    "admit",
                    now=now,
                    job=job,
                    terms=terms,
                    authorization=envelope,
                    bootstrap_payer=self.bootstrap_payer,
                )
                return 202 if old is None else 200, {
                    "status": "admitted",
                    "job": job,
                    "terms": digest(terms),
                }
            finally:
                ledger.close()
        except (KeyError, TypeError, ValueError):
            return 400, {"code": "invalid_direct_funding_request"}
