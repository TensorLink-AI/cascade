"""Satellite-side verification (LANE_D_PLAN §14.3, §1.5).

A satellite re-derives what the main validator signed, from the signed
records alone. Invariant 12: nothing in this package imports the producer's
eligibility, verdict or weight code; see :mod:`cascade.satellite.lane_d_replay`.
"""
