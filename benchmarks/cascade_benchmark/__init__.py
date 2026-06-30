"""cascade-benchmark — isolated sidecar for public time-series benchmarks.

Run *out of process* from the validator (its deps conflict with the cascade
core), this package loads a trained cascade checkpoint, wraps it as a gluonts
predictor, scores it on GIFT-Eval / BOOM / TIME, and writes the metrics to a
JSON file the validator reads back and logs. Nothing here is imported by the
cascade package itself.
"""
