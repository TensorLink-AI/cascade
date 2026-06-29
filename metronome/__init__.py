"""metronome — a Bittensor subnet for synthetic time-series training data.

King-of-the-hill on *data*, not models. Miners submit synthetic time-series
**data generators** (code, optionally with safetensors weights — the generator
may itself be a trained model). The subnet owner runs a deterministic
**trainer** that trains a fresh copy of the owner-supplied base TSFM on the
reigning king's generator output and on a challenger's generator output under
an *identical* training contract — same architecture, seed, epochs, budget —
so the only thing that varies between the two trained models is the data.
Both trained checkpoints are pushed to the Hippius Hub registry and recorded
in a signed training manifest (published to Hippius S3 alongside per-round
training logs). Validators pull both checkpoints, evaluate them on a shared
held-out real-world eval set, and the challenger dethrones the king only if its
trained model beats the king's by a confidence-bounded margin (paired bootstrap
LCB on geomean(CRPS, MASE)). Weights are pure winner-takes-all.

Companion subnet ``horizon`` is the dual of this one: there miners submit
trained models directly. metronome instead isolates *data quality* as the
competitive axis by holding the training process fixed.
"""

__version__ = "0.0.1"
