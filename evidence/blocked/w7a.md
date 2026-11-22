Reason: W7a's required `params` fields (horizons, max_mirror_lag_hours, gate_suites, tau_public, lowfreq_freqs) make W1.5's `check_lane_d` (cascade/audit/lane_d.py, outside tests/accept/w7a/PATHS) FAIL with "stated params differ from config", so `make laned-e2e` cannot stay green within PATHS.

## The conflict

- `tests/accept/w7a/test_w7a_report_fields.py::test_e2e_params_carry_every_satellite_value`
  and `test_w7a_validate.py::test_every_params_field_is_required` require
  `report["params"]` to carry the 22 §14.3 fields, including the five above.
- `cascade/audit/lane_d.py::check_lane_d` compares `report["params"]` to its
  own `_expected_params(cfg)` with dict equality (`report.get("params") != want`),
  and `_expected_params` lists only the 17 W1.5 fields.
- So any report that passes W7a fails the audit in
  `tests/accept/w1_5/test_w1_5_e2e.py` (`test_e2e_audit_passes[synthetic]`,
  `test_e2e_tampered_row_fails[synthetic]`,
  `test_inv9_share_zero_vector_armed_nonempty_lane_d_effect[synthetic]`) and
  `tests/accept/w8/test_w8_settlement.py::test_inv9_share_zero_vector_armed_nonempty_lane_d_effect`.
  The two requirements are mutually exclusive; no change inside PATHS satisfies both.

## State of the build (branch laned/w7a-build)

- The three PATHS files are implemented: `tests/accept/w7a/` 120 passed,
  `tests/accept/w5a/` 30 passed, stub_scan clean, ruff clean.
- The only red tests are the four audit cases above, all failing on the
  params diff `['gate_suites', 'horizons', 'lowfreq_freqs',
  'max_mirror_lag_hours', 'tau_public']`.

## Unblock (needs a PATHS amendment by the planner)

Add `cascade/audit/lane_d.py` to `tests/accept/w7a/PATHS` and extend its
`_expected_params(cfg)` with the same five values the producer states:

    "horizons": [int(cfg.horizon)],
    "max_mirror_lag_hours": float(cfg.max_mirror_lag_hours),
    "gate_suites": [str(s) for s in cfg.gate_suites],
    "tau_public": float(cfg.tau_public),
    "lowfreq_freqs": [str(f) for f in cfg.lowfreq_freqs],

The audit re-derives the values from `cfg` independently (invariant 12), so it
stays independent of the producer.
