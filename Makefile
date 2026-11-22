# Lane D build targets (LANE_D_PLAN §3.0). CI runs them after
# `uv sync --frozen --all-extras`; the [model] cases need torch.

.PHONY: laned-e2e

# W1.5 walking skeleton: pool_d -> checkpoint -> intake -> ingest -> scorer
# -> signed report -> lane_d_state.json -> emission -> check_lane_d.
laned-e2e:
	uv run --frozen pytest -q -p no:cacheprovider tests/accept/w1_5
