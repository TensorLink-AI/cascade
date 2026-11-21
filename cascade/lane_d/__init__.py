"""Lane D intake service (LANE_D_PLAN §1.2): the storage interface the vault
sits behind and the submit/ready half of the miner flow.

Kept free of torch and of the validator package: the intake runs on the box
as its own service (``deploy_gate`` area ``lane_d_intake``).
"""
