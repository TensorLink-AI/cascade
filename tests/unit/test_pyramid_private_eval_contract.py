"""The v2 Q21 model runs through the private offload protocol on CPU fixtures."""
from __future__ import annotations

import numpy as np


def test_pyramid_private_request_qualifies_sequential_and_parallel_forecasts(tmp_path):
    import torch

    from cascade.eval.window import EvalWindow
    from cascade.shared.config import parse_chain_config
    from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint
    from cascade.validator.private_worker import read_forecasts, run_request, write_request
    from scripts.pyramid_config import local_config

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        contract = parse_chain_config(local_config("2m")).training
        config = CascadeMoEConfig.from_contract(contract)
        with torch.random.fork_rng():
            torch.manual_seed(123)
            model = CascadeMoEModel(config)
        directory = tmp_path / "checkpoint"
        directory.mkdir()
        save_checkpoint(directory, model.state_dict(), config, contract)
        rng = np.random.default_rng(42)
        windows = [EvalWindow(str(i), rng.normal(size=(c, 64)), rng.normal(size=(c, 8)),
                             {"source": "private-source"}) for i, c in enumerate((1, 2))]
        request, output = tmp_path / "request.npz", tmp_path / "samples.npz"
        identity = write_request(request, directory, windows, contract=contract, num_samples=4)
        with np.load(request, allow_pickle=False) as payload:
            assert set(payload.files) == {"metadata", "h_0", "h_1"}
            assert "private-source" not in payload["metadata"].tobytes().decode()
        report = run_request(directory, request, output, device="cpu", workers=2, qualify=True)
        assert report["qualified"] and report["workers"] == 2
        samples = read_forecasts(output, identity, windows, 4)
        assert [x.shape for x in samples] == [(1, 4, 8), (2, 4, 8)]
        assert all(np.isfinite(x).all() for x in samples)
    finally:
        torch.set_num_threads(old)
