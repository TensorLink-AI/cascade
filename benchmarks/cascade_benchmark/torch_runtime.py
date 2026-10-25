"""Numerical switch handoff in the isolated sidecar.

Mirrors cascade.eval.torch_runtime: this environment cannot depend on core.
"""

from __future__ import annotations


def capture_runtime():
    import numpy as np
    import torch

    return {"torch": torch.__version__, "numpy": np.__version__,
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled()}


def apply_runtime(runtime):
    import numpy as np
    import torch

    if (runtime["torch"].split("+")[0] != torch.__version__.split("+")[0]
            or runtime["numpy"] != np.__version__):
        raise ValueError("inference torch/numpy runtime differs from the validator")
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    torch.set_default_dtype(dtypes[runtime["default_dtype"]])
    torch.use_deterministic_algorithms(runtime["deterministic"], warn_only=runtime["warn_only"])
    torch.set_float32_matmul_precision(runtime["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_tf32"]
    torch.backends.cudnn.enabled = runtime["cudnn_enabled"]
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.mkldnn.enabled = runtime["mkldnn_enabled"]
    torch.backends.cuda.enable_flash_sdp(runtime["flash_sdp"])
    torch.backends.cuda.enable_mem_efficient_sdp(runtime["efficient_sdp"])
    torch.backends.cuda.enable_math_sdp(runtime["math_sdp"])
