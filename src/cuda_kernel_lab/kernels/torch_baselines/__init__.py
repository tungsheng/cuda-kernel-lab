"""PyTorch reference implementations for correctness and baseline benchmarks."""

from cuda_kernel_lab.kernels.torch_baselines.matmul import matmul
from cuda_kernel_lab.kernels.torch_baselines.memory import (
    copy,
    reduction_sum,
    scale,
    vector_add,
)
from cuda_kernel_lab.kernels.torch_baselines.norms import layernorm, rmsnorm
from cuda_kernel_lab.kernels.torch_baselines.softmax import softmax
from cuda_kernel_lab.kernels.torch_baselines.swiglu import swiglu

__all__ = [
    "copy",
    "layernorm",
    "matmul",
    "reduction_sum",
    "rmsnorm",
    "scale",
    "softmax",
    "swiglu",
    "vector_add",
]
