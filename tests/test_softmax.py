from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from inference_kernel_lab.kernels.torch_baselines import softmax as torch_softmax
from inference_kernel_lab.kernels.triton.softmax import is_available as triton_is_available
from inference_kernel_lab.kernels.triton.softmax import softmax as triton_softmax
from inference_kernel_lab.ops.softmax import flop_count, memory_traffic_bytes

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def available_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("torch", torch_softmax)]
    if triton_is_available():
        backends.append(("triton", triton_softmax))
    return backends


@requires_torch
@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
@pytest.mark.parametrize("shape", [(4, 16), (7, 33), (32, 128)])
def test_softmax_matches_torch(backend_name: str, backend: object, shape: tuple[int, int]) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    x = torch.randn(shape, device=device, dtype=torch.float32)

    torch.testing.assert_close(backend(x), torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-6)


@requires_torch
def test_softmax_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2D tensor"):
        torch_softmax(torch.randn(2, 3, 4))


def test_softmax_memory_traffic_estimates() -> None:
    assert memory_traffic_bytes(rows=4, cols=8, dtype_size=4, model="fused") == 256
    assert memory_traffic_bytes(rows=4, cols=8, dtype_size=4, model="naive") == 512


def test_softmax_flop_estimate() -> None:
    assert flop_count(rows=4, cols=8) == 4 * (2 * 7 + 3 * 8)
