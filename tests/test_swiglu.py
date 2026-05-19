from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from cuda_kernel_lab.kernels.torch_baselines import swiglu as torch_swiglu
from cuda_kernel_lab.kernels.triton.swiglu import is_available as triton_is_available
from cuda_kernel_lab.kernels.triton.swiglu import swiglu as triton_swiglu
from cuda_kernel_lab.ops.swiglu import flop_count, memory_traffic_bytes

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def available_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("torch", torch_swiglu)]
    if triton_is_available():
        backends.append(("triton", triton_swiglu))
    return backends


@requires_torch
@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
@pytest.mark.parametrize("shape", [(4, 16), (7, 33), (32, 128)])
def test_swiglu_matches_reference(
    backend_name: str,
    backend: object,
    shape: tuple[int, int],
) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    gate = torch.randn(shape, device=device, dtype=torch.float32)
    up = torch.randn(shape, device=device, dtype=torch.float32)
    expected = gate * gate.sigmoid() * up

    torch.testing.assert_close(backend(gate, up), expected, rtol=1e-5, atol=1e-6)


@requires_torch
def test_swiglu_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        torch_swiglu(torch.randn(2, 4), torch.randn(2, 5))


def test_swiglu_memory_traffic_estimate() -> None:
    assert memory_traffic_bytes(numel=10, dtype_size=4) == 120


def test_swiglu_flop_estimate() -> None:
    assert flop_count(numel=10) == 50
