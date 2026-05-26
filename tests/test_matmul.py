from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from cuda_kernel_lab.kernels.torch_baselines import matmul as torch_matmul
from cuda_kernel_lab.kernels.triton.matmul import is_available as triton_is_available
from cuda_kernel_lab.kernels.triton.matmul import matmul as triton_matmul
from cuda_kernel_lab.ops.matmul import flop_count, memory_traffic_bytes

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def available_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("torch", torch_matmul)]
    if triton_is_available():
        backends.append(("triton", triton_matmul))
    return backends


@requires_torch
@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
@pytest.mark.parametrize(("m", "n", "k"), [(4, 8, 16), (7, 9, 33), (16, 12, 32)])
def test_matmul_matches_reference(
    backend_name: str,
    backend: object,
    m: int,
    n: int,
    k: int,
) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    a = torch.randn((m, k), device=device, dtype=torch.float32)
    b = torch.randn((k, n), device=device, dtype=torch.float32)

    torch.testing.assert_close(backend(a, b), a @ b, rtol=1e-4, atol=1e-5)


@requires_torch
def test_matmul_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        torch_matmul(torch.randn(2, 4), torch.randn(5, 2))


def test_triton_matmul_rejects_unknown_schedule_before_runtime_checks() -> None:
    with pytest.raises(ValueError, match="schedule"):
        triton_matmul(None, None, schedule="round-robin")


def test_triton_matmul_rejects_invalid_persistent_waves_before_runtime_checks() -> None:
    with pytest.raises(ValueError, match="persistent_waves"):
        triton_matmul(None, None, persistent_waves=0)


def test_matmul_memory_traffic_estimate() -> None:
    assert memory_traffic_bytes(m=2, n=3, k=4, dtype_size=4) == 104


def test_matmul_flop_estimate() -> None:
    assert flop_count(m=2, n=3, k=4) == 48
