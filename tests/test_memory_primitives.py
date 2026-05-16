from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kernels.torch_baselines import memory as torch_memory  # noqa: E402
from kernels.torch_baselines.memory import (  # noqa: E402
    flop_count,
    memory_traffic_bytes,
)
from kernels.triton import memory as triton_memory  # noqa: E402


def available_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("torch", torch_memory)]
    if triton_memory.is_available():
        backends.append(("triton", triton_memory))
    return backends


@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
def test_vector_add_matches_torch(backend_name: str, backend: object) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    a = torch.randn(256, device=device, dtype=torch.float32)
    b = torch.randn(256, device=device, dtype=torch.float32)

    torch.testing.assert_close(backend.vector_add(a, b), a + b)


@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
def test_copy_materializes_equal_tensor(backend_name: str, backend: object) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    x = torch.randn(128, device=device, dtype=torch.float32)
    out = backend.copy(x)

    torch.testing.assert_close(out, x)
    assert out.data_ptr() != x.data_ptr()


@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
def test_scale_matches_torch(backend_name: str, backend: object) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    x = torch.randn(128, device=device, dtype=torch.float32)

    torch.testing.assert_close(backend.scale(x, 0.25), x * 0.25)


@pytest.mark.parametrize(("backend_name", "backend"), available_backends())
def test_reduction_sum_matches_torch(backend_name: str, backend: object) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    x = torch.randn(32, 16, device=device, dtype=torch.float32)

    torch.testing.assert_close(backend.reduction_sum(x), x.sum(), rtol=1e-5, atol=1e-5)


def test_memory_traffic_estimates() -> None:
    assert memory_traffic_bytes("vector_add", numel=10, dtype_size=4) == 120
    assert memory_traffic_bytes("copy", numel=10, dtype_size=4) == 80
    assert memory_traffic_bytes("scale", numel=10, dtype_size=4) == 80
    assert memory_traffic_bytes("reduction_sum", numel=10, dtype_size=4) == 44


def test_flop_estimates() -> None:
    assert flop_count("vector_add", numel=10) == 10
    assert flop_count("scale", numel=10) == 10
    assert flop_count("copy", numel=10) == 0
    assert flop_count("reduction_sum", numel=10) == 9


def test_vector_add_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        torch_memory.vector_add(torch.randn(4), torch.randn(5))
