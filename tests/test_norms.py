from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from kernels.torch_baselines import layernorm as torch_layernorm
from kernels.torch_baselines import rmsnorm as torch_rmsnorm
from kernels.torch_baselines.norms import flop_count, memory_traffic_bytes
from kernels.triton.norms import is_available as triton_is_available
from kernels.triton.norms import layernorm as triton_layernorm
from kernels.triton.norms import rmsnorm as triton_rmsnorm

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def available_rmsnorm_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("torch", torch_rmsnorm)]
    if triton_is_available():
        backends.append(("triton", triton_rmsnorm))
    return backends


def available_layernorm_backends() -> list[tuple[str, object]]:
    backends: list[tuple[str, object]] = [("torch", torch_layernorm)]
    if triton_is_available():
        backends.append(("triton", triton_layernorm))
    return backends


@requires_torch
@pytest.mark.parametrize(("backend_name", "backend"), available_rmsnorm_backends())
@pytest.mark.parametrize("shape", [(4, 16), (7, 33), (16, 128)])
def test_rmsnorm_matches_reference(
    backend_name: str,
    backend: object,
    shape: tuple[int, int],
) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    x = torch.randn(shape, device=device, dtype=torch.float32)
    weight = torch.randn((shape[-1],), device=device, dtype=torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + 1e-6) * weight

    torch.testing.assert_close(backend(x, weight, eps=1e-6), expected, rtol=1e-5, atol=1e-6)


@requires_torch
@pytest.mark.parametrize(("backend_name", "backend"), available_layernorm_backends())
@pytest.mark.parametrize("shape", [(4, 16), (7, 33), (16, 128)])
def test_layernorm_matches_torch(
    backend_name: str,
    backend: object,
    shape: tuple[int, int],
) -> None:
    device = "cuda" if backend_name == "triton" else "cpu"
    x = torch.randn(shape, device=device, dtype=torch.float32)
    weight = torch.randn((shape[-1],), device=device, dtype=torch.float32)
    bias = torch.randn((shape[-1],), device=device, dtype=torch.float32)
    expected = torch.nn.functional.layer_norm(x, (shape[-1],), weight, bias, eps=1e-5)

    torch.testing.assert_close(backend(x, weight, bias, eps=1e-5), expected, rtol=1e-5, atol=1e-6)


@requires_torch
def test_rmsnorm_rejects_bad_weight_shape() -> None:
    with pytest.raises(ValueError, match="weight length"):
        torch_rmsnorm(torch.randn(2, 4), torch.randn(5))


@requires_torch
def test_layernorm_rejects_bad_bias_shape() -> None:
    with pytest.raises(ValueError, match="bias shape"):
        torch_layernorm(torch.randn(2, 4), torch.randn(4), torch.randn(5))


def test_norm_memory_traffic_estimates() -> None:
    assert memory_traffic_bytes("rmsnorm", rows=4, cols=8, dtype_size=4) == 384
    assert memory_traffic_bytes("layernorm", rows=4, cols=8, dtype_size=4) == 512


def test_norm_flop_estimates() -> None:
    assert flop_count("rmsnorm", rows=4, cols=8) == 4 * (7 + 4 * 8)
    assert flop_count("layernorm", rows=4, cols=8) == 4 * (2 * 7 + 6 * 8)
