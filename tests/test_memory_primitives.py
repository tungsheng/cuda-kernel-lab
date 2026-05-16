from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kernels.torch_baselines.memory import (  # noqa: E402
    copy,
    flop_count,
    memory_traffic_bytes,
    reduction_sum,
    scale,
    vector_add,
)


def test_vector_add_matches_torch() -> None:
    a = torch.randn(256, dtype=torch.float32)
    b = torch.randn(256, dtype=torch.float32)

    torch.testing.assert_close(vector_add(a, b), a + b)


def test_copy_materializes_equal_tensor() -> None:
    x = torch.randn(128, dtype=torch.float32)
    out = copy(x)

    torch.testing.assert_close(out, x)
    assert out.data_ptr() != x.data_ptr()


def test_scale_matches_torch() -> None:
    x = torch.randn(128, dtype=torch.float32)

    torch.testing.assert_close(scale(x, 0.25), x * 0.25)


def test_reduction_sum_matches_torch() -> None:
    x = torch.randn(32, 16, dtype=torch.float32)

    torch.testing.assert_close(reduction_sum(x), x.sum())
    torch.testing.assert_close(reduction_sum(x, dim=1), x.sum(dim=1))


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
        vector_add(torch.randn(4), torch.randn(5))

