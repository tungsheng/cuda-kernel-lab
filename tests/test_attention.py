from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from cuda_kernel_lab.benchmarks.attention import run_one
from cuda_kernel_lab.kernels.torch_baselines import decode_attention
from cuda_kernel_lab.ops.attention import (
    decode_attention_flop_count,
    decode_attention_memory_traffic_bytes,
)

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


@requires_torch
def test_decode_attention_matches_reference() -> None:
    query = torch.randn((2, 4), dtype=torch.float32)
    key_cache = torch.randn((5, 2, 4), dtype=torch.float32)
    value_cache = torch.randn((5, 2, 4), dtype=torch.float32)
    scale = 0.25

    expected_heads = []
    for head in range(query.shape[0]):
        scores = (key_cache[:, head, :].float() * query[head].float()).sum(dim=-1) * scale
        probs = scores.softmax(dim=-1)
        expected_heads.append((probs[:, None] * value_cache[:, head, :]).sum(dim=0))
    expected = torch.stack(expected_heads)

    torch.testing.assert_close(
        decode_attention(query, key_cache, value_cache, scale=scale),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


@requires_torch
def test_decode_attention_supports_out() -> None:
    query = torch.randn((2, 4), dtype=torch.float32)
    key_cache = torch.randn((5, 2, 4), dtype=torch.float32)
    value_cache = torch.randn((5, 2, 4), dtype=torch.float32)
    out = torch.empty_like(query)

    result = decode_attention(query, key_cache, value_cache, out=out)

    assert result is out
    torch.testing.assert_close(out, decode_attention(query, key_cache, value_cache))


@requires_torch
def test_decode_attention_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="query shape"):
        decode_attention(
            torch.randn((3, 4)),
            torch.randn((5, 2, 4)),
            torch.randn((5, 2, 4)),
        )


@requires_torch
def test_attention_benchmark_records_metadata() -> None:
    result = run_one(
        torch=torch,
        backend="torch",
        seq_len=4,
        num_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
        scale=None,
        warmup=0,
        iterations=1,
        skip_correctness=False,
    )

    assert result.name == "torch:decode_attention"
    assert result.shape == (4, 2, 4)
    assert result.strategy == "torch-baseline"
    assert result.parameters == {
        "seq_len": 4,
        "num_heads": 2,
        "head_dim": 4,
        "scale": 0.5,
    }
    assert result.correctness is not None
    assert result.correctness.passed is True


def test_decode_attention_memory_traffic_estimate() -> None:
    assert (
        decode_attention_memory_traffic_bytes(
            seq_len=4,
            num_heads=2,
            head_dim=3,
            dtype_size=2,
        )
        == 120
    )


def test_decode_attention_flop_estimate() -> None:
    assert decode_attention_flop_count(seq_len=4, num_heads=2, head_dim=3) == 136


def test_decode_attention_estimates_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        decode_attention_memory_traffic_bytes(seq_len=0, num_heads=2, head_dim=3, dtype_size=2)

    with pytest.raises(ValueError, match="dtype_size"):
        decode_attention_memory_traffic_bytes(seq_len=4, num_heads=2, head_dim=3, dtype_size=0)
