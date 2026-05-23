from __future__ import annotations

import argparse

import pytest

try:
    import torch
except ImportError:
    torch = None

from cuda_kernel_lab.benchmark import (
    BenchmarkResult,
    CorrectnessResult,
    check_tensors_close,
    collect_run_metadata,
)
from cuda_kernel_lab.optimization import memory_optimization

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def test_benchmark_result_includes_strategy_optimization_and_correctness_metadata() -> None:
    result = BenchmarkResult(
        name="triton:vector_add",
        device="cuda",
        dtype="float32",
        shape=(1024,),
        latencies_ms=[1.0, 2.0, 3.0],
        bytes_moved=12_288,
        flops=1024,
        strategy="triton-block-size",
        variant="block_size=1024",
        parameters={"block_size": 1024},
        metrics={"tokens_per_second_p50": 1000.0},
        optimization=memory_optimization(
            backend="triton",
            op_name="vector_add",
            reduction_strategy="iterative",
        ),
        correctness=CorrectnessResult(
            checked=True,
            passed=True,
            reference_backend="torch",
            max_abs_error=0.0,
            max_rel_error=0.0,
            atol=1e-5,
            rtol=1e-4,
        ),
    )

    payload = result.as_dict()

    assert payload["strategy"] == "triton-block-size"
    assert payload["variant"] == "block_size=1024"
    assert payload["parameters"] == {"block_size": 1024}
    assert payload["metrics"] == {"tokens_per_second_p50": 1000.0}
    assert payload["optimization"]["method_family"] == "launch tuning"
    assert payload["optimization"]["technique"] == "Coalesced block-size tuning"
    assert "knobs" not in payload["optimization"]
    assert payload["correctness"]["checked"] is True
    assert payload["correctness"]["passed"] is True


def test_run_metadata_can_use_exported_git_context(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_KERNEL_LAB_GIT_COMMIT", "abc123")
    monkeypatch.setenv("CUDA_KERNEL_LAB_GIT_DIRTY", "false")

    metadata = collect_run_metadata("memory_bandwidth", argparse.Namespace(output="results.jsonl"))

    assert metadata.git_commit == "abc123"
    assert metadata.git_dirty is False


def test_run_metadata_includes_optional_provider_context(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_KERNEL_LAB_PROVIDER", "runpod")
    monkeypatch.setenv("CUDA_KERNEL_LAB_PROVIDER_ID", "pod-123")
    monkeypatch.setenv("CUDA_KERNEL_LAB_PROVIDER_GPU_ID", "NVIDIA L4")
    monkeypatch.setenv("CUDA_KERNEL_LAB_PROVIDER_CLOUD_TYPE", "SECURE")

    metadata = collect_run_metadata("memory_bandwidth", argparse.Namespace(output="results.jsonl"))

    assert metadata.as_dict()["provider"] == {
        "name": "runpod",
        "id": "pod-123",
        "gpu_id": "NVIDIA L4",
        "cloud_type": "SECURE",
    }


def test_run_metadata_omits_provider_context_by_default(monkeypatch) -> None:
    for name in (
        "CUDA_KERNEL_LAB_PROVIDER",
        "CUDA_KERNEL_LAB_PROVIDER_ID",
        "CUDA_KERNEL_LAB_PROVIDER_GPU_ID",
        "CUDA_KERNEL_LAB_PROVIDER_CLOUD_TYPE",
        "CUDA_KERNEL_LAB_PROVIDER_IMAGE",
        "CUDA_KERNEL_LAB_PROVIDER_TEMPLATE_ID",
        "CUDA_KERNEL_LAB_PROVIDER_REGION",
    ):
        monkeypatch.delenv(name, raising=False)

    metadata = collect_run_metadata("memory_bandwidth", argparse.Namespace(output="results.jsonl"))

    assert "provider" not in metadata.as_dict()


@requires_torch
def test_check_tensors_close_reports_nonfinite_values_without_nan_errors() -> None:
    result = check_tensors_close(
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, 2.0]),
        torch=torch,
        rtol=1e-5,
        atol=1e-6,
    )

    assert result.passed is False
    assert result.max_abs_error == 0.0
    assert result.max_rel_error == 0.0
    assert result.message == "non-finite values: actual=1, expected=0"
