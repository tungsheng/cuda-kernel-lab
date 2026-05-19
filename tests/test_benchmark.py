from __future__ import annotations

import argparse

from cuda_kernel_lab.benchmark import BenchmarkResult, CorrectnessResult, collect_run_metadata


def test_benchmark_result_includes_strategy_and_correctness_metadata() -> None:
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
    assert payload["correctness"]["checked"] is True
    assert payload["correctness"]["passed"] is True


def test_run_metadata_can_use_exported_git_context(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_KERNEL_LAB_GIT_COMMIT", "abc123")
    monkeypatch.setenv("CUDA_KERNEL_LAB_GIT_DIRTY", "false")

    metadata = collect_run_metadata("memory_bandwidth", argparse.Namespace(output="results.jsonl"))

    assert metadata.git_commit == "abc123"
    assert metadata.git_dirty is False
