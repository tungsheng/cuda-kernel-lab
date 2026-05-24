from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuda_kernel_lab import benchmark_compare


def test_compare_detects_throughput_regression(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    _write_jsonl(
        baseline_dir / "matmul.jsonl",
        [_record(name="triton:matmul", tflops=100.0)],
    )
    _write_jsonl(
        candidate_dir / "matmul.jsonl",
        [_record(name="triton:matmul", tflops=91.0)],
    )

    comparison = benchmark_compare.compare_result_dirs(
        baseline_dir,
        candidate_dir,
        max_regression_pct=5.0,
    )

    assert comparison.passed is False
    assert comparison.regressions[0].metric == "TFLOP/s"
    assert comparison.regressions[0].regression_pct == pytest.approx(9.0)
    assert "Status: fail" in benchmark_compare.render_markdown(comparison)


def test_compare_passes_when_candidate_is_within_threshold(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    _write_jsonl(
        baseline_dir / "matmul.jsonl",
        [_record(name="triton:matmul", tflops=100.0)],
    )
    _write_jsonl(
        candidate_dir / "matmul.jsonl",
        [_record(name="triton:matmul", tflops=97.0)],
    )

    comparison = benchmark_compare.compare_result_dirs(
        baseline_dir,
        candidate_dir,
        max_regression_pct=5.0,
    )

    assert comparison.passed is True
    assert "| none |" in benchmark_compare.render_markdown(comparison)


def test_compare_main_exits_nonzero_for_regression(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    _write_jsonl(
        baseline_dir / "matmul.jsonl",
        [_record(name="triton:matmul", tflops=100.0)],
    )
    _write_jsonl(
        candidate_dir / "matmul.jsonl",
        [_record(name="triton:matmul", tflops=90.0)],
    )

    with pytest.raises(SystemExit) as exc:
        benchmark_compare.main(
            [
                "--baseline-dir",
                str(baseline_dir),
                "--candidate-dir",
                str(candidate_dir),
                "--max-regression-pct",
                "5",
            ]
        )

    assert exc.value.code == 1


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _record(
    *,
    name: str,
    tflops: float,
    correctness_passed: bool = True,
) -> dict[str, object]:
    return {
        "run": {
            "benchmark": "matmul",
            "args": _matmul_params(),
            "command": "uv run benchmark-matmul",
            "timestamp_utc": "2026-05-24T00:00:00+00:00",
            "git_commit": "abc123",
            "git_dirty": False,
            "host": {"python": "3.13.2", "platform": "Linux"},
            "packages": {"torch": "2.9.1", "triton": "3.5.1"},
            "cuda_devices": [{"index": 0, "name": "NVIDIA H200"}],
        },
        "result": {
            "name": name,
            "device": "cuda",
            "dtype": "float16",
            "shape": [4096, 4096, 4096],
            "p50_ms": 1.0,
            "p95_ms": 1.05,
            "p99_ms": 1.1,
            "bytes_moved": 1,
            "bandwidth_gbps": 1.0,
            "flops": 1,
            "tflops": tflops,
            "latencies_ms": [1.0, 1.05, 1.1],
            "strategy": "triton-tiled-dot" if name.startswith("triton") else "torch-baseline",
            "variant": "block_m=128, block_n=128, block_k=64, group_m=1",
            "parameters": _matmul_params(),
            "metrics": {},
            "correctness": {
                "checked": True,
                "passed": correctness_passed,
                "reference_backend": "torch",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": 1e-2,
                "rtol": 1e-2,
                "message": None,
            },
        },
    }


def _matmul_params() -> dict[str, object]:
    return {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "num_warps": 4,
        "num_stages": 4,
        "input_precision": "tf32",
        "group_m": 1,
    }
