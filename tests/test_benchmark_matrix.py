from __future__ import annotations

from pathlib import Path

import pytest

from cuda_kernel_lab import benchmark_matrix


def test_default_matrix_covers_expected_primitives_and_dtypes() -> None:
    entries = benchmark_matrix.build_matrix()

    assert len(entries) == 6
    assert [(entry.primitive, entry.dtype) for entry in entries] == [
        ("memory", "float32"),
        ("softmax", "float32"),
        ("norms", "float32"),
        ("memory", "float16"),
        ("softmax", "float16"),
        ("norms", "float16"),
    ]


def test_matrix_commands_include_shapes_and_output_paths() -> None:
    entries = benchmark_matrix.build_matrix(output_dir=Path("results"), warmup=7, iterations=11)
    command_lines = [entry.shell_line() for entry in entries]

    assert (
        "uv run benchmark-memory --backend all --device cuda --op all --numel 16777216 "
        "--dtype float32 --warmup 7 --iterations 11 --output results/memory.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-softmax --backend all --device cuda --rows 4096 --cols 1024 "
        "--dtype float16 --warmup 7 --iterations 11 --output results/softmax.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-norms --backend all --device cuda --op all --rows 4096 "
        "--cols 4096 --dtype float16 --warmup 7 --iterations 11 --output results/norms.jsonl"
    ) in command_lines


def test_dry_run_prints_without_executing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr(benchmark_matrix.subprocess, "run", fail_run)

    benchmark_matrix.main(["--dry-run", "--output-dir", "results"])

    output = capsys.readouterr().out
    assert "uv run benchmark-memory" in output
    assert "uv run benchmark-softmax" in output
    assert "uv run benchmark-norms" in output
    assert "results/memory.jsonl" in output


def test_matrix_rejects_invalid_timing_values() -> None:
    with pytest.raises(ValueError, match="warmup"):
        benchmark_matrix.build_matrix(warmup=-1)

    with pytest.raises(ValueError, match="iterations"):
        benchmark_matrix.build_matrix(iterations=0)
