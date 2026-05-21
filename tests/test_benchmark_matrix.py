from __future__ import annotations

from pathlib import Path

import pytest

from cuda_kernel_lab import benchmark_matrix


def test_default_matrix_covers_expected_primitives_and_dtypes() -> None:
    entries = benchmark_matrix.build_matrix()

    assert len(entries) == 8
    assert [(entry.primitive, entry.dtype) for entry in entries] == [
        ("memory", "float32"),
        ("softmax", "float32"),
        ("norms", "float32"),
        ("swiglu", "float32"),
        ("memory", "float16"),
        ("softmax", "float16"),
        ("norms", "float16"),
        ("swiglu", "float16"),
    ]


def test_matrix_commands_include_shapes_and_output_paths() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        warmup=7,
        iterations=11,
        memory_block_size=2048,
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert (
        "uv run benchmark-memory --backend all --device cuda --op all --numel 16777216 "
        "--dtype float32 --block-size 2048 --reduction-strategy iterative "
        "--warmup 7 --iterations 11 "
        "--output results/memory.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-softmax --backend all --device cuda --rows 4096 --cols 1024 "
        "--dtype float16 --warmup 7 --iterations 11 --output results/softmax.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-norms --backend all --device cuda --op all --rows 4096 "
        "--cols 4096 --dtype float16 --warmup 7 --iterations 11 --output results/norms.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-swiglu --backend all --device cuda --rows 4096 --cols 4096 "
        "--dtype float16 --block-size 1024 --warmup 7 --iterations 11 "
        "--output results/swiglu.jsonl"
    ) in command_lines


def test_matrix_can_include_vector_add_strategy_sweep() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_vector_add_sweep=True,
        vector_add_sweep_block_sizes=(512, 2048),
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 10
    assert (
        "uv run benchmark-memory --backend all --device cuda --op vector_add "
        "--numel 16777216 --dtype float32 --block-size 512 --reduction-strategy iterative "
        "--warmup 25 --iterations 100 "
        "--output results/vector-add-block-size.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-memory --backend all --device cuda --op vector_add "
        "--numel 16777216 --dtype float32 --block-size 2048 --reduction-strategy iterative "
        "--warmup 25 --iterations 100 "
        "--output results/vector-add-block-size.jsonl"
    ) in command_lines


def test_matrix_sweep_does_not_repeat_baseline_block_size() -> None:
    entries = benchmark_matrix.build_matrix(include_vector_add_sweep=True)
    command_lines = [entry.shell_line() for entry in entries]
    sweep_lines = [line for line in command_lines if "vector-add-block-size.jsonl" in line]

    assert len(entries) == 10
    assert len(sweep_lines) == 2
    assert all("--block-size 1024" not in line for line in sweep_lines)


def test_matrix_can_include_reduction_strategy_sweep() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_reduction_sweep=True,
        reduction_sweep_strategies=("iterative", "two_pass"),
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 9
    assert (
        "uv run benchmark-memory --backend all --device cuda --op reduction_sum "
        "--numel 16777216 --dtype float32 --block-size 1024 --reduction-strategy two_pass "
        "--warmup 25 --iterations 100 --output results/reduction-strategy.jsonl"
    ) in command_lines


def test_matrix_can_include_matmul_progression() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_matmul=True,
        matmul_block_m=32,
        matmul_block_n=16,
        matmul_block_k=64,
        matmul_num_warps=8,
        matmul_num_stages=4,
        matmul_input_precision="ieee",
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 10
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024 "
        "--dtype float32 --block-m 32 --block-n 16 --block-k 64 "
        "--num-warps 8 --num-stages 4 --input-precision ieee "
        "--warmup 25 --iterations 100 --output results/matmul.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024 "
        "--dtype float16 --block-m 32 --block-n 16 --block-k 64 "
        "--num-warps 8 --num-stages 4 --input-precision ieee "
        "--warmup 25 --iterations 100 --output results/matmul.jsonl"
    ) in command_lines


def test_matrix_can_include_matmul_strategy_sweep() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_matmul_sweep=True,
        matmul_sweep_tile_shapes=((16, 16, 32), (16, 32, 32), (32, 16, 32)),
        matmul_sweep_launch_configs=((4, 3),),
    )
    command_lines = [entry.shell_line() for entry in entries]
    sweep_lines = [line for line in command_lines if "matmul-tile-shape.jsonl" in line]

    assert len(entries) == 11
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024 "
        "--dtype float16 --block-m 16 --block-n 16 --block-k 32 "
        "--num-warps 4 --num-stages 3 --input-precision ieee "
        "--warmup 25 --iterations 100 --output results/matmul.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024 "
        "--dtype float16 --block-m 16 --block-n 32 --block-k 32 "
        "--num-warps 4 --num-stages 3 --input-precision ieee "
        "--warmup 25 --iterations 100 --output results/matmul-tile-shape.jsonl"
    ) in sweep_lines
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024 "
        "--dtype float16 --block-m 32 --block-n 16 --block-k 32 "
        "--num-warps 4 --num-stages 3 --input-precision ieee "
        "--warmup 25 --iterations 100 --output results/matmul-tile-shape.jsonl"
    ) in sweep_lines
    assert all("--block-m 16 --block-n 16 --block-k 32" not in line for line in sweep_lines)


def test_matrix_can_include_rmsnorm_shape_sweep() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_rmsnorm_shape_sweep=True,
        rmsnorm_shape_sweep_shapes=((512, 1024), (4096, 8192)),
    )
    command_lines = [entry.shell_line() for entry in entries]
    sweep_lines = [line for line in command_lines if "rmsnorm-shape-sweep.jsonl" in line]

    assert len(entries) == 10
    assert (
        "uv run benchmark-norms --backend all --device cuda --op rmsnorm "
        "--rows 512 --cols 1024 --dtype float16 --warmup 25 --iterations 100 "
        "--output results/rmsnorm-shape-sweep.jsonl"
    ) in sweep_lines
    assert (
        "uv run benchmark-norms --backend all --device cuda --op rmsnorm "
        "--rows 4096 --cols 8192 --dtype float16 --warmup 25 --iterations 100 "
        "--output results/rmsnorm-shape-sweep.jsonl"
    ) in sweep_lines


def test_matrix_can_include_attention_baseline() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_attention_baseline=True,
        attention_seq_len=4096,
        attention_num_heads=8,
        attention_head_dim=64,
        attention_dtype="float16",
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 9
    assert (
        "uv run benchmark-attention --backend torch --device cuda --seq-len 4096 "
        "--num-heads 8 --head-dim 64 --dtype float16 --warmup 25 --iterations 100 "
        "--output results/attention.jsonl"
    ) in command_lines


def test_matrix_can_include_decode_step_graph_workflow() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_decode_step=True,
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 9
    assert (
        "uv run benchmark-decode-step --mode all --device cuda --dtype float16 "
        "--warmup 25 --iterations 100 --output results/decode-step.jsonl"
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
    assert "uv run benchmark-swiglu" in output
    assert "results/memory.jsonl" in output
    assert "results/swiglu.jsonl" in output


def test_matrix_rejects_invalid_timing_values() -> None:
    with pytest.raises(ValueError, match="warmup"):
        benchmark_matrix.build_matrix(warmup=-1)

    with pytest.raises(ValueError, match="iterations"):
        benchmark_matrix.build_matrix(iterations=0)

    with pytest.raises(ValueError, match="memory_block_size"):
        benchmark_matrix.build_matrix(memory_block_size=0)

    with pytest.raises(ValueError, match="swiglu_block_size"):
        benchmark_matrix.build_matrix(swiglu_block_size=0)

    with pytest.raises(ValueError, match="matmul block sizes"):
        benchmark_matrix.build_matrix(matmul_block_m=0)

    with pytest.raises(ValueError, match="matmul launch settings"):
        benchmark_matrix.build_matrix(matmul_num_warps=0)

    with pytest.raises(ValueError, match="matmul_input_precision"):
        benchmark_matrix.build_matrix(matmul_input_precision="fast")

    with pytest.raises(ValueError, match="matmul_sweep_tile_shapes"):
        benchmark_matrix.build_matrix(matmul_sweep_tile_shapes=((16, 16),))

    with pytest.raises(ValueError, match="matmul_sweep_tile_shapes"):
        benchmark_matrix.build_matrix(matmul_sweep_tile_shapes=((16, 16, 0),))

    with pytest.raises(ValueError, match="matmul_sweep_launch_configs"):
        benchmark_matrix.build_matrix(matmul_sweep_launch_configs=((4,),))

    with pytest.raises(ValueError, match="matmul_sweep_launch_configs"):
        benchmark_matrix.build_matrix(matmul_sweep_launch_configs=((4, 0),))

    with pytest.raises(ValueError, match="rmsnorm_shape_sweep_shapes"):
        benchmark_matrix.build_matrix(rmsnorm_shape_sweep_shapes=((1024,),))

    with pytest.raises(ValueError, match="rmsnorm_shape_sweep_shapes"):
        benchmark_matrix.build_matrix(rmsnorm_shape_sweep_shapes=((1024, 0),))

    with pytest.raises(ValueError, match="rmsnorm_shape_sweep_dtype"):
        benchmark_matrix.build_matrix(rmsnorm_shape_sweep_dtype="int8")

    with pytest.raises(ValueError, match="attention shape"):
        benchmark_matrix.build_matrix(attention_seq_len=0)

    with pytest.raises(ValueError, match="attention_dtype"):
        benchmark_matrix.build_matrix(attention_dtype="int8")

    with pytest.raises(ValueError, match="vector_add_sweep_block_sizes"):
        benchmark_matrix.build_matrix(vector_add_sweep_block_sizes=(512, 0))

    with pytest.raises(ValueError, match="reduction_strategy"):
        benchmark_matrix.build_matrix(reduction_strategy="")


def test_matrix_parses_strategy_sweep_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_matrix.main(
        [
            "--dry-run",
            "--output-dir",
            "results",
            "--include-vector-add-sweep",
            "--vector-add-sweep-block-sizes",
            "256,512",
            "--include-reduction-sweep",
            "--reduction-sweep-strategies",
            "iterative,two_pass",
            "--include-matmul",
            "--include-matmul-sweep",
            "--matmul-sweep-tile-shapes",
            "16x32x32,32x16x32",
            "--matmul-sweep-launch-configs",
            "4x3,8x4",
            "--include-rmsnorm-shape-sweep",
            "--rmsnorm-shape-sweep-shapes",
            "512x1024,4096x8192",
            "--include-attention-baseline",
            "--attention-seq-len",
            "4096",
            "--attention-num-heads",
            "8",
            "--attention-head-dim",
            "64",
            "--include-decode-step",
        ]
    )

    output = capsys.readouterr().out
    assert "--block-size 256" in output
    assert "--block-size 512" in output
    assert "results/vector-add-block-size.jsonl" in output
    assert "--reduction-strategy two_pass" in output
    assert "results/reduction-strategy.jsonl" in output
    assert "uv run benchmark-matmul" in output
    assert "results/matmul.jsonl" in output
    assert "results/matmul-tile-shape.jsonl" in output
    assert "--block-m 16 --block-n 32 --block-k 32" in output
    assert "--block-m 32 --block-n 16 --block-k 32" in output
    assert "--num-warps 8 --num-stages 4" in output
    assert "results/rmsnorm-shape-sweep.jsonl" in output
    assert "--rows 512 --cols 1024 --dtype float16" in output
    assert "--rows 4096 --cols 8192 --dtype float16" in output
    assert "results/attention.jsonl" in output
    assert "uv run benchmark-attention --backend torch" in output
    assert "--seq-len 4096 --num-heads 8 --head-dim 64" in output
    assert "results/decode-step.jsonl" in output
    assert "uv run benchmark-decode-step --mode all" in output
