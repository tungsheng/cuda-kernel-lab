from __future__ import annotations

import json
import subprocess
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


def test_default_output_dir_uses_runpod_root() -> None:
    assert benchmark_matrix.DEFAULT_OUTPUT_DIR == Path("experiments/results/runpod/manual-run")


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


def test_matrix_can_select_grouped_matmul_program_ordering() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_matmul=True,
        matmul_group_m=4,
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert any(
        "--input-precision ieee --group-m 4 --warmup 25 --iterations 100 "
        "--output results/matmul.jsonl" in line
        for line in command_lines
    )


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


def test_matrix_can_include_tensor_core_suite() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        include_tensor_core_suite=True,
        tensor_core_matmul_shapes=((2048, 2048, 2048),),
        tensor_core_dtypes=("float16", "bfloat16"),
    )
    command_lines = [entry.shell_line() for entry in entries]
    tensor_lines = [line for line in command_lines if "matmul-tensor-core.jsonl" in line]

    assert len(entries) == 10
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 2048 --n 2048 --k 2048 "
        "--dtype float16 --block-m 128 --block-n 128 --block-k 64 "
        "--num-warps 4 --num-stages 4 --input-precision tf32 "
        "--warmup 25 --iterations 100 --output results/matmul-tensor-core.jsonl"
    ) in tensor_lines
    assert (
        "uv run benchmark-matmul --backend all --device cuda --m 2048 --n 2048 --k 2048 "
        "--dtype bfloat16 --block-m 128 --block-n 128 --block-k 64 "
        "--num-warps 4 --num-stages 4 --input-precision tf32 "
        "--warmup 25 --iterations 100 --output results/matmul-tensor-core.jsonl"
    ) in tensor_lines


def test_h200_roofline_suite_adds_focused_benchmark_tracks() -> None:
    entries = benchmark_matrix.build_matrix(
        suite="h200-roofline",
        output_dir=Path("results"),
        tensor_core_matmul_shapes=((2048, 2048, 2048),),
        tensor_core_dtypes=("bfloat16",),
        matmul_sweep_tile_shapes=((16, 16, 32), (32, 32, 32)),
        matmul_sweep_launch_configs=((4, 3),),
    )
    command_lines = [entry.shell_line() for entry in entries]
    matmul_lines = [line for line in command_lines if "results/matmul.jsonl" in line]
    tile_lines = [line for line in command_lines if "results/matmul-tile-shape.jsonl" in line]
    tuning_lines = [line for line in command_lines if "results/matmul-tuning.jsonl" in line]
    impact_lines = [line for line in command_lines if "results/matmul-llm-impact.jsonl" in line]

    assert tile_lines
    assert tuning_lines
    assert impact_lines
    accelerator_matmul_lines = matmul_lines + tile_lines + tuning_lines
    assert all("--input-precision tf32" in line for line in accelerator_matmul_lines)
    assert any("--m 512 --n 11008 --k 4096 --dtype bfloat16" in line for line in tuning_lines)
    assert any("--m 512 --n 11008 --k 4096 --dtype bfloat16" in line for line in impact_lines)
    assert any("--group-m 4" in line for line in impact_lines)
    assert any("--group-m 8" in line for line in impact_lines)
    assert any("--dtype bfloat16" in line for line in command_lines)
    assert any("results/rmsnorm-shape-sweep.jsonl" in line for line in command_lines)
    assert any("results/attention.jsonl" in line for line in command_lines)


def test_h200_matmul_autotune_suite_runs_only_repeated_matmul_candidates() -> None:
    entries = benchmark_matrix.build_matrix(
        suite="h200-matmul-autotune",
        output_dir=Path("results"),
        matmul_autotune_shapes=((512, 11008, 4096),),
        matmul_autotune_dtypes=("float16",),
        matmul_autotune_schedules=("standard", "persistent"),
        matmul_autotune_configs=((128, 128, 64, 4, 4, 1), (128, 128, 64, 4, 4, 4)),
        matmul_autotune_repeats=2,
        matmul_autotune_seed=7,
    )
    second_entries = benchmark_matrix.build_matrix(
        suite="h200-matmul-autotune",
        output_dir=Path("results"),
        matmul_autotune_shapes=((512, 11008, 4096),),
        matmul_autotune_dtypes=("float16",),
        matmul_autotune_schedules=("standard", "persistent"),
        matmul_autotune_configs=((128, 128, 64, 4, 4, 1), (128, 128, 64, 4, 4, 4)),
        matmul_autotune_repeats=2,
        matmul_autotune_seed=7,
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 8
    assert command_lines == [entry.shell_line() for entry in second_entries]
    assert all("benchmark-matmul" in line for line in command_lines)
    assert all("results/matmul-autotune.jsonl" in line for line in command_lines)
    assert not any("benchmark-memory" in line for line in command_lines)
    assert sum("--group-m 4" in line for line in command_lines) == 4
    assert sum("--schedule persistent" in line for line in command_lines) == 4


def test_h200_matmul_autotune_defaults_avoid_over_shared_memory_candidate() -> None:
    entries = benchmark_matrix.build_matrix(
        suite="h200-matmul-autotune",
        output_dir=Path("results"),
        matmul_autotune_shapes=((4096, 4096, 4096),),
        matmul_autotune_dtypes=("bfloat16",),
        matmul_autotune_repeats=1,
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert command_lines
    assert not any("--block-k 128" in line for line in command_lines)
    assert any("--num-warps 8 --num-stages 4" in line for line in command_lines)
    assert any("--num-warps 8 --num-stages 5" in line for line in command_lines)
    assert any("--num-warps 8 --num-stages 6" in line for line in command_lines)


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

    assert len(entries) == 10
    assert (
        "uv run benchmark-decode-step --mode all --device cuda --dtype float16 "
        "--warmup 25 --iterations 100 --output results/decode-step.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-decode-step --dynamic-trace --mode all --device cuda "
        "--dtype float16 --warmup 25 --iterations 100 "
        "--output results/decode-step-dynamic.jsonl"
    ) in command_lines


def test_matrix_can_run_only_decode_step_workflow() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        only_decode_step=True,
        include_vector_add_sweep=True,
        include_reduction_sweep=True,
        include_matmul_sweep=True,
    )
    command_lines = [entry.shell_line() for entry in entries]

    assert len(entries) == 2
    assert all("benchmark-memory" not in line for line in command_lines)
    assert all("benchmark-matmul" not in line for line in command_lines)
    assert (
        "uv run benchmark-decode-step --mode all --device cuda --dtype float16 "
        "--warmup 25 --iterations 100 --output results/decode-step.jsonl"
    ) in command_lines
    assert (
        "uv run benchmark-decode-step --dynamic-trace --mode all --device cuda "
        "--dtype float16 --warmup 25 --iterations 100 "
        "--output results/decode-step-dynamic.jsonl"
    ) in command_lines


def test_matrix_can_include_decode_bucket_sweep() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        only_decode_step=True,
        include_decode_bucket_sweep=True,
        decode_bucket_sweep_values=("1,2,4,8", "1,2,3,4,6,8"),
    )
    command_lines = [entry.shell_line() for entry in entries]
    bucket_lines = [
        line for line in command_lines if "decode-step-dynamic-buckets.jsonl" in line
    ]

    assert len(entries) == 4
    assert len(bucket_lines) == 2
    assert any("--batch-buckets 1,2,4,8" in line for line in bucket_lines)
    assert any("--batch-buckets 1,2,3,4,6,8" in line for line in bucket_lines)


def test_matrix_can_include_decode_tail_sweep() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        only_decode_step=True,
        include_decode_tail_sweep=True,
        decode_tail_iterations=300,
        decode_tail_seeds=(0, 4),
    )
    command_lines = [entry.shell_line() for entry in entries]
    tail_lines = [
        line for line in command_lines if "decode-step-dynamic-tail.jsonl" in line
    ]

    assert len(entries) == 8
    assert len(tail_lines) == 6
    assert all("--mode dynamic-piecewise-graph-same-stream" in line for line in tail_lines)
    assert any("--batch-buckets 1,2,3,4,6,8" in line for line in tail_lines)
    assert any("--batch-buckets 1,2,3,4,5,6,8" in line for line in tail_lines)
    assert any("--batch-buckets 1,2,3,4,5,6,7,8" in line for line in tail_lines)
    assert all("--iterations 300" in line for line in tail_lines)
    assert sum("--seed 0" in line for line in tail_lines) == 3
    assert sum("--seed 4" in line for line in tail_lines) == 3


def test_matrix_can_select_decode_dynamic_copy_mode() -> None:
    entries = benchmark_matrix.build_matrix(
        output_dir=Path("results"),
        only_decode_step=True,
        include_decode_tail_sweep=True,
        decode_tail_bucket_values=("1,2,4,8",),
        decode_tail_seeds=(0,),
        decode_attention_backend="sdpa-head-major",
        decode_dynamic_copy_mode="resident",
        decode_piecewise_post_mode="eager",
        decode_orchestration_timing="off",
    )
    command_lines = [entry.shell_line() for entry in entries]
    dynamic_lines = [
        line for line in command_lines if "benchmark-decode-step --dynamic-trace" in line
    ]
    static_lines = [
        line for line in command_lines if "benchmark-decode-step --mode all" in line
    ]

    assert dynamic_lines
    assert all("--attention-backend sdpa-head-major" in line for line in dynamic_lines)
    assert all("--dynamic-copy-mode resident" in line for line in dynamic_lines)
    assert all("--piecewise-post-mode eager" in line for line in dynamic_lines)
    assert all("--orchestration-timing off" in line for line in dynamic_lines)
    assert all("--piecewise-post-mode eager" in line for line in static_lines)


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


def test_dry_run_can_select_h200_matmul_autotune_suite(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_matrix.main(
        [
            "--dry-run",
            "--suite",
            "h200-matmul-autotune",
            "--output-dir",
            "results",
            "--matmul-autotune-shapes",
            "512x11008x4096",
            "--matmul-autotune-dtypes",
            "float16",
            "--matmul-autotune-schedules",
            "standard,persistent",
            "--matmul-autotune-configs",
            "128x128x64x4x4x4",
            "--matmul-autotune-repeats",
            "2",
            "--matmul-autotune-seed",
            "11",
        ]
    )

    output = capsys.readouterr().out
    assert "results/matmul-autotune.jsonl" in output
    assert "--m 512 --n 11008 --k 4096 --dtype float16" in output
    assert "--group-m 4" in output
    assert "--schedule persistent" in output
    assert "benchmark-memory" not in output


def test_keep_going_records_failed_matrix_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def fake_run(command: tuple[str, ...], *, check: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.CalledProcessError(returncode=42, cmd=command)

    monkeypatch.setattr(benchmark_matrix.subprocess, "run", fake_run)

    benchmark_matrix.main(["--output-dir", str(tmp_path), "--keep-going"])

    failure_manifest = json.loads((tmp_path / "benchmark-failures.json").read_text())
    assert calls == 8
    assert "continuing" in capsys.readouterr().out
    assert failure_manifest["kind"] == "benchmark-matrix-failures"
    assert failure_manifest["failures"][0]["returncode"] == 42
    assert failure_manifest["failures"][0]["primitive"] == "memory"


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

    with pytest.raises(ValueError, match="matmul launch settings"):
        benchmark_matrix.build_matrix(matmul_group_m=0)

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

    with pytest.raises(ValueError, match="tensor_core_matmul_shapes"):
        benchmark_matrix.build_matrix(tensor_core_matmul_shapes=((1024, 1024),))

    with pytest.raises(ValueError, match="tensor_core_matmul_shapes"):
        benchmark_matrix.build_matrix(tensor_core_matmul_shapes=((1024, 1024, 0),))

    with pytest.raises(ValueError, match="tensor_core_dtypes"):
        benchmark_matrix.build_matrix(tensor_core_dtypes=("float8",))

    with pytest.raises(ValueError, match="matmul_tuning_configs"):
        benchmark_matrix.build_matrix(matmul_tuning_configs=((128, 128, 64),))

    with pytest.raises(ValueError, match="matmul_llm_impact_configs"):
        benchmark_matrix.build_matrix(matmul_llm_impact_configs=((128, 128, 64),))

    with pytest.raises(ValueError, match="matmul_autotune_shapes"):
        benchmark_matrix.build_matrix(matmul_autotune_shapes=((512, 4096),))

    with pytest.raises(ValueError, match="matmul_autotune_dtypes"):
        benchmark_matrix.build_matrix(matmul_autotune_dtypes=("float8",))

    with pytest.raises(ValueError, match="matmul_autotune_schedules"):
        benchmark_matrix.build_matrix(matmul_autotune_schedules=("round-robin",))

    with pytest.raises(ValueError, match="matmul_autotune_configs"):
        benchmark_matrix.build_matrix(matmul_autotune_configs=((128, 128, 64),))

    with pytest.raises(ValueError, match="matmul_autotune_repeats"):
        benchmark_matrix.build_matrix(matmul_autotune_repeats=0)

    with pytest.raises(ValueError, match="matmul_autotune_seed"):
        benchmark_matrix.build_matrix(matmul_autotune_seed=-1)

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

    with pytest.raises(ValueError, match="decode bucket"):
        benchmark_matrix.build_matrix(decode_bucket_sweep_values=("1,4,2",))

    with pytest.raises(ValueError, match="decode bucket"):
        benchmark_matrix.build_matrix(decode_tail_bucket_values=("2,1",))

    with pytest.raises(ValueError, match="decode_tail_iterations"):
        benchmark_matrix.build_matrix(decode_tail_iterations=0)

    with pytest.raises(ValueError, match="decode_tail_seeds"):
        benchmark_matrix.build_matrix(decode_tail_seeds=())

    with pytest.raises(ValueError, match="decode_tail_seeds"):
        benchmark_matrix.build_matrix(decode_tail_seeds=(-1,))

    with pytest.raises(ValueError, match="decode_attention_backend"):
        benchmark_matrix.build_matrix(decode_attention_backend="flash")

    with pytest.raises(ValueError, match="decode_dynamic_copy_mode"):
        benchmark_matrix.build_matrix(decode_dynamic_copy_mode="kv-only")

    with pytest.raises(ValueError, match="decode_piecewise_post_mode"):
        benchmark_matrix.build_matrix(decode_piecewise_post_mode="skip")

    with pytest.raises(ValueError, match="decode_orchestration_timing"):
        benchmark_matrix.build_matrix(decode_orchestration_timing="verbose")

    with pytest.raises(ValueError, match="suite"):
        benchmark_matrix.build_matrix(suite="broad")


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
            "--include-tensor-core-suite",
            "--tensor-core-matmul-shapes",
            "2048x2048x2048",
            "--tensor-core-dtypes",
            "bfloat16",
            "--include-matmul-tuning-suite",
            "--matmul-tuning-shapes",
            "512x4096x11008",
            "--matmul-tuning-configs",
            "64x128x32x8x4",
            "--include-matmul-llm-impact-suite",
            "--matmul-llm-impact-shapes",
            "512x11008x4096",
            "--matmul-llm-impact-configs",
            "128x128x64x4x4x4",
            "--include-attention-baseline",
            "--attention-seq-len",
            "4096",
            "--attention-num-heads",
            "8",
            "--attention-head-dim",
            "64",
            "--include-decode-step",
            "--include-decode-tail-sweep",
            "--decode-tail-seeds",
            "2,3",
            "--decode-tail-iterations",
            "300",
            "--decode-tail-buckets",
            "1,2,4,8;1,2,3,4,6,8",
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
    assert "results/matmul-tensor-core.jsonl" in output
    assert "results/matmul-tuning.jsonl" in output
    assert "results/matmul-llm-impact.jsonl" in output
    assert "--m 2048 --n 2048 --k 2048 --dtype bfloat16" in output
    assert "--m 512 --n 4096 --k 11008 --dtype bfloat16" in output
    assert "--m 512 --n 11008 --k 4096 --dtype bfloat16" in output
    assert "--block-m 16 --block-n 32 --block-k 32" in output
    assert "--block-m 32 --block-n 16 --block-k 32" in output
    assert "--block-m 64 --block-n 128 --block-k 32 --num-warps 8 --num-stages 4" in output
    assert (
        "--block-m 128 --block-n 128 --block-k 64 --num-warps 4 --num-stages 4 "
        "--input-precision tf32 --group-m 4"
    ) in output
    assert "--num-warps 8 --num-stages 4" in output
    assert "results/rmsnorm-shape-sweep.jsonl" in output
    assert "--rows 512 --cols 1024 --dtype float16" in output
    assert "--rows 4096 --cols 8192 --dtype float16" in output
    assert "results/attention.jsonl" in output
    assert "uv run benchmark-attention --backend torch" in output
    assert "--seq-len 4096 --num-heads 8 --head-dim 64" in output
    assert "results/decode-step.jsonl" in output
    assert "uv run benchmark-decode-step --mode all" in output
    assert "results/decode-step-dynamic.jsonl" in output
    assert "uv run benchmark-decode-step --dynamic-trace --mode all" in output
    assert "results/decode-step-dynamic-tail.jsonl" in output
    assert "--mode dynamic-piecewise-graph-same-stream" in output
    assert "--batch-buckets 1,2,4,8" in output
    assert "--batch-buckets 1,2,3,4,6,8" in output
    assert "--seed 2" in output
    assert "--seed 3" in output


def test_matrix_parses_decode_only_bucket_sweep_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_matrix.main(
        [
            "--dry-run",
            "--output-dir",
            "results",
            "--include-vector-add-sweep",
            "--only-decode-step",
            "--include-decode-bucket-sweep",
            "--decode-bucket-sweep-values",
            "1,2,4,8;1,2,3,4,6,8",
        ]
    )

    output = capsys.readouterr().out
    assert "uv run benchmark-memory" not in output
    assert "results/vector-add-block-size.jsonl" not in output
    assert "results/decode-step.jsonl" in output
    assert "results/decode-step-dynamic-buckets.jsonl" in output
    assert "--batch-buckets 1,2,4,8" in output
    assert "--batch-buckets 1,2,3,4,6,8" in output
