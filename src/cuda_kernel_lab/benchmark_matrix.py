"""Run or print the standard live-GPU benchmark matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("experiments/results/runpod/manual-run")
DEFAULT_SUITE = "standard"
SUITES = ("standard", "h200-roofline", "tensor-core")
DEFAULT_WARMUP = 25
DEFAULT_ITERATIONS = 100
DEFAULT_MEMORY_BLOCK_SIZE = 1024
DEFAULT_SWIGLU_BLOCK_SIZE = 1024
DEFAULT_MATMUL_BLOCK_M = 16
DEFAULT_MATMUL_BLOCK_N = 16
DEFAULT_MATMUL_BLOCK_K = 32
DEFAULT_MATMUL_NUM_WARPS = 4
DEFAULT_MATMUL_NUM_STAGES = 3
DEFAULT_MATMUL_INPUT_PRECISION = "ieee"
DEFAULT_MATMUL_SWEEP_TILE_SHAPES = (
    (16, 32, 32),
    (32, 16, 32),
    (32, 32, 32),
    (32, 32, 64),
    (64, 64, 32),
    (64, 128, 32),
    (128, 64, 32),
    (128, 128, 32),
)
DEFAULT_MATMUL_SWEEP_LAUNCH_CONFIGS = (
    (4, 3),
    (4, 4),
    (8, 3),
    (8, 4),
)
DEFAULT_TENSOR_CORE_MATMUL_SHAPES = (
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (512, 4096, 11008),
    (512, 11008, 4096),
)
DEFAULT_TENSOR_CORE_DTYPES = ("float16", "bfloat16")
DEFAULT_TENSOR_CORE_BLOCK_M = 128
DEFAULT_TENSOR_CORE_BLOCK_N = 128
DEFAULT_TENSOR_CORE_BLOCK_K = 64
DEFAULT_TENSOR_CORE_NUM_WARPS = 4
DEFAULT_TENSOR_CORE_NUM_STAGES = 4
DEFAULT_RMSNORM_SHAPE_SWEEP_SHAPES = (
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 4096),
    (4096, 8192),
)
DEFAULT_RMSNORM_SHAPE_SWEEP_DTYPE = "float16"
DEFAULT_ATTENTION_SEQ_LEN = 2048
DEFAULT_ATTENTION_NUM_HEADS = 16
DEFAULT_ATTENTION_HEAD_DIM = 128
DEFAULT_ATTENTION_DTYPE = "float16"
DEFAULT_DECODE_BUCKET_SWEEP_VALUES = (
    "1,2,3,4,6,8",
    "1,2,3,4,5,6,8",
    "1,2,3,4,5,6,7,8",
    "1,2,4,6,8",
    "1,2,4,8",
)
DEFAULT_DECODE_TAIL_BUCKET_VALUES = (
    "1,2,3,4,6,8",
    "1,2,3,4,5,6,8",
    "1,2,3,4,5,6,7,8",
)
DEFAULT_DECODE_TAIL_ITERATIONS = 500
DEFAULT_DECODE_TAIL_SEEDS = (0, 1, 2)
DEFAULT_DECODE_ATTENTION_BACKEND = "einsum"
DECODE_ATTENTION_BACKENDS = ("einsum", "sdpa", "sdpa-head-major")
DEFAULT_DECODE_DYNAMIC_COPY_MODE = "full"
DECODE_DYNAMIC_COPY_MODES = ("full", "x-only", "resident")
DEFAULT_DECODE_PIECEWISE_POST_MODE = "graph"
DECODE_PIECEWISE_POST_MODES = ("graph", "eager")
DEFAULT_DECODE_ORCHESTRATION_TIMING = "on"
DECODE_ORCHESTRATION_TIMINGS = ("on", "off")
DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES = (512, 1024, 2048)
DEFAULT_REDUCTION_STRATEGY = "iterative"
DEFAULT_REDUCTION_SWEEP_STRATEGIES = ("iterative", "two_pass")
REDUCTION_STRATEGIES = ("iterative", "two_pass")
MATMUL_INPUT_PRECISIONS = ("tf32", "tf32x3", "ieee")
DEFAULT_DEVICE = "cuda"
DTYPES = ("float32", "float16")
SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class MatrixCommand:
    """One benchmark command in the live-GPU evidence matrix."""

    primitive: str
    dtype: str
    command: tuple[str, ...]

    def shell_line(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


def build_matrix(
    *,
    suite: str = DEFAULT_SUITE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = DEFAULT_DEVICE,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    memory_block_size: int = DEFAULT_MEMORY_BLOCK_SIZE,
    swiglu_block_size: int = DEFAULT_SWIGLU_BLOCK_SIZE,
    include_matmul: bool = False,
    matmul_block_m: int = DEFAULT_MATMUL_BLOCK_M,
    matmul_block_n: int = DEFAULT_MATMUL_BLOCK_N,
    matmul_block_k: int = DEFAULT_MATMUL_BLOCK_K,
    matmul_num_warps: int = DEFAULT_MATMUL_NUM_WARPS,
    matmul_num_stages: int = DEFAULT_MATMUL_NUM_STAGES,
    matmul_input_precision: str = DEFAULT_MATMUL_INPUT_PRECISION,
    include_matmul_sweep: bool = False,
    matmul_sweep_tile_shapes: tuple[tuple[int, ...], ...] = DEFAULT_MATMUL_SWEEP_TILE_SHAPES,
    matmul_sweep_launch_configs: tuple[tuple[int, ...], ...] = DEFAULT_MATMUL_SWEEP_LAUNCH_CONFIGS,
    include_tensor_core_suite: bool = False,
    tensor_core_matmul_shapes: tuple[tuple[int, ...], ...] = DEFAULT_TENSOR_CORE_MATMUL_SHAPES,
    tensor_core_dtypes: tuple[str, ...] = DEFAULT_TENSOR_CORE_DTYPES,
    include_rmsnorm_shape_sweep: bool = False,
    rmsnorm_shape_sweep_shapes: tuple[tuple[int, ...], ...] = DEFAULT_RMSNORM_SHAPE_SWEEP_SHAPES,
    rmsnorm_shape_sweep_dtype: str = DEFAULT_RMSNORM_SHAPE_SWEEP_DTYPE,
    include_attention_baseline: bool = False,
    attention_seq_len: int = DEFAULT_ATTENTION_SEQ_LEN,
    attention_num_heads: int = DEFAULT_ATTENTION_NUM_HEADS,
    attention_head_dim: int = DEFAULT_ATTENTION_HEAD_DIM,
    attention_dtype: str = DEFAULT_ATTENTION_DTYPE,
    include_decode_step: bool = False,
    only_decode_step: bool = False,
    include_decode_bucket_sweep: bool = False,
    decode_bucket_sweep_values: tuple[str, ...] = DEFAULT_DECODE_BUCKET_SWEEP_VALUES,
    include_decode_tail_sweep: bool = False,
    decode_tail_buckets: str | None = None,
    decode_tail_bucket_values: tuple[str, ...] | None = None,
    decode_tail_iterations: int = DEFAULT_DECODE_TAIL_ITERATIONS,
    decode_tail_seeds: tuple[int, ...] = DEFAULT_DECODE_TAIL_SEEDS,
    decode_attention_backend: str = DEFAULT_DECODE_ATTENTION_BACKEND,
    decode_dynamic_copy_mode: str = DEFAULT_DECODE_DYNAMIC_COPY_MODE,
    decode_piecewise_post_mode: str = DEFAULT_DECODE_PIECEWISE_POST_MODE,
    decode_orchestration_timing: str = DEFAULT_DECODE_ORCHESTRATION_TIMING,
    include_vector_add_sweep: bool = False,
    vector_add_sweep_block_sizes: tuple[int, ...] = DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES,
    reduction_strategy: str = DEFAULT_REDUCTION_STRATEGY,
    include_reduction_sweep: bool = False,
    reduction_sweep_strategies: tuple[str, ...] = DEFAULT_REDUCTION_SWEEP_STRATEGIES,
) -> tuple[MatrixCommand, ...]:
    """Build the default live-GPU benchmark command matrix."""

    suite = _normalize_suite(suite)
    if suite in {"h200-roofline", "tensor-core"}:
        include_tensor_core_suite = True
        include_matmul_sweep = True
        include_rmsnorm_shape_sweep = True
        include_attention_baseline = True

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if memory_block_size <= 0:
        raise ValueError("memory_block_size must be positive")
    if swiglu_block_size <= 0:
        raise ValueError("swiglu_block_size must be positive")
    if matmul_block_m <= 0 or matmul_block_n <= 0 or matmul_block_k <= 0:
        raise ValueError("matmul block sizes must be positive")
    if matmul_num_warps <= 0 or matmul_num_stages <= 0:
        raise ValueError("matmul launch settings must be positive")
    if matmul_input_precision not in MATMUL_INPUT_PRECISIONS:
        raise ValueError("matmul_input_precision must be one of tf32, tf32x3, ieee")
    if any(len(tile_shape) != 3 for tile_shape in matmul_sweep_tile_shapes):
        raise ValueError("matmul_sweep_tile_shapes must be MxNxK triples")
    if any(any(dim <= 0 for dim in tile_shape) for tile_shape in matmul_sweep_tile_shapes):
        raise ValueError("matmul_sweep_tile_shapes must be positive")
    if any(len(launch_config) != 2 for launch_config in matmul_sweep_launch_configs):
        raise ValueError("matmul_sweep_launch_configs must be WARPSxSTAGES pairs")
    if any(any(dim <= 0 for dim in launch_config) for launch_config in matmul_sweep_launch_configs):
        raise ValueError("matmul_sweep_launch_configs must be positive")
    if any(len(shape) != 3 for shape in tensor_core_matmul_shapes):
        raise ValueError("tensor_core_matmul_shapes must be MxNxK triples")
    if any(any(dim <= 0 for dim in shape) for shape in tensor_core_matmul_shapes):
        raise ValueError("tensor_core_matmul_shapes must be positive")
    if not tensor_core_dtypes:
        raise ValueError("tensor_core_dtypes must not be empty")
    if any(dtype not in SUPPORTED_DTYPES for dtype in tensor_core_dtypes):
        raise ValueError("tensor_core_dtypes must be one of float32, float16, bfloat16")
    if any(len(shape) != 2 for shape in rmsnorm_shape_sweep_shapes):
        raise ValueError("rmsnorm_shape_sweep_shapes must be ROWSxCOLS pairs")
    if any(any(dim <= 0 for dim in shape) for shape in rmsnorm_shape_sweep_shapes):
        raise ValueError("rmsnorm_shape_sweep_shapes must be positive")
    if rmsnorm_shape_sweep_dtype not in DTYPES:
        raise ValueError("rmsnorm_shape_sweep_dtype must be one of float32, float16")
    if attention_seq_len <= 0 or attention_num_heads <= 0 or attention_head_dim <= 0:
        raise ValueError("attention shape values must be positive")
    if attention_dtype not in DTYPES:
        raise ValueError("attention_dtype must be one of float32, float16")
    if decode_tail_bucket_values is None:
        decode_tail_bucket_values = (
            _parse_decode_bucket_values(decode_tail_buckets)
            if decode_tail_buckets is not None
            else DEFAULT_DECODE_TAIL_BUCKET_VALUES
        )

    for buckets in decode_bucket_sweep_values:
        _validate_batch_bucket_value(buckets)
    for buckets in decode_tail_bucket_values:
        _validate_batch_bucket_value(buckets)
    if decode_tail_iterations <= 0:
        raise ValueError("decode_tail_iterations must be positive")
    if not decode_tail_seeds:
        raise ValueError("decode_tail_seeds must include at least one seed")
    if any(seed < 0 for seed in decode_tail_seeds):
        raise ValueError("decode_tail_seeds must be non-negative")
    if decode_attention_backend not in DECODE_ATTENTION_BACKENDS:
        raise ValueError("decode_attention_backend must be one of einsum, sdpa, sdpa-head-major")
    if decode_dynamic_copy_mode not in DECODE_DYNAMIC_COPY_MODES:
        raise ValueError("decode_dynamic_copy_mode must be one of full, x-only, resident")
    if decode_piecewise_post_mode not in DECODE_PIECEWISE_POST_MODES:
        raise ValueError("decode_piecewise_post_mode must be one of graph, eager")
    if decode_orchestration_timing not in DECODE_ORCHESTRATION_TIMINGS:
        raise ValueError("decode_orchestration_timing must be one of on, off")
    if any(block_size <= 0 for block_size in vector_add_sweep_block_sizes):
        raise ValueError("vector_add_sweep_block_sizes must be positive")
    if reduction_strategy not in REDUCTION_STRATEGIES:
        raise ValueError("reduction_strategy must be one of iterative, two_pass")
    if any(strategy not in REDUCTION_STRATEGIES for strategy in reduction_sweep_strategies):
        raise ValueError("reduction_sweep_strategies must be one of iterative, two_pass")

    commands: list[MatrixCommand] = []
    matmul_baseline_config = (
        matmul_block_m,
        matmul_block_n,
        matmul_block_k,
        matmul_num_warps,
        matmul_num_stages,
        matmul_input_precision,
    )
    if not only_decode_step:
        for dtype in DTYPES:
            commands.append(
                MatrixCommand(
                    primitive="memory",
                    dtype=dtype,
                    command=(
                        "uv",
                        "run",
                        "benchmark-memory",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--op",
                        "all",
                        "--numel",
                        "16777216",
                        "--dtype",
                        dtype,
                        "--block-size",
                        str(memory_block_size),
                        "--reduction-strategy",
                        reduction_strategy,
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "memory.jsonl"),
                    ),
                )
            )
            commands.append(
                MatrixCommand(
                    primitive="softmax",
                    dtype=dtype,
                    command=(
                        "uv",
                        "run",
                        "benchmark-softmax",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--rows",
                        "4096",
                        "--cols",
                        "1024",
                        "--dtype",
                        dtype,
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "softmax.jsonl"),
                    ),
                )
            )
            commands.append(
                MatrixCommand(
                    primitive="norms",
                    dtype=dtype,
                    command=(
                        "uv",
                        "run",
                        "benchmark-norms",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--op",
                        "all",
                        "--rows",
                        "4096",
                        "--cols",
                        "4096",
                        "--dtype",
                        dtype,
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "norms.jsonl"),
                    ),
                )
            )
            commands.append(
                MatrixCommand(
                    primitive="swiglu",
                    dtype=dtype,
                    command=(
                        "uv",
                        "run",
                        "benchmark-swiglu",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--rows",
                        "4096",
                        "--cols",
                        "4096",
                        "--dtype",
                        dtype,
                        "--block-size",
                        str(swiglu_block_size),
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "swiglu.jsonl"),
                    ),
                )
            )
            if include_matmul or (include_matmul_sweep and dtype == "float16"):
                commands.append(
                    MatrixCommand(
                        primitive="matmul",
                        dtype=dtype,
                        command=_matmul_command(
                            device=device,
                            dtype=dtype,
                            block_m=matmul_block_m,
                            block_n=matmul_block_n,
                            block_k=matmul_block_k,
                            num_warps=matmul_num_warps,
                            num_stages=matmul_num_stages,
                            input_precision=matmul_input_precision,
                            warmup=warmup,
                            iterations=iterations,
                            output=output_dir / "matmul.jsonl",
                        ),
                    )
                )

    if include_vector_add_sweep and not only_decode_step:
        for block_size in _extra_vector_add_block_sizes(
            vector_add_sweep_block_sizes,
            baseline_block_size=memory_block_size,
        ):
            commands.append(
                MatrixCommand(
                    primitive="memory",
                    dtype="float32",
                    command=(
                        "uv",
                        "run",
                        "benchmark-memory",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--op",
                        "vector_add",
                        "--numel",
                        "16777216",
                        "--dtype",
                        "float32",
                        "--block-size",
                        str(block_size),
                        "--reduction-strategy",
                        reduction_strategy,
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "vector-add-block-size.jsonl"),
                    ),
                )
            )

    if include_reduction_sweep and not only_decode_step:
        for strategy in _extra_values(
            reduction_sweep_strategies,
            baseline_value=reduction_strategy,
        ):
            commands.append(
                MatrixCommand(
                    primitive="memory",
                    dtype="float32",
                    command=(
                        "uv",
                        "run",
                        "benchmark-memory",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--op",
                        "reduction_sum",
                        "--numel",
                        "16777216",
                        "--dtype",
                        "float32",
                        "--block-size",
                        str(memory_block_size),
                        "--reduction-strategy",
                        strategy,
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "reduction-strategy.jsonl"),
                    ),
                )
            )

    if include_matmul_sweep and not only_decode_step:
        for (
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            input_precision,
        ) in _extra_matmul_configs(
            matmul_sweep_tile_shapes,
            matmul_sweep_launch_configs,
            input_precision=matmul_input_precision,
            baseline_config=matmul_baseline_config,
        ):
            commands.append(
                MatrixCommand(
                    primitive="matmul",
                    dtype="float16",
                    command=_matmul_command(
                        device=device,
                        dtype="float16",
                        block_m=block_m,
                        block_n=block_n,
                        block_k=block_k,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        input_precision=input_precision,
                        warmup=warmup,
                        iterations=iterations,
                        output=output_dir / "matmul-tile-shape.jsonl",
                    ),
                )
            )
    if include_tensor_core_suite and not only_decode_step:
        for dtype in tensor_core_dtypes:
            for m, n, k in tensor_core_matmul_shapes:
                commands.append(
                    MatrixCommand(
                        primitive="matmul",
                        dtype=dtype,
                        command=_matmul_command(
                            device=device,
                            dtype=dtype,
                            block_m=DEFAULT_TENSOR_CORE_BLOCK_M,
                            block_n=DEFAULT_TENSOR_CORE_BLOCK_N,
                            block_k=DEFAULT_TENSOR_CORE_BLOCK_K,
                            num_warps=DEFAULT_TENSOR_CORE_NUM_WARPS,
                            num_stages=DEFAULT_TENSOR_CORE_NUM_STAGES,
                            input_precision="tf32",
                            warmup=warmup,
                            iterations=iterations,
                            output=output_dir / "matmul-tensor-core.jsonl",
                            m=m,
                            n=n,
                            k=k,
                        ),
                    )
                )
    if include_rmsnorm_shape_sweep and not only_decode_step:
        for rows, cols in rmsnorm_shape_sweep_shapes:
            commands.append(
                MatrixCommand(
                    primitive="norms",
                    dtype=rmsnorm_shape_sweep_dtype,
                    command=(
                        "uv",
                        "run",
                        "benchmark-norms",
                        "--backend",
                        "all",
                        "--device",
                        device,
                        "--op",
                        "rmsnorm",
                        "--rows",
                        str(rows),
                        "--cols",
                        str(cols),
                        "--dtype",
                        rmsnorm_shape_sweep_dtype,
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "rmsnorm-shape-sweep.jsonl"),
                    ),
                )
            )
    if include_attention_baseline and not only_decode_step:
        commands.append(
            MatrixCommand(
                primitive="attention",
                dtype=attention_dtype,
                command=(
                    "uv",
                    "run",
                    "benchmark-attention",
                    "--backend",
                    "torch",
                    "--device",
                    device,
                    "--seq-len",
                    str(attention_seq_len),
                    "--num-heads",
                    str(attention_num_heads),
                    "--head-dim",
                    str(attention_head_dim),
                    "--dtype",
                    attention_dtype,
                    "--warmup",
                    str(warmup),
                    "--iterations",
                    str(iterations),
                    "--output",
                    str(output_dir / "attention.jsonl"),
                ),
            )
        )
    if include_decode_step or only_decode_step:
        commands.extend(
            _decode_step_commands(
                device=device,
                output_dir=output_dir,
                warmup=warmup,
                iterations=iterations,
                attention_backend=decode_attention_backend,
                dynamic_copy_mode=decode_dynamic_copy_mode,
                piecewise_post_mode=decode_piecewise_post_mode,
                orchestration_timing=decode_orchestration_timing,
            )
        )
    if include_decode_bucket_sweep:
        for batch_buckets in decode_bucket_sweep_values:
            commands.append(
                _decode_step_dynamic_command(
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                    output=output_dir / "decode-step-dynamic-buckets.jsonl",
                    batch_buckets=batch_buckets,
                    attention_backend=decode_attention_backend,
                    dynamic_copy_mode=decode_dynamic_copy_mode,
                    piecewise_post_mode=decode_piecewise_post_mode,
                    orchestration_timing=decode_orchestration_timing,
                )
            )
    if include_decode_tail_sweep:
        for batch_buckets in decode_tail_bucket_values:
            for seed in decode_tail_seeds:
                commands.append(
                    _decode_step_dynamic_command(
                        device=device,
                        warmup=warmup,
                        iterations=decode_tail_iterations,
                        output=output_dir / "decode-step-dynamic-tail.jsonl",
                        mode="dynamic-piecewise-graph-same-stream",
                        batch_buckets=batch_buckets,
                        seed=seed,
                        attention_backend=decode_attention_backend,
                        dynamic_copy_mode=decode_dynamic_copy_mode,
                        piecewise_post_mode=decode_piecewise_post_mode,
                        orchestration_timing=decode_orchestration_timing,
                    )
                )
    return tuple(commands)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default=DEFAULT_SUITE,
        help=(
            "Benchmark suite preset. h200-roofline and tensor-core add larger "
            "matmul, BF16, RMSNorm shape, and attention baseline coverage."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=DEFAULT_DEVICE,
        help="Device argument passed to each benchmark command.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSONL benchmark outputs.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--memory-block-size",
        type=int,
        default=DEFAULT_MEMORY_BLOCK_SIZE,
        help="Triton block size passed to benchmark-memory commands.",
    )
    parser.add_argument(
        "--swiglu-block-size",
        type=int,
        default=DEFAULT_SWIGLU_BLOCK_SIZE,
        help="Triton block size passed to benchmark-swiglu commands.",
    )
    parser.add_argument(
        "--include-matmul",
        action="store_true",
        help="Add tiled matmul benchmark commands.",
    )
    parser.add_argument("--matmul-block-m", type=int, default=DEFAULT_MATMUL_BLOCK_M)
    parser.add_argument("--matmul-block-n", type=int, default=DEFAULT_MATMUL_BLOCK_N)
    parser.add_argument("--matmul-block-k", type=int, default=DEFAULT_MATMUL_BLOCK_K)
    parser.add_argument("--matmul-num-warps", type=int, default=DEFAULT_MATMUL_NUM_WARPS)
    parser.add_argument("--matmul-num-stages", type=int, default=DEFAULT_MATMUL_NUM_STAGES)
    parser.add_argument(
        "--matmul-input-precision",
        choices=MATMUL_INPUT_PRECISIONS,
        default=DEFAULT_MATMUL_INPUT_PRECISION,
        help="Triton tl.dot input precision passed to benchmark-matmul.",
    )
    parser.add_argument(
        "--include-matmul-sweep",
        action="store_true",
        help="Add the float16 matmul tile-shape and launch-configuration strategy sweep.",
    )
    parser.add_argument(
        "--matmul-sweep-tile-shapes",
        default=_join_tile_shapes(DEFAULT_MATMUL_SWEEP_TILE_SHAPES),
        help="Comma-separated MxNxK tile shapes for --include-matmul-sweep.",
    )
    parser.add_argument(
        "--matmul-sweep-launch-configs",
        default=_join_launch_configs(DEFAULT_MATMUL_SWEEP_LAUNCH_CONFIGS),
        help="Comma-separated WARPSxSTAGES launch configs for --include-matmul-sweep.",
    )
    parser.add_argument(
        "--include-tensor-core-suite",
        action="store_true",
        help="Add larger FP16/BF16 matmul rows for Tensor Core and roofline evidence.",
    )
    parser.add_argument(
        "--tensor-core-matmul-shapes",
        default=_join_tile_shapes(DEFAULT_TENSOR_CORE_MATMUL_SHAPES),
        help="Comma-separated MxNxK matmul shapes for the Tensor Core suite.",
    )
    parser.add_argument(
        "--tensor-core-dtypes",
        default=_join_values(DEFAULT_TENSOR_CORE_DTYPES),
        help="Comma-separated dtypes for the Tensor Core suite.",
    )
    parser.add_argument(
        "--include-rmsnorm-shape-sweep",
        action="store_true",
        help="Add a focused RMSNorm shape sweep for hidden-size and batch-size evidence.",
    )
    parser.add_argument(
        "--rmsnorm-shape-sweep-shapes",
        default=_join_shapes(DEFAULT_RMSNORM_SHAPE_SWEEP_SHAPES),
        help="Comma-separated ROWSxCOLS shapes for --include-rmsnorm-shape-sweep.",
    )
    parser.add_argument(
        "--rmsnorm-shape-sweep-dtype",
        choices=DTYPES,
        default=DEFAULT_RMSNORM_SHAPE_SWEEP_DTYPE,
        help="Dtype for --include-rmsnorm-shape-sweep.",
    )
    parser.add_argument(
        "--include-attention-baseline",
        action="store_true",
        help="Add the PyTorch contiguous KV-cache decode-attention baseline.",
    )
    parser.add_argument("--attention-seq-len", type=int, default=DEFAULT_ATTENTION_SEQ_LEN)
    parser.add_argument("--attention-num-heads", type=int, default=DEFAULT_ATTENTION_NUM_HEADS)
    parser.add_argument("--attention-head-dim", type=int, default=DEFAULT_ATTENTION_HEAD_DIM)
    parser.add_argument(
        "--attention-dtype",
        choices=DTYPES,
        default=DEFAULT_ATTENTION_DTYPE,
        help="Dtype for --include-attention-baseline.",
    )
    parser.add_argument(
        "--include-decode-step",
        action="store_true",
        help="Add synthetic decode-step graph, piecewise graph, and dynamic trace benchmarks.",
    )
    parser.add_argument(
        "--only-decode-step",
        action="store_true",
        help="Run only synthetic decode-step static and dynamic trace benchmarks.",
    )
    parser.add_argument(
        "--include-decode-bucket-sweep",
        action="store_true",
        help="Add dynamic decode-step traces for alternate batch-bucket sets.",
    )
    parser.add_argument(
        "--decode-bucket-sweep-values",
        default=_join_decode_bucket_values(DEFAULT_DECODE_BUCKET_SWEEP_VALUES),
        help=(
            "Semicolon-separated batch-bucket lists for --include-decode-bucket-sweep, "
            "for example '1,2,4,8;1,2,3,4,6,8'."
        ),
    )
    parser.add_argument(
        "--include-decode-tail-sweep",
        action="store_true",
        help="Add longer multi-seed dynamic decode traces for tail-latency evidence.",
    )
    parser.add_argument(
        "--decode-tail-buckets",
        default=_join_decode_bucket_values(DEFAULT_DECODE_TAIL_BUCKET_VALUES),
        help=(
            "Semicolon-separated batch-bucket lists for --include-decode-tail-sweep, "
            "for example '1,2,3,4,6,8;1,2,3,4,5,6,7,8'."
        ),
    )
    parser.add_argument(
        "--decode-tail-iterations",
        type=int,
        default=DEFAULT_DECODE_TAIL_ITERATIONS,
        help="Iterations per seed for --include-decode-tail-sweep.",
    )
    parser.add_argument(
        "--decode-tail-seeds",
        default=_join_int_values(DEFAULT_DECODE_TAIL_SEEDS),
        help="Comma-separated seeds for --include-decode-tail-sweep.",
    )
    parser.add_argument(
        "--decode-attention-backend",
        choices=DECODE_ATTENTION_BACKENDS,
        default=DEFAULT_DECODE_ATTENTION_BACKEND,
        help="Attention backend passed to decode-step benchmark commands.",
    )
    parser.add_argument(
        "--decode-dynamic-copy-mode",
        choices=DECODE_DYNAMIC_COPY_MODES,
        default=DEFAULT_DECODE_DYNAMIC_COPY_MODE,
        help="Dynamic piecewise decode-step input staging mode.",
    )
    parser.add_argument(
        "--decode-piecewise-post-mode",
        choices=DECODE_PIECEWISE_POST_MODES,
        default=DEFAULT_DECODE_PIECEWISE_POST_MODE,
        help="Post-attention add mode for dynamic piecewise decode-step replay.",
    )
    parser.add_argument(
        "--decode-orchestration-timing",
        choices=DECODE_ORCHESTRATION_TIMINGS,
        default=DEFAULT_DECODE_ORCHESTRATION_TIMING,
        help="Per-region host orchestration timing mode for dynamic decode-step traces.",
    )
    parser.add_argument(
        "--reduction-strategy",
        choices=REDUCTION_STRATEGIES,
        default=DEFAULT_REDUCTION_STRATEGY,
        help="Reduction strategy passed to benchmark-memory commands.",
    )
    parser.add_argument(
        "--include-vector-add-sweep",
        action="store_true",
        help="Add the vector_add block-size strategy sweep.",
    )
    parser.add_argument(
        "--vector-add-sweep-block-sizes",
        default=_join_block_sizes(DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES),
        help="Comma-separated block sizes for --include-vector-add-sweep.",
    )
    parser.add_argument(
        "--include-reduction-sweep",
        action="store_true",
        help="Add the reduction_sum strategy sweep.",
    )
    parser.add_argument(
        "--reduction-sweep-strategies",
        default=_join_values(DEFAULT_REDUCTION_SWEEP_STRATEGIES),
        help="Comma-separated strategies for --include-reduction-sweep.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    commands = build_matrix(
        suite=args.suite,
        output_dir=args.output_dir,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        memory_block_size=args.memory_block_size,
        swiglu_block_size=args.swiglu_block_size,
        include_matmul=args.include_matmul,
        matmul_block_m=args.matmul_block_m,
        matmul_block_n=args.matmul_block_n,
        matmul_block_k=args.matmul_block_k,
        matmul_num_warps=args.matmul_num_warps,
        matmul_num_stages=args.matmul_num_stages,
        matmul_input_precision=args.matmul_input_precision,
        include_matmul_sweep=args.include_matmul_sweep,
        matmul_sweep_tile_shapes=_parse_tile_shapes(args.matmul_sweep_tile_shapes),
        matmul_sweep_launch_configs=_parse_launch_configs(args.matmul_sweep_launch_configs),
        include_tensor_core_suite=args.include_tensor_core_suite,
        tensor_core_matmul_shapes=_parse_tile_shapes(args.tensor_core_matmul_shapes),
        tensor_core_dtypes=_parse_values(args.tensor_core_dtypes),
        include_rmsnorm_shape_sweep=args.include_rmsnorm_shape_sweep,
        rmsnorm_shape_sweep_shapes=_parse_shapes(args.rmsnorm_shape_sweep_shapes),
        rmsnorm_shape_sweep_dtype=args.rmsnorm_shape_sweep_dtype,
        include_attention_baseline=args.include_attention_baseline,
        attention_seq_len=args.attention_seq_len,
        attention_num_heads=args.attention_num_heads,
        attention_head_dim=args.attention_head_dim,
        attention_dtype=args.attention_dtype,
        include_decode_step=args.include_decode_step,
        only_decode_step=args.only_decode_step,
        include_decode_bucket_sweep=args.include_decode_bucket_sweep,
        decode_bucket_sweep_values=_parse_decode_bucket_values(
            args.decode_bucket_sweep_values
        ),
        include_decode_tail_sweep=args.include_decode_tail_sweep,
        decode_tail_bucket_values=_parse_decode_bucket_values(args.decode_tail_buckets),
        decode_tail_iterations=args.decode_tail_iterations,
        decode_tail_seeds=_parse_int_values(args.decode_tail_seeds, label="decode_tail_seeds"),
        decode_attention_backend=args.decode_attention_backend,
        decode_dynamic_copy_mode=args.decode_dynamic_copy_mode,
        decode_piecewise_post_mode=args.decode_piecewise_post_mode,
        decode_orchestration_timing=args.decode_orchestration_timing,
        include_vector_add_sweep=args.include_vector_add_sweep,
        vector_add_sweep_block_sizes=_parse_block_sizes(args.vector_add_sweep_block_sizes),
        reduction_strategy=args.reduction_strategy,
        include_reduction_sweep=args.include_reduction_sweep,
        reduction_sweep_strategies=_parse_values(args.reduction_sweep_strategies),
    )

    if args.dry_run:
        for entry in commands:
            print(entry.shell_line())
        return

    for entry in commands:
        print(entry.shell_line(), flush=True)
        subprocess.run(entry.command, check=True)


def _matmul_command(
    *,
    device: str,
    dtype: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    input_precision: str,
    warmup: int,
    iterations: int,
    output: Path,
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "benchmark-matmul",
        "--backend",
        "all",
        "--device",
        device,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--dtype",
        dtype,
        "--block-m",
        str(block_m),
        "--block-n",
        str(block_n),
        "--block-k",
        str(block_k),
        "--num-warps",
        str(num_warps),
        "--num-stages",
        str(num_stages),
        "--input-precision",
        input_precision,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--output",
        str(output),
    )


def _decode_step_commands(
    *,
    device: str,
    output_dir: Path,
    warmup: int,
    iterations: int,
    attention_backend: str,
    dynamic_copy_mode: str,
    piecewise_post_mode: str,
    orchestration_timing: str,
) -> tuple[MatrixCommand, ...]:
    return (
        _decode_step_static_command(
            device=device,
            warmup=warmup,
            iterations=iterations,
            output=output_dir / "decode-step.jsonl",
            attention_backend=attention_backend,
            piecewise_post_mode=piecewise_post_mode,
        ),
        _decode_step_dynamic_command(
            device=device,
            warmup=warmup,
            iterations=iterations,
            output=output_dir / "decode-step-dynamic.jsonl",
            attention_backend=attention_backend,
            dynamic_copy_mode=dynamic_copy_mode,
            piecewise_post_mode=piecewise_post_mode,
            orchestration_timing=orchestration_timing,
        ),
    )


def _decode_step_static_command(
    *,
    device: str,
    warmup: int,
    iterations: int,
    output: Path,
    attention_backend: str,
    piecewise_post_mode: str,
) -> MatrixCommand:
    command = [
        "uv",
        "run",
        "benchmark-decode-step",
        "--mode",
        "all",
        "--device",
        device,
        "--dtype",
        "float16",
    ]
    if attention_backend != DEFAULT_DECODE_ATTENTION_BACKEND:
        command.extend(("--attention-backend", attention_backend))
    if piecewise_post_mode != DEFAULT_DECODE_PIECEWISE_POST_MODE:
        command.extend(("--piecewise-post-mode", piecewise_post_mode))
    command.extend(
        (
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--output",
            str(output),
        )
    )
    return MatrixCommand(
        primitive="decode_step",
        dtype="float16",
        command=tuple(command),
    )


def _decode_step_dynamic_command(
    *,
    device: str,
    warmup: int,
    iterations: int,
    output: Path,
    mode: str = "all",
    batch_buckets: str | None = None,
    seed: int | None = None,
    attention_backend: str = DEFAULT_DECODE_ATTENTION_BACKEND,
    dynamic_copy_mode: str = DEFAULT_DECODE_DYNAMIC_COPY_MODE,
    piecewise_post_mode: str = DEFAULT_DECODE_PIECEWISE_POST_MODE,
    orchestration_timing: str = DEFAULT_DECODE_ORCHESTRATION_TIMING,
) -> MatrixCommand:
    command = [
        "uv",
        "run",
        "benchmark-decode-step",
        "--dynamic-trace",
        "--mode",
        mode,
        "--device",
        device,
        "--dtype",
        "float16",
    ]
    if attention_backend != DEFAULT_DECODE_ATTENTION_BACKEND:
        command.extend(("--attention-backend", attention_backend))
    if dynamic_copy_mode != DEFAULT_DECODE_DYNAMIC_COPY_MODE:
        command.extend(("--dynamic-copy-mode", dynamic_copy_mode))
    if piecewise_post_mode != DEFAULT_DECODE_PIECEWISE_POST_MODE:
        command.extend(("--piecewise-post-mode", piecewise_post_mode))
    if orchestration_timing != DEFAULT_DECODE_ORCHESTRATION_TIMING:
        command.extend(("--orchestration-timing", orchestration_timing))
    if batch_buckets is not None:
        command.extend(("--batch-buckets", batch_buckets))
    if seed is not None:
        command.extend(("--seed", str(seed)))
    command.extend(
        (
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--output",
            str(output),
        )
    )
    return MatrixCommand(
        primitive="decode_step",
        dtype="float16",
        command=tuple(command),
    )


def _parse_block_sizes(value: str) -> tuple[int, ...]:
    try:
        block_sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("vector_add_sweep_block_sizes must be comma-separated integers") from exc
    if not block_sizes:
        raise ValueError("vector_add_sweep_block_sizes must not be empty")
    return block_sizes


def _normalize_suite(value: str) -> str:
    if value not in SUITES:
        raise ValueError("suite must be one of standard, h200-roofline, tensor-core")
    return value


def _parse_tile_shapes(value: str) -> tuple[tuple[int, ...], ...]:
    tile_shapes = []
    try:
        for token in (part.strip() for part in value.split(",")):
            if not token:
                continue
            dimensions = tuple(int(part.strip()) for part in token.lower().split("x"))
            if len(dimensions) != 3:
                raise ValueError
            tile_shapes.append(dimensions)
    except ValueError as exc:
        raise ValueError("matmul_sweep_tile_shapes must be comma-separated MxNxK triples") from exc
    if not tile_shapes:
        raise ValueError("matmul_sweep_tile_shapes must not be empty")
    return tuple(tile_shapes)


def _parse_launch_configs(value: str) -> tuple[tuple[int, ...], ...]:
    launch_configs = []
    try:
        for token in (part.strip() for part in value.split(",")):
            if not token:
                continue
            dimensions = tuple(int(part.strip()) for part in token.lower().split("x"))
            if len(dimensions) != 2:
                raise ValueError
            launch_configs.append(dimensions)
    except ValueError as exc:
        raise ValueError(
            "matmul_sweep_launch_configs must be comma-separated WARPSxSTAGES pairs"
        ) from exc
    if not launch_configs:
        raise ValueError("matmul_sweep_launch_configs must not be empty")
    return tuple(launch_configs)


def _parse_shapes(value: str) -> tuple[tuple[int, ...], ...]:
    shapes = []
    try:
        for token in (part.strip() for part in value.split(",")):
            if not token:
                continue
            dimensions = tuple(int(part.strip()) for part in token.lower().split("x"))
            if len(dimensions) != 2:
                raise ValueError
            shapes.append(dimensions)
    except ValueError as exc:
        raise ValueError(
            "rmsnorm_shape_sweep_shapes must be comma-separated ROWSxCOLS pairs"
        ) from exc
    if not shapes:
        raise ValueError("rmsnorm_shape_sweep_shapes must not be empty")
    return tuple(shapes)


def _extra_vector_add_block_sizes(
    block_sizes: tuple[int, ...],
    *,
    baseline_block_size: int,
) -> tuple[int, ...]:
    seen = {baseline_block_size}
    extras = []
    for block_size in block_sizes:
        if block_size in seen:
            continue
        extras.append(block_size)
        seen.add(block_size)
    return tuple(extras)


def _extra_matmul_configs(
    tile_shapes: tuple[tuple[int, ...], ...],
    launch_configs: tuple[tuple[int, ...], ...],
    *,
    input_precision: str,
    baseline_config: tuple[int, int, int, int, int, str],
) -> tuple[tuple[int, int, int, int, int, str], ...]:
    seen = {baseline_config}
    extras = []
    for tile_shape in tile_shapes:
        block_m, block_n, block_k = tile_shape
        for launch_config in launch_configs:
            num_warps, num_stages = launch_config
            normalized = (block_m, block_n, block_k, num_warps, num_stages, input_precision)
            if normalized in seen:
                continue
            extras.append(normalized)
            seen.add(normalized)
    return tuple(extras)


def _parse_values(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("comma-separated values must not be empty")
    return values


def _parse_int_values(value: str, *, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers") from exc
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def _parse_decode_bucket_values(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(";") if part.strip())
    if not values:
        raise ValueError("decode_bucket_sweep_values must not be empty")
    for buckets in values:
        _validate_batch_bucket_value(buckets)
    return values


def _validate_batch_bucket_value(value: str) -> None:
    try:
        buckets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("decode bucket values must be comma-separated integers") from exc
    if not buckets:
        raise ValueError("decode bucket values must not be empty")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("decode bucket values must be positive")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("decode bucket values must be unique and ascending")


def _extra_values(values: tuple[str, ...], *, baseline_value: str) -> tuple[str, ...]:
    seen = {baseline_value}
    extras = []
    for value in values:
        if value in seen:
            continue
        extras.append(value)
        seen.add(value)
    return tuple(extras)


def _join_block_sizes(block_sizes: tuple[int, ...]) -> str:
    return ",".join(str(block_size) for block_size in block_sizes)


def _join_tile_shapes(tile_shapes: tuple[tuple[int, int, int], ...]) -> str:
    return ",".join(f"{block_m}x{block_n}x{block_k}" for block_m, block_n, block_k in tile_shapes)


def _join_launch_configs(launch_configs: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{num_warps}x{num_stages}" for num_warps, num_stages in launch_configs)


def _join_shapes(shapes: tuple[tuple[int, int], ...]) -> str:
    return ",".join(f"{rows}x{cols}" for rows, cols in shapes)


def _join_values(values: tuple[str, ...]) -> str:
    return ",".join(values)


def _join_int_values(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _join_decode_bucket_values(values: tuple[str, ...]) -> str:
    return ";".join(values)


if __name__ == "__main__":
    main()
