"""Run or print the standard live-GPU benchmark matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("experiments/results/aws-ec2/manual-run")
DEFAULT_WARMUP = 25
DEFAULT_ITERATIONS = 100
DEFAULT_MEMORY_BLOCK_SIZE = 1024
DEFAULT_SWIGLU_BLOCK_SIZE = 1024
DEFAULT_MATMUL_BLOCK_M = 16
DEFAULT_MATMUL_BLOCK_N = 16
DEFAULT_MATMUL_BLOCK_K = 32
DEFAULT_MATMUL_SWEEP_TILE_SHAPES = (
    (16, 32, 32),
    (32, 16, 32),
    (32, 32, 32),
    (32, 32, 64),
)
DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES = (512, 1024, 2048)
DEFAULT_REDUCTION_STRATEGY = "iterative"
DEFAULT_REDUCTION_SWEEP_STRATEGIES = ("iterative", "two_pass")
REDUCTION_STRATEGIES = ("iterative", "two_pass")
DEFAULT_DEVICE = "cuda"
DTYPES = ("float32", "float16")


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
    include_matmul_sweep: bool = False,
    matmul_sweep_tile_shapes: tuple[tuple[int, ...], ...] = DEFAULT_MATMUL_SWEEP_TILE_SHAPES,
    include_vector_add_sweep: bool = False,
    vector_add_sweep_block_sizes: tuple[int, ...] = DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES,
    reduction_strategy: str = DEFAULT_REDUCTION_STRATEGY,
    include_reduction_sweep: bool = False,
    reduction_sweep_strategies: tuple[str, ...] = DEFAULT_REDUCTION_SWEEP_STRATEGIES,
) -> tuple[MatrixCommand, ...]:
    """Build the default live-GPU benchmark command matrix."""

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
    if any(len(tile_shape) != 3 for tile_shape in matmul_sweep_tile_shapes):
        raise ValueError("matmul_sweep_tile_shapes must be MxNxK triples")
    if any(any(dim <= 0 for dim in tile_shape) for tile_shape in matmul_sweep_tile_shapes):
        raise ValueError("matmul_sweep_tile_shapes must be positive")
    if any(block_size <= 0 for block_size in vector_add_sweep_block_sizes):
        raise ValueError("vector_add_sweep_block_sizes must be positive")
    if reduction_strategy not in REDUCTION_STRATEGIES:
        raise ValueError("reduction_strategy must be one of iterative, two_pass")
    if any(strategy not in REDUCTION_STRATEGIES for strategy in reduction_sweep_strategies):
        raise ValueError("reduction_sweep_strategies must be one of iterative, two_pass")

    commands: list[MatrixCommand] = []
    matmul_baseline_tile_shape = (matmul_block_m, matmul_block_n, matmul_block_k)
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
        if include_matmul or include_matmul_sweep:
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
                        warmup=warmup,
                        iterations=iterations,
                        output=output_dir / "matmul.jsonl",
                    ),
                )
            )

    if include_vector_add_sweep:
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

    if include_reduction_sweep:
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

    if include_matmul_sweep:
        for block_m, block_n, block_k in _extra_matmul_tile_shapes(
            matmul_sweep_tile_shapes,
            baseline_tile_shape=matmul_baseline_tile_shape,
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
                        warmup=warmup,
                        iterations=iterations,
                        output=output_dir / "matmul-tile-shape.jsonl",
                    ),
                )
            )
    return tuple(commands)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--include-matmul-sweep",
        action="store_true",
        help="Add the float16 matmul tile-shape strategy sweep.",
    )
    parser.add_argument(
        "--matmul-sweep-tile-shapes",
        default=_join_tile_shapes(DEFAULT_MATMUL_SWEEP_TILE_SHAPES),
        help="Comma-separated MxNxK tile shapes for --include-matmul-sweep.",
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
        include_matmul_sweep=args.include_matmul_sweep,
        matmul_sweep_tile_shapes=_parse_tile_shapes(args.matmul_sweep_tile_shapes),
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
    warmup: int,
    iterations: int,
    output: Path,
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
        "1024",
        "--n",
        "1024",
        "--k",
        "1024",
        "--dtype",
        dtype,
        "--block-m",
        str(block_m),
        "--block-n",
        str(block_n),
        "--block-k",
        str(block_k),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--output",
        str(output),
    )


def _parse_block_sizes(value: str) -> tuple[int, ...]:
    try:
        block_sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("vector_add_sweep_block_sizes must be comma-separated integers") from exc
    if not block_sizes:
        raise ValueError("vector_add_sweep_block_sizes must not be empty")
    return block_sizes


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
        raise ValueError(
            "matmul_sweep_tile_shapes must be comma-separated MxNxK triples"
        ) from exc
    if not tile_shapes:
        raise ValueError("matmul_sweep_tile_shapes must not be empty")
    return tuple(tile_shapes)


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


def _extra_matmul_tile_shapes(
    tile_shapes: tuple[tuple[int, ...], ...],
    *,
    baseline_tile_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    seen = {baseline_tile_shape}
    extras = []
    for tile_shape in tile_shapes:
        block_m, block_n, block_k = tile_shape
        normalized = (block_m, block_n, block_k)
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


def _join_values(values: tuple[str, ...]) -> str:
    return ",".join(values)


if __name__ == "__main__":
    main()
