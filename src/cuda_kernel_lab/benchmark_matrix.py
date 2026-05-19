"""Run or print the first live-GPU benchmark matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("experiments/results/aws-ec2-first-run")
DEFAULT_WARMUP = 25
DEFAULT_ITERATIONS = 100
DEFAULT_MEMORY_BLOCK_SIZE = 1024
DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES = (512, 1024, 2048)
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
    include_vector_add_sweep: bool = False,
    vector_add_sweep_block_sizes: tuple[int, ...] = DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES,
) -> tuple[MatrixCommand, ...]:
    """Build the default live-GPU benchmark command matrix."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if memory_block_size <= 0:
        raise ValueError("memory_block_size must be positive")
    if any(block_size <= 0 for block_size in vector_add_sweep_block_sizes):
        raise ValueError("vector_add_sweep_block_sizes must be positive")

    commands: list[MatrixCommand] = []
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
                        "--warmup",
                        str(warmup),
                        "--iterations",
                        str(iterations),
                        "--output",
                        str(output_dir / "vector-add-block-size.jsonl"),
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
        "--include-vector-add-sweep",
        action="store_true",
        help="Add the first vector_add block-size strategy sweep.",
    )
    parser.add_argument(
        "--vector-add-sweep-block-sizes",
        default=_join_block_sizes(DEFAULT_VECTOR_ADD_SWEEP_BLOCK_SIZES),
        help="Comma-separated block sizes for --include-vector-add-sweep.",
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
        include_vector_add_sweep=args.include_vector_add_sweep,
        vector_add_sweep_block_sizes=_parse_block_sizes(args.vector_add_sweep_block_sizes),
    )

    if args.dry_run:
        for entry in commands:
            print(entry.shell_line())
        return

    for entry in commands:
        print(entry.shell_line(), flush=True)
        subprocess.run(entry.command, check=True)


def _parse_block_sizes(value: str) -> tuple[int, ...]:
    try:
        block_sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("vector_add_sweep_block_sizes must be comma-separated integers") from exc
    if not block_sizes:
        raise ValueError("vector_add_sweep_block_sizes must not be empty")
    return block_sizes


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


def _join_block_sizes(block_sizes: tuple[int, ...]) -> str:
    return ",".join(str(block_size) for block_size in block_sizes)


if __name__ == "__main__":
    main()
