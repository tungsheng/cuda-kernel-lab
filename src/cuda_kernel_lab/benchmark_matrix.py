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
) -> tuple[MatrixCommand, ...]:
    """Build the default live-GPU benchmark command matrix."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    commands = build_matrix(
        output_dir=args.output_dir,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    if args.dry_run:
        for entry in commands:
            print(entry.shell_line())
        return

    for entry in commands:
        print(entry.shell_line(), flush=True)
        subprocess.run(entry.command, check=True)


if __name__ == "__main__":
    main()
