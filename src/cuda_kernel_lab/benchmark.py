"""Small benchmark helpers used by kernel experiments."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from cuda_kernel_lab.device import collect_cuda_devices
from cuda_kernel_lab.metrics import percentile


@dataclass(frozen=True)
class CorrectnessResult:
    """Numerical agreement result for one benchmarked operation."""

    checked: bool
    passed: bool | None
    reference_backend: str | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    atol: float | None = None
    rtol: float | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "passed": self.passed,
            "reference_backend": self.reference_backend,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "atol": self.atol,
            "rtol": self.rtol,
            "message": self.message,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """Latency and derived throughput metrics for one benchmarked operation."""

    name: str
    device: str
    dtype: str
    shape: tuple[int, ...]
    latencies_ms: list[float]
    bytes_moved: int
    flops: int
    strategy: str = "baseline"
    variant: str = "default"
    parameters: dict[str, Any] | None = None
    correctness: CorrectnessResult | None = None

    @property
    def p50_ms(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def p99_ms(self) -> float:
        return percentile(self.latencies_ms, 99)

    @property
    def bandwidth_gbps(self) -> float:
        seconds = self.p50_ms / 1_000
        return self.bytes_moved / seconds / 1e9 if seconds > 0 else 0.0

    @property
    def tflops(self) -> float:
        seconds = self.p50_ms / 1_000
        return self.flops / seconds / 1e12 if seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "shape": self.shape,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "bytes_moved": self.bytes_moved,
            "bandwidth_gbps": self.bandwidth_gbps,
            "flops": self.flops,
            "tflops": self.tflops,
            "latencies_ms": self.latencies_ms,
            "strategy": self.strategy,
            "variant": self.variant,
            "parameters": _jsonable(self.parameters or {}),
            "correctness": (
                self.correctness.as_dict()
                if self.correctness is not None
                else CorrectnessResult(checked=False, passed=None).as_dict()
            ),
        }


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    """Metadata that makes a benchmark run reproducible after the terminal scrolls away."""

    benchmark: str
    args: dict[str, Any]
    command: str
    timestamp_utc: str
    git_commit: str | None
    git_dirty: bool | None
    host: dict[str, str | int | None]
    packages: dict[str, str | None]
    cuda_devices: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "args": self.args,
            "command": self.command,
            "timestamp_utc": self.timestamp_utc,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "host": self.host,
            "packages": self.packages,
            "cuda_devices": self.cuda_devices,
        }


def collect_run_metadata(benchmark: str, args: Any) -> BenchmarkRunMetadata:
    """Collect stable benchmark-run context for JSON and JSONL outputs."""

    return BenchmarkRunMetadata(
        benchmark=benchmark,
        args=_jsonable_mapping(vars(args)),
        command=" ".join(shlex.quote(part) for part in sys.argv),
        timestamp_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        host={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "pid": os.getpid(),
        },
        packages={
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "triton": _package_version("triton"),
        },
        cuda_devices=[
            {
                "index": device.index,
                "name": device.name,
                "capability": list(device.capability),
                "total_memory_bytes": device.total_memory_bytes,
                "multiprocessor_count": device.multiprocessor_count,
            }
            for device in collect_cuda_devices()
        ],
    )


def benchmark_records(
    results: list[BenchmarkResult],
    metadata: BenchmarkRunMetadata,
) -> list[dict[str, Any]]:
    """Return one self-contained JSON-ready record per benchmark result."""

    run = metadata.as_dict()
    return [{"run": run, "result": result.as_dict()} for result in results]


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write benchmark records as JSON Lines and return the resolved path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return path


def benchmark_callable(
    name: str,
    fn: Callable[[], Any],
    *,
    device: str,
    dtype: str,
    shape: tuple[int, ...],
    bytes_moved: int,
    flops: int,
    warmup: int = 25,
    iterations: int = 100,
    strategy: str = "baseline",
    variant: str = "default",
    parameters: dict[str, Any] | None = None,
    correctness: CorrectnessResult | None = None,
) -> BenchmarkResult:
    """Benchmark a callable with CUDA events when available and wall time otherwise."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    torch = _try_import_torch()
    use_cuda_events = torch is not None and device.startswith("cuda") and torch.cuda.is_available()

    for _ in range(warmup):
        fn()
    _synchronize(torch, device)

    latencies_ms: list[float] = []
    if use_cuda_events:
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            latencies_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(iterations):
            start_time = perf_counter()
            fn()
            _synchronize(torch, device)
            latencies_ms.append((perf_counter() - start_time) * 1_000)

    return BenchmarkResult(
        name=name,
        device=device,
        dtype=dtype,
        shape=shape,
        latencies_ms=latencies_ms,
        bytes_moved=bytes_moved,
        flops=flops,
        strategy=strategy,
        variant=variant,
        parameters=parameters,
        correctness=correctness,
    )


def check_tensors_close(
    actual: Any,
    expected: Any,
    *,
    torch: Any,
    rtol: float,
    atol: float,
    reference_backend: str = "torch",
) -> CorrectnessResult:
    """Return numerical agreement metadata for tensor-like benchmark outputs."""

    try:
        actual_tensor = actual.detach()
        expected_tensor = expected.detach().to(device=actual_tensor.device)
        if expected_tensor.dtype != actual_tensor.dtype:
            expected_tensor = expected_tensor.to(dtype=actual_tensor.dtype)

        diff = (actual_tensor - expected_tensor).abs()
        max_abs_error = float(diff.max().item()) if diff.numel() else 0.0
        denominator = expected_tensor.abs().clamp_min(1e-12)
        max_rel_error = float((diff / denominator).max().item()) if diff.numel() else 0.0
        passed = bool(torch.allclose(actual_tensor, expected_tensor, rtol=rtol, atol=atol))
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return CorrectnessResult(
            checked=True,
            passed=False,
            reference_backend=reference_backend,
            atol=atol,
            rtol=rtol,
            message=str(exc),
        )

    return CorrectnessResult(
        checked=True,
        passed=passed,
        reference_backend=reference_backend,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        atol=atol,
        rtol=rtol,
    )


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _synchronize(torch: Any | None, device: str) -> None:
    if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _jsonable_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in values.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_commit() -> str | None:
    return os.environ.get("CUDA_KERNEL_LAB_GIT_COMMIT") or _git_output("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    env_value = os.environ.get("CUDA_KERNEL_LAB_GIT_DIRTY")
    if env_value is not None:
        return _env_bool(env_value)

    status = _git_output("status", "--short")
    if status is None:
        return None
    return bool(status)


def _env_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "dirty"}:
        return True
    if normalized in {"0", "false", "no", "clean"}:
        return False
    return None
