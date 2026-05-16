"""Small benchmark helpers used by kernel experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from inference_kernel_lab.metrics import percentile


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
        }


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
) -> BenchmarkResult:
    """Benchmark a callable with CUDA events when available and wall time otherwise."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    torch = _try_import_torch()
    use_cuda_events = (
        torch is not None
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )

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

