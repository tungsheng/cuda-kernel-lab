"""Small roofline helpers for benchmark reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RooflineSpec:
    """Nominal peak hardware values used for roofline context."""

    name: str
    memory_bandwidth_gbps: float
    tflops_by_dtype: dict[str, float]
    source: str

    def peak_tflops_for_dtype(self, dtype: str) -> float | None:
        """Return the closest peak Tensor Core throughput for a benchmark dtype."""

        normalized = dtype.replace("torch.", "").lower()
        if normalized == "float32":
            normalized = "tf32"
        return self.tflops_by_dtype.get(normalized)


H200_SXM = RooflineSpec(
    name="NVIDIA H200 SXM",
    memory_bandwidth_gbps=4_800.0,
    tflops_by_dtype={
        "tf32": 989.0,
        "float16": 1_979.0,
        "bfloat16": 1_979.0,
        "fp8": 3_958.0,
    },
    source="NVIDIA H200 published peak specs",
)


def spec_for_device_name(device_name: str | None) -> RooflineSpec | None:
    """Return a known roofline spec for a CUDA device or provider GPU label."""

    if not device_name:
        return None
    normalized = device_name.upper()
    if "H200" in normalized:
        return H200_SXM
    return None


def arithmetic_intensity(flops: int | float, bytes_moved: int | float) -> float | None:
    """Return FLOPs per byte when both quantities are positive."""

    if flops <= 0 or bytes_moved <= 0:
        return None
    return float(flops) / float(bytes_moved)


def ridge_point_flops_per_byte(spec: RooflineSpec, dtype: str) -> float | None:
    """Return the compute-vs-memory ridge point for a dtype."""

    peak_tflops = spec.peak_tflops_for_dtype(dtype)
    if peak_tflops is None or spec.memory_bandwidth_gbps <= 0:
        return None
    return peak_tflops * 1_000.0 / spec.memory_bandwidth_gbps
