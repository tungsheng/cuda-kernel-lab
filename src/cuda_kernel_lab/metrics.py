"""Metric helpers shared by benchmarks and tests."""

from __future__ import annotations

from collections.abc import Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    """Return an interpolated percentile for a non-empty sequence."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if pct < 0 or pct > 100:
        raise ValueError("pct must be between 0 and 100")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    rank = pct / 100 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def dtype_size_bytes(dtype: object) -> int:
    """Return the byte width for a torch dtype-like object."""

    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int):
        return itemsize

    text = str(dtype)
    if text.endswith("float16") or text.endswith("bfloat16"):
        return 2
    if text.endswith("float32") or text.endswith("int32"):
        return 4
    if text.endswith("float64") or text.endswith("int64"):
        return 8
    raise ValueError(f"unsupported dtype for byte-size estimate: {dtype}")
