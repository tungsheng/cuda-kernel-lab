"""Device inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CudaDeviceInfo:
    index: int
    name: str
    capability: tuple[int, int]
    total_memory_bytes: int
    multiprocessor_count: int | None

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / 1024**3


def collect_cuda_devices() -> list[CudaDeviceInfo]:
    """Return visible CUDA device metadata, or an empty list when torch/CUDA is absent."""

    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return []

    devices: list[CudaDeviceInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            CudaDeviceInfo(
                index=index,
                name=props.name,
                capability=(props.major, props.minor),
                total_memory_bytes=props.total_memory,
                multiprocessor_count=getattr(props, "multi_processor_count", None),
            )
        )
    return devices


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch

