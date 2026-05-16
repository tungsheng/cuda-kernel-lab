"""Print visible CUDA device information."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_kernel_lab.device import collect_cuda_devices


def main() -> None:
    devices = collect_cuda_devices()
    if not devices:
        print("No CUDA devices visible through PyTorch.")
        return

    for device in devices:
        capability = ".".join(str(part) for part in device.capability)
        sm_count = "unknown" if device.multiprocessor_count is None else device.multiprocessor_count
        print(f"cuda:{device.index}")
        print(f"  name: {device.name}")
        print(f"  compute capability: {capability}")
        print(f"  memory: {device.total_memory_gib:.2f} GiB")
        print(f"  multiprocessors: {sm_count}")


if __name__ == "__main__":
    main()
