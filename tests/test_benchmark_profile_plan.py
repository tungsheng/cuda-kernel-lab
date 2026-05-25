from __future__ import annotations

import json
from pathlib import Path

from cuda_kernel_lab import benchmark_profile_plan


def test_profile_targets_from_manifest_encodes_autotune_winners() -> None:
    manifest = {
        "winners": [
            {
                "dtype": "bfloat16",
                "shape": [512, 11008, 4096],
                "parameters": {
                    "block_m": 128,
                    "block_n": 128,
                    "block_k": 64,
                    "num_warps": 8,
                    "num_stages": 4,
                    "group_m": 8,
                    "input_precision": "tf32",
                },
            }
        ]
    }

    assert benchmark_profile_plan.profile_targets_from_manifest(manifest) == (
        "matmul-autotune-bfloat16-512x11008x4096-bm128-bn128-bk64-w8-s4-gm8-iptf32",
    )


def test_profile_targets_from_manifest_filters_dtype_shape_and_limit() -> None:
    manifest = {
        "winners": [
            _winner("float16", [512, 4096, 11008], group_m=4),
            _winner("bfloat16", [512, 11008, 4096], group_m=8),
            _winner("bfloat16", [4096, 4096, 4096], group_m=8),
        ]
    }

    assert benchmark_profile_plan.profile_targets_from_manifest(
        manifest,
        dtypes=("bfloat16",),
        shapes=((4096, 4096, 4096),),
        limit=1,
    ) == (
        "matmul-autotune-bfloat16-4096x4096x4096-bm128-bn128-bk64-w8-s4-gm8-iptf32",
    )


def test_load_profile_targets_reads_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "best.json"
    manifest_path.write_text(
        json.dumps({"winners": [_winner("float16", [512, 4096, 11008], group_m=4)]}),
        encoding="utf-8",
    )

    assert benchmark_profile_plan.load_profile_targets(manifest_path) == (
        "matmul-autotune-float16-512x4096x11008-bm128-bn128-bk64-w8-s4-gm4-iptf32",
    )


def _winner(dtype: str, shape: list[int], *, group_m: int) -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": shape,
        "parameters": {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "num_warps": 8,
            "num_stages": 4,
            "group_m": group_m,
            "input_precision": "tf32",
        },
    }
