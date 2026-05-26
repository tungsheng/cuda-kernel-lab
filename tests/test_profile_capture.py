from __future__ import annotations

import pytest

from cuda_kernel_lab import profile_capture


def test_parse_matmul_profile_capture_args() -> None:
    args = profile_capture.parse_args(
        [
            "matmul",
            "--m",
            "512",
            "--n",
            "4096",
            "--k",
            "11008",
            "--dtype",
            "bfloat16",
            "--block-m",
            "128",
            "--block-n",
            "128",
            "--block-k",
            "64",
            "--num-warps",
            "8",
            "--num-stages",
            "5",
            "--input-precision",
            "tf32",
            "--group-m",
            "8",
            "--schedule",
            "persistent",
            "--persistent-waves",
            "4",
        ]
    )

    assert args.command == "matmul"
    assert args.m == 512
    assert args.n == 4096
    assert args.k == 11008
    assert args.dtype == "bfloat16"
    assert args.schedule == "persistent"
    assert args.persistent_waves == 4
    assert args.warmup == 2
    assert args.profile_iterations == 1


def test_parse_matmul_profile_capture_requires_positive_profile_iterations() -> None:
    args = profile_capture.parse_args(
        [
            "matmul",
            "--m",
            "512",
            "--n",
            "4096",
            "--k",
            "11008",
            "--dtype",
            "float16",
            "--block-m",
            "128",
            "--block-n",
            "128",
            "--block-k",
            "64",
            "--num-warps",
            "8",
            "--num-stages",
            "5",
            "--input-precision",
            "tf32",
            "--group-m",
            "8",
            "--profile-iterations",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="profile-iterations"):
        profile_capture.run_matmul(args)


def test_parse_matmul_profile_capture_requires_positive_persistent_waves() -> None:
    args = profile_capture.parse_args(
        [
            "matmul",
            "--m",
            "512",
            "--n",
            "4096",
            "--k",
            "11008",
            "--dtype",
            "float16",
            "--block-m",
            "128",
            "--block-n",
            "128",
            "--block-k",
            "64",
            "--num-warps",
            "8",
            "--num-stages",
            "5",
            "--input-precision",
            "tf32",
            "--group-m",
            "8",
            "--schedule",
            "persistent",
            "--persistent-waves",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="persistent-waves"):
        profile_capture.run_matmul(args)
