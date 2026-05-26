from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shell_scripts_parse_cleanly() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            "scripts/up",
            "scripts/down",
            "scripts/benchmark",
            "scripts/runpod-up",
            "scripts/runpod-down",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_benchmark_dry_run_uses_runpod_connection(tmp_path: Path) -> None:
    connection_file = tmp_path / "runpod.env"
    connection_file.write_text(
        "\n".join(
            [
                "PROVIDER_PLATFORM=runpod",
                "RUNPOD_POD_ID=pod123",
                "RUNPOD_GPU_ID='NVIDIA L4'",
                "RUNPOD_CLOUD_TYPE=SECURE",
                "KEY_FILE=/tmp/fake-runpod-key",
                "SSH_USER=root",
                "SSH_HOST=203.0.113.10",
                "SSH_PORT=2222",
                "REMOTE_DIR=cuda-kernel-lab",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark",
            "--dry-run",
            "--platform",
            "runpod",
            "--connection-file",
            str(connection_file),
            "--run-id",
            "test-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Platform: runpod" in result.stdout
    assert "Runpod pod: pod123" in result.stdout
    assert "Raw results: experiments/results/runpod/test-run" in result.stdout
    assert "ssh -p 2222 root@203.0.113.10" in result.stdout
    assert "terraform -chdir" not in result.stdout


def test_benchmark_dry_run_can_select_h200_roofline_suite(tmp_path: Path) -> None:
    connection_file = tmp_path / "runpod.env"
    connection_file.write_text(
        "\n".join(
            [
                "PROVIDER_PLATFORM=runpod",
                "RUNPOD_POD_ID=pod123",
                "RUNPOD_GPU_ID='NVIDIA H200'",
                "KEY_FILE=/tmp/fake-runpod-key",
                "SSH_USER=root",
                "SSH_HOST=203.0.113.10",
                "SSH_PORT=2222",
                "REMOTE_DIR=cuda-kernel-lab",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark",
            "--dry-run",
            "--platform",
            "runpod",
            "--connection-file",
            str(connection_file),
            "--run-id",
            "test-run",
            "--suite",
            "h200-roofline",
            "--with-profiling",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Benchmark suite: h200-roofline" in result.stdout
    assert "Profile mode: light" in result.stdout
    assert "Profile preset: auto" in result.stdout
    assert "Profile timeout seconds: 120" in result.stdout
    assert (
        "benchmark-matrix\\ --output-dir\\ experiments/results/runpod/test-run\\ "
        "--suite\\ h200-roofline"
    ) in result.stdout
    assert "matmul-tensor-core-bfloat16" in result.stdout
    assert "matmul-llm-down-bfloat16" in result.stdout
    assert "memory-vector-add-float32" not in result.stdout
    assert "find_ncu_bin" in result.stdout
    system_path = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    assert system_path in result.stdout
    assert 'ncu_runner=\\\'env\\\'' in result.stdout
    assert 'HOME=\\"\\$HOME\\"' in result.stdout
    assert 'PATH=\\"\\$PATH\\"' in result.stdout
    assert '\\"\\$\\{ncu_bin\\}\\"' in result.stdout
    assert "profile_csv_has_metrics" in result.stdout
    assert "profile_failure_reason" in result.stdout
    assert "profile_counter_preflight" in result.stdout
    assert "profile_python=\\'.venv/bin/python\\'" in result.stdout
    assert "cuda_kernel_lab.profile_capture" in result.stdout
    assert "--profile-from-start\\ off" in result.stdout
    assert "thenif" not in result.stdout
    assert "gpu__time_duration.sum" in result.stdout
    assert "timeout\\ 120" in result.stdout
    assert "Profile summaries: profiling/reports/test-run" in result.stdout


def test_runpod_up_profile_counters_defaults_to_community() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/up",
            "--dry-run",
            "--profile-counters",
            "--gpu-id",
            "NVIDIA H200",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Cloud type: COMMUNITY" in result.stdout
    assert "Nsight counters: validate access during bootstrap" in result.stdout
    assert "--cloud-type COMMUNITY" in result.stdout


def test_runpod_up_profile_counters_respects_explicit_cloud_type() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/up",
            "--dry-run",
            "--profile-counters",
            "--cloud-type",
            "SECURE",
            "--gpu-id",
            "NVIDIA H200",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Cloud type: SECURE" in result.stdout
    assert "--cloud-type SECURE" in result.stdout


def test_benchmark_dry_run_can_select_h200_matmul_autotune_suite(tmp_path: Path) -> None:
    connection_file = tmp_path / "runpod.env"
    connection_file.write_text(
        "\n".join(
            [
                "PROVIDER_PLATFORM=runpod",
                "RUNPOD_POD_ID=pod123",
                "RUNPOD_GPU_ID='NVIDIA H200'",
                "KEY_FILE=/tmp/fake-runpod-key",
                "SSH_USER=root",
                "SSH_HOST=203.0.113.10",
                "SSH_PORT=2222",
                "REMOTE_DIR=cuda-kernel-lab",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark",
            "--dry-run",
            "--platform",
            "runpod",
            "--connection-file",
            str(connection_file),
            "--run-id",
            "test-autotune",
            "--suite",
            "h200-matmul-autotune",
            "--matmul-autotune-shapes",
            "512x11008x4096",
            "--matmul-autotune-configs",
            "128x128x64x4x4x4,128x128x64x4x4x8",
            "--matmul-autotune-repeats",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Benchmark suite: h200-matmul-autotune" in result.stdout
    assert (
        "benchmark-matrix\\ --output-dir\\ experiments/results/runpod/test-autotune"
        in result.stdout
    )
    assert "h200-matmul-autotune" in result.stdout
    assert "Matrix keep-going: enabled" in result.stdout
    assert "Matmul autotune shapes: 512x11008x4096" in result.stdout
    assert (
        "Matmul autotune configs: 128x128x64x4x4x4,128x128x64x4x4x8"
        in result.stdout
    )
    assert "Matmul autotune repeats: 2" in result.stdout
    assert "--keep-going" in result.stdout
    assert "--matmul-autotune-configs" in result.stdout
    assert "--include-vector-add-sweep" not in result.stdout
    assert "--include-reduction-sweep" not in result.stdout
    assert "benchmark-autotune" in result.stdout
    assert "h200-matmul-best.json" in result.stdout
    assert "h200-matmul-best.md" in result.stdout


def test_benchmark_dry_run_can_run_profile_only_target(tmp_path: Path) -> None:
    connection_file = tmp_path / "runpod.env"
    connection_file.write_text(
        "\n".join(
            [
                "PROVIDER_PLATFORM=runpod",
                "RUNPOD_POD_ID=pod123",
                "RUNPOD_GPU_ID='NVIDIA H200'",
                "KEY_FILE=/tmp/fake-runpod-key",
                "SSH_USER=root",
                "SSH_HOST=203.0.113.10",
                "SSH_PORT=2222",
                "REMOTE_DIR=cuda-kernel-lab",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark",
            "--dry-run",
            "--platform",
            "runpod",
            "--connection-file",
            str(connection_file),
            "--run-id",
            "test-profile-only",
            "--profile-only",
            "--profile-mode",
            "full",
            "--profile-timeout-seconds",
            "120",
            "--profile-targets",
            "matmul-llm-down-bfloat16",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Profile-only: enabled" in result.stdout
    assert "Profile mode: full" in result.stdout
    assert "Profile timeout seconds: 120" in result.stdout
    assert "Profile targets: matmul-llm-down-bfloat16" in result.stdout
    assert "benchmark-matrix" not in result.stdout
    assert "matmul-llm-down-bfloat16" in result.stdout
    assert "matmul-llm-up-bfloat16" not in result.stdout
    assert "--set" in result.stdout
    assert "full" in result.stdout
    assert "timeout\\ 120" in result.stdout


def test_benchmark_dry_run_profiles_autotune_winners(tmp_path: Path) -> None:
    connection_file = tmp_path / "runpod.env"
    connection_file.write_text(
        "\n".join(
            [
                "PROVIDER_PLATFORM=runpod",
                "RUNPOD_POD_ID=pod123",
                "RUNPOD_GPU_ID='NVIDIA H200'",
                "KEY_FILE=/tmp/fake-runpod-key",
                "SSH_USER=root",
                "SSH_HOST=203.0.113.10",
                "SSH_PORT=2222",
                "REMOTE_DIR=cuda-kernel-lab",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "h200-matmul-best.json"
    manifest.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark",
            "--dry-run",
            "--platform",
            "runpod",
            "--connection-file",
            str(connection_file),
            "--run-id",
            "test-profile-winners",
            "--profile-only",
            "--profile-preset",
            "autotune-winners",
            "--profile-autotune-manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    target = (
        "matmul-autotune-bfloat16-512x11008x4096-"
        "bm128-bn128-bk64-w8-s4-gm8-iptf32"
    )
    assert "Profile preset: autotune-winners" in result.stdout
    assert f"Profile autotune manifest: {manifest}" in result.stdout
    assert target in result.stdout
    assert "--num-warps" in result.stdout
    assert "--num-stages" in result.stdout
    assert "--group-m" in result.stdout
    assert "cuda_kernel_lab.profile_capture" in result.stdout
    assert "--profile-from-start\\ off" in result.stdout
    assert "triton-autotune-block-128x128x64-warps-8-stages-4-groupm-8" in result.stdout


def test_benchmark_dry_run_preserves_aws_connection(tmp_path: Path) -> None:
    connection_file = tmp_path / "aws.env"
    connection_file.write_text(
        "\n".join(
            [
                "KEY_FILE=/tmp/fake-aws-key",
                "SSH_USER=ubuntu",
                "SSH_HOST=198.51.100.10",
                "REMOTE_DIR=cuda-kernel-lab",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "scripts/benchmark",
            "--dry-run",
            "--platform",
            "aws",
            "--connection-file",
            str(connection_file),
            "--run-id",
            "test-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Platform: aws" in result.stdout
    assert "Raw results: experiments/results/aws-ec2/test-run" in result.stdout
    assert "terraform -chdir=" in result.stdout
    assert "ssh -p 22 ubuntu@<terraform ssh_host>" in result.stdout
