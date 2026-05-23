from __future__ import annotations

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
