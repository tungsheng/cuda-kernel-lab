from __future__ import annotations

import pytest

from cuda_kernel_lab import aws_ec2_live_gpu


def test_launch_plan_uses_safe_defaults_and_required_tags() -> None:
    config = aws_ec2_live_gpu.LaunchConfig(
        key_name="kernel-key",
        subnet_id="subnet-123",
        security_group_id="sg-123",
    )

    plan = aws_ec2_live_gpu.render_launch_plan(config)

    assert "--region us-west-2" in plan
    assert "--instance-type g5.xlarge" in plan
    assert "--key-name kernel-key" in plan
    assert "--subnet-id subnet-123" in plan
    assert "--security-group-ids sg-123" in plan
    assert "Project,Value=cuda-kernel-lab" in plan
    assert "Purpose,Value=live-gpu-benchmark" in plan
    assert aws_ec2_live_gpu.DEFAULT_AMI_SSM_PARAMETER in plan
    assert "--image-id ${AMI_ID}" in plan


def test_launch_print_mode_does_not_execute(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("launch without --execute must not run subprocesses")

    monkeypatch.setattr(aws_ec2_live_gpu.subprocess, "run", fail_run)

    aws_ec2_live_gpu.main(
        [
            "launch",
            "--key-name",
            "kernel-key",
            "--subnet-id",
            "subnet-123",
            "--security-group-id",
            "sg-123",
        ]
    )

    output = capsys.readouterr().out
    assert "aws ssm get-parameter" in output
    assert "aws ec2 run-instances" in output
    assert "g5.xlarge" in output


def test_terminate_print_mode_does_not_execute(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("terminate without --execute must not run subprocesses")

    monkeypatch.setattr(aws_ec2_live_gpu.subprocess, "run", fail_run)

    aws_ec2_live_gpu.main(["terminate", "--instance-id", "i-123"])

    output = capsys.readouterr().out
    assert "aws ec2 terminate-instances" in output
    assert "--region us-west-2" in output
    assert "--instance-ids i-123" in output


def test_execute_launch_resolves_ami_before_run_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    class Result:
        stdout = "ami-123\n"

    def fake_run(command: tuple[str, ...], **_kwargs: object) -> Result:
        calls.append(command)
        return Result()

    monkeypatch.setattr(aws_ec2_live_gpu.subprocess, "run", fake_run)

    aws_ec2_live_gpu.main(
        [
            "launch",
            "--key-name",
            "kernel-key",
            "--subnet-id",
            "subnet-123",
            "--security-group-id",
            "sg-123",
            "--execute",
        ]
    )

    assert calls[0][:3] == ("aws", "ssm", "get-parameter")
    assert calls[1][:3] == ("aws", "ec2", "run-instances")
    assert "--image-id" in calls[1]
    assert "ami-123" in calls[1]
