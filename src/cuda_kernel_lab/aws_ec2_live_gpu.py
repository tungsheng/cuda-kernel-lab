"""Print or run AWS EC2 commands for a disposable live-GPU benchmark host."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass

DEFAULT_REGION = "us-west-2"
DEFAULT_INSTANCE_TYPE = "g5.xlarge"
DEFAULT_NAME = "cuda-kernel-lab-live-gpu"
DEFAULT_VOLUME_SIZE_GB = 100
DEFAULT_AMI_SSM_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-pytorch-2.5-ubuntu-22.04/latest/ami-id"
)
PROJECT_TAG = "cuda-kernel-lab"
PURPOSE_TAG = "live-gpu-benchmark"


@dataclass(frozen=True)
class LaunchConfig:
    key_name: str
    subnet_id: str
    security_group_id: str
    region: str = DEFAULT_REGION
    instance_type: str = DEFAULT_INSTANCE_TYPE
    name: str = DEFAULT_NAME
    ami_ssm_parameter: str = DEFAULT_AMI_SSM_PARAMETER
    volume_size_gb: int = DEFAULT_VOLUME_SIZE_GB
    profile: str | None = None
    public_ip: bool = True
    iam_instance_profile: str | None = None


@dataclass(frozen=True)
class TerminateConfig:
    instance_id: str
    region: str = DEFAULT_REGION
    profile: str | None = None


def ssm_ami_command(config: LaunchConfig) -> tuple[str, ...]:
    return (
        "aws",
        *_profile_args(config.profile),
        "ssm",
        "get-parameter",
        "--region",
        config.region,
        "--name",
        config.ami_ssm_parameter,
        "--query",
        "Parameter.Value",
        "--output",
        "text",
    )


def run_instances_command(config: LaunchConfig, *, ami_id: str = "${AMI_ID}") -> tuple[str, ...]:
    command = [
        "aws",
        *_profile_args(config.profile),
        "ec2",
        "run-instances",
        "--region",
        config.region,
        "--image-id",
        ami_id,
        "--instance-type",
        config.instance_type,
        "--key-name",
        config.key_name,
        "--subnet-id",
        config.subnet_id,
        "--security-group-ids",
        config.security_group_id,
        "--block-device-mappings",
        (
            "DeviceName=/dev/sda1,"
            f"Ebs={{VolumeSize={config.volume_size_gb},VolumeType=gp3,DeleteOnTermination=true}}"
        ),
        "--tag-specifications",
        _tag_spec("instance", config.name),
        _tag_spec("volume", config.name),
    ]
    if config.public_ip:
        command.append("--associate-public-ip-address")
    if config.iam_instance_profile:
        command.extend(
            (
                "--iam-instance-profile",
                f"Name={config.iam_instance_profile}",
            )
        )
    return tuple(command)


def terminate_instances_command(config: TerminateConfig) -> tuple[str, ...]:
    return (
        "aws",
        *_profile_args(config.profile),
        "ec2",
        "terminate-instances",
        "--region",
        config.region,
        "--instance-ids",
        config.instance_id,
    )


def render_command(command: tuple[str, ...]) -> str:
    return " ".join(_shell_part(part) for part in command)


def render_launch_plan(config: LaunchConfig) -> str:
    ami_command = render_command(ssm_ami_command(config))
    run_command = render_command(run_instances_command(config))
    return "\n".join(
        (
            "# Resolve the AWS Deep Learning AMI.",
            f"AMI_ID=$({ami_command})",
            "",
            "# Launch the disposable GPU benchmark host.",
            run_command,
        )
    )


def execute_launch(config: LaunchConfig) -> None:
    ami_result = subprocess.run(
        ssm_ami_command(config),
        capture_output=True,
        check=True,
        text=True,
    )
    ami_id = ami_result.stdout.strip()
    if not ami_id:
        raise SystemExit("AWS SSM returned an empty AMI id.")
    subprocess.run(run_instances_command(config, ami_id=ami_id), check=True)


def execute_terminate(config: TerminateConfig) -> None:
    subprocess.run(terminate_instances_command(config), check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="Print or run an EC2 launch command.")
    launch.add_argument("--key-name", required=True)
    launch.add_argument("--subnet-id", required=True)
    launch.add_argument("--security-group-id", required=True)
    launch.add_argument("--region", default=DEFAULT_REGION)
    launch.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    launch.add_argument("--name", default=DEFAULT_NAME)
    launch.add_argument("--ami-ssm-parameter", default=DEFAULT_AMI_SSM_PARAMETER)
    launch.add_argument("--volume-size-gb", type=int, default=DEFAULT_VOLUME_SIZE_GB)
    launch.add_argument("--profile")
    launch.add_argument("--iam-instance-profile")
    launch.add_argument("--no-public-ip", action="store_true")
    launch.add_argument("--execute", action="store_true")

    terminate = subparsers.add_parser("terminate", help="Print or run an EC2 terminate command.")
    terminate.add_argument("--instance-id", required=True)
    terminate.add_argument("--region", default=DEFAULT_REGION)
    terminate.add_argument("--profile")
    terminate.add_argument("--execute", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "launch":
        config = LaunchConfig(
            key_name=args.key_name,
            subnet_id=args.subnet_id,
            security_group_id=args.security_group_id,
            region=args.region,
            instance_type=args.instance_type,
            name=args.name,
            ami_ssm_parameter=args.ami_ssm_parameter,
            volume_size_gb=args.volume_size_gb,
            profile=args.profile,
            public_ip=not args.no_public_ip,
            iam_instance_profile=args.iam_instance_profile,
        )
        if args.execute:
            execute_launch(config)
        else:
            print(render_launch_plan(config))
        return

    config = TerminateConfig(
        instance_id=args.instance_id,
        region=args.region,
        profile=args.profile,
    )
    if args.execute:
        execute_terminate(config)
    else:
        print(render_command(terminate_instances_command(config)))


def _profile_args(profile: str | None) -> tuple[str, ...]:
    return ("--profile", profile) if profile else ()


def _shell_part(part: str) -> str:
    if part.startswith("${") and part.endswith("}"):
        return part
    return shlex.quote(part)


def _tag_spec(resource_type: str, name: str) -> str:
    return (
        f"ResourceType={resource_type},"
        "Tags=["
        f"{{Key=Name,Value={name}}},"
        f"{{Key=Project,Value={PROJECT_TAG}}},"
        f"{{Key=Purpose,Value={PURPOSE_TAG}}}"
        "]"
    )


if __name__ == "__main__":
    main()
