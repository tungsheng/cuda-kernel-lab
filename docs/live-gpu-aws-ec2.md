# Live GPU On AWS EC2

Use AWS EC2 as a disposable single-GPU Linux host for benchmark evidence. This
repo does not need an EKS cluster or a long-running GPU machine.

Defaults:

- region: `us-west-2`
- instance type: `g5.xlarge`
- AMI source: AWS Deep Learning OSS NVIDIA Driver GPU PyTorch Ubuntu 22.04,
  resolved from AWS SSM Parameter Store

## Prerequisites

- AWS CLI with credentials for `us-west-2`
- EC2 GPU quota for `g5.xlarge`
- an EC2 key pair
- a subnet that can be reached by SSH
- a security group that allows SSH from your IP
- local `uv` installed on the EC2 host after launch

## Launch

Print the launch commands first:

```bash
uv run aws-ec2-live-gpu launch \
  --key-name <key-pair-name> \
  --subnet-id <subnet-id> \
  --security-group-id <security-group-id>
```

Add `--profile <aws-profile>` if you use a named AWS profile.

When the commands look correct, run the launch:

```bash
uv run aws-ec2-live-gpu launch \
  --key-name <key-pair-name> \
  --subnet-id <subnet-id> \
  --security-group-id <security-group-id> \
  --execute
```

Record the instance ID and public IP from the AWS output.

## Prepare The Host

SSH to the instance, then install `uv` and clone this repo:

```bash
ssh -i <key-file.pem> ubuntu@<public-ip>
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo-url>
cd cuda-kernel-lab
uv sync --group dev --extra gpu
```

Confirm the host is ready:

```bash
uv run gpu-info
uv run pytest
uv run benchmark-matrix --include-vector-add-sweep --dry-run
```

## Run The First Evidence Matrix

```bash
uv run benchmark-matrix --include-vector-add-sweep
uv run benchmark-report --input-dir experiments/results/aws-ec2-first-run
```

This writes JSONL records to:

```text
experiments/results/aws-ec2-first-run/
```

Copy the JSONL files back to your workstation if you want to inspect them
locally. Commit compact summaries, not large raw result dumps.

## First Profiler Pass

Start with one memory primitive and one fused kernel:

```bash
ncu --set full --target-processes all uv run benchmark-memory --backend triton --device cuda --op vector_add --dtype float32 --output experiments/results/aws-ec2-first-run-profiled/memory-profiled.jsonl
ncu --set full --target-processes all uv run benchmark-softmax --backend triton --device cuda --rows 4096 --cols 1024 --dtype float32 --output experiments/results/aws-ec2-first-run-profiled/softmax-profiled.jsonl
```

Save compact profiler notes under `profiling/reports/`.

## Terminate

Print the terminate command first:

```bash
uv run aws-ec2-live-gpu terminate --instance-id <instance-id>
```

Then terminate explicitly:

```bash
uv run aws-ec2-live-gpu terminate --instance-id <instance-id> --execute
```

Terminate the instance as soon as benchmark evidence is collected.
