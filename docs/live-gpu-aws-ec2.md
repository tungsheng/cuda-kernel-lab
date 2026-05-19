# Live GPU On AWS EC2

Use AWS EC2 as a disposable single-GPU Linux host for benchmark evidence. This
repo does not need an EKS cluster or a long-running GPU machine. Terraform owns
the benchmark instance, security group, AMI lookup, and local state under
`infra/env/aws-gpu`.

Defaults:

- region: `us-west-2`
- instance type: `g5.xlarge`
- AMI source: AWS Deep Learning Base OSS NVIDIA Driver GPU Ubuntu 22.04,
  resolved from AWS SSM Parameter Store

## Prerequisites

- Terraform `>= 1.6`
- AWS credentials for `us-west-2`
- EC2 GPU quota for `g5.xlarge`
- an EC2 key pair and matching local private key
- a default VPC, or an explicit subnet passed with `--subnet-id`
- local `ssh` and `tar` when using the bootstrap path

## Recommended Scripted Launch

Use the repo script when you want the closest equivalent to
`gpu-inference-lab`'s `scripts/up`, but backed by a small Terraform EC2
environment instead of an EKS platform:

```bash
./scripts/up \
  --key-name <key-pair-name> \
  --key-file <key-file.pem>
```

By default, the script:

- writes `infra/env/aws-gpu/local.auto.tfvars`
- runs `terraform -chdir=infra/env/aws-gpu init`
- runs `terraform -chdir=infra/env/aws-gpu apply -auto-approve`
- lets Terraform discover the default subnet, create the SSH security group,
  resolve the AMI from SSM Parameter Store, and launch one `g5.xlarge` instance
- syncs this working tree to `~/cuda-kernel-lab`
- installs `uv`, syncs GPU dependencies, runs `gpu-info`, and prints the
  benchmark matrix dry run

Use a named AWS profile or a different GPU shape explicitly:

```bash
./scripts/up \
  --profile <aws-profile> \
  --instance-type g6.xlarge \
  --key-name <key-pair-name> \
  --key-file <key-file.pem>
```

Use existing network controls when the default VPC path is not appropriate:

```bash
./scripts/up \
  --key-name <key-pair-name> \
  --key-file <key-file.pem> \
  --subnet-id <subnet-id> \
  --security-group-id <security-group-id>
```

After launch, SSH to the host and run the benchmark evidence pass:

```bash
ssh -i <key-file.pem> ubuntu@<public-ip>
cd ~/cuda-kernel-lab
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep
uv run benchmark-report --input-dir experiments/results/aws-ec2-first-run
```

Terminate the instance and remove the temporary security group:

```bash
./scripts/down
```

Run `./scripts/up --help` for all options, including `--skip-bootstrap`,
`--no-public-ip`, `--ingress-cidr`, and `--tf-vars-file`.

## Direct Terraform

Use Terraform directly when you want to inspect the plan before creating
anything:

```bash
terraform -chdir=infra/env/aws-gpu init
terraform -chdir=infra/env/aws-gpu plan \
  -out tfplan \
  -var 'key_name=<key-pair-name>' \
  -var 'ssh_ingress_cidr=<your-ip>/32'
terraform -chdir=infra/env/aws-gpu apply tfplan
```

Common overrides:

```bash
terraform -chdir=infra/env/aws-gpu plan \
  -var 'key_name=<key-pair-name>' \
  -var 'ssh_ingress_cidr=<your-ip>/32' \
  -var 'instance_type=g6.xlarge' \
  -var 'subnet_id=<subnet-id>'
```

## Manual Host Preparation

When you use direct Terraform or `./scripts/up --skip-bootstrap`, SSH to the
instance, then install `uv` and clone or copy this repo:

```bash
ssh -i <key-file.pem> ubuntu@<public-ip>
uv_installer=$(mktemp)
curl -LsSf https://astral.sh/uv/install.sh -o "$uv_installer"
sh "$uv_installer"
rm -f "$uv_installer"
git clone <repo-url>
cd cuda-kernel-lab
uv sync --group dev --extra gpu
```

Confirm the host is ready:

```bash
uv run gpu-info
uv run pytest
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep --dry-run
```

## Run The First Evidence Matrix

```bash
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep
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

Generate a compact Markdown summary from a CSV/text Nsight Compute export:

```bash
uv run nsight-summary \
  --input profiling/nsight_compute/vector-add.csv \
  --output profiling/reports/vector-add-a10g.md \
  --operation vector_add \
  --strategy triton-block-size
```

## Terminate

```bash
./scripts/down
```

Or destroy directly with Terraform:

```bash
terraform -chdir=infra/env/aws-gpu destroy
```

Terminate the instance as soon as benchmark evidence is collected.
