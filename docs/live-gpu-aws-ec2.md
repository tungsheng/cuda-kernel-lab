# Legacy Live GPU On AWS EC2

Runpod is the default live-GPU provider. Use AWS EC2 only when you need to
compare against historical A10G evidence or inspect the Terraform fallback.

## Defaults

- region: `us-west-2`
- instance type: `g5.xlarge`
- AMI: AWS Deep Learning Base OSS NVIDIA Driver GPU Ubuntu 22.04 from SSM
- Terraform environment: `infra/env/aws-gpu`
- raw JSONL: `experiments/results/aws-ec2/<run-id>/`
- report: `experiments/reports/aws-ec2/<run-id>.md`
- profile summaries: `profiling/reports/<run-id>/`

## Prerequisites

- Terraform `>= 1.6`
- AWS credentials and EC2 GPU quota in `us-west-2`
- local `aws`, `ssh`, and `tar`
- passwordless `sudo` on the remote host for Nsight Compute counters when using
  `--with-profiling`

## Provider Loop

Preview the host launch without touching AWS:

```bash
./scripts/up --platform aws --dry-run
```

Start the GPU host. With no key arguments, the script creates or reuses the
project key under `.aws-gpu/keys/` and writes `.aws-gpu/connection.env`:

```bash
./scripts/up --platform aws
```

Run one or more benchmark experiments:

```bash
./scripts/benchmark --platform aws --run-id <run-id>
./scripts/benchmark --platform aws --run-id <second-run-id> --include-matmul-sweep
```

Tear the host down:

```bash
./scripts/down --platform aws
```

The default teardown removes generated Terraform variables and generated
connection metadata. The reusable dev key under `.aws-gpu/keys/` is left in
place.

## Useful Overrides

Use an existing EC2 key pair only when SSH access must align with your own key
inventory:

```bash
./scripts/up --platform aws \
  --key-name <key-pair-name> \
  --key-file <key-file.pem>
```

If SSH times out because HTTPS IP discovery and TCP/22 use different NAT egress
addresses, pass the SSH-visible CIDR:

```bash
./scripts/up --platform aws --ingress-cidr <ssh-egress-cidr>
```

Use `--skip-bootstrap` for an infra-only host without SSH sync or dependency
installation:

```bash
./scripts/up --platform aws --skip-bootstrap
```

Use direct Terraform only when you need to inspect or customize the plan:

```bash
terraform -chdir=infra/env/aws-gpu init
terraform -chdir=infra/env/aws-gpu plan \
  -out tfplan \
  -var 'key_name=cuda-kernel-lab-${USER}' \
  -var 'ssh_ingress_cidr=<your-ip>/32'
terraform -chdir=infra/env/aws-gpu apply tfplan
terraform -chdir=infra/env/aws-gpu destroy
```

After manual teardown, confirm local Terraform state is empty:

```bash
terraform -chdir=infra/env/aws-gpu state list
```

## Benchmark And Profiling Notes

Use the same benchmark flags documented in [Benchmark Workflow](benchmark-workflow.md),
but add `--platform aws` to shell-script commands.

When `--with-profiling` is set, the benchmark script runs `ncu` through
passwordless `sudo` because NVIDIA performance counters are restricted for
normal users on the AWS Deep Learning AMI.
