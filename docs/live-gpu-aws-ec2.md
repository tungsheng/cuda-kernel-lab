# Live GPU On AWS EC2

Use AWS EC2 for disposable single-GPU benchmark evidence. The normal path is
one command: create a temporary EC2 key pair, launch a `g5.xlarge`, run the
benchmark matrix, copy artifacts back, destroy Terraform resources, and delete
the temporary key.

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
- passwordless `sudo` on the remote host for Nsight Compute performance
  counters when using `--with-profiling`

## Recommended Flow

Preview the plan without touching AWS:

```bash
./scripts/live-benchmark --run-id <run-id> --dry-run
```

Run the full benchmark:

```bash
./scripts/live-benchmark --run-id <run-id>
```

If SSH times out on a network where HTTPS and SSH use different carrier NAT
egress addresses, override the auto-discovered `/32` with the SSH-visible CIDR:

```bash
./scripts/live-benchmark --run-id <run-id> --ingress-cidr <ssh-egress-cidr>
```

Add matmul progression numbers when needed:

```bash
./scripts/live-benchmark --run-id <run-id> --include-matmul
```

Collect the next recommended matmul evidence set, including the float16
tile-shape plus launch-configuration sweep and a focused matmul Nsight Compute
profile:

```bash
./scripts/live-benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

Capture focused Nsight Compute evidence for the current memory bottleneck, a
known fused-kernel win, and the current matmul tiled-dot target:

```bash
./scripts/live-benchmark --run-id <run-id> --with-profiling
```

Use an existing EC2 key pair only when you need to keep SSH access aligned with
your own key inventory:

```bash
./scripts/live-benchmark \
  --run-id <run-id> \
  --key-name <key-pair-name> \
  --key-file <key-file.pem>
```

## Manual Host Flow

Use `scripts/up` and `scripts/down` when you want to inspect or operate the host
manually:

```bash
./scripts/up --key-name <key-pair-name> --key-file <key-file.pem>
ssh -i <key-file.pem> ubuntu@<public-ip>
cd ~/cuda-kernel-lab
uv run benchmark-matrix \
  --output-dir experiments/results/aws-ec2/<run-id> \
  --include-vector-add-sweep \
  --include-reduction-sweep \
  --include-matmul-sweep
uv run benchmark-report --input-dir experiments/results/aws-ec2/<run-id>
./scripts/down
```

Use `./scripts/up --skip-bootstrap` for an infra-only host without SSH
bootstrap. Use direct Terraform only when you need to inspect or customize the
plan:

```bash
terraform -chdir=infra/env/aws-gpu init
terraform -chdir=infra/env/aws-gpu plan \
  -out tfplan \
  -var 'key_name=<key-pair-name>' \
  -var 'ssh_ingress_cidr=<your-ip>/32'
terraform -chdir=infra/env/aws-gpu apply tfplan
terraform -chdir=infra/env/aws-gpu destroy
```

## Cleanup Verification

After a live run, confirm local Terraform state is empty:

```bash
terraform -chdir=infra/env/aws-gpu state list
```

The live wrapper also removes its run-specific tfvars file and temporary private
key under `.aws-gpu/live-benchmark/`.

## Profiler Starting Point

Prefer `--with-profiling` for disposable evidence runs. It captures:

- `memory-vector-add-float32`
- `memory-reduction-iterative-float32`
- `memory-reduction-two-pass-float32`
- `norms-rmsnorm-float16`
- `matmul-tiled-float16`

The matmul target uses the current Tensor Core candidate launch settings from
the benchmark CLI and should be read alongside the `--include-matmul-sweep`
rows before promoting a final tile choice.

The wrapper runs `ncu` through passwordless `sudo` because NVIDIA performance
counters are restricted for normal users on the AWS Deep Learning AMI.

For manual host work, start with the memory bottleneck and the strongest fused
win:

```bash
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-memory --backend triton --device cuda --op vector_add \
  --dtype float32 \
  --output experiments/results/aws-ec2/<run-id>-profiled/memory-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-norms --backend triton --device cuda --op rmsnorm \
  --rows 4096 --cols 4096 --dtype float16 \
  --output experiments/results/aws-ec2/<run-id>-profiled/rmsnorm-profiled.jsonl
```

Save compact profiler notes under `profiling/reports/`.

## SSH Timeout Troubleshooting

`scripts/up` discovers the default SSH ingress CIDR with
`https://checkip.amazonaws.com` and opens only that IPv4 `/32`. Some networks
route HTTPS discovery and TCP/22 through different NAT addresses. In that case
the EC2 instance boots normally, `sshd` starts, but SSH attempts time out
because the security group never sees traffic from the discovered `/32`.

Use `./scripts/up --ingress-cidr <cidr>` or
`./scripts/live-benchmark --ingress-cidr <cidr>` when your SSH egress address is
known to differ from HTTPS discovery. Keep the CIDR as narrow as your network
allows; avoid opening SSH broadly.
