# Live GPU On AWS EC2

Use AWS EC2 for disposable single-GPU benchmark evidence. The normal path is a
three-step loop: start a `g5.xlarge`, run one or more benchmark experiments, and
tear the host down when you are done.

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

Preview the host launch without touching AWS:

```bash
./scripts/up --dry-run
```

Start the GPU host. With no key arguments, `scripts/up` creates or reuses the
default project key at `.aws-gpu/keys/cuda-kernel-lab-${USER}.pem` and writes
connection metadata to `.aws-gpu/connection.env`.

```bash
./scripts/up
```

Run a benchmark against the host:

```bash
./scripts/benchmark --run-id <run-id>
```

Run more benchmark experiments without another Terraform apply/destroy cycle:

```bash
./scripts/benchmark --run-id <second-run-id> --include-matmul-sweep
```

For decode-step graph and dynamic batching iterations, skip the full matrix:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --only-decode-step \
  --include-decode-bucket-sweep \
  --include-decode-tail-sweep
```

Capture the current recommended evidence bundle before moving from matmul into
attention work:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --include-matmul-sweep \
  --include-rmsnorm-shape-sweep \
  --include-attention-baseline \
  --include-decode-step \
  --with-profiling
```

If you edit local kernels or benchmark code while the host is still running,
rerun `./scripts/up` to resync and re-bootstrap the remote repo before the next
`./scripts/benchmark`.

Tear the host down:

```bash
./scripts/down
```

The default teardown removes generated Terraform variables and generated
connection metadata, so the next `./scripts/up` starts with fresh host outputs.

If SSH times out on a network where HTTPS and SSH use different carrier NAT
egress addresses, override the auto-discovered `/32` with the SSH-visible CIDR:

```bash
./scripts/up --ingress-cidr <ssh-egress-cidr>
```

Add matmul progression numbers when needed:

```bash
./scripts/benchmark --run-id <run-id> --include-matmul
```

Collect the next recommended matmul evidence set, including the float16
tile-shape plus launch-configuration sweep and a focused matmul Nsight Compute
profile:

```bash
./scripts/benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

Capture focused Nsight Compute evidence for the current memory bottleneck, a
known fused-kernel win, and the current matmul tiled-dot target:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

Use an existing EC2 key pair only when you need to keep SSH access aligned with
your own key inventory:

```bash
./scripts/up \
  --key-name <key-pair-name> \
  --key-file <key-file.pem>
./scripts/benchmark --run-id <run-id> --key-file <key-file.pem>
./scripts/down
```

## Manual Host Flow

Use `scripts/up` and `scripts/down` when you want to inspect or operate the host
manually:

```bash
./scripts/up
ssh -i .aws-gpu/keys/cuda-kernel-lab-${USER}.pem ubuntu@<public-ip>
cd ~/cuda-kernel-lab
uv run benchmark-matrix \
  --output-dir experiments/results/aws-ec2/<run-id> \
  --include-vector-add-sweep \
  --include-reduction-sweep \
  --include-matmul-sweep \
  --include-rmsnorm-shape-sweep \
  --include-attention-baseline \
  --include-decode-step
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
  -var 'key_name=cuda-kernel-lab-${USER}' \
  -var 'ssh_ingress_cidr=<your-ip>/32'
terraform -chdir=infra/env/aws-gpu apply tfplan
terraform -chdir=infra/env/aws-gpu destroy
```

## Cleanup Verification

After `./scripts/down`, confirm local Terraform state is empty:

```bash
terraform -chdir=infra/env/aws-gpu state list
```

The reusable dev key under `.aws-gpu/keys/` is left in place for the next
`./scripts/up`.

## Profiler Starting Point

Prefer `--with-profiling` when a benchmark needs profiler evidence. It captures:

- `memory-vector-add-float32`
- `memory-reduction-iterative-float32`
- `memory-reduction-two-pass-float32`
- `norms-rmsnorm-float16`
- `matmul-tiled-float16`

The matmul target uses the current Tensor Core candidate launch settings from
the benchmark CLI and should be read alongside the `--include-matmul-sweep`
rows before promoting a final tile choice.

When `--include-decode-step` or `--only-decode-step` is set, profiling adds the
fixed-shape synthetic decode-step modes plus dynamic eager, ordered piecewise
graph, and same-stream piecewise graph targets. Use those Nsight summaries for
the occupancy and HBM-counter side of the graph replay comparison. The matrix
also writes a dynamic trace to `decode-step-dynamic.jsonl` for scheduler metrics
such as graph hit rate, padding waste, scheduler decision latency,
phase/bucket latency breakdowns, and host-side orchestration timing. Add
`--include-decode-tail-sweep` when p95/p99 stability is the experiment
question; it runs longer same-stream dynamic traces across multiple seeds for
the default low-padding, middle, and dense bucket policies.
Use `--decode-tail-buckets '1,2,4,8;1,2,3,4,6,8'` to compare a custom policy
set without editing the matrix code.
Use `--decode-attention-backend sdpa --decode-dynamic-copy-mode x-only` for
the resident-KV-cache dynamic piecewise graph experiment, where only the
current activation is staged into graph-owned buffers. Switch to
`--decode-dynamic-copy-mode resident` for a synthetic fully-resident upper-bound
run with no per-step input staging, and add `--decode-piecewise-post-mode eager`
when testing whether the post-attention add is better left outside the captured
graph.
Add `--decode-orchestration-timing off` when comparing the production-like
dynamic hot loop without per-region host timing probes.

The benchmark script runs `ncu` through passwordless `sudo` because NVIDIA performance
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

Use `./scripts/up --ingress-cidr <cidr>` when your SSH egress address is known
to differ from HTTPS discovery. Keep the CIDR as narrow as your network allows;
avoid opening SSH broadly.
