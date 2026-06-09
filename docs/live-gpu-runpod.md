# Live GPU On Runpod

Runpod is the default provider for disposable single-GPU benchmark evidence.
This page covers provider lifecycle and Runpod-specific options. Benchmark
suite choices live in [Benchmark Workflow](benchmark-workflow.md), and profiler
details live in [Profiling Workflow](profiling-workflow.md).

## Defaults

- platform: `runpod`
- GPU id: `NVIDIA L4`
- cloud type: `SECURE`
- image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- connection metadata: `.runpod/connection.env`
- SSH key: `.runpod/keys/cuda-kernel-lab-runpod-${USER}`
- raw JSONL: `experiments/results/runpod/<run-id>/`
- report: `experiments/reports/runpod/<run-id>.md`
- profile summaries: `profiling/reports/<run-id>/`

## Prerequisites

- `runpodctl`, configured with `runpodctl doctor`
- local `ssh`, `ssh-keygen`, `tar`, and `python3`
- Runpod availability for the selected `--gpu-id`

## Provider Loop

Preview the Pod launch without touching Runpod:

```bash
./scripts/up --dry-run
```

Start a Pod. The first run creates or reuses the project-local SSH key and
registers the public key with Runpod:

```bash
./scripts/up
```

Run one or more benchmark experiments against the same Pod:

```bash
./scripts/benchmark --run-id <run-id>
./scripts/benchmark --run-id <second-run-id> --include-matmul-sweep
```

Tear the Pod down:

```bash
./scripts/down
```

## Common Overrides

Choose a different GPU or cloud type:

```bash
./scripts/up --gpu-id "NVIDIA H200" --profile-counters
./scripts/up --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY
```

Use `--profile-counters` for profiler runs. It validates NVIDIA counter access
and defaults to Community Cloud unless `--cloud-type` is set. If validation
fails, the new Pod is deleted and `.runpod/connection.env` is removed unless
`--keep-failed-pod` is passed.

Use `--timing-only` for timing runs that should skip Nsight setup:

```bash
./scripts/up --gpu-id "NVIDIA H200" --timing-only
```

Use an image, template, or network volume when the default image is not enough:

```bash
./scripts/up --image <image>
./scripts/up --template-id <template-id>
./scripts/up --network-volume-id <network-volume-id>
```

Use `--no-install-nsight-compute` only for benchmark-only Pods where profiling
will not be needed.

## Legacy Provider

Use AWS only when you need historical A10G comparison data or Terraform
fallback behavior:

```bash
./scripts/up --platform aws
./scripts/benchmark --platform aws --run-id <run-id>
./scripts/down --platform aws
```

See [Live GPU On AWS EC2](live-gpu-aws-ec2.md) for the EC2-specific workflow.
