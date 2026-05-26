# Live GPU On Runpod

Use Runpod Pods for disposable single-GPU benchmark evidence. The normal path is
the same three-step loop as local development: start a Pod, run one or more
benchmark experiments, and tear the Pod down when you are done.

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

- local `runpodctl`, configured with `runpodctl doctor`
- local `ssh`, `ssh-keygen`, `tar`, and `python3`
- Runpod GPU availability for the selected `--gpu-id`

## Recommended Flow

Preview the Pod launch without touching Runpod:

```bash
./scripts/up --dry-run
```

Start the Pod. The first run creates or reuses the project-local SSH key and
registers the public key with Runpod.

```bash
./scripts/up
```

Run a benchmark against the Pod:

```bash
./scripts/benchmark --run-id <run-id>
```

Run more benchmark experiments without another Pod create/delete cycle:

```bash
./scripts/benchmark --run-id <second-run-id> --include-matmul-sweep
```

For the H200 Tensor Core/roofline path, launch an H200 Pod and run the named
suite with profiling:

```bash
./scripts/up --gpu-id "NVIDIA H200" --profile-counters
./scripts/benchmark --run-id <run-id> --suite h200-roofline --with-profiling
```

For timing-only H200 evidence, skip Nsight setup and counter checks explicitly:

```bash
./scripts/up --gpu-id "NVIDIA H200" --timing-only
./scripts/benchmark --run-id <run-id> --suite h200-matmul-autotune
```

Runpod bootstrap installs and validates Nsight Compute by default so
`--with-profiling` can collect `ncu` CSVs. Profiling defaults to a lightweight
counter set, a `120` second timeout, and `--profile-preset auto`; H200 suites
resolve that preset to focused matmul gap targets for roofline runs and the
stable autotune winners for `h200-matmul-autotune` runs. Rerun a single target
with `--profile-only` when the report points to one kernel:

Use `--profile-counters` for profiler runs. It validates NVIDIA performance
counter access during bootstrap and defaults the Pod to Community Cloud unless
`--cloud-type` is explicitly set. If the benchmark preflight reports
`ERR_NVGPUCTRPERM`, recreate the Pod with `--profile-counters` or choose a
datacenter/host that allows performance counters. A failed profile-counter
bootstrap deletes the newly created Pod and removes `.runpod/connection.env` by
default; pass `--keep-failed-pod` when you need to inspect that failed Pod.

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-targets matmul-llm-down-bfloat16 \
  --profile-timeout-seconds 120
```

Use `./scripts/up --no-install-nsight-compute` only for benchmark-only Pods.

After a roofline run, use the autotune suite to select stable shape-specific
matmul configs:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-matmul-autotune --with-profiling
uv run benchmark-compare \
  --baseline-dir experiments/results/runpod/<baseline-run-id> \
  --candidate-dir experiments/results/runpod/<run-id>
```

The profile phase reads the run's `h200-matmul-best.json` and profiles those
exact winner parameters, so the Nsight reports explain the selected configs
rather than the older fixed matmul probes. Matmul targets are captured through a
direct Python harness that warms Triton before enabling CUDA profiler markers,
which avoids landing Nsight on setup work or missing the profiled kernel. To
replay only those profiles:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-preset autotune-winners \
  --profile-autotune-manifest experiments/results/runpod/<run-id>/h200-matmul-best.json
```

Pass a focused candidate set through the same shell workflow when you want to
retest the remaining H200 matmul gap directly:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --suite h200-matmul-autotune \
  --matmul-autotune-shapes 512x11008x4096 \
  --matmul-autotune-configs 128x128x64x4x4x4,128x128x64x4x4x8
```

Autotune runs continue after an individual candidate failure and write
`benchmark-failures.json` beside the JSONL results.

For the current decode-step graph and dynamic batching track, skip the full
matrix and run the resident head-major KV path directly:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --only-decode-step \
  --include-decode-bucket-sweep \
  --include-decode-tail-sweep \
  --decode-attention-backend sdpa-head-major \
  --decode-dynamic-copy-mode resident \
  --decode-piecewise-post-mode eager \
  --decode-orchestration-timing off \
  --decode-tail-buckets '1,2,3,4,5,6,7,8'
```

Tear the Pod down:

```bash
./scripts/down
```

## Useful Overrides

Choose a different GPU or use Community Cloud explicitly:

```bash
./scripts/up --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY
```

Use a Runpod template instead of the default image:

```bash
./scripts/up --template-id runpod-torch-v21
```

Attach an existing network volume when you want dependencies or captures to
survive Pod replacement:

```bash
./scripts/up --network-volume-id <network-volume-id>
```

Use the legacy AWS provider only when you need to compare against the saved EC2
A10G evidence:

```bash
./scripts/up --platform aws
./scripts/benchmark --platform aws --run-id <run-id>
./scripts/down --platform aws
```

## Profiling

`./scripts/benchmark --with-profiling` still captures focused Nsight Compute
CSV exports and compact summaries. On Runpod, the script runs `ncu` directly
when the container user is root, or through passwordless `sudo` when available.

If profiling fails because `ncu` is absent, recreate the Pod with the default
bootstrap path or use an image/template that already includes Nsight Compute.
