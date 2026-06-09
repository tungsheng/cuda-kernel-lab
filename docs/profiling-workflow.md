# Profiling Workflow

Profile after a benchmark result is interesting enough to explain. Profiler
evidence should validate or challenge the benchmark interpretation, not replace
correctness checks or repeated timing.

## Before Profiling

Record the benchmark command, JSONL path, device, shape, dtype, backend,
optimization technique, hypothesis, changed knobs, and expected bottleneck.

## Runpod Setup

Runpod bootstrap installs and validates Nsight Compute by default. For profile
runs, prefer a Pod that validates NVIDIA performance-counter access during
bootstrap:

```bash
./scripts/up --profile-counters
```

Use `./scripts/up --timing-only` for benchmark-only Pods that should skip Nsight
setup and counter checks.

## Capture Profiles

Capture focused profiles alongside a benchmark run:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

Defaults:

- `--profile-mode light`: captures counters used by `nsight-summary`.
- `--profile-timeout-seconds 120`: keeps live runs from hanging indefinitely.
- `--profile-preset auto`: resolves to `broad`, `decode`, `matmul-gaps`, or
  `autotune-winners` based on the selected suite.

Use H200 suite commands from [Benchmark Workflow](benchmark-workflow.md) when
profiling roofline or autotune evidence. The automatic profile preset focuses
H200 roofline runs on matmul gaps and H200 autotune runs on selected winners.

## Profile-Only Replay

Rerun one target after a timeout or after a report identifies a specific kernel:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-targets matmul-llm-down-bfloat16 \
  --profile-timeout-seconds 120
```

Rerun only H200 autotune winner profiles:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-preset autotune-winners \
  --profile-autotune-manifest experiments/results/runpod/<run-id>/h200-matmul-best.json
```

Matmul profile targets use `cuda_kernel_lab.profile_capture` with CUDA profiler
start/stop markers, so Triton JIT warmup stays outside the profiled region.

## Artifacts

- Raw Nsight exports and stderr logs: `profiling/nsight_compute/<run-id>/`.
- Compact profiler notes: `profiling/reports/<run-id>/`.
- Large binary captures: ignored by default; do not commit them.

Convert a small Nsight CSV or text export into a starter note:

```bash
uv run nsight-summary \
  --input profiling/nsight_compute/vector-add.csv \
  --output profiling/reports/vector-add-a10g.md \
  --benchmark-command "uv run benchmark-memory --backend triton --device cuda --op vector_add --dtype float32" \
  --result-jsonl experiments/results/runpod/<run-id>-profiled/memory-profiled.jsonl \
  --operation vector_add \
  --strategy triton-block-size
```

## What To Look For

- achieved memory throughput and load/store behavior
- occupancy and launch configuration
- register pressure and shared memory usage
- Tensor Core or tensor-pipe utilization for matmul validation
- cache behavior when parameter vectors are reused
- whether measured counters agree with the analytical traffic model

Use [profiling/reports/TEMPLATE.md](../profiling/reports/TEMPLATE.md) for the
final writeup.
