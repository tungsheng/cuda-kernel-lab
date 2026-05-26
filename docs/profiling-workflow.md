# Profiling Workflow

Use profiler runs after a benchmark result is interesting enough to explain.
The profiler should validate or challenge the benchmark interpretation.

## Before Profiling

Record:

- benchmark command
- result JSONL path
- device and driver context
- shape, dtype, backend, and operation
- optimization technique, method family, changed knobs, and hypothesis
- expected bottleneck

## Nsight Compute

For the standard Runpod evidence path, start a Pod with `./scripts/up`, then let
the benchmark script collect focused profiles and compact summaries:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

The default profile mode is `light`: it captures the counters used by
`nsight-summary` and applies a `120` second per-target timeout so live runs fail
fast enough to keep the Pod useful. `--profile-preset auto` resolves H200
roofline and Tensor Core suites to `matmul-gaps`, H200 autotune suites to
`autotune-winners`, standard suites to `broad`, and decode-only runs to
`decode`. Use `--profile-mode full` only when a target already justifies the
longer Nsight Compute collection.

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-targets matmul-llm-down-bfloat16 \
  --profile-timeout-seconds 120
```

`--profile-only` skips `benchmark-matrix` and reruns just the selected Nsight
targets, which is the fastest loop after an H200 benchmark report identifies a
specific kernel gap.

For H200 Tensor Core/roofline evidence, use the named suite. This adds larger
FP16/BF16 matmul rows, LLM-shape tuning rows, grouped-impact rows, and extra
matmul profile targets for Tensor Core counters. The automatic profile preset
captures only the matmul gap targets for this suite:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-roofline --with-profiling
```

For H200 autotune evidence, `--with-profiling` profiles the stable winners
selected in `experiments/results/runpod/<run-id>/h200-matmul-best.json`:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-matmul-autotune --with-profiling
```

When the benchmark already exists, rerun only winner profiles by pointing to the
manifest explicitly:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-preset autotune-winners \
  --profile-autotune-manifest experiments/results/runpod/<run-id>/h200-matmul-best.json
```

`scripts/benchmark --with-profiling` checks for `ncu` and runs a small Nsight
counter preflight before running the matrix. Runpod Pods created by
`./scripts/up` install and validate Nsight Compute during bootstrap unless
`--no-install-nsight-compute` is passed. For profile runs, prefer
`./scripts/up --profile-counters`; it validates NVIDIA performance-counter
access during bootstrap and defaults the Pod to Community Cloud unless
`--cloud-type` is explicitly set.

Matmul profile targets use a direct Python capture harness under Nsight Compute:
the normal benchmark command first writes the JSONL result, then `ncu` runs
`.venv/bin/python -m cuda_kernel_lab.profile_capture` with CUDA profiler
start/stop markers. That keeps Triton JIT warmup outside the profiled region and
fails the profile when the CSV only reports `No kernels were profiled`.

Add `--include-decode-step` to profile the naive/fused, full-graph, and
piecewise-graph decode-step modes alongside the standard kernel targets. Use
`--only-decode-step --with-profiling` when you want only the static and dynamic
decode-step profile targets. Use `--include-decode-tail-sweep` outside Nsight
Compute when the question is p95/p99 stability; those rows provide longer
multi-seed, multi-policy timing evidence while the profiler rows explain
individual kernels.

Example full-capture command shape:

```bash
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-memory --backend triton --device cuda --op vector_add
```

On Runpod, the benchmark script runs `ncu` directly as root or through
passwordless `sudo` when available. On the legacy AWS Deep Learning AMI, run
`ncu` with passwordless `sudo`; otherwise Nsight Compute can fail with NVIDIA
performance-counter permission errors.

Suggested profiler targets:

```bash
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-memory --backend triton --device cuda --op vector_add \
  --dtype float32 \
  --output experiments/results/runpod/<run-id>-profiled/memory-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-softmax --backend triton --device cuda \
  --rows 4096 --cols 1024 --dtype float32 \
  --output experiments/results/runpod/<run-id>-profiled/softmax-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-swiglu --backend triton --device cuda \
  --rows 4096 --cols 4096 --dtype float32 \
  --output experiments/results/runpod/<run-id>-profiled/swiglu-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-matmul --backend triton --device cuda \
  --m 1024 --n 1024 --k 1024 --dtype float16 \
  --block-m 64 --block-n 64 --block-k 32 \
  --num-warps 4 --num-stages 3 --input-precision tf32 \
  --output experiments/results/runpod/<run-id>-profiled/matmul-profiled.jsonl
```

Large binary captures are ignored by default. Commit compact text summaries in
`profiling/reports/`.

If you export a CSV or text summary from Nsight Compute, convert it to a compact
repo note:

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

- achieved memory throughput
- global load/store efficiency
- occupancy and launch configuration
- register pressure
- shared memory usage
- Tensor Core or tensor-pipe utilization for matmul
- cache behavior when parameter vectors are reused
- whether measured traffic agrees with the analytical model

Use [profiling/reports/TEMPLATE.md](../profiling/reports/TEMPLATE.md) for the
writeup.
