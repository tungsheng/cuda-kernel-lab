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

For the standard AWS evidence path, start a host with `./scripts/up`, then let
the benchmark script collect focused profiles and compact summaries:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

Example command shape:

```bash
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-memory --backend triton --device cuda --op vector_add
```

On the AWS Deep Learning AMI, run `ncu` with passwordless `sudo`; otherwise
Nsight Compute can fail with NVIDIA performance-counter permission errors.

Suggested profiler targets:

```bash
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-memory --backend triton --device cuda --op vector_add \
  --dtype float32 \
  --output experiments/results/aws-ec2/<run-id>-profiled/memory-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-softmax --backend triton --device cuda \
  --rows 4096 --cols 1024 --dtype float32 \
  --output experiments/results/aws-ec2/<run-id>-profiled/softmax-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-swiglu --backend triton --device cuda \
  --rows 4096 --cols 4096 --dtype float32 \
  --output experiments/results/aws-ec2/<run-id>-profiled/swiglu-profiled.jsonl
sudo -n env HOME="$HOME" PATH="$PATH" ncu --set full --target-processes all \
  uv run benchmark-matmul --backend triton --device cuda \
  --m 1024 --n 1024 --k 1024 --dtype float16 \
  --block-m 64 --block-n 64 --block-k 32 \
  --num-warps 4 --num-stages 3 --input-precision tf32 \
  --output experiments/results/aws-ec2/<run-id>-profiled/matmul-profiled.jsonl
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
  --result-jsonl experiments/results/aws-ec2/<run-id>-profiled/memory-profiled.jsonl \
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
