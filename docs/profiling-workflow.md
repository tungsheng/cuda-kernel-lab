# Profiling Workflow

Use profiler runs after a benchmark result is interesting enough to explain.
The profiler should validate or challenge the benchmark interpretation.

## Before Profiling

Record:

- benchmark command
- result JSONL path
- device and driver context
- shape, dtype, backend, and operation
- optimization strategy being tested
- expected bottleneck

## Nsight Compute

Example command shape:

```bash
ncu --set full --target-processes all uv run benchmark-memory --backend triton --device cuda --op vector_add
```

First profiler targets:

```bash
ncu --set full --target-processes all uv run benchmark-memory --backend triton --device cuda --op vector_add --dtype float32 --output experiments/results/aws-ec2-first-run-profiled/memory-profiled.jsonl
ncu --set full --target-processes all uv run benchmark-softmax --backend triton --device cuda --rows 4096 --cols 1024 --dtype float32 --output experiments/results/aws-ec2-first-run-profiled/softmax-profiled.jsonl
```

Large binary captures are ignored by default. Commit compact text summaries in
`profiling/reports/`.

## What To Look For

- achieved memory throughput
- global load/store efficiency
- occupancy and launch configuration
- register pressure
- shared memory usage
- cache behavior when parameter vectors are reused
- whether measured traffic agrees with the analytical model

Use [profiling/reports/TEMPLATE.md](../profiling/reports/TEMPLATE.md) for the
writeup.
