# Profiling Workflow

Use profiler runs after a benchmark result is interesting enough to explain.
The profiler should validate or challenge the benchmark interpretation.

## Before Profiling

Record:

- benchmark command
- result JSONL path
- device and driver context
- shape, dtype, backend, and operation
- expected bottleneck

## Nsight Compute

Example command shape:

```bash
ncu --set full --target-processes all uv run benchmark-memory --backend triton --device cuda --op vector_add
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
