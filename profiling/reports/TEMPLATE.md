# Profiling Report Template

## Summary

One sentence with the conclusion.

## Benchmark Context

- Command:
- JSONL result:
- Operation:
- Backend:
- Optimization strategy:
- Shape:
- Dtype:
- Device:
- Git commit:

## Profiler Command

```bash

```

## Key Metrics

Record only counters that affect the conclusion.

- Latency:
- Memory throughput:
- Global load/store behavior:
- Occupancy:
- Registers:
- Shared memory:
- Tensor Core utilization:
- Cache notes:

## Interpretation

Is the kernel memory-bound or compute-bound? Did profiler counters confirm the
benchmark traffic model and the optimization strategy?

## Follow-Up

What should change or be measured next?
