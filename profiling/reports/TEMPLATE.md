# Profiling Report Template

## Summary

One sentence with the conclusion.

## Context

- Benchmark command:
- JSONL result:
- Profiler command:
- Operation/backend:
- Shape and dtype:
- Device:
- Method family and technique:
- Hypothesis:
- Knobs changed:
- Git commit:

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

Did profiler counters confirm the benchmark traffic model and the technique
hypothesis? Is the kernel memory-bound, compute-bound, launch-bound, or limited
by another resource?

## Follow-Up

What should change or be measured next?
