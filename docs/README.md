# Documentation

Use the root README to get running. Use these docs when you need the method,
context, or reporting shape behind a benchmark.

## Start Here

- [Benchmark Workflow](benchmark-workflow.md): run benchmarks and save JSONL
  records.
- [Interpreting Results](interpreting-results.md): read latency, bandwidth, and
  traffic-model columns.
- [Optimization Strategies](optimization-strategies.md): compare a baseline to
  a kernel change with evidence.
- [Profiling Workflow](profiling-workflow.md): validate benchmark conclusions
  with compact profiler notes.
- [Milestones](milestones.md): project roadmap and current implementation status.

## Concept Notes

- [GPU Execution Model](concepts/gpu-execution-model.md)
- [Roofline Analysis](concepts/roofline-analysis.md)
- [Memory Hierarchy](concepts/memory-hierarchy.md)
- [Kernel Fusion](concepts/kernel-fusion.md)
- [Occupancy and Registers](concepts/occupancy-registers.md)
- [Tensor Cores](concepts/tensor-cores.md)
- [KV Cache Layout](concepts/kv-cache-layout.md)
- [Kernel/System Boundary](concepts/kernel-system-boundary.md)

## Rule Of Thumb

Keep commands in workflow docs, interpretation in concept docs, and raw
observations in experiment or profiling notes.
