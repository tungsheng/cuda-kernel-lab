# Documentation

Use the root README to get running. Use these docs to understand what to
measure, how to save results, and how to turn runs into conclusions.

## Workflows

- [Benchmark Workflow](benchmark-workflow.md): run benchmarks locally or on CUDA
  hosts and save JSONL results.
- [Interpreting Results](interpreting-results.md): read latency, bandwidth,
  FLOP, and traffic-model numbers.
- [Profiling Workflow](profiling-workflow.md): collect compact Nsight notes and
  profiler summaries.
- [Milestones](milestones.md): project roadmap and current implementation status.

## Concepts

- [GPU Execution Model](concepts/gpu-execution-model.md)
- [Roofline Analysis](concepts/roofline-analysis.md)
- [Memory Hierarchy](concepts/memory-hierarchy.md)
- [Kernel Fusion](concepts/kernel-fusion.md)
- [KV Cache Layout](concepts/kv-cache-layout.md)
- [Inference System Lessons](concepts/inference-system-lessons.md)

## Documentation Rule

Keep docs simple:

- root README explains how to run the project
- workflow docs explain how to do repeatable work
- concept docs explain why a result matters
- experiment and profiling templates keep raw observations consistent
