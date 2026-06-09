# Documentation

Use the root README to get running. Use this map when you need the method,
commands, or interpretation rules behind a benchmark.

## Start

- [Project Architecture](project-architecture.md): the evidence loop, track
  boundaries, and artifact locations.
- [Benchmark Workflow](benchmark-workflow.md): local benchmark commands,
  matrix suites, output files, and result promotion.
- [Live GPU On Runpod](live-gpu-runpod.md): the default disposable GPU provider
  lifecycle and provider-specific overrides.

## Run GPU Benchmarks

- [Profiling Workflow](profiling-workflow.md): Nsight setup, presets,
  profile-only replay, and compact profiler notes.
- [Live GPU On AWS EC2](live-gpu-aws-ec2.md): legacy `g5.xlarge` fallback for
  historical A10G comparisons or Terraform inspection.

## Interpret Evidence

- [Interpreting Results](interpreting-results.md): latency, bandwidth,
  TFLOP/s, traffic-model, and decode dynamic fields.
- [Optimization Techniques](optimization-techniques.md): method family,
  technique, hypothesis, knobs, and expected profiler signal.
- [Optimization Strategies](optimization-strategies.md): comparison patterns
  and recommended evidence tracks.

## Concept Notes

- [GPU Execution Model](concepts/gpu-execution-model.md)
- [Roofline Analysis](concepts/roofline-analysis.md)
- [Memory Hierarchy](concepts/memory-hierarchy.md)
- [Kernel Fusion](concepts/kernel-fusion.md)
- [Occupancy and Registers](concepts/occupancy-registers.md)
- [Tensor Cores](concepts/tensor-cores.md)
- [KV Cache Layout](concepts/kv-cache-layout.md)
- [Kernel/System Boundary](concepts/kernel-system-boundary.md)

## Reference

- [Milestones](milestones.md): roadmap and implementation status.

Keep commands in workflow docs, interpretation in concept docs, and raw
observations in experiment or profiling notes.
