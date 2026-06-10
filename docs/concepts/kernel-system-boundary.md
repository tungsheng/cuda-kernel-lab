# Kernel/System Boundary

This repo studies kernel optimization strategies. It uses inference-shaped
primitives because they are realistic GPU workloads, not because this repo is an
inference serving lab.

## In Scope Here

- memory bandwidth and coalescing
- reductions
- fusion
- launch/config tuning
- occupancy and register tradeoffs
- Tensor Core matmul strategy
- profiler-backed kernel interpretation
- synthetic decode-step replay for measuring kernel and launch behavior

## Out Of Scope Here

Keep these in `gpu-inference-lab`:

- request scheduling
- queueing
- dynamic batching policy
- service-level tail latency
- cluster deployment
- full inference-loop experiments

## Boundary Rule

If the question is "How do we make this kernel faster and prove why?", it
belongs here.

If the question is "How does this affect users, queues, batches, or serving
throughput?", it belongs in `gpu-inference-lab`.

Synthetic dynamic-shape traces are in scope when they replay kernel paths,
bucket choices, or CUDA Graph launch behavior. Service-level policies that
decide how real requests wait, batch, or shed load are out of scope.
