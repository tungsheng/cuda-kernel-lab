# CUDA Kernel Lab

CUDA Kernel Lab is a profiling-driven GPU optimization lab for LLM-shaped
primitives. It focuses on reproducible benchmark evidence, profiler validation,
and clear performance conclusions.

This repo studies kernel paths: memory traffic, reductions, fusion,
launch/config tuning, Tensor Core validation, and synthetic decode-step replay.
Full inference serving, request scheduling, queueing, and cluster experiments
belong outside this repo.

## Quick Start

Create the development environment:

```bash
uv sync --group dev
```

Run local checks:

```bash
uv run pytest
uv run ruff check .
```

Install GPU dependencies before CUDA/Triton benchmarks:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
```

Run a small local benchmark:

```bash
uv run benchmark-memory --backend all --op all --numel 16777216 --dtype float32
```

## Live GPU Loop

Runpod is the default live-GPU provider. A standard evidence run is:

```bash
./scripts/up
./scripts/benchmark --run-id <run-id>
./scripts/down
```

The benchmark script copies raw JSONL to `experiments/results/runpod/<run-id>/`
and writes a generated report to `experiments/reports/runpod/<run-id>.md`.
Use `--platform aws` only for the legacy EC2 fallback.

For focused suites, decode-step runs, H200 matmul autotune, and profiling
replay, use the workflow docs linked below.

## What Is Here

```text
src/cuda_kernel_lab/
├── benchmarks/     # benchmark entry points
├── kernels/        # PyTorch, Triton, and CUDA kernel trees
├── ops/            # backend-neutral traffic and FLOP models
├── benchmark.py    # timing and result metadata
└── benchmark_cli.py
```

Implemented kernel evidence tracks:

- memory primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
- fused row-wise softmax
- RMSNorm and LayerNorm forward kernels
- fused SwiGLU elementwise activation
- tiled matmul progression and Tensor Core validation sweeps
- contiguous KV-cache decode-attention PyTorch baseline
- synthetic decode-step benchmark for eager, full-graph, piecewise-graph, and
  dynamic trace replay

## Read Next

- [Documentation map](docs/README.md)
- [Project architecture](docs/project-architecture.md)
- [Benchmark workflow](docs/benchmark-workflow.md)
- [Runpod live GPU workflow](docs/live-gpu-runpod.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Interpreting results](docs/interpreting-results.md)
- [Optimization strategies](docs/optimization-strategies.md)
- [Optimization techniques](docs/optimization-techniques.md)
- [Legacy AWS EC2 workflow](docs/live-gpu-aws-ec2.md)

## License

MIT
