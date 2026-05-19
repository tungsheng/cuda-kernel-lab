# CUDA Kernel Lab

CUDA kernel optimization strategy lab with reproducible performance benchmarks
on LLM-shaped primitives.

## Quick Start

Create the development environment:

```bash
uv sync --group dev
```

Run the local checks:

```bash
uv run pytest
uv run ruff check .
```

Install GPU dependencies before running CUDA/Triton benchmarks:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
```

Run the first benchmark:

```bash
uv run benchmark-memory --backend all --op all --numel 16777216 --dtype float32
```

On a CUDA host, collect GPU numbers explicitly:

```bash
uv run benchmark-memory --backend all --device cuda --op all
uv run benchmark-softmax --backend all --device cuda --rows 4096 --cols 1024
uv run benchmark-norms --backend all --device cuda --op all --rows 4096 --cols 4096
```

Save benchmark records for later analysis:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

## What Is Here

```text
src/cuda_kernel_lab/
├── benchmarks/     # benchmark entry points
├── kernels/        # PyTorch, Triton, and CUDA kernel trees
├── ops/            # backend-neutral traffic and FLOP models
├── benchmark.py    # timing and result metadata
└── benchmark_cli.py
```

Current implemented kernels:

- memory primitives: `copy`, `scale`, `vector_add`, `reduction_sum`
- fused row-wise softmax
- RMSNorm and LayerNorm forward kernels

This repo focuses on kernel optimization strategy: memory coalescing,
vectorization, reductions, fusion, launch tuning, profiler validation, and
eventually Tensor Core matmul. Broader inference-system experiments belong in
`gpu-inference-lab`.

## Read Next

- [Docs map](docs/README.md)
- [Benchmark workflow](docs/benchmark-workflow.md)
- [Optimization strategies](docs/optimization-strategies.md)
- [Interpreting results](docs/interpreting-results.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Milestones](docs/milestones.md)

## License

MIT
