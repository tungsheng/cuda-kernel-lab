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

Run a local benchmark:

```bash
uv run benchmark-memory --backend all --op all --numel 16777216 --dtype float32
```

On a CUDA host, collect the standard matrix and report:

```bash
uv run benchmark-matrix --dry-run
uv run benchmark-matrix --output-dir experiments/results/aws-ec2/<run-id>
uv run benchmark-report --input-dir experiments/results/aws-ec2/<run-id>
```

For AWS evidence runs, start one GPU host, run as many benchmark experiments as
you need, then tear it down. The first `up` creates or reuses a project-local
SSH key under `.aws-gpu/keys/`.

```bash
./scripts/up
./scripts/benchmark --run-id <run-id>
./scripts/down
```

Add focused Nsight Compute profiler captures to a benchmark run:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

Move into the matmul/Tensor Core track with the focused tile and launch sweep:

```bash
./scripts/benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

Use explicit key arguments only when you need to align with an existing EC2 key
pair:

```bash
./scripts/up --key-name <key-pair-name> --key-file <key-file.pem>
./scripts/benchmark --run-id <run-id> --key-file <key-file.pem>
./scripts/down
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
- fused SwiGLU elementwise activation
- tiled matmul progression kernel
- contiguous KV-cache decode-attention PyTorch baseline
- synthetic decode-step benchmark for eager launches vs CUDA Graph replay

This repo focuses on kernel optimization strategy: memory coalescing,
vectorization, reductions, fusion, launch tuning, profiler validation, and
eventually Tensor Core matmul. Broader inference-system experiments belong in
`gpu-inference-lab`.

## Read Next

- [Docs map](docs/README.md)
- [Benchmark workflow](docs/benchmark-workflow.md)
- [Optimization techniques](docs/optimization-techniques.md)
- [AWS EC2 live GPU workflow](docs/live-gpu-aws-ec2.md)
- [Optimization strategies](docs/optimization-strategies.md)
- [Interpreting results](docs/interpreting-results.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Milestones](docs/milestones.md)

## License

MIT
