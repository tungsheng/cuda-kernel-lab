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

Print the first live-GPU benchmark matrix:

```bash
uv run benchmark-matrix --dry-run
```

Include the first `vector_add` block-size sweep when collecting strategy
evidence:

```bash
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep --dry-run
```

Generate a report after a live GPU matrix run:

```bash
uv run benchmark-report --input-dir experiments/results/aws-ec2-first-run
```

On a CUDA host, collect GPU numbers explicitly:

```bash
uv run benchmark-memory --backend all --device cuda --op all
uv run benchmark-softmax --backend all --device cuda --rows 4096 --cols 1024
uv run benchmark-norms --backend all --device cuda --op all --rows 4096 --cols 4096
uv run benchmark-swiglu --backend all --device cuda --rows 4096 --cols 4096
uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024
```

Spin up a Terraform-managed disposable AWS EC2 GPU host when you do not have
local CUDA:

```bash
./scripts/up --key-name <key-pair-name> --key-file <key-file.pem>
```

Tear it down after collecting evidence:

```bash
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

This repo focuses on kernel optimization strategy: memory coalescing,
vectorization, reductions, fusion, launch tuning, profiler validation, and
eventually Tensor Core matmul. Broader inference-system experiments belong in
`gpu-inference-lab`.

## Read Next

- [Docs map](docs/README.md)
- [Benchmark workflow](docs/benchmark-workflow.md)
- [AWS EC2 live GPU workflow](docs/live-gpu-aws-ec2.md)
- [Optimization strategies](docs/optimization-strategies.md)
- [Interpreting results](docs/interpreting-results.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Milestones](docs/milestones.md)

## License

MIT
