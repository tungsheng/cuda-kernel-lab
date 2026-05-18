# Inference Kernel Lab

Hands-on GPU kernel engineering lab for LLM inference primitives, benchmarks,
and profiling.

The project connects kernel implementations to inference bottlenecks. Each
primitive should answer a small performance question:

- Is the kernel memory-bound or compute-bound?
- How much high bandwidth memory traffic does it move?
- What fraction of practical bandwidth or throughput does it reach?
- What does fusion, tiling, or layout change save?

## Current Scope

Implemented:

- PyTorch baselines for memory primitives, softmax, RMSNorm, and LayerNorm
- Triton kernels for memory primitives, fused row-wise softmax, RMSNorm, and
  LayerNorm
- benchmark CLIs with p50, p95, p99, GB/s, TFLOP/s, and optional JSONL output
- docs for roofline analysis, memory hierarchy, fusion, KV cache layout, and
  serving-system lessons

Planned:

- SwiGLU fusion
- matmul progression from naive to tiled/Tensor Core-aware variants
- KV cache layout experiments
- decode attention microkernel
- mini scheduler simulator

## Layout

```text
inference-kernel-lab/
├── src/inference_kernel_lab/
│   ├── benchmark.py          # timing and result metadata
│   ├── benchmark_cli.py      # shared benchmark CLI helpers
│   ├── ops/                  # backend-neutral traffic/FLOP models
│   ├── kernels/              # PyTorch, Triton, and CUDA kernel trees
│   └── benchmarks/           # benchmark entry points
├── tests/                    # correctness and accounting tests
├── docs/                     # research notes and analysis guides
├── experiments/              # local result workflow notes
├── profiling/                # profiler command notes and summaries
└── scripts/
```

## Quick Start

Create the development environment:

```bash
uv sync --group dev
```

Install GPU dependencies before running kernel tests or benchmarks:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
```

Triton is installed only on Linux CUDA hosts. On other platforms, the PyTorch
baseline still works once the `gpu` extra installs PyTorch.

Run local checks:

```bash
uv run pytest
uv run ruff check .
```

Run benchmarks:

```bash
uv run benchmark-memory --backend all --op all --numel 16777216 --dtype float32
uv run benchmark-softmax --backend all --rows 4096 --cols 1024 --dtype float32
uv run benchmark-norms --backend all --op all --rows 4096 --cols 4096 --dtype float32
```

Use `--device cuda` on a CUDA host when collecting GPU numbers. `--backend all`
runs PyTorch and Triton when CUDA/Triton are available; otherwise it falls back
to the PyTorch backend.

## Capturing Results

Append durable benchmark records with `--output`:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

Each JSONL record includes:

- command and parsed arguments
- git commit and dirty flag
- host, Python, NumPy, PyTorch, and Triton versions
- visible CUDA device metadata
- raw latencies and derived p50/p95/p99, GB/s, and TFLOP/s metrics

Keep large local result files under `experiments/results/`. Promote compact
summaries into `profiling/reports/` or `docs/` when the conclusion is worth
checking into the repo.

## Documentation

Start with [docs/README.md](docs/README.md) for the research-note map.

Useful entry points:

- [docs/02-roofline-analysis.md](docs/02-roofline-analysis.md): bandwidth,
  arithmetic intensity, and benchmark interpretation
- [docs/03-memory-hierarchy.md](docs/03-memory-hierarchy.md): what to record
  from memory-system behavior
- [docs/04-kernel-fusion.md](docs/04-kernel-fusion.md): fusion questions and
  softmax traffic model
- [experiments/README.md](experiments/README.md): result capture workflow
- [profiling/reports/README.md](profiling/reports/README.md): profiler report
  checklist

## Milestones

| Milestone | Focus | Status |
| --- | --- | --- |
| 0 | setup, tests, benchmark discipline, GPU info | implemented |
| 1 | copy, scale, vector add, reduction sum | implemented |
| 2 | fused row-wise softmax | implemented |
| 3 | RMSNorm and LayerNorm forward kernels | implemented |
| 4 | SwiGLU elementwise fusion | planned |
| 5 | matmul progression and tiling | planned |
| 6 | contiguous and paged KV cache layout | planned |
| 7 | decode attention microkernel | planned |
| 8 | mini inference scheduler simulator | planned |
| 9 | integration demo with dashboard/report | planned |

## Success Criteria

The project is successful if the code and notes make it easy to explain:

- why decode is memory-bound
- why kernel fusion helps
- why tiled matmul improves reuse
- why KV cache layout affects serving throughput
- why faster kernels do not automatically solve tail latency

## License

MIT
