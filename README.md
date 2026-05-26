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
uv run benchmark-matrix --output-dir experiments/results/runpod/<run-id>
uv run benchmark-report --input-dir experiments/results/runpod/<run-id>
```

For Runpod evidence runs, start one GPU Pod, run as many benchmark experiments
as you need, then tear it down. The first `up` creates or reuses a project-local
SSH key under `.runpod/keys/`. Pass `--platform aws` to use the legacy EC2
provider.

```bash
./scripts/up
./scripts/benchmark --run-id <run-id>
./scripts/down
```

Run the current decode-step graph and dynamic batching workflow without the full
matrix:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --only-decode-step \
  --include-decode-bucket-sweep \
  --include-decode-tail-sweep \
  --decode-attention-backend sdpa-head-major \
  --decode-dynamic-copy-mode resident \
  --decode-piecewise-post-mode eager \
  --decode-orchestration-timing off \
  --decode-tail-buckets '1,2,3,4,5,6,7,8'
```

That command matches the latest saved A10G decode evidence: resident
head-major KV views, same-stream piecewise CUDA Graph replay, eager post-add,
and production-like hot-loop timing. Keep the full `1,2,3,4,5,6,7,8` bucket set
when the experiment should avoid padding; pass a semicolon-separated
`--decode-tail-buckets` list when comparing coarser bucket policies:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --only-decode-step \
  --include-decode-tail-sweep \
  --decode-tail-buckets '1,2,4,8;1,2,3,4,6,8'
```

Use `--decode-dynamic-copy-mode x-only` only when the experiment should model a
resident KV cache while still staging the current activation. Use the default
orchestration timing when you need per-region host breakdowns.

Add focused Nsight Compute profiler captures to a benchmark run:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

Profiling defaults to a lightweight metric set, a `120` second per-target
timeout, and an automatic target preset. H200 roofline and Tensor Core suites
default to the `matmul-gaps` preset instead of profiling every broad smoke
target. The H200 autotune suite defaults to `autotune-winners`, which profiles
the exact configs selected in `h200-matmul-best.json`. Rerun only the profiler
for a specific target when the benchmark results already exist or a previous
profile run timed out:

```bash
./scripts/benchmark \
  --run-id <run-id> \
  --profile-only \
  --profile-targets matmul-llm-down-bfloat16 \
  --profile-timeout-seconds 120
```

Move into the matmul/Tensor Core track with the focused tile and launch sweep:

```bash
./scripts/benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

Run the H200 roofline/Tensor Core suite when the goal is CUDA kernel benchmark
depth rather than the default broad smoke matrix:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-roofline --with-profiling
```

Runpod bootstrap installs Nsight Compute by default so profiling can preflight
`ncu` before the benchmark starts. Use `./scripts/up --timing-only` for
benchmark-only H200 Pods that should skip Nsight setup, and use
`./scripts/up --profile-counters` for profiler runs that must validate NVIDIA
performance-counter access during bootstrap. The H200 suite also includes
focused matmul tuning rows for the square and asymmetric LLM GEMM shapes, plus
grouped program-ordering rows in `matmul-llm-impact.jsonl`.

Use the H200 autotune suite when the next question is the best stable matmul
configuration for the measured shapes:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-matmul-autotune --with-profiling
uv run benchmark-compare \
  --baseline-dir experiments/results/runpod/<baseline-run-id> \
  --candidate-dir experiments/results/runpod/<run-id>
```

The autotune run writes repeated shuffled rows to `matmul-autotune.jsonl` and
selects stable winners in `h200-matmul-best.json`; with `--with-profiling`,
Nsight summaries are captured for those winners under
`profiling/reports/<run-id>/`. The default H200 candidate
set avoids the shared-memory-heavy `block_k=128` tile that exceeds the observed
H200 per-block shared-memory limit; pass `--matmul-autotune-configs` to
`scripts/benchmark` when you want to test a custom list. Autotune runs use
matrix keep-going mode, so an invalid candidate is recorded in
`benchmark-failures.json` and the remaining candidates still produce a report.

Use `--platform aws` and explicit key arguments only when you need to align with
an existing EC2 key pair:

```bash
./scripts/up --platform aws --key-name <key-pair-name> --key-file <key-file.pem>
./scripts/benchmark --platform aws --run-id <run-id> --key-file <key-file.pem>
./scripts/down --platform aws
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
- synthetic decode-step benchmark for eager, full-graph, piecewise-graph, and dynamic trace replay

This repo focuses on kernel optimization strategy: memory coalescing,
vectorization, reductions, fusion, launch tuning, profiler validation, and
eventually Tensor Core matmul. Broader inference-system experiments belong in
`gpu-inference-lab`.

## Read Next

- [Docs map](docs/README.md)
- [Benchmark workflow](docs/benchmark-workflow.md)
- [Optimization techniques](docs/optimization-techniques.md)
- [Runpod live GPU workflow](docs/live-gpu-runpod.md)
- [AWS EC2 legacy GPU workflow](docs/live-gpu-aws-ec2.md)
- [Optimization strategies](docs/optimization-strategies.md)
- [Interpreting results](docs/interpreting-results.md)
- [Profiling workflow](docs/profiling-workflow.md)
- [Milestones](docs/milestones.md)

## License

MIT
