# Inference Kernel Lab

A hands-on GPU kernel engineering lab for LLM inference primitives, focused on
memory bandwidth, kernel fusion, Tensor Core utilization, KV cache layout, and
decode-time bottlenecks.

This project studies LLM inference performance from GPU primitives to
serving-system behavior:

- PyTorch baseline
- Triton kernel
- CUDA C++ kernel
- benchmark
- profiler analysis
- inference-system lesson

The goal is not to collect random CUDA exercises. Each kernel should connect
back to an inference bottleneck and answer a small set of performance questions:

- Is it memory-bound or compute-bound?
- How much high bandwidth memory traffic does it move?
- How much theoretical bandwidth did it achieve?
- What did fusion save?

## Project Layout

```text
inference-kernel-lab/
├── README.md
├── pyproject.toml
├── src/
│   └── inference_kernel_lab/
│       ├── benchmark.py
│       ├── benchmark_cli.py
│       ├── device.py
│       ├── gpu_info.py
│       ├── metrics.py
│       ├── ops/
│       ├── kernels/
│       └── benchmarks/
├── tests/
├── profiling/
│   ├── nsight_compute/
│   └── reports/
├── experiments/
├── docs/
└── scripts/
```

## Quick Start

Create the local environment and install development tools:

```bash
uv sync --group dev
```

Install the PyTorch/Triton extra before running kernel tests or benchmarks:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
```

Triton is installed only on Linux CUDA hosts. On other platforms the PyTorch
baseline can still run once the `gpu` extra installs PyTorch.

Run the correctness tests:

```bash
uv run pytest
```

Inspect the available device:

```bash
uv run gpu-info
```

Run the Milestone 0/1 memory benchmark:

```bash
uv run python -m inference_kernel_lab.benchmarks.memory_bandwidth --backend all --op all --numel 16777216 --dtype float32
```

Run the Milestone 2 softmax benchmark:

```bash
uv run python -m inference_kernel_lab.benchmarks.softmax --backend all --rows 4096 --cols 1024 --dtype float32
```

Run the Milestone 3 normalization benchmark:

```bash
uv run python -m inference_kernel_lab.benchmarks.norms --backend all --op all --rows 4096 --cols 4096 --dtype float32
```

Append reproducible JSONL records with run metadata:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

Example output columns:

```text
name                   device    dtype      p50_ms   p95_ms   p99_ms   GB/s    TFLOP/s
torch:copy             cuda      float32    ...
triton:copy            cuda      float32    ...
torch:scale            cuda      float32    ...
triton:scale           cuda      float32    ...
```

## CUDA Host Workflow

Use this checklist before collecting benchmark numbers:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
uv run pytest
uv run python -m inference_kernel_lab.benchmarks.memory_bandwidth --backend all --device cuda --op all
uv run python -m inference_kernel_lab.benchmarks.softmax --backend all --device cuda
uv run python -m inference_kernel_lab.benchmarks.norms --backend all --device cuda --op all
```

`--backend all` runs PyTorch and Triton when CUDA/Triton are available. On a
CPU-only host it runs the PyTorch backend only, so local development still stays
fast and boring in the best possible way.

Use `--output <path>.jsonl` for benchmark runs that should be compared later.
Each output line includes the command, arguments, git commit, dirty flag, host,
package versions, visible CUDA devices, raw latencies, and derived metrics.

## Milestones

### Milestone 0: Setup and Benchmarking Discipline

Deliverables:

- pytest correctness tests
- benchmark runner
- GPU info script
- p50 / p95 / p99 latency output
- bandwidth estimate
- TFLOPS estimate

### Milestone 1: Memory Bandwidth Primitives

Initial kernels:

- `copy`
- `scale`
- `vector_add`
- `reduction_sum`

Current backends:

- PyTorch baseline
- Triton CUDA kernels for the same primitive set

Expected lesson: simple elementwise kernels are usually limited by high
bandwidth memory traffic, not floating point throughput.

### Milestone 2: Softmax

Compare a PyTorch baseline, Triton implementation, and CUDA implementation.
The core lesson is row-wise reduction and kernel fusion:

```text
naive softmax:
read input -> write intermediate -> read intermediate -> write output

fused softmax:
read input -> reduce -> normalize -> write output
```

Current backends:

- PyTorch baseline
- Triton fused row-wise softmax

Benchmark:

```bash
uv run python -m inference_kernel_lab.benchmarks.softmax --backend all --device cuda --rows 4096 --cols 1024
```

### Milestone 3: RMSNorm / LayerNorm

Forward-only normalization kernels for inference. Focus on reductions,
vectorized loads, epsilon stability, and FP16/BF16 behavior.

Current backends:

- PyTorch RMSNorm and LayerNorm baselines
- Triton fused row-wise RMSNorm and LayerNorm forward kernels

Benchmark:

```bash
uv run python -m inference_kernel_lab.benchmarks.norms --backend all --device cuda --op all --rows 4096 --cols 4096
```

### Milestone 4: SwiGLU Fusion

Start with the elementwise fusion:

```text
silu(gate) * up
```

The point is to quantify memory traffic saved by keeping intermediate values in
registers instead of writing and rereading activation tensors.

### Milestone 5: Matrix Multiplication Progression

Implement naive matmul, shared-memory tiled matmul, Triton matmul, and
optionally a Tensor Core version. The purpose is to explain why tiling improves
arithmetic intensity, not to beat cuBLAS early.

### Milestone 6: KV Cache Layout Experiments

Build a small simulator for contiguous and paged KV cache layouts, block table
lookup, and fragmentation behavior.

### Milestone 7: Attention Decode Microkernel

Start with single-batch, single-head, fixed-sequence decode attention, then
expand toward multi-head, batched sequences, and paged KV cache integration.

### Milestone 8: Mini Inference Scheduler Simulator

Model requests, prompt length, decode length, batching, active sequences, KV
cache usage, queue depth, time-to-first-token, inter-token latency, and p95
latency.

### Milestone 9: Final Integration Demo

Build a controlled toy inference loop using selected custom kernels with a
benchmark dashboard and profiling report.

## Success Criteria

This project is successful if the results and docs make it easy to explain:

- why decode is memory-bound
- why kernel fusion helps
- why tiled matmul improves reuse
- why KV cache layout affects serving throughput
- why faster kernels do not automatically solve tail latency

## License

MIT
