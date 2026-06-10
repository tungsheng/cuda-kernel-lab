# Benchmark Workflow

Benchmarks answer one question at a time: how fast did this backend run this
primitive for this shape and dtype, and which optimization strategy explains the
result?

A single CUDA host is enough for the kernel microbenchmarks in this repo. Use
Runpod only when you need disposable live-GPU evidence.

## Local Checks

Run these before collecting or comparing numbers:

```bash
uv run pytest
uv run ruff check .
```

On a CUDA host, install GPU dependencies and confirm the visible device:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
```

## Primitive Benchmarks

Use `--backend all` when comparing PyTorch controls against implemented Triton
kernels. Use `--backend torch` or `--backend triton` when isolating one
implementation. The attention benchmark currently supports the PyTorch
contiguous-KV baseline only.

```bash
uv run benchmark-memory --backend all --device cuda --op all
uv run benchmark-softmax --backend all --device cuda --rows 4096 --cols 1024
uv run benchmark-norms --backend all --device cuda --op all --rows 4096 --cols 4096
uv run benchmark-swiglu --backend all --device cuda --rows 4096 --cols 4096
uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024
uv run benchmark-attention --backend torch --device cuda --seq-len 2048 --num-heads 16 --head-dim 128 --dtype float16
```

For Triton matmul, record tile shape, launch configuration, and `tl.dot` input
precision:

```bash
uv run benchmark-matmul --backend all --device cuda \
  --m 1024 --n 1024 --k 1024 --dtype float16 \
  --block-m 64 --block-n 64 --block-k 32 \
  --num-warps 4 --num-stages 4 --input-precision tf32
```

## Matrix Runs

Preview the matrix first:

```bash
uv run benchmark-matrix --dry-run
```

Run the default matrix and generate a report:

```bash
uv run benchmark-matrix --output-dir experiments/results/runpod/<run-id>
uv run benchmark-report --input-dir experiments/results/runpod/<run-id>
```

The default matrix covers memory, softmax, normalization, and SwiGLU for
`float32` and `float16`. Write related sweeps into the same run directory so
`benchmark-report` can summarize one evidence note.

Optional matrix flags:

- `--include-matmul`: default tiled matmul progression rows.
- `--include-matmul-sweep`: focused float16 tile and launch sweep for Tensor
  Core validation.
- `--include-rmsnorm-shape-sweep`: hidden-size and batch-size coverage for the
  normalization fusion track.
- `--include-attention-baseline`: contiguous KV-cache decode-attention baseline.
- `--include-decode-step`: synthetic naive/fused eager, whole-step graph,
  piecewise graph, and dynamic trace rows.

## Focused Suites

Run the primary synthetic resident-KV decode path when the question is CUDA
Graph replay, dynamic buckets, padding waste, or hot-loop timing:

```bash
uv run benchmark-matrix \
  --output-dir experiments/results/runpod/<run-id> \
  --only-decode-step \
  --include-decode-bucket-sweep \
  --include-decode-tail-sweep \
  --decode-attention-backend sdpa-head-major \
  --decode-dynamic-copy-mode resident \
  --decode-piecewise-post-mode eager \
  --decode-orchestration-timing off \
  --decode-tail-buckets '1,2,3,4,5,6,7,8'
```

In this focused path, `sdpa-head-major` means PyTorch SDPA over resident
head-major K/V views. The fused decode-step variants fuse the synthetic RMSNorm
and SwiGLU regions with Triton kernels; attention remains an eager PyTorch/SDPA
region between captured graph segments.

Use the H200 roofline suite for larger FP16/BF16 matmul rows, LLM-shape tuning,
grouped program-ordering candidates, attention context, and profiler-backed
roofline interpretation:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-roofline --with-profiling
```

Use the H200 matmul autotune suite after a roofline run identifies the matmul
gap:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-matmul-autotune --with-profiling
```

Autotune writes repeated candidates to `matmul-autotune.jsonl`, stable winners
to `h200-matmul-best.json`, and candidate failures to
`benchmark-failures.json` while keeping the remaining rows reportable.

Compare a candidate run against a previous baseline before promoting a new
kernel or config:

```bash
uv run benchmark-compare \
  --baseline-dir experiments/results/runpod/<baseline-run-id> \
  --candidate-dir experiments/results/runpod/<run-id> \
  --max-regression-pct 5
```

Use [Profiling Workflow](profiling-workflow.md) for profile-only replay and
Nsight details.

## Output Files

Common outputs:

- `experiments/results/<provider>/<run-id>/*.jsonl`: raw benchmark records.
- `experiments/reports/<provider>/<run-id>.md`: generated benchmark report.
- `profiling/nsight_compute/<run-id>/`: Nsight CSV, stderr, and benchmark logs.
- `profiling/reports/<run-id>/`: compact profiler notes.

Important matrix files:

| File | Contents |
| --- | --- |
| `memory.jsonl` | default memory primitive rows |
| `vector-add-block-size.jsonl` | vector-add launch/block-size sweep |
| `reduction-strategy.jsonl` | iterative vs two-pass reduction rows |
| `matmul.jsonl` | default matmul rows |
| `matmul-tile-shape.jsonl` | focused float16 tile and launch sweep |
| `matmul-tensor-core.jsonl` | large FP16/BF16 Tensor Core validation rows |
| `matmul-tuning.jsonl` | shape-specific matmul tuning rows |
| `matmul-llm-impact.jsonl` | grouped-ordering LLM projection comparison |
| `matmul-autotune.jsonl` | repeated H200 autotune candidates |
| `rmsnorm-shape-sweep.jsonl` | RMSNorm shape scaling rows |
| `attention.jsonl` | contiguous KV-cache attention baseline |
| `decode-step.jsonl` | fixed-shape synthetic decode rows |
| `decode-step-dynamic*.jsonl` | dynamic traces, bucket sweeps, and tail sweeps |

Each JSONL record includes command metadata, git state, host/package/device
metadata, raw latencies, p50/p95/p99, GB/s, TFLOP/s, optimization metadata, and
correctness status.

## Promote A Result

When a result is worth keeping:

1. Keep large raw records under ignored `experiments/results/`.
2. Write a short note with [experiments/TEMPLATE.md](../experiments/TEMPLATE.md).
3. Add profiler details with [profiling/reports/TEMPLATE.md](../profiling/reports/TEMPLATE.md)
   if profiler data was collected.
4. Summarize only durable lessons in `docs/` concept or workflow pages.
