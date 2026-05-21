# Benchmark Workflow

Benchmarks answer one question at a time: how fast did this backend run this
primitive for this shape and dtype, and which optimization strategy explains the
result?

You do not need a live AWS cluster. A single Linux CUDA host is enough for the
kernel microbenchmarks in this repo.

## Local Checks

Run these before collecting numbers:

```bash
uv run pytest
uv run ruff check .
```

On a CUDA host, confirm the visible device:

```bash
uv sync --group dev --extra gpu
uv run gpu-info
```

## Run Benchmarks

Memory primitives:

```bash
uv run benchmark-memory --backend all --device cuda --op all
```

Softmax:

```bash
uv run benchmark-softmax --backend all --device cuda --rows 4096 --cols 1024
```

Normalization:

```bash
uv run benchmark-norms --backend all --device cuda --op all --rows 4096 --cols 4096
```

SwiGLU:

```bash
uv run benchmark-swiglu --backend all --device cuda --rows 4096 --cols 4096
```

Matmul:

```bash
uv run benchmark-matmul --backend all --device cuda --m 1024 --n 1024 --k 1024
```

For the Triton matmul path, the benchmark records tile shape, launch
configuration, and `tl.dot` input precision:

```bash
uv run benchmark-matmul --backend all --device cuda \
  --m 1024 --n 1024 --k 1024 --dtype float16 \
  --block-m 64 --block-n 64 --block-k 32 \
  --num-warps 4 --num-stages 4 --input-precision tf32
```

The default `input_precision` is `ieee` so standalone float32 correctness checks
compare strict FP32 semantics. Use `tf32` or `tf32x3` only when the experiment is
explicitly about approximate float32 Tensor Core behavior.

Use `--backend torch` for a PyTorch-only baseline. Use `--backend triton` when
you only want the custom Triton implementation.

## Run The Matrix

For a baseline-only run, print the matrix first:

```bash
uv run benchmark-matrix --dry-run
```

Then run it on the CUDA host. Without `--output-dir`, the matrix writes to
`experiments/results/aws-ec2/manual-run`.

```bash
uv run benchmark-matrix --output-dir experiments/results/aws-ec2/<run-id>
```

The default matrix runs memory, softmax, normalization, and SwiGLU benchmarks for
`float32` and `float16`. Use a run-id directory under
`experiments/results/aws-ec2/` for live GPU evidence.

Generate a report from those JSONL records:

```bash
uv run benchmark-report --input-dir experiments/results/aws-ec2/<run-id>
```

`benchmark-report` reads every `.jsonl` file in the input directory, so write
small sweeps into the same run directory when they belong to the same evidence
note.

For AWS EC2 evidence, collect the baseline matrix and strategy sweeps in one
disposable GPU session:

```bash
./scripts/live-benchmark --run-id <run-id>
```

Add `--include-matmul` when you want the default tiled matmul progression
numbers in the same run directory. Add `--include-matmul-sweep` when you want
the float16 tile-shape and launch-configuration strategy sweep that moves the
evidence track toward Tensor Core validation:

```bash
uv run benchmark-matrix \
  --output-dir experiments/results/aws-ec2/<run-id> \
  --include-matmul-sweep
```

When running the matrix manually, pass
`--include-vector-add-sweep --include-reduction-sweep --include-matmul-sweep`
and the same `--output-dir experiments/results/aws-ec2/<run-id>`.

The baseline matrix already captures the default memory block size, `1024`.
The sweep appends the additional PyTorch and Triton `vector_add` comparison
runs, `512` and `2048`, to the run's `vector-add-block-size.jsonl`. It also
appends the non-default `reduction_sum` strategy comparison to
`reduction-strategy.jsonl`. The matmul sweep appends additional float16 tile
shape, `num_warps`, and `num_stages` variants to
`matmul-tile-shape.jsonl` while keeping the default float16 matmul baseline in
`matmul.jsonl`. The sweep is scoped to float16 because the Tensor Core
milestone is about HMMA utilization; use standalone `benchmark-matmul` runs for
float32 precision experiments.

## Save Results

Append JSONL records with `--output`:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

Each record includes:

- command and parsed arguments
- git commit and dirty flag
- host and package versions
- visible CUDA device metadata
- raw latencies
- p50, p95, p99, GB/s, and TFLOP/s
- explicit strategy, variant, parameter, and optimization technique metadata
- correctness check status and max error values against the PyTorch reference

Keep large result files under `experiments/results/`. That directory is ignored
by default.

## Promote A Result

When a result is worth keeping:

1. Write a short note with [experiments/TEMPLATE.md](../experiments/TEMPLATE.md).
2. Add profiler details with [profiling/reports/TEMPLATE.md](../profiling/reports/TEMPLATE.md)
   if profiler data was collected.
3. Summarize only the smallest useful table in a concept doc.
