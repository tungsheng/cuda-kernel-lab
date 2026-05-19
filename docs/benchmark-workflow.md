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

Use `--backend torch` for a PyTorch-only baseline. Use `--backend triton` when
you only want the custom Triton implementation.

## Run The First Matrix

For the first AWS EC2 evidence run, print the full matrix first:

```bash
uv run benchmark-matrix --dry-run
```

Then run it on the CUDA host:

```bash
uv run benchmark-matrix
```

The default matrix runs memory, softmax, and normalization benchmarks for
`float32` and `float16`, writing JSONL records under
`experiments/results/aws-ec2-first-run/`.

Generate the first report from those JSONL records:

```bash
uv run benchmark-report --input-dir experiments/results/aws-ec2-first-run
```

For the first `vector_add` strategy sweep, vary the Triton block size:

```bash
uv run benchmark-memory --backend triton --device cuda --op vector_add --dtype float32 --block-size 512
uv run benchmark-memory --backend triton --device cuda --op vector_add --dtype float32 --block-size 1024
uv run benchmark-memory --backend triton --device cuda --op vector_add --dtype float32 --block-size 2048
```

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

Keep large result files under `experiments/results/`. That directory is ignored
by default.

## Promote A Result

When a result is worth keeping:

1. Write a short note with [experiments/TEMPLATE.md](../experiments/TEMPLATE.md).
2. Add profiler details with [profiling/reports/TEMPLATE.md](../profiling/reports/TEMPLATE.md)
   if profiler data was collected.
3. Summarize only the smallest useful table in a concept doc.
