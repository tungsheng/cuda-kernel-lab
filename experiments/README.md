# Experiments

Use this directory for local benchmark result records and short experiment notes
that tie measurements back to a specific question.

Recommended result output:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

Live-GPU evidence run on a disposable Runpod Pod:

```bash
./scripts/up
./scripts/benchmark --run-id <run-id>
./scripts/down
```

This writes raw JSONL under `experiments/results/runpod/<run-id>/` and a
compact report under `experiments/reports/runpod/<run-id>.md`.

Add profiler evidence when the benchmark report points at a specific bottleneck:

```bash
./scripts/benchmark --run-id <run-id> --with-profiling
```

This also writes Nsight Compute CSV exports under
`profiling/nsight_compute/<run-id>/` and compact profiler notes under
`profiling/reports/<run-id>/`.

For the matmul/Tensor Core track, include the focused tile-shape sweep:

```bash
./scripts/benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

For the H200 Tensor Core/roofline track, use the named suite:

```bash
./scripts/benchmark --run-id <run-id> --suite h200-roofline --with-profiling
```

For the current decode-step graph track, keep the note focused on resident
head-major KV cache, same-stream piecewise graph replay, and multi-seed tail
latency:

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

Manual live-GPU matrix collection:

```bash
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep --dry-run
uv run benchmark-matrix \
  --output-dir experiments/results/runpod/<run-id> \
  --include-vector-add-sweep \
  --include-reduction-sweep
uv run benchmark-report \
  --input-dir experiments/results/runpod/<run-id>
```

The `experiments/results/` directory is ignored by default because JSONL runs
can grow quickly. Use [TEMPLATE.md](TEMPLATE.md) for short experiment notes.
Promote a compact summary into `profiling/reports/` or `docs/` when the result
is worth preserving in the repo.

## Keep Notes Small

Each note should answer one question, point to the exact command/result file,
and end with the next question to test.
