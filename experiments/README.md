# Experiments

Use this directory for local benchmark result records and short experiment notes
that tie measurements back to a specific question.

Recommended result output:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

Live-GPU evidence run on disposable AWS EC2:

```bash
./scripts/live-benchmark --run-id <run-id>
```

This writes raw JSONL under `experiments/results/aws-ec2/<run-id>/` and a
compact report under `experiments/reports/aws-ec2/<run-id>.md`.

Add profiler evidence when the benchmark report points at a specific bottleneck:

```bash
./scripts/live-benchmark --run-id <run-id> --with-profiling
```

This also writes Nsight Compute CSV exports under
`profiling/nsight_compute/<run-id>/` and compact profiler notes under
`profiling/reports/<run-id>/`.

For the matmul/Tensor Core track, include the focused tile-shape sweep:

```bash
./scripts/live-benchmark --run-id <run-id> --include-matmul-sweep --with-profiling
```

Manual live-GPU matrix collection:

```bash
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep --dry-run
uv run benchmark-matrix \
  --output-dir experiments/results/aws-ec2/<run-id> \
  --include-vector-add-sweep \
  --include-reduction-sweep
uv run benchmark-report \
  --input-dir experiments/results/aws-ec2/<run-id>
```

The `experiments/results/` directory is ignored by default because JSONL runs
can grow quickly. Use [TEMPLATE.md](TEMPLATE.md) for short experiment notes.
Promote a compact summary into `profiling/reports/` or `docs/` when the result
is worth preserving in the repo.

## Keep Notes Small

Each note should answer one question, point to the exact command/result file,
and end with the next question to test.
