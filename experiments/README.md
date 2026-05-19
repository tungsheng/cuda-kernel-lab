# Experiments

Use this directory for local benchmark result records and short experiment notes
that tie measurements back to a specific question.

Recommended result output:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

First live-GPU evidence run:

```bash
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep --dry-run
uv run benchmark-matrix --include-vector-add-sweep --include-reduction-sweep
uv run benchmark-report --input-dir experiments/results/aws-ec2-first-run
```

The `experiments/results/` directory is ignored by default because JSONL runs
can grow quickly. Use [TEMPLATE.md](TEMPLATE.md) for short experiment notes.
Promote a compact summary into `profiling/reports/` or `docs/` when the result
is worth preserving in the repo.

## Keep Notes Small

Each note should answer one question, point to the exact command/result file,
and end with the next question to test.
