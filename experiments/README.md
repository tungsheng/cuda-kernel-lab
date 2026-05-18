# Experiments

Use this directory for benchmark result records and short experiment notes that
tie measurements back to a specific question.

Recommended result output:

```bash
uv run benchmark-memory --backend all --device cuda --op all --output experiments/results/memory.jsonl
```

The `experiments/results/` directory is ignored by default because JSONL runs
can grow quickly. Promote a compact summary into `profiling/reports/` or `docs/`
when the result is worth preserving in the repo.
