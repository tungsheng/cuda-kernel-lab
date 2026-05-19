# Nsight Compute

Store Nsight Compute command notes and small exported summaries here.

Large binary profiler captures are ignored by default. Prefer committing compact
text summaries in `profiling/reports/` using `profiling/reports/TEMPLATE.md`.

Example command shape:

```bash
ncu --set full --target-processes all uv run benchmark-memory --backend triton --device cuda --op vector_add
```

When exporting a small text or CSV summary, convert it with:

```bash
uv run nsight-summary --input profiling/nsight_compute/vector-add.csv --output profiling/reports/vector-add-a10g.md
```
