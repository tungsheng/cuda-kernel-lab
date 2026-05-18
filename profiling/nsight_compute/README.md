# Nsight Compute

Store Nsight Compute command notes and small exported summaries here.

Large binary profiler captures are ignored by default. Prefer committing compact
text summaries in `profiling/reports/`.

Example command shape:

```bash
ncu --set full --target-processes all uv run benchmark-memory --backend triton --device cuda --op vector_add
```
