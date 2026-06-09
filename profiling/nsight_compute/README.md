# Nsight Compute Artifacts

Store small Nsight Compute CSV exports, stderr logs, and benchmark logs here
when they support a compact profiler report.

Large binary profiler captures are ignored by default. Prefer committing the
human-readable conclusion in `profiling/reports/` with
`profiling/reports/TEMPLATE.md`.

Convert a small export into a starter note with:

```bash
uv run nsight-summary \
  --input profiling/nsight_compute/vector-add.csv \
  --output profiling/reports/vector-add-a10g.md
```
