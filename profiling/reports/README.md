# Profiling Reports

Commit short profiler writeups here. Use [TEMPLATE.md](TEMPLATE.md) for new
reports.

Each report should include:

- hardware and driver context
- command used
- JSONL result path or pasted result record when available
- kernel name and input shape
- latency summary
- raw latency distribution notes when p95/p99 look noisy
- memory throughput summary
- occupancy or launch configuration notes
- interpretation: memory-bound or compute-bound

Suggested filename:

```text
YYYY-MM-DD-kernel-shape-device.md
```
