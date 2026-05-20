# Profiling Reports

Commit short profiler writeups here. Use [TEMPLATE.md](TEMPLATE.md) for new
reports.

Use `uv run nsight-summary` to turn a small Nsight Compute CSV or text export
into a Markdown starter report, then edit the interpretation by hand.

Each report should answer:

- What command and result are being explained?
- Which optimization technique and hypothesis are being validated?
- Which profiler counters matter?
- Is the kernel memory-bound or compute-bound?
- Did the counters confirm the benchmark interpretation?

Suggested filename:

```text
YYYY-MM-DD-kernel-shape-device.md
```
