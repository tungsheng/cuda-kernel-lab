# Profiling Reports

Commit compact profiler writeups here. Use [TEMPLATE.md](TEMPLATE.md) for new
reports.

Raw Nsight CSVs, stderr logs, and benchmark logs belong under
`profiling/nsight_compute/<run-id>/`. Large binary profiler captures are ignored
and should stay out of git.

Use `uv run nsight-summary` to turn a small Nsight Compute CSV or text export
into a Markdown starter report, then edit the interpretation by hand.

Each report should answer:

- What command and result are being explained?
- Which optimization technique and hypothesis are being validated?
- Which counters matter?
- Is the result memory-bound, compute-bound, launch-bound, or otherwise limited?
- Did the counters confirm the benchmark interpretation?
