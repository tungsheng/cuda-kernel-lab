# Interpreting Results

Benchmark output is useful when it explains a bottleneck, not just when it shows
a lower latency.

## Columns

- `p50_ms`: median latency.
- `p95_ms` and `p99_ms`: tail latency and run stability.
- `GB/s`: estimated bytes moved divided by p50 latency.
- `TFLOP/s`: estimated FLOPs divided by p50 latency.
- `bytes_moved`: analytical traffic model, not a profiler counter.
- `flops`: scalar accounting model, not instruction-level analysis.
- `optimization`: method family, concrete technique, hypothesis, and expected
  profiler signal for the experiment. Tuned knobs remain in `parameters`.

## First Questions

Ask these in order:

1. Is the result correct against the PyTorch baseline?
2. Is p95/p99 close to p50, or is the run noisy?
3. Is the primitive low arithmetic intensity?
4. Does GB/s explain the result better than TFLOP/s?
5. Did fusion remove reads or writes?
6. Which optimization strategy changed the metric?
7. Which concrete optimization technique was tested?
8. Do profiler counters support the analytical traffic model and technique
   hypothesis?

## Common Mistakes

- Comparing latency without matching shape, dtype, device, and backend.
- Treating estimated `bytes_moved` as measured HBM transactions.
- Reporting TFLOP/s for a memory-bound primitive as the headline number.
- Forgetting that output allocation can distort small-kernel timings.
- Treating a `naive` traffic model run as a different softmax kernel. It only
  changes the denominator used for effective bandwidth.

## Decode-Step Dynamic Rows

For dynamic decode rows, read these fields together:

- `p50_ms`, `p95_ms`, and `p99_ms`: hot-loop latency for one synthetic scheduler step.
- `tokens_per_second` and `tokens_per_second_at_host_p95`: synthetic throughput from
  active tokens per step, not a full model serving number.
- `graph_hit_rate_pct`: whether the trace used already-captured graph buckets.
- `padding_waste_pct`: extra bucket capacity used to avoid recapture.
- `scheduler_cpu_p95_us`: host-side scheduler/hot-loop cost.
- `bucket_breakdown`: which active batch size caused the worst tail latency.

When `orchestration_timing` is `off`, per-region host breakdowns are absent by
design, but the row still reports total host step and scheduler timing. Use
`orchestration_timing=on` first to find a bottleneck, then turn it off for the
production-like timing comparison.

Saved A10G decode evidence in
`experiments/reports/aws-ec2/2026-05-22-round12-kv-active-views.md` shows the
saved dynamic path as `dynamic-piecewise-graph-same-stream` with
`sdpa-head-major`, `resident`, eager post-add, orchestration timing off, and
dense `1,2,3,4,5,6,7,8` buckets. The three tail seeds landed around
`0.155-0.158 ms` p50 and `0.228-0.232 ms` p95 with zero padding.

## Result Summary Template

```text
Question:
Command:
Device:
Shape and dtype:
Backend comparison:
Best p50:
Tail behavior:
Bandwidth interpretation:
Profiler confirmation, if collected:
Conclusion:
```
