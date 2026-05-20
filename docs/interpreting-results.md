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
