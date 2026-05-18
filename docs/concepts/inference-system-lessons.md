# Inference System Lessons

This note connects kernel-level measurements to serving-level behavior.

## Questions

- When does a faster kernel improve tokens per second?
- When does batching improve throughput but hurt tail latency?
- How does prompt length affect time-to-first-token?
- How does decode length affect inter-token latency?
- How does KV cache memory pressure limit concurrency?

## Expected Lesson

A faster kernel helps, but scheduling, batching, queueing, and KV cache memory
management can still dominate p95 latency.

## Measurement Shape

When the simulator milestones land, keep kernel speedups and serving metrics in
the same report:

- tokens per second
- time to first token
- inter-token latency
- p95 and p99 request latency
- active sequences and KV cache usage
- queue depth and batching policy
