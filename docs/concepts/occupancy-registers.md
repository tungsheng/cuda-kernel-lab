# Occupancy and Registers

Occupancy is the amount of GPU work that can be resident on an SM at once.
Register pressure is one common reason occupancy drops.

## Why It Matters

More occupancy can help hide memory latency, but maximum occupancy is not always
the fastest point. A kernel with more registers may run faster if those registers
avoid extra global memory traffic or reduce instruction count.

## What To Record

When profiler data is available, record:

- block size and grid shape
- registers per thread
- shared memory per block
- achieved occupancy
- limiting resource
- p50 latency and tail stability

## Interpretation

Ask:

- Did the optimization increase register use?
- Did occupancy fall?
- Did latency improve anyway?
- Did reduced memory traffic or fusion explain the improvement?

Use this note to avoid treating occupancy as a goal by itself. The goal is
measured performance with a clear bottleneck explanation.
