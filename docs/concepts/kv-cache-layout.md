# KV Cache Layout

KV cache layout is both a kernel concern and a serving-system memory management
concern.

## Layouts

- Contiguous KV cache: simple addressing, but can waste memory when request
  lengths vary.
- Paged KV cache: stores sequences in fixed-size blocks and uses a block table
  lookup to map logical token positions to physical cache blocks.

## Experiments

Planned experiments:

- contiguous lookup benchmark
- paged lookup benchmark
- block allocator simulation
- fragmentation benchmark
- decode attention over contiguous and paged cache layouts

Record for each experiment:

- block size and cache layout
- sequence length distribution
- allocated versus live tokens
- lookup cost and fragmentation
- impact on active batch size or memory headroom

## Core Lesson

Paged KV cache is not just a data structure. It lets an inference server pack
variable-length active sequences into GPU memory more flexibly, which can improve
batching and throughput under real request mixes.
