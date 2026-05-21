# KV Cache Layout

KV cache layout is included here as a kernel-layout case study: the useful
question is how addressing, memory access, and layout affect kernel behavior.
End-to-end serving and scheduler experiments belong in `gpu-inference-lab`.

## Layouts

- Contiguous KV cache: simple addressing, but can waste memory when request
  lengths vary.
- Paged KV cache: stores sequences in fixed-size blocks and uses a block table
  lookup to map logical token positions to physical cache blocks.

## Experiments

Active baseline:

- contiguous decode attention benchmark with PyTorch over
  `(seq_len, num_heads, head_dim)` K/V caches

Planned experiments:

- contiguous lookup benchmark
- paged lookup benchmark
- block allocator simulation
- fragmentation benchmark
- custom decode attention over contiguous and paged cache layouts

Record for each experiment:

- block size and cache layout
- sequence length distribution
- allocated versus live tokens
- lookup cost and fragmentation
- memory headroom and allocation waste

## Core Lesson

Paged KV cache is not just a data structure. At the kernel level, it changes
address calculation, memory coalescing, cache behavior, and the cost of lookup
indirection.
