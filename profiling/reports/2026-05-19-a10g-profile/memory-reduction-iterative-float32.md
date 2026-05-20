# A10G reduction_sum iterative float32 profile

## Context

- Benchmark command: `uv run benchmark-memory --backend triton --device cuda --op reduction_sum --numel 16777216 --dtype float32 --block-size 1024 --reduction-strategy iterative --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/memory-reduction-iterative-float32.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/memory-reduction-iterative-float32.jsonl`
- Operation: `reduction_sum`
- Strategy: `triton-reduction-iterative`

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 136832 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 93.48 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 74212736 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 2431616 | byte | `dram__bytes_write.sum` |
| Occupancy | 99.01 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 16 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 16 | byte/block | `launch__shared_mem_per_block_dynamic` |

## Interpretation

- The first-pass Triton reduction is DRAM-bound and well occupied: 93.48% DRAM throughput and 99.01% occupancy with only 16 registers per thread.
- DRAM reads dominate the profile, while writes are limited to partial sums. That matches the expected shape of a streaming reduction.
- The benchmark report still favors torch for the full reduction, so the remaining gap is more likely reduction orchestration and finalization than first-pass bandwidth.

## Follow-Up

Treat the first-pass kernel as healthy. Future reduction work should focus on end-to-end launch count, final reduction strategy, and whether a custom second stage beats `partials.sum()`.
