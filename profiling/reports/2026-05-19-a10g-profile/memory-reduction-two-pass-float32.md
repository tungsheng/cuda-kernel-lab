# A10G reduction_sum two-pass float32 profile

## Context

- Benchmark command: `uv run benchmark-memory --backend triton --device cuda --op reduction_sum --numel 16777216 --dtype float32 --block-size 1024 --reduction-strategy two_pass --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/memory-reduction-two-pass-float32.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/memory-reduction-two-pass-float32.jsonl`
- Operation: `reduction_sum`
- Strategy: `triton-reduction-two-pass`

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 137024 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 93.22 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 74127232 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 2419968 | byte | `dram__bytes_write.sum` |
| Occupancy | 98.93 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 16 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 16 | byte/block | `launch__shared_mem_per_block_dynamic` |

## Interpretation

- The profiled first pass is effectively the same bottleneck as the iterative variant: 93.22% DRAM throughput, 98.93% occupancy, and the same low register footprint.
- This confirms the target-specific launch skip captured the large Triton partial-sum kernel rather than the tiny PyTorch cleanup reduction.
- Since the benchmark two-pass path is slower than torch despite a strong first pass, the end-to-end cost is in orchestration and the second-stage reduction, not raw memory bandwidth.

## Follow-Up

If reductions stay in scope, add a dedicated Triton second-stage reduction before trying more block-size tuning. Otherwise, move this evidence into the matmul milestone as the memory-bound contrast case.
