# A10G RMSNorm float16 profile

## Context

- Benchmark command: `uv run benchmark-norms --backend triton --device cuda --op rmsnorm --rows 4096 --cols 4096 --dtype float16 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/norms-rmsnorm-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/norms-rmsnorm-float16.jsonl`
- Operation: `rmsnorm`
- Strategy: `triton-fused-rmsnorm`

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 137280 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 91.41 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 37925760 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 37261568 | byte | `dram__bytes_write.sum` |
| Occupancy | 93.12 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 40 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 32 | byte/block | `launch__shared_mem_per_block_dynamic` |

## Interpretation

- Fused RMSNorm is the strongest validated Triton win in this run: 5.539x faster than torch for float16 in the benchmark report.
- The kernel is still primarily memory-bound, sustaining 91.41% DRAM throughput while reading the input/weight data and writing the normalized output in one fused pass.
- The higher register footprint is acceptable here because occupancy remains above 93% and the fusion removes enough framework and intermediate-memory cost to dominate the tradeoff.

## Follow-Up

Use RMSNorm as the positive fusion example in future reports. The next fundamental work should compare this memory-bound fused win against matmul kernels where tile reuse and Tensor Core utilization matter.
