# A10G RMSNorm float16 profile

## Context

- Benchmark command: `uv run benchmark-norms --backend triton --device cuda --op rmsnorm --rows 4096 --cols 4096 --dtype float16 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-20-technique-metadata-verify-profiled/norms-rmsnorm-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-20-technique-metadata-verify-profiled/norms-rmsnorm-float16.jsonl`
- Operation: `rmsnorm`
- Strategy label: `triton-fused-rmsnorm`
- Method family: `fusion`
- Optimization technique: `Row-wise RMSNorm fusion`
- Hypothesis: Fusing row reductions, normalization, parameter loads, and affine writeback should remove framework overhead and avoid intermediate normalization tensors.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 137344 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 91.42 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 37926144 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 37313280 | byte | `dram__bytes_write.sum` |
| Occupancy | 92.80 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 40 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 32 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
