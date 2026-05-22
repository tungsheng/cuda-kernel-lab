# A10G reduction_sum two-pass float32 profile

## Context

- Benchmark command: `uv run benchmark-memory --backend triton --device cuda --op reduction_sum --numel 16777216 --dtype float32 --block-size 1024 --reduction-strategy two_pass --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode-profiled/memory-reduction-two-pass-float32.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-22-dynamic-piecewise-decode-profiled/memory-reduction-two-pass-float32.jsonl`
- Operation: `reduction_sum`
- Strategy label: `triton-reduction-two-pass`
- Method family: `reduction`
- Optimization technique: `Two-pass block reduction`
- Hypothesis: Reducing to FP32 partial sums with Triton and finalizing in a second step can cut repeated launches, but may pay partial-traffic or framework cleanup cost.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 136704 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 93.40 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 74116352 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 2398208 | byte | `dram__bytes_write.sum` |
| Occupancy | 98.52 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 16 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 16 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
