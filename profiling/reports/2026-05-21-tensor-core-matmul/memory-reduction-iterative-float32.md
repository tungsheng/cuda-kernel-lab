# A10G reduction_sum iterative float32 profile

## Context

- Benchmark command: `uv run benchmark-memory --backend triton --device cuda --op reduction_sum --numel 16777216 --dtype float32 --block-size 1024 --reduction-strategy iterative --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-21-tensor-core-matmul-profiled/memory-reduction-iterative-float32.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-21-tensor-core-matmul-profiled/memory-reduction-iterative-float32.jsonl`
- Operation: `reduction_sum`
- Strategy label: `triton-reduction-iterative`
- Method family: `reduction`
- Optimization technique: `Iterative block reduction`
- Hypothesis: Repeated Triton block reductions over FP32 partial sums should stream memory efficiently, while repeated launches expose orchestration overhead.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 136416 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 93.75 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 74209536 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 2432384 | byte | `dram__bytes_write.sum` |
| Occupancy | 99.01 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 16 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 16 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
