# A10G vector_add float32 profile

## Context

- Benchmark command: `uv run benchmark-memory --backend triton --device cuda --op vector_add --numel 16777216 --dtype float32 --block-size 1024 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-21-strategy-next-profiled/memory-vector-add-float32.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-21-strategy-next-profiled/memory-vector-add-float32.jsonl`
- Operation: `vector_add`
- Strategy label: `triton-block-size`
- Method family: `launch tuning`
- Optimization technique: `Coalesced block-size tuning`
- Hypothesis: Varying Triton block size for contiguous streaming kernels can improve occupancy and memory throughput.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 421568 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 91.58 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 154447360 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 76962816 | byte | `dram__bytes_write.sum` |
| Occupancy | 81.80 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 26 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
