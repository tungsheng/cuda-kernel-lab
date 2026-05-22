# A10G dynamic decode step eager float16 profile

## Context

- Benchmark command: `uv run benchmark-decode-step --dynamic-trace --mode dynamic-eager --device cuda --dtype float16 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-22-tail-sweep-live-profiled/decode-step-dynamic-eager-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-22-tail-sweep-live-profiled/decode-step-dynamic-eager-float16.jsonl`
- Operation: `decode_step`
- Strategy label: `dynamic-eager`
- Method family: `fusion`
- Optimization technique: `Fused eager decode step`
- Hypothesis: Replacing decomposed normalization and activation work with fused kernels should reduce kernel count and intermediate memory traffic before graph replay.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 34688 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 85.58 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 9299968 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 8465280 | byte | `dram__bytes_write.sum` |
| Occupancy | 89.80 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 30 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
