# A10G decode step fused same-stream piecewise CUDA Graph float16 profile

## Context

- Benchmark command: `uv run benchmark-decode-step --mode fused-piecewise-graph-same-stream --device cuda --dtype float16 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-22-tail-sweep-live-profiled/decode-step-fused-piecewise-graph-same-stream-float16.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-22-tail-sweep-live-profiled/decode-step-fused-piecewise-graph-same-stream-float16.jsonl`
- Operation: `decode_step`
- Strategy label: `fused-piecewise-graph-same-stream`
- Method family: `launch replay`
- Optimization technique: `Fused same-stream piecewise CUDA Graph replay`
- Hypothesis: Replaying captured fused pre/post-attention regions on the caller stream should preserve dynamic-shape graph reuse while removing explicit stream handoff cost.

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 34752 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 85.62 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 9300224 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 8508160 | byte | `dram__bytes_write.sum` |
| Occupancy | 90.77 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 30 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_dynamic` |
| Tensor pipe utilization | 0 | % | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| Tensor Core utilization | 0 | % | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` |

## Interpretation

Compare these counters against the benchmark result and technique hypothesis before choosing the next change.

## Follow-Up

Record the next technique change or profiler counter to inspect.
