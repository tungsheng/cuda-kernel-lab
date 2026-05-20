# A10G vector_add float32 profile

## Context

- Benchmark command: `uv run benchmark-memory --backend triton --device cuda --op vector_add --numel 16777216 --dtype float32 --block-size 1024 --warmup 5 --iterations 10 --skip-correctness --output experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/memory-vector-add-float32.jsonl`
- JSONL result: `experiments/results/aws-ec2/2026-05-19-a10g-profile-profiled/memory-vector-add-float32.jsonl`
- Operation: `vector_add`
- Strategy: `triton-block-size`

## Key Metrics

| Metric | Value | Unit | Nsight Name |
| --- | ---: | --- | --- |
| Kernel time | 418784 | ns | `gpu__time_duration.sum` |
| DRAM throughput | 92.20 | % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read | 154459904 | byte | `dram__bytes_read.sum` |
| DRAM bytes written | 76982912 | byte | `dram__bytes_write.sum` |
| Occupancy | 81.16 | % | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Registers per thread | 26 | register/thread | `launch__registers_per_thread` |
| Static shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_static` |
| Dynamic shared memory per block | 0 | byte/block | `launch__shared_mem_per_block_dynamic` |

## Interpretation

- The Triton `vector_add` kernel is DRAM-bound: it sustains 92.20% of peak DRAM throughput with no shared-memory usage and moderate register pressure.
- The profiler captured about 231 MB of DRAM traffic for one launch, matching the expected two float32 input reads plus one output write for 16,777,216 elements.
- Because PyTorch is still faster in the benchmark report while Triton is already near peak DRAM throughput, another broad block-size sweep is unlikely to change the main conclusion.

## Follow-Up

Use this as the memory-traffic baseline. The next meaningful comparison should reduce bytes moved through fusion or move to matmul reuse/Tensor Core work.
