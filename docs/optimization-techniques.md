# Optimization Techniques

This catalog names the optimization technique behind each experiment. Use these
terms in experiment notes, benchmark summaries, and profiler reports so a result
explains the method being tested, not only the faster backend.

## Technique Catalog

| Method Family | Technique | Used By | What It Tests |
| --- | --- | --- | --- |
| baseline | PyTorch reference baseline | all primitives | Establishes the correctness and performance control for each shape and dtype. |
| launch tuning | Coalesced block-size tuning | `copy`, `scale`, `vector_add` | Varies Triton block size for simple contiguous global-memory kernels to test occupancy and DRAM-throughput sensitivity. |
| reduction | Iterative block reduction | `reduction_sum` | Repeatedly reduces FP32 partial sums with Triton until one scalar remains. |
| reduction | Two-pass block reduction | `reduction_sum` | Reduces to FP32 partial sums with Triton, then finalizes those partials in a second reduction step. |
| fusion | Row-wise softmax fusion | `softmax` | Keeps max, subtract, exp, sum, divide, and output write inside one row-wise kernel. |
| fusion | Row-wise normalization fusion | `rmsnorm`, `layernorm` | Fuses row reductions, normalization, parameter loads, and affine writeback. |
| fusion | Elementwise SwiGLU fusion | `swiglu` | Fuses sigmoid/SwiLU gating and multiply without materializing activation intermediates. |
| tiling | Tiled dot-product reuse | `matmul` | Uses tiled `tl.dot` matmul plus tile-shape and launch-configuration sweeps to study reuse, occupancy, pipeline staging, register pressure, and Tensor Core utilization. |
| tiling | Persistent tiled dot-product scheduling | `matmul` | Keeps a bounded set of Triton programs resident across output tiles to study persistent scheduling, wave count, occupancy, and Tensor Core utilization. |
| baseline | Contiguous KV-cache decode attention baseline | `attention` | Establishes the PyTorch control for one-token decode attention with contiguous key/value cache layout before future custom fused attention work. |
| launch replay | CUDA Graph decode-step replay | `decode_step` | Replays static and piecewise synthetic decode steps to separate CPU/driver launch overhead from GPU kernel time and dynamic-shape graph reuse. |

## How To Describe An Experiment

Each experiment should answer these fields explicitly:

- Method family: broad class such as `fusion`, `reduction`, or `tiling`.
- Technique: the concrete method from the catalog.
- Hypothesis: the metric expected to move and why.
- Control: usually the PyTorch baseline for the same operation, shape, dtype, and device.
- Knobs changed: block size, reduction strategy, traffic model, tile shape, or epsilon.
- Expected profiler signal: counters that should confirm or challenge the hypothesis.

Benchmark JSONL stores the technique fields under `optimization` and keeps
changed knobs in the existing `parameters` object.

Example:

```text
Method family: fusion
Technique: Elementwise SwiGLU fusion
Hypothesis: fusing SiLU and multiply avoids intermediate activation traffic and
reduces launch overhead, so p50 latency should fall and effective GB/s should rise.
Control: PyTorch SwiGLU baseline for the same shape and dtype.
Knobs changed: block_size=1024
Expected profiler signal: traffic close to two input reads plus one output write,
high DRAM throughput, and no intermediate activation stores.
```

## Interpreting By Technique

Launch tuning is meaningful only when shape, dtype, backend, and traffic model
match. If the profiler already shows high DRAM throughput, wider block-size
sweeps are unlikely to change the conclusion.

Reduction experiments need both first-pass profiler evidence and end-to-end
timing. A healthy first pass can still lose if launch count or finalization cost
dominates.

Fusion experiments should be interpreted as traffic-removal tests. The main
question is whether avoiding intermediate tensors or framework launches moves
p50 latency and effective bandwidth.

Tiling experiments should be interpreted with TFLOP/s and profiler counters, not
GB/s alone. For float16 matmul, Tensor Core/HMMA utilization, occupancy, shared
memory, registers, warp count, and pipeline stages explain the tile-shape
tradeoff.

Attention experiments should first pin down the PyTorch contiguous-KV baseline
and shape sensitivity. Future custom attention kernels should be evaluated as
fused decode targets where K/V cache reads dominate traffic and
score/probability intermediate writes are avoided.

CUDA Graph experiments should be read as launch-path tests. Compare host wall
time, CUDA event time, CPU utilization, and launch-overhead estimates across
the naive/fused and eager/graph matrix before attributing any change to kernel
code. For `decode_step`, `fused` means the synthetic RMSNorm and SwiGLU regions
use Triton fused kernels; it does not mean attention itself is a custom fused
kernel.

Piecewise CUDA Graph experiments should be read as dynamic-shape compromise
tests. Compare graph hit rate and padding waste against the fully eager dynamic
trace before treating lower launch overhead as a net win.
