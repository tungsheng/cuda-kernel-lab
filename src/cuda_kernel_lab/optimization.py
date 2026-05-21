"""Optimization technique metadata used by benchmarks and reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizationTechnique:
    """Human-readable optimization method metadata for one benchmark result."""

    method_family: str
    method_id: str
    technique: str
    hypothesis: str
    expected_profiler_signal: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_family": self.method_family,
            "method_id": self.method_id,
            "technique": self.technique,
            "hypothesis": self.hypothesis,
            "expected_profiler_signal": self.expected_profiler_signal,
        }


def torch_reference_baseline() -> OptimizationTechnique:
    """Return the control technique used for PyTorch rows."""

    return OptimizationTechnique(
        method_family="baseline",
        method_id="torch.reference_baseline",
        technique="PyTorch reference baseline",
        hypothesis="Establish the latency, bandwidth, and correctness baseline for comparison.",
        expected_profiler_signal="Profiler evidence is optional for baseline rows.",
    )


def memory_optimization(
    *,
    backend: str,
    op_name: str,
    reduction_strategy: str,
) -> OptimizationTechnique:
    """Return optimization metadata for memory-bandwidth benchmarks."""

    if backend == "torch":
        return torch_reference_baseline()

    if op_name == "reduction_sum":
        if reduction_strategy == "two_pass":
            return OptimizationTechnique(
                method_family="reduction",
                method_id="triton.reduction_two_pass",
                technique="Two-pass block reduction",
                hypothesis=(
                    "Reducing to FP32 partial sums with Triton and finalizing in a second step "
                    "can cut repeated launches, but may pay partial-traffic or framework cleanup "
                    "cost."
                ),
                expected_profiler_signal=(
                    "High first-pass DRAM throughput, low register pressure, and visible cost in "
                    "the second-stage/finalization path if end-to-end latency trails the baseline."
                ),
            )
        return OptimizationTechnique(
            method_family="reduction",
            method_id="triton.reduction_iterative",
            technique="Iterative block reduction",
            hypothesis=(
                "Repeated Triton block reductions over FP32 partial sums should stream memory "
                "efficiently, while repeated launches expose orchestration overhead."
            ),
            expected_profiler_signal=(
                "High DRAM throughput and occupancy in the large first pass, with launch-count "
                "cost visible in end-to-end timing."
            ),
        )

    return OptimizationTechnique(
        method_family="launch tuning",
        method_id="triton.coalesced_block_size",
        technique="Coalesced block-size tuning",
        hypothesis=(
            "Varying Triton block size for contiguous streaming kernels can improve occupancy "
            "and memory throughput."
        ),
        expected_profiler_signal=(
            "Coalesced global loads/stores, high DRAM throughput, no shared-memory pressure, and "
            "similar traffic to the analytical model."
        ),
    )


def softmax_optimization(*, backend: str) -> OptimizationTechnique:
    """Return optimization metadata for row-wise softmax benchmarks."""

    if backend == "torch":
        return torch_reference_baseline()

    return OptimizationTechnique(
        method_family="fusion",
        method_id="triton.rowwise_softmax_fusion",
        technique="Row-wise softmax fusion",
        hypothesis=(
            "Keeping row max, subtract, exp, sum, divide, and store inside one kernel should "
            "reduce global-memory traffic and launch overhead versus a naive multi-kernel path."
        ),
        expected_profiler_signal=(
            "Global traffic close to one input read plus one output write, with no intermediate "
            "tensor writes."
        ),
    )


def norms_optimization(*, backend: str, op_name: str) -> OptimizationTechnique:
    """Return optimization metadata for row-wise normalization benchmarks."""

    if backend == "torch":
        return torch_reference_baseline()

    technique = "Row-wise RMSNorm fusion" if op_name == "rmsnorm" else "Row-wise LayerNorm fusion"

    return OptimizationTechnique(
        method_family="fusion",
        method_id=f"triton.rowwise_{op_name}_fusion",
        technique=technique,
        hypothesis=(
            "Fusing row reductions, normalization, parameter loads, and affine writeback should "
            "remove framework overhead and avoid intermediate normalization tensors."
        ),
        expected_profiler_signal=(
            "High DRAM throughput, no large intermediate writes, and acceptable occupancy despite "
            "the per-row reduction register footprint."
        ),
    )


def swiglu_optimization(*, backend: str) -> OptimizationTechnique:
    """Return optimization metadata for SwiGLU benchmarks."""

    if backend == "torch":
        return torch_reference_baseline()

    return OptimizationTechnique(
        method_family="fusion",
        method_id="triton.elementwise_swiglu_fusion",
        technique="Elementwise SwiGLU fusion",
        hypothesis=(
            "Fusing sigmoid, SiLU gating, multiply, and store should avoid materialized "
            "activation intermediates, lowering memory traffic and launch overhead."
        ),
        expected_profiler_signal=(
            "Traffic close to two input reads plus one output write, with strong DRAM throughput "
            "and no intermediate activation stores."
        ),
    )


def matmul_optimization(
    *,
    backend: str,
) -> OptimizationTechnique:
    """Return optimization metadata for matmul benchmarks."""

    if backend == "torch":
        return torch_reference_baseline()

    return OptimizationTechnique(
        method_family="tiling",
        method_id="triton.tiled_dot",
        technique="Tiled dot-product reuse",
        hypothesis=(
            "Triton tile-shape and launch-configuration sweeps with `tl.dot` can increase "
            "arithmetic intensity and Tensor Core utilization, but may trade off occupancy, "
            "pipeline depth, and register pressure."
        ),
        expected_profiler_signal=(
            "High TFLOP/s for float16, Tensor Core/HMMA utilization, and tile-dependent changes "
            "in occupancy, register pressure, shared memory, and pipeline staging."
        ),
    )


def attention_optimization(
    *,
    backend: str,
) -> OptimizationTechnique:
    """Return optimization metadata for decode attention benchmarks."""

    if backend == "torch":
        return torch_reference_baseline()

    return OptimizationTechnique(
        method_family="fusion",
        method_id="triton.decode_attention_fusion",
        technique="One-token decode attention fusion",
        hypothesis=(
            "Fusing score calculation, softmax, and value accumulation for one decode token "
            "should avoid materialized score/probability tensors and make KV-cache traffic the "
            "dominant cost."
        ),
        expected_profiler_signal=(
            "Global traffic dominated by contiguous K/V cache reads, little intermediate write "
            "traffic, and occupancy/register pressure shaped by head_dim and sequence length."
        ),
    )


def technique_from_result(
    *,
    backend: str,
    primitive: str,
    operation: str,
    strategy: str,
    parameters: dict[str, Any],
) -> OptimizationTechnique:
    """Best-effort technique metadata for old result records."""

    if backend == "torch" or strategy == "torch-baseline":
        return torch_reference_baseline()

    if primitive == "memory":
        return memory_optimization(
            backend=backend,
            op_name=operation,
            reduction_strategy=str(parameters.get("reduction_strategy") or "iterative"),
        )
    if primitive == "softmax":
        return softmax_optimization(backend=backend)
    if primitive == "norms":
        return norms_optimization(backend=backend, op_name=operation)
    if primitive == "swiglu":
        return swiglu_optimization(backend=backend)
    if primitive == "matmul":
        return matmul_optimization(backend=backend)
    if primitive == "attention":
        return attention_optimization(backend=backend)

    return OptimizationTechnique(
        method_family="custom",
        method_id=strategy,
        technique=strategy,
        hypothesis="Review the benchmark command and parameters to interpret this strategy.",
        expected_profiler_signal="Choose profiler counters that match the custom strategy.",
    )


def technique_from_strategy(strategy: str | None) -> OptimizationTechnique | None:
    """Infer technique metadata from a compact strategy label."""

    if not strategy:
        return None
    if strategy == "torch-baseline":
        return torch_reference_baseline()
    if strategy == "triton-block-size":
        return memory_optimization(
            backend="triton",
            op_name="vector_add",
            reduction_strategy="iterative",
        )
    if strategy == "triton-reduction-iterative":
        return memory_optimization(
            backend="triton",
            op_name="reduction_sum",
            reduction_strategy="iterative",
        )
    if strategy == "triton-reduction-two-pass":
        return memory_optimization(
            backend="triton",
            op_name="reduction_sum",
            reduction_strategy="two_pass",
        )
    if strategy == "triton-fused-row-softmax":
        return softmax_optimization(backend="triton")
    if strategy == "triton-fused-rmsnorm":
        return norms_optimization(backend="triton", op_name="rmsnorm")
    if strategy == "triton-fused-layernorm":
        return norms_optimization(backend="triton", op_name="layernorm")
    if strategy == "triton-fused-swiglu":
        return swiglu_optimization(backend="triton")
    if strategy.startswith("triton-tiled-dot"):
        return matmul_optimization(backend="triton")
    if strategy == "triton-decode-attention":
        return attention_optimization(backend="triton")
    return None


def technique_from_mapping(value: Any) -> OptimizationTechnique | None:
    """Parse optimization metadata from a JSON-like mapping."""

    if not isinstance(value, dict):
        return None

    return OptimizationTechnique(
        method_family=str(value.get("method_family") or "custom"),
        method_id=str(value.get("method_id") or value.get("technique") or "custom"),
        technique=str(value.get("technique") or value.get("method_id") or "custom"),
        hypothesis=str(value.get("hypothesis") or value.get("description") or ""),
        expected_profiler_signal=str(value.get("expected_profiler_signal") or ""),
    )
