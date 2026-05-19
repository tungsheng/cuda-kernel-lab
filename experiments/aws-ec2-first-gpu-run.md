# AWS EC2 First GPU Run

Status: pending live GPU run

## Question

What are the first baseline PyTorch and Triton measurements for the CUDA Kernel
Lab benchmark matrix on a disposable AWS EC2 `g5.xlarge` host in `us-west-2`?

## Optimization Strategy

Baseline measurement. Strategy variants come after this run establishes
trustworthy reference numbers.

## Environment

- Region: `us-west-2`
- Instance type: `g5.xlarge`
- GPU:
- Driver/CUDA:
- AMI:
- Python:
- PyTorch:
- Triton:
- Git commit:

## Commands

```bash
uv sync --group dev --extra gpu
uv run gpu-info
uv run pytest
uv run benchmark-matrix --dry-run
uv run benchmark-matrix
```

## Expected Result Files

- `experiments/results/aws-ec2-first-run/memory.jsonl`
- `experiments/results/aws-ec2-first-run/softmax.jsonl`
- `experiments/results/aws-ec2-first-run/norms.jsonl`

## Results

Fill this after the live GPU run. Do not enter estimated or synthetic numbers.

| Primitive | Dtype | Fastest Backend | p50 | p95 | p99 | GB/s | TFLOP/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| memory | float32 |  |  |  |  |  |  |
| memory | float16 |  |  |  |  |  |  |
| softmax | float32 |  |  |  |  |  |  |
| softmax | float16 |  |  |  |  |  |  |
| norms | float32 |  |  |  |  |  |  |
| norms | float16 |  |  |  |  |  |  |

## Observation

Pending.

## Interpretation

Pending.

## Next Question

Which memory-bandwidth strategy variant should be tested first for
`vector_add`?
