# Experiments

Use this directory for benchmark result records and short notes that tie a
measurement back to one question.

## Artifact Map

- `experiments/results/`: ignored raw JSONL benchmark records.
- `experiments/reports/`: generated Markdown reports from `benchmark-report`.
- `experiments/TEMPLATE.md`: short hand-written note template.
- `profiling/nsight_compute/`: Nsight CSV, stderr, and benchmark logs.
- `profiling/reports/`: compact profiler writeups.

## Standard Flow

Run a disposable GPU evidence loop with Runpod:

```bash
./scripts/up
./scripts/benchmark --run-id <run-id>
./scripts/down
```

This writes JSONL under `experiments/results/runpod/<run-id>/` and a generated
report under `experiments/reports/runpod/<run-id>.md`.

For suite choices, decode-step runs, H200 autotune, and profiling, use
[docs/benchmark-workflow.md](../docs/benchmark-workflow.md) and
[docs/profiling-workflow.md](../docs/profiling-workflow.md).

## Keep Notes Small

Each hand-written note should answer one question, point to the exact command
and result file, summarize only the key numbers, and end with the next question
to test.

Promote durable lessons into `docs/`; keep raw records in ignored result
directories.
