"""Generate compact Markdown summaries from Nsight Compute exports."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from cuda_kernel_lab.optimization import technique_from_strategy

KEY_METRICS = {
    "gpu__time_duration.sum": "Kernel time",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM throughput",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM throughput",
    "dram__bytes_read.sum": "DRAM bytes read",
    "dram__bytes_write.sum": "DRAM bytes written",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Occupancy",
    "launch__registers_per_thread": "Registers per thread",
    "launch__shared_mem_per_block_static": "Static shared memory per block",
    "launch__shared_mem_per_block_dynamic": "Dynamic shared memory per block",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "Tensor pipe utilization",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": (
        "Tensor Core utilization"
    ),
    "smsp__inst_executed_pipe_tensor.sum": "Tensor pipe instructions",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum": "HMMA instructions",
}


@dataclass(frozen=True)
class NsightMetric:
    label: str
    name: str
    value: str
    unit: str


def parse_metrics(path: Path) -> list[NsightMetric]:
    """Extract the small set of profiler counters used in evidence notes."""

    text = path.read_text(encoding="utf-8")
    return _metrics_from_csv(text) or _metrics_from_text(text)


def render_markdown(
    metrics: list[NsightMetric],
    *,
    title: str,
    benchmark_command: str | None,
    result_jsonl: str | None,
    operation: str | None,
    strategy: str | None,
) -> str:
    """Render a compact profiler summary."""

    optimization = technique_from_strategy(strategy)
    lines = [
        f"# {title}",
        "",
        "## Context",
        "",
        f"- Benchmark command: `{benchmark_command or ''}`",
        f"- JSONL result: `{result_jsonl or ''}`",
        f"- Operation: `{operation or ''}`",
        f"- Strategy label: `{strategy or ''}`",
        f"- Method family: `{optimization.method_family if optimization else ''}`",
        f"- Optimization technique: `{optimization.technique if optimization else ''}`",
        f"- Hypothesis: {optimization.hypothesis if optimization else ''}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value | Unit | Nsight Name |",
        "| --- | ---: | --- | --- |",
    ]
    if metrics:
        for metric in metrics:
            lines.append(
                f"| {metric.label} | {metric.value} | {metric.unit} | `{metric.name}` |"
            )
    else:
        lines.append("| No key metrics found |  |  |  |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Compare these counters against the benchmark result and technique hypothesis "
            "before choosing the next change.",
            "",
            "## Follow-Up",
            "",
            "Record the next technique change or profiler counter to inspect.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Nsight Compute CSV/text export.")
    parser.add_argument("--output", type=Path, help="Markdown summary path. Prints when omitted.")
    parser.add_argument("--title", default="Nsight Compute Summary")
    parser.add_argument("--benchmark-command")
    parser.add_argument("--result-jsonl")
    parser.add_argument("--operation")
    parser.add_argument("--strategy")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = render_markdown(
        parse_metrics(args.input),
        title=args.title,
        benchmark_command=args.benchmark_command,
        result_jsonl=args.result_jsonl,
        operation=args.operation,
        strategy=args.strategy,
    )
    if args.output is None:
        print(summary)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(f"Wrote profiler summary to {args.output}", file=sys.stderr)


def _metrics_from_csv(text: str) -> list[NsightMetric]:
    sample = _csv_sample(text)
    if sample is None:
        return []

    wide_metrics = _metrics_from_wide_csv(sample)
    if wide_metrics:
        return wide_metrics

    reader = csv.DictReader(sample.splitlines())
    if reader.fieldnames is None:
        return []

    metrics = []
    for row in reader:
        name = _first_value(row, "Metric Name", "Name", "Metric")
        if name not in KEY_METRICS:
            continue
        value = _first_value(row, "Metric Value", "Value", "Avg", "Average")
        unit = _first_value(row, "Metric Unit", "Unit")
        metrics.append(
            NsightMetric(
                label=KEY_METRICS[name],
                name=name,
                value=value,
                unit=unit,
            )
        )
    return metrics


def _metrics_from_wide_csv(text: str) -> list[NsightMetric]:
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 3:
        return []

    for index, header in enumerate(rows[:-2]):
        if "ID" not in header and "Kernel Name" not in header:
            continue
        if not any(metric_name in header for metric_name in KEY_METRICS):
            continue

        units = rows[index + 1]
        values = rows[index + 2]
        metrics = []
        for metric_name, label in KEY_METRICS.items():
            if metric_name not in header:
                continue
            column = header.index(metric_name)
            value = values[column].strip() if column < len(values) else ""
            if not value:
                continue
            unit = units[column].strip() if column < len(units) else ""
            metrics.append(
                NsightMetric(
                    label=label,
                    name=metric_name,
                    value=value,
                    unit=unit,
                )
            )
        return metrics

    return []


def _csv_sample(text: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        fields = {field.strip() for field in line.split(",")}
        if {"Metric Name", "Metric Value"}.issubset(fields):
            return "\n".join(lines[index:])
    return text.lstrip() or None


def _metrics_from_text(text: str) -> list[NsightMetric]:
    metrics = []
    for name, label in KEY_METRICS.items():
        match = re.search(rf"{re.escape(name)}\s+([0-9.+\-eE]+)\s*([A-Za-z/%]*)", text)
        if match is None:
            continue
        metrics.append(
            NsightMetric(
                label=label,
                name=name,
                value=match.group(1),
                unit=match.group(2),
            )
        )
    return metrics


def _first_value(row: dict[str, str | None], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    return ""


if __name__ == "__main__":
    main()
