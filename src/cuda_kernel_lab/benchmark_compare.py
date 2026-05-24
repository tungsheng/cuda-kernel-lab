"""Compare two benchmark result directories and gate regressions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from cuda_kernel_lab.benchmark_report import ReportRow, load_report_rows

DEFAULT_MAX_REGRESSION_PCT = 5.0
CompareKey = tuple[str, str, str, tuple[int, ...], str]


@dataclass(frozen=True)
class Regression:
    key: CompareKey
    baseline: ReportRow
    candidate: ReportRow
    metric: str
    baseline_value: float
    candidate_value: float
    regression_pct: float


@dataclass(frozen=True)
class Comparison:
    baseline_dir: Path
    candidate_dir: Path
    max_regression_pct: float
    compared_keys: int
    regressions: tuple[Regression, ...]
    correctness_failures: tuple[ReportRow, ...]
    missing_candidate_keys: tuple[CompareKey, ...]

    @property
    def passed(self) -> bool:
        return (
            not self.regressions
            and not self.correctness_failures
            and not self.missing_candidate_keys
        )


def compare_result_dirs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    max_regression_pct: float = DEFAULT_MAX_REGRESSION_PCT,
    require_matching_keys: bool = False,
) -> Comparison:
    """Compare best rows for matching backend/shape keys."""

    if max_regression_pct < 0:
        raise ValueError("max_regression_pct must be non-negative")

    baseline_rows = load_report_rows(baseline_dir)
    candidate_rows = load_report_rows(candidate_dir)
    baseline_best = _best_rows_by_key(baseline_rows)
    candidate_best = _best_rows_by_key(candidate_rows)

    regressions = []
    for key in sorted(set(baseline_best) & set(candidate_best)):
        baseline = baseline_best[key]
        candidate = candidate_best[key]
        metric, baseline_value, candidate_value, regression_pct = _regression_metric(
            baseline,
            candidate,
        )
        if regression_pct > max_regression_pct:
            regressions.append(
                Regression(
                    key=key,
                    baseline=baseline,
                    candidate=candidate,
                    metric=metric,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    regression_pct=regression_pct,
                )
            )

    missing_candidate_keys: tuple[CompareKey, ...] = ()
    if require_matching_keys:
        missing_candidate_keys = tuple(sorted(set(baseline_best) - set(candidate_best)))

    correctness_failures = tuple(
        row for row in candidate_rows if row.correctness in {"fail", "unknown"}
    )
    return Comparison(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        max_regression_pct=max_regression_pct,
        compared_keys=len(set(baseline_best) & set(candidate_best)),
        regressions=tuple(regressions),
        correctness_failures=correctness_failures,
        missing_candidate_keys=missing_candidate_keys,
    )


def render_markdown(comparison: Comparison) -> str:
    status = "pass" if comparison.passed else "fail"
    lines = [
        "# Benchmark Compare Report",
        "",
        f"Status: {status}",
        "",
        "## Inputs",
        "",
        f"- Baseline: `{comparison.baseline_dir}`",
        f"- Candidate: `{comparison.candidate_dir}`",
        f"- Max regression: `{comparison.max_regression_pct:.4g}%`",
        f"- Compared keys: `{comparison.compared_keys}`",
        "",
    ]

    if comparison.correctness_failures:
        lines.extend(
            [
                "## Correctness Failures",
                "",
                "| Source | Backend | Operation | Dtype | Shape | Variant | Correctness |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in comparison.correctness_failures:
            lines.append(
                "| "
                f"`{row.source}` | {row.backend} | {row.operation} | {row.dtype} | "
                f"{_shape_label(row.shape)} | {_escape_cell(row.variant)} | {row.correctness} |"
            )
        lines.append("")

    if comparison.missing_candidate_keys:
        lines.extend(
            [
                "## Missing Candidate Keys",
                "",
                "| Primitive | Operation | Dtype | Shape | Backend |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for primitive, operation, dtype, shape, backend in comparison.missing_candidate_keys:
            lines.append(
                "| "
                f"{primitive} | {operation} | {dtype} | {_shape_label(shape)} | {backend} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Performance Regressions",
            "",
            "| Primitive | Operation | Dtype | Shape | Backend | Metric | Baseline | Candidate | "
            "Regression % | Baseline Variant | Candidate Variant |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if comparison.regressions:
        for regression in sorted(
            comparison.regressions,
            key=lambda item: item.regression_pct,
            reverse=True,
        ):
            primitive, operation, dtype, shape, backend = regression.key
            lines.append(
                "| "
                f"{primitive} | {operation} | {dtype} | {_shape_label(shape)} | {backend} | "
                f"{regression.metric} | {_fmt(regression.baseline_value)} | "
                f"{_fmt(regression.candidate_value)} | {_fmt(regression.regression_pct)} | "
                f"{_escape_cell(regression.baseline.variant)} | "
                f"{_escape_cell(regression.candidate.variant)} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |")

    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-regression-pct", type=float, default=DEFAULT_MAX_REGRESSION_PCT)
    parser.add_argument(
        "--require-matching-keys",
        action="store_true",
        help="Fail when a baseline backend/shape key is missing from the candidate.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Always exit 0 after writing the comparison report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    comparison = compare_result_dirs(
        args.baseline_dir,
        args.candidate_dir,
        max_regression_pct=args.max_regression_pct,
        require_matching_keys=args.require_matching_keys,
    )
    report = render_markdown(comparison)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote benchmark compare report to {args.output}", file=sys.stderr)

    if not comparison.passed and not args.warn_only:
        raise SystemExit(1)


def _best_rows_by_key(rows: list[ReportRow]) -> dict[CompareKey, ReportRow]:
    best: dict[CompareKey, ReportRow] = {}
    for row in rows:
        key = (row.primitive, row.operation, row.dtype, row.shape, row.backend)
        current = best.get(key)
        if current is None or _score(row) > _score(current):
            best[key] = row
    return best


def _score(row: ReportRow) -> float:
    if row.tflops > 0:
        return row.tflops
    if row.bandwidth_gbps > 0:
        return row.bandwidth_gbps
    if row.p50_ms > 0:
        return 1.0 / row.p50_ms
    return 0.0


def _regression_metric(
    baseline: ReportRow,
    candidate: ReportRow,
) -> tuple[str, float, float, float]:
    if baseline.tflops > 0 or candidate.tflops > 0:
        baseline_value = baseline.tflops
        candidate_value = candidate.tflops
        return (
            "TFLOP/s",
            baseline_value,
            candidate_value,
            _throughput_regression_pct(baseline_value, candidate_value),
        )
    if baseline.bandwidth_gbps > 0 or candidate.bandwidth_gbps > 0:
        baseline_value = baseline.bandwidth_gbps
        candidate_value = candidate.bandwidth_gbps
        return (
            "GB/s",
            baseline_value,
            candidate_value,
            _throughput_regression_pct(baseline_value, candidate_value),
        )
    return (
        "p50 ms",
        baseline.p50_ms,
        candidate.p50_ms,
        _latency_regression_pct(baseline.p50_ms, candidate.p50_ms),
    )


def _throughput_regression_pct(baseline_value: float, candidate_value: float) -> float:
    if baseline_value <= 0:
        return 0.0
    return max(0.0, (baseline_value - candidate_value) / baseline_value * 100.0)


def _latency_regression_pct(baseline_value: float, candidate_value: float) -> float:
    if baseline_value <= 0:
        return 0.0
    return max(0.0, (candidate_value - baseline_value) / baseline_value * 100.0)


def _shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()
