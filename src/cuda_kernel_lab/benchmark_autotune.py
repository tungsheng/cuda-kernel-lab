"""Summarize repeated matmul autotune rows into stable best-config artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from cuda_kernel_lab.benchmark_report import ReportRow, load_report_rows

DEFAULT_INPUT_DIR = Path("experiments/results/runpod/manual-run")
DEFAULT_OUTPUT_NAME = "h200-matmul-best.json"
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MAX_SPREAD_PCT = 12.0
DEFAULT_MAX_NOISE_RATIO = 1.25


@dataclass(frozen=True)
class CandidateSummary:
    dtype: str
    shape: tuple[int, ...]
    variant: str
    parameters: dict[str, Any]
    samples: int
    median_tflops: float
    best_tflops: float
    worst_tflops: float
    median_p50_ms: float
    best_p50_ms: float
    median_noise_ratio: float
    spread_pct: float
    stable: bool
    torch_best_tflops: float | None

    @property
    def triton_torch_pct(self) -> float | None:
        if self.torch_best_tflops is None or self.torch_best_tflops <= 0:
            return None
        return self.median_tflops / self.torch_best_tflops * 100.0

    def as_dict(self, *, selected: bool) -> dict[str, Any]:
        values = {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "variant": self.variant,
            "parameters": self.parameters,
            "samples": self.samples,
            "median_tflops": self.median_tflops,
            "best_tflops": self.best_tflops,
            "worst_tflops": self.worst_tflops,
            "median_p50_ms": self.median_p50_ms,
            "best_p50_ms": self.best_p50_ms,
            "median_noise_ratio": self.median_noise_ratio,
            "spread_pct": self.spread_pct,
            "stable": self.stable,
            "selected": selected,
            "torch_best_tflops": self.torch_best_tflops,
        }
        triton_torch_pct = self.triton_torch_pct
        if triton_torch_pct is not None:
            values["triton_torch_pct"] = triton_torch_pct
        return values


def summarize_autotune(
    input_dir: Path,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
    max_noise_ratio: float = DEFAULT_MAX_NOISE_RATIO,
) -> dict[str, Any]:
    """Return a best-config manifest for repeated Triton matmul rows."""

    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if max_spread_pct < 0:
        raise ValueError("max_spread_pct must be non-negative")
    if max_noise_ratio <= 0:
        raise ValueError("max_noise_ratio must be positive")

    rows = load_report_rows(input_dir)
    candidates = _candidate_summaries(
        rows,
        min_samples=min_samples,
        max_spread_pct=max_spread_pct,
        max_noise_ratio=max_noise_ratio,
    )
    winners = _winner_by_shape(candidates)

    return {
        "schema_version": 1,
        "kind": "h200-matmul-autotune",
        "input_dir": str(input_dir),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection": {
            "metric": "median_tflops",
            "min_samples": min_samples,
            "max_spread_pct": max_spread_pct,
            "max_noise_ratio": max_noise_ratio,
        },
        "winners": [
            summary.as_dict(selected=True)
            for summary in sorted(winners, key=lambda item: (item.dtype, item.shape))
        ],
        "candidates": [
            summary.as_dict(selected=summary in winners)
            for summary in sorted(
                candidates,
                key=lambda item: (
                    item.dtype,
                    item.shape,
                    -item.median_tflops,
                    item.variant,
                ),
            )
        ],
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render a compact autotune summary from a manifest."""

    lines = [
        "# H200 Matmul Autotune Summary",
        "",
        f"Status: selected {len(manifest.get('winners', []))} best configs",
        "",
        "## Selection",
        "",
    ]
    selection = manifest.get("selection")
    if isinstance(selection, dict):
        lines.extend(
            [
                f"- Metric: `{selection.get('metric')}`",
                f"- Minimum samples: `{selection.get('min_samples')}`",
                f"- Maximum spread: `{selection.get('max_spread_pct')}%`",
                f"- Maximum median noise ratio: `{selection.get('max_noise_ratio')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Winners",
            "",
            "| Dtype | Shape | Stable | Samples | Median TFLOP/s | Best TFLOP/s | "
            "Spread % | Median Noise | Triton/Torch % | Variant |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in manifest.get("winners", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            f"{row.get('dtype')} | {_shape_label(row.get('shape'))} | "
            f"{'yes' if row.get('stable') else 'no'} | "
            f"{row.get('samples')} | {_fmt(row.get('median_tflops'))} | "
            f"{_fmt(row.get('best_tflops'))} | {_fmt(row.get('spread_pct'))} | "
            f"{_fmt(row.get('median_noise_ratio'))} | "
            f"{_fmt(row.get('triton_torch_pct'))} | "
            f"{_escape_cell(str(row.get('variant') or ''))} |"
        )
    return "\n".join(lines) + "\n"


def default_output_for(input_dir: Path) -> Path:
    return input_dir / DEFAULT_OUTPUT_NAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--max-spread-pct", type=float, default=DEFAULT_MAX_SPREAD_PCT)
    parser.add_argument("--max-noise-ratio", type=float, default=DEFAULT_MAX_NOISE_RATIO)
    parser.add_argument("--dry-run", action="store_true", help="Print the JSON manifest.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = summarize_autotune(
        args.input_dir,
        min_samples=args.min_samples,
        max_spread_pct=args.max_spread_pct,
        max_noise_ratio=args.max_noise_ratio,
    )
    output = args.output or default_output_for(args.input_dir)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote autotune manifest to {output}", file=sys.stderr)

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(manifest), encoding="utf-8")
        print(f"Wrote autotune summary to {args.markdown_output}", file=sys.stderr)


def _candidate_summaries(
    rows: list[ReportRow],
    *,
    min_samples: int,
    max_spread_pct: float,
    max_noise_ratio: float,
) -> list[CandidateSummary]:
    triton_groups: dict[tuple[str, tuple[int, ...], str, str], list[ReportRow]] = defaultdict(list)
    torch_best: dict[tuple[str, tuple[int, ...]], float] = {}
    for row in rows:
        if row.primitive != "matmul" or row.operation != "matmul":
            continue
        shape_key = (row.dtype, row.shape)
        if row.backend == "torch":
            torch_best[shape_key] = max(torch_best.get(shape_key, 0.0), row.tflops)
        elif row.backend == "triton":
            triton_groups[
                (
                    row.dtype,
                    row.shape,
                    row.variant,
                    json.dumps(row.parameters, sort_keys=True),
                )
            ].append(row)

    summaries = []
    for (dtype, shape, variant, _parameter_json), group_rows in triton_groups.items():
        tflops = [row.tflops for row in group_rows]
        p50_values = [row.p50_ms for row in group_rows]
        noise_values = [row.noise_ratio or 1.0 for row in group_rows]
        median_tflops = float(median(tflops))
        best_tflops = max(tflops)
        worst_tflops = min(tflops)
        spread_pct = (
            (best_tflops - worst_tflops) / median_tflops * 100.0
            if median_tflops > 0
            else 0.0
        )
        median_noise_ratio = float(median(noise_values))
        stable = (
            len(group_rows) >= min_samples
            and spread_pct <= max_spread_pct
            and median_noise_ratio <= max_noise_ratio
        )
        summaries.append(
            CandidateSummary(
                dtype=dtype,
                shape=shape,
                variant=variant,
                parameters=dict(group_rows[0].parameters),
                samples=len(group_rows),
                median_tflops=median_tflops,
                best_tflops=best_tflops,
                worst_tflops=worst_tflops,
                median_p50_ms=float(median(p50_values)),
                best_p50_ms=min(p50_values),
                median_noise_ratio=median_noise_ratio,
                spread_pct=spread_pct,
                stable=stable,
                torch_best_tflops=torch_best.get((dtype, shape)),
            )
        )
    return summaries


def _winner_by_shape(candidates: list[CandidateSummary]) -> list[CandidateSummary]:
    grouped: dict[tuple[str, tuple[int, ...]], list[CandidateSummary]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.dtype, candidate.shape)].append(candidate)

    winners = []
    for shape_candidates in grouped.values():
        stable_candidates = [candidate for candidate in shape_candidates if candidate.stable]
        pool = stable_candidates or shape_candidates
        winners.append(max(pool, key=lambda item: (item.median_tflops, item.best_tflops)))
    return winners


def _shape_label(shape: object) -> str:
    if isinstance(shape, list | tuple):
        return "x".join(str(dim) for dim in shape)
    return ""


def _fmt(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4g}"
    return ""


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    main()
