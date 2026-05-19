from __future__ import annotations

from pathlib import Path

from cuda_kernel_lab import nsight_summary


def test_parse_metrics_reads_key_nsight_csv_rows(tmp_path: Path) -> None:
    export = tmp_path / "ncu.csv"
    export.write_text(
        "\n".join(
            [
                "==PROF== Connected to process",
                "Metric Name,Metric Unit,Metric Value",
                "gpu__time_duration.sum,usec,42.5",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed,%,71.2",
                "ignored_metric,%,1",
            ]
        ),
        encoding="utf-8",
    )

    metrics = nsight_summary.parse_metrics(export)

    assert [(metric.label, metric.value) for metric in metrics] == [
        ("Kernel time", "42.5"),
        ("DRAM throughput", "71.2"),
    ]


def test_parse_metrics_reads_wide_nsight_raw_csv(tmp_path: Path) -> None:
    export = tmp_path / "ncu-raw.csv"
    export.write_text(
        "\n".join(
            [
                "==PROF== Connected to process",
                '"ID","Kernel Name","gpu__time_duration.sum",'
                '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",'
                '"dram__bytes_read.sum","launch__registers_per_thread"',
                '"","","ns","%","byte","register/thread"',
                '"0","_vector_add_kernel","419072","92.10","154414208","26"',
            ]
        ),
        encoding="utf-8",
    )

    metrics = nsight_summary.parse_metrics(export)

    assert [(metric.label, metric.value, metric.unit) for metric in metrics] == [
        ("Kernel time", "419072", "ns"),
        ("DRAM throughput", "92.10", "%"),
        ("DRAM bytes read", "154414208", "byte"),
        ("Registers per thread", "26", "register/thread"),
    ]


def test_render_markdown_includes_context_and_metrics(tmp_path: Path) -> None:
    export = tmp_path / "ncu.csv"
    export.write_text(
        "Metric Name,Metric Unit,Metric Value\n"
        "launch__registers_per_thread,register/thread,32\n",
        encoding="utf-8",
    )

    report = nsight_summary.render_markdown(
        nsight_summary.parse_metrics(export),
        title="Vector Add Profile",
        benchmark_command="uv run benchmark-memory",
        result_jsonl="experiments/results/profiled/memory.jsonl",
        operation="vector_add",
        strategy="triton-block-size",
    )

    assert "# Vector Add Profile" in report
    assert "- Operation: `vector_add`" in report
    assert "| Registers per thread | 32 | register/thread |" in report
