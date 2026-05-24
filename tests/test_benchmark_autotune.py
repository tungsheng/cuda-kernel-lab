from __future__ import annotations

import json
from pathlib import Path

from cuda_kernel_lab import benchmark_autotune


def test_autotune_prefers_stable_candidate_over_noisy_higher_peak(tmp_path: Path) -> None:
    params_g1 = _matmul_params(group_m=1)
    params_g4 = _matmul_params(group_m=4)
    records = [_record(name="torch:matmul", tflops=500.0, params=params_g1)]
    records.extend(
        _record(name="triton:matmul", tflops=tflops, params=params_g1)
        for tflops in (300.0, 305.0, 295.0)
    )
    records.extend(
        _record(name="triton:matmul", tflops=tflops, params=params_g4)
        for tflops in (250.0, 360.0, 370.0)
    )
    _write_jsonl(tmp_path / "matmul-autotune.jsonl", records)

    manifest = benchmark_autotune.summarize_autotune(tmp_path)
    winner = manifest["winners"][0]

    assert winner["parameters"]["group_m"] == 1
    assert winner["stable"] is True
    assert winner["median_tflops"] == 300.0
    assert winner["triton_torch_pct"] == 60.0
    assert any(
        candidate["parameters"]["group_m"] == 4 and candidate["stable"] is False
        for candidate in manifest["candidates"]
    )


def test_autotune_writes_manifest_and_markdown(tmp_path: Path) -> None:
    params = _matmul_params(group_m=4)
    _write_jsonl(
        tmp_path / "matmul-autotune.jsonl",
        [
            _record(name="torch:matmul", tflops=500.0, params=params),
            _record(name="triton:matmul", tflops=350.0, params=params),
            _record(name="triton:matmul", tflops=352.0, params=params),
            _record(name="triton:matmul", tflops=351.0, params=params),
        ],
    )
    output = tmp_path / "best.json"
    markdown_output = tmp_path / "best.md"

    benchmark_autotune.main(
        [
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    assert json.loads(output.read_text(encoding="utf-8"))["winners"][0]["stable"] is True
    assert "# H200 Matmul Autotune Summary" in markdown_output.read_text(encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _record(
    *,
    name: str,
    tflops: float,
    params: dict[str, object],
) -> dict[str, object]:
    return {
        "run": {
            "benchmark": "matmul",
            "args": params,
            "command": "uv run benchmark-matmul",
            "timestamp_utc": "2026-05-24T00:00:00+00:00",
            "git_commit": "abc123",
            "git_dirty": False,
            "host": {"python": "3.13.2", "platform": "Linux"},
            "packages": {"torch": "2.9.1", "triton": "3.5.1"},
            "cuda_devices": [{"index": 0, "name": "NVIDIA H200"}],
            "provider": {"name": "runpod", "gpu_id": "NVIDIA H200"},
        },
        "result": {
            "name": name,
            "device": "cuda",
            "dtype": "float16",
            "shape": [512, 11008, 4096],
            "p50_ms": 1.0,
            "p95_ms": 1.05,
            "p99_ms": 1.1,
            "bytes_moved": 1,
            "bandwidth_gbps": 1.0,
            "flops": 1,
            "tflops": tflops,
            "latencies_ms": [1.0, 1.05, 1.1],
            "strategy": "triton-tiled-dot" if name.startswith("triton") else "torch-baseline",
            "variant": _variant(params),
            "parameters": params,
            "metrics": {},
            "correctness": {
                "checked": True,
                "passed": True,
                "reference_backend": "torch",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": 1e-2,
                "rtol": 1e-2,
                "message": None,
            },
        },
    }


def _matmul_params(*, group_m: int) -> dict[str, object]:
    return {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "num_warps": 4,
        "num_stages": 4,
        "input_precision": "tf32",
        "group_m": group_m,
    }


def _variant(params: dict[str, object]) -> str:
    return (
        f"block_m={params['block_m']}, block_n={params['block_n']}, "
        f"block_k={params['block_k']}, num_warps={params['num_warps']}, "
        f"num_stages={params['num_stages']}, input_precision={params['input_precision']}, "
        f"group_m={params['group_m']}"
    )
