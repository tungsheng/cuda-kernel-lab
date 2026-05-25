"""Build profile target names from benchmark autotune winner manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

TARGET_PREFIX = "matmul-autotune"
REQUIRED_PARAMETER_KEYS = (
    "block_m",
    "block_n",
    "block_k",
    "num_warps",
    "num_stages",
    "group_m",
)


def profile_targets_from_manifest(
    manifest: dict[str, Any],
    *,
    dtypes: tuple[str, ...] = (),
    shapes: tuple[tuple[int, int, int], ...] = (),
    limit: int | None = None,
) -> tuple[str, ...]:
    """Return encoded profile target names for selected autotune winners."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    dtype_filter = set(dtypes)
    shape_filter = set(shapes)
    targets = []
    for winner in manifest.get("winners", []):
        if not isinstance(winner, dict):
            continue
        dtype = str(winner.get("dtype") or "")
        shape = _shape(winner)
        if dtype_filter and dtype not in dtype_filter:
            continue
        if shape_filter and shape not in shape_filter:
            continue
        targets.append(_target_for_winner(winner, dtype=dtype, shape=shape))
        if limit is not None and len(targets) >= limit:
            break

    return tuple(targets)


def load_profile_targets(
    manifest_path: Path,
    *,
    dtypes: tuple[str, ...] = (),
    shapes: tuple[tuple[int, int, int], ...] = (),
    limit: int | None = None,
) -> tuple[str, ...]:
    """Load a manifest and return encoded profile target names."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return profile_targets_from_manifest(
        manifest,
        dtypes=dtypes,
        shapes=shapes,
        limit=limit,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        action="append",
        default=[],
        help="Winner dtype to include. Repeat to include more than one dtype.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Winner shape to include as MxNxK. Repeat to include more than one shape.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--format",
        choices=("targets", "json"),
        default="targets",
        help="Output newline-separated targets or JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    targets = load_profile_targets(
        args.manifest,
        dtypes=tuple(args.dtype),
        shapes=tuple(_parse_shape(value) for value in args.shape),
        limit=args.limit,
    )

    if args.format == "json":
        print(json.dumps({"targets": list(targets)}, indent=2, sort_keys=True))
    else:
        for target in targets:
            print(target)

    if not targets:
        print(f"warning: no profile targets found in {args.manifest}", file=sys.stderr)


def _target_for_winner(
    winner: dict[str, Any],
    *,
    dtype: str,
    shape: tuple[int, int, int],
) -> str:
    parameters = winner.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("winner is missing parameters")

    missing = [key for key in REQUIRED_PARAMETER_KEYS if key not in parameters]
    if missing:
        raise ValueError(f"winner is missing parameter(s): {', '.join(missing)}")

    input_precision = str(parameters.get("input_precision") or "tf32")
    m, n, k = shape
    return (
        f"{TARGET_PREFIX}-{dtype}-{m}x{n}x{k}"
        f"-bm{_positive_int(parameters, 'block_m')}"
        f"-bn{_positive_int(parameters, 'block_n')}"
        f"-bk{_positive_int(parameters, 'block_k')}"
        f"-w{_positive_int(parameters, 'num_warps')}"
        f"-s{_positive_int(parameters, 'num_stages')}"
        f"-gm{_positive_int(parameters, 'group_m')}"
        f"-ip{input_precision}"
    )


def _shape(winner: dict[str, Any]) -> tuple[int, int, int]:
    raw_shape = winner.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 3:
        raise ValueError("winner shape must be an MxNxK triple")
    return cast(tuple[int, int, int], tuple(int(dim) for dim in raw_shape))


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError("shape must be MxNxK") from exc
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("shape must be a positive MxNxK triple")
    return cast(tuple[int, int, int], parts)


def _positive_int(parameters: dict[str, Any], key: str) -> int:
    value = int(parameters[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


if __name__ == "__main__":
    main()
