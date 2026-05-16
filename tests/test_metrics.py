from __future__ import annotations

import pytest

from inference_kernel_lab.metrics import percentile


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 50)


def test_percentile_rejects_invalid_percent() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1], 101)

