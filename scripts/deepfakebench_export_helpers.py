from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_DIMS = {
    "UCF": 256,
    "LSDA": 512,
    "Xception": 2048,
    "EfficientNet-B4": 1792,
    "SPSL": 2048,
}


def global_average_pool(array: Any) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 4:
        value = value.mean(axis=(-2, -1))
    if value.ndim != 2:
        raise ValueError(f"Expected [rows, features], got {value.shape}")
    return value


def validate_feature_batch(detector: str, features: Any) -> np.ndarray:
    if detector not in EXPECTED_DIMS:
        raise ValueError(f"Unknown detector: {detector}")
    pooled = global_average_pool(features)
    if pooled.shape[1] != EXPECTED_DIMS[detector]:
        raise ValueError(
            f"{detector} feature dimension is {pooled.shape[1]}, "
            f"expected {EXPECTED_DIMS[detector]}"
        )
    if not np.isfinite(pooled).all():
        raise ValueError("Feature batch contains NaN or Inf")
    return pooled


def save_export(
    output: str | Path,
    detector: str,
    scores: Any,
    labels: Any,
    features: Any,
) -> None:
    scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    features_array = validate_feature_batch(detector, features).astype(
        np.float32,
        copy=False,
    )
    if not (
        scores_array.shape[0]
        == labels_array.shape[0]
        == features_array.shape[0]
    ):
        raise ValueError("scores, labels, and features have different row counts")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        scores=scores_array,
        labels=labels_array,
        features=features_array,
    )
