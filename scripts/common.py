from __future__ import annotations

import csv
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import yaml


EPSILON = 1e-8


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def read_identity(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "dataset",
                "row_index",
                "label",
                "video_id",
                "frame_id",
                "frame_path",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"Identity line {line_number} is missing {sorted(missing)}"
                )
            row["label"] = int(row["label"])
            if row["label"] not in (0, 1):
                raise ValueError(f"Invalid label on identity line {line_number}")
            rows_by_dataset.setdefault(str(row["dataset"]), []).append(row)

    for dataset, rows in rows_by_dataset.items():
        indices = [int(row["row_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(
                f"row_index must be contiguous within {dataset}: "
                f"expected 0..{len(rows) - 1}"
            )
    return rows_by_dataset


def dataset_rows(
    rows_by_dataset: dict[str, list[dict[str, Any]]],
    display_name: str,
) -> list[dict[str, Any]]:
    if display_name not in rows_by_dataset:
        raise ValueError(f"Identity manifest has no dataset {display_name!r}")
    return rows_by_dataset[display_name]


def labels_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def condition_path(
    raw_root: str | Path,
    combination: dict[str, Any],
    condition: dict[str, Any],
    purpose: str = "combined",
) -> Path:
    template_key = {
        "combined": "file_template",
        "score": "score_file_template",
        "auc_score": "auc_score_file_template",
        "feature": "feature_file_template",
    }[purpose]
    template = combination.get(
        template_key,
        combination.get("file_template"),
    )
    if template is None:
        raise ValueError(
            f"No {purpose} file template for "
            f"{combination.get('dataset')}/{combination.get('detector')}"
        )
    filename = str(template).format(
        suffix=condition["file_suffix"],
        condition=condition["id"],
    )
    path = Path(raw_root) / filename
    lowered = path.name.lower()
    if "order_ablation" in lowered or "alt_jpeg30_resize50" in lowered:
        raise ValueError(
            f"Order-ablation file is not a main-condition input: {path.name}"
        )
    return path


def load_npz_output(
    path: str | Path,
    expected_labels: np.ndarray,
    expected_feature_dim: int,
    require_features: bool = True,
) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        score_key = "scores" if "scores" in files else "logits"
        required = {"labels", score_key}
        if require_features:
            required.add("features")
        missing = required.difference(files)
        if missing:
            raise ValueError(f"{path} is missing arrays {sorted(missing)}")
        labels = np.asarray(archive["labels"]).reshape(-1).astype(np.int64)
        scores = np.asarray(archive[score_key], dtype=np.float64).reshape(-1)
        result = {"labels": labels, "scores": scores}
        if require_features:
            result["features"] = np.asarray(
                archive["features"], dtype=np.float64
            )

    if labels.shape != expected_labels.shape:
        raise ValueError(f"Row count mismatch in {path}")
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f"Labels do not match the identity manifest: {path}")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"Score count mismatch in {path}")
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite score in {path}")
    if require_features:
        features = result["features"]
        if features.ndim != 2:
            raise ValueError(f"Features must be a 2D matrix: {path}")
        if features.shape != (labels.shape[0], int(expected_feature_dim)):
            raise ValueError(
                f"Feature shape mismatch in {path}: {features.shape}; "
                f"expected ({labels.shape[0]}, {expected_feature_dim})"
            )
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite feature in {path}")
    return result


def full_video_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    grouped: dict[str, list[int]] = {}
    labels: dict[str, int] = {}
    for index, row in enumerate(rows):
        video_id = str(row["video_id"])
        label = int(row["label"])
        if video_id in labels and labels[video_id] != label:
            raise ValueError(f"Video has conflicting labels: {video_id}")
        labels[video_id] = label
        grouped.setdefault(video_id, []).append(index)
    video_ids = list(grouped)
    indices = [
        np.asarray(grouped[video_id], dtype=np.int64)
        for video_id in video_ids
    ]
    video_labels = np.asarray(
        [labels[video_id] for video_id in video_ids],
        dtype=np.int64,
    )
    return video_ids, indices, video_labels


def legacy_video_key(row: dict[str, Any]) -> str:
    if row.get("legacy_video_id"):
        return str(row["legacy_video_id"])
    normalized = str(row["frame_path"]).replace("\\", "/")
    return PurePosixPath(normalized).parent.name


def aggregate_scores(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    grouped_label: dict[str, int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["video_id"])
            if mode == "full_identity"
            else legacy_video_key(row)
        )
        grouped.setdefault(key, []).append(index)
        label = int(row["label"])
        if key in grouped_label and grouped_label[key] != label:
            raise ValueError(f"Video group has conflicting labels: {key}")
        grouped_label[key] = label
    labels = []
    means = []
    for key, indices in grouped.items():
        labels.append(grouped_label[key])
        means.append(float(scores[np.asarray(indices, dtype=np.int64)].mean()))
    return np.asarray(labels, dtype=np.int64), np.asarray(means, dtype=np.float64)


def fdr(features: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    real = np.asarray(features[labels == 0], dtype=np.float64)
    fake = np.asarray(features[labels == 1], dtype=np.float64)
    if real.shape[0] == 0 or fake.shape[0] == 0:
        raise ValueError("FDR requires both real and fake samples")
    real_mean = real.mean(axis=0)
    fake_mean = fake.mean(axis=0)
    real_var = real.var(axis=0, ddof=0)
    fake_var = fake.var(axis=0, ddof=0)
    variance_sum = real_var + fake_var
    values = (real_mean - fake_mean) ** 2 / (variance_sum + EPSILON)
    result = float(values.mean())
    if not np.isfinite(result):
        raise ValueError("FDR is not finite")
    return result, int(np.count_nonzero(variance_sum == 0.0))


def safe_key(*parts: str) -> str:
    text = "__".join(parts).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def write_csv(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
