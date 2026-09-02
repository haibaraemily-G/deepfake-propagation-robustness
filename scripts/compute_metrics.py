from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn import metrics

from common import (
    aggregate_scores,
    condition_path,
    dataset_rows,
    fdr,
    labels_from_rows,
    load_npz_output,
    load_yaml,
    read_identity,
    write_csv,
)


PERFORMANCE_FIELDS = [
    "dataset",
    "detector",
    "condition",
    "auc",
    "acc",
    "auc_drop_pp",
    "eer",
    "ap",
    "video_auc",
]

DIAGNOSIS_FIELDS = [
    "dataset",
    "detector",
    "condition",
    "fdr_clean",
    "fdr_degraded",
    "fdr_ratio",
    "score_shift_real",
    "score_shift_fake",
    "failure_type",
]


def frame_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = float(metrics.auc(fpr, tpr))
    fnr = 1.0 - tpr
    eer = float(fpr[np.nanargmin(np.abs(fnr - fpr))])
    ap = float(metrics.average_precision_score(labels, scores))
    prediction = (scores > 0.5).astype(np.int64)
    acc = float(np.mean(prediction == labels))
    return {"auc": auc, "acc": acc, "eer": eer, "ap": ap}


def video_auc_legacy(
    rows: list[dict],
    scores: np.ndarray,
) -> float:
    labels, video_scores = aggregate_scores(rows, scores, "legacy")
    fpr, tpr, _ = metrics.roc_curve(labels, video_scores, pos_label=1)
    return float(metrics.auc(fpr, tpr))


def failure_type(ratio: float, baseline: bool) -> str:
    if baseline:
        return "baseline"
    if ratio >= 0.8:
        return "A_threshold_or_score_shift"
    if ratio >= 0.3:
        return "B_feature_degradation"
    return "C_feature_collapse"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute frame metrics, legacy video AUC, FDR, and score shifts."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    registry = load_yaml(args.registry)
    identity = read_identity(args.identity)
    dataset_specs = {item["key"]: item for item in registry["datasets"]}
    output_dir = Path(args.output_dir)
    performance_rows = []
    diagnosis_rows = []

    for combination in registry["combinations"]:
        dataset_spec = dataset_specs[combination["dataset"]]
        display_name = dataset_spec["display_name"]
        rows = dataset_rows(identity, display_name)
        labels = labels_from_rows(rows)
        expected_rows = dataset_spec.get("expected_rows")
        if expected_rows is not None and len(rows) != int(expected_rows):
            raise ValueError(
                f"{display_name} has {len(rows)} rows; expected {expected_rows}"
            )

        loaded = {}
        for condition in registry["conditions"]:
            score_path = condition_path(
                args.raw_root, combination, condition, purpose="score"
            )
            feature_path = condition_path(
                args.raw_root, combination, condition, purpose="feature"
            )
            score_data = load_npz_output(
                score_path,
                labels,
                int(combination["feature_dim"]),
                require_features=False,
            )
            feature_data = load_npz_output(
                feature_path,
                labels,
                int(combination["feature_dim"]),
                require_features=True,
            )
            loaded[condition["id"]] = {
                "labels": labels,
                "scores": score_data["scores"],
                "features": feature_data["features"],
            }

        clean = loaded["D0_clean"]
        clean_frame = frame_metrics(labels, clean["scores"])
        clean_fdr, _ = fdr(clean["features"], labels)

        for condition in registry["conditions"]:
            condition_id = condition["id"]
            data = loaded[condition_id]
            current_frame = frame_metrics(labels, data["scores"])
            current_fdr, _ = fdr(data["features"], labels)
            ratio = current_fdr / clean_fdr
            if not np.isfinite(ratio):
                raise ValueError(
                    f"Non-finite FDR ratio for {display_name}/"
                    f"{combination['detector']}/{condition_id}"
                )
            real = labels == 0
            fake = labels == 1
            shift_real = float(
                data["scores"][real].mean() - clean["scores"][real].mean()
            )
            shift_fake = float(
                data["scores"][fake].mean() - clean["scores"][fake].mean()
            )
            performance_rows.append(
                {
                    "dataset": display_name,
                    "detector": combination["detector"],
                    "condition": condition_id,
                    "auc": current_frame["auc"],
                    "acc": current_frame["acc"],
                    "auc_drop_pp": (
                        clean_frame["auc"] - current_frame["auc"]
                    )
                    * 100.0,
                    "eer": current_frame["eer"],
                    "ap": current_frame["ap"],
                    "video_auc": video_auc_legacy(rows, data["scores"]),
                }
            )
            diagnosis_rows.append(
                {
                    "dataset": display_name,
                    "detector": combination["detector"],
                    "condition": condition_id,
                    "fdr_clean": clean_fdr,
                    "fdr_degraded": current_fdr,
                    "fdr_ratio": ratio,
                    "score_shift_real": shift_real,
                    "score_shift_fake": shift_fake,
                    "failure_type": failure_type(
                        ratio,
                        baseline=condition_id == "D0_clean",
                    ),
                }
            )

    write_csv(
        output_dir / "performance_long.csv",
        performance_rows,
        PERFORMANCE_FIELDS,
    )
    write_csv(
        output_dir / "diagnosis_long.csv",
        diagnosis_rows,
        DIAGNOSIS_FIELDS,
    )


if __name__ == "__main__":
    main()
