from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from common import (
    condition_path,
    dataset_rows,
    full_video_groups,
    labels_from_rows,
    load_npz_output,
    load_yaml,
    read_identity,
    write_csv,
)


FIELDS = [
    "dataset",
    "detector",
    "n_videos",
    "n_real_videos",
    "n_fake_videos",
    "n_frames",
    "clean_video_auc",
    "clean_video_auc_ci_low",
    "clean_video_auc_ci_high",
    "d6_video_auc",
    "d6_video_auc_ci_low",
    "d6_video_auc_ci_high",
    "auc_drop",
    "auc_drop_pp",
    "auc_drop_ci_low",
    "auc_drop_ci_high",
    "auc_drop_ci_low_pp",
    "auc_drop_ci_high_pp",
    "ci_excludes_zero",
    "bootstrap_n",
    "bootstrap_seed",
    "identity_seed",
    "clean_npz",
    "d6_npz",
]


def video_scores(
    score: np.ndarray,
    groups: list[np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [float(score[indices].mean()) for indices in groups],
        dtype=np.float64,
    )


def bootstrap_pair(
    labels: np.ndarray,
    clean: np.ndarray,
    d6: np.ndarray,
    iterations: int,
    seed: int,
    quantile_method: str,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    clean_values = np.empty(iterations, dtype=np.float64)
    d6_values = np.empty(iterations, dtype=np.float64)
    drop_values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(
            0,
            labels.shape[0],
            size=labels.shape[0],
        )
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            raise RuntimeError(
                f"Bootstrap replicate {iteration} contains only one class"
            )
        clean_values[iteration] = roc_auc_score(
            sampled_labels,
            clean[indices],
        )
        d6_values[iteration] = roc_auc_score(
            sampled_labels,
            d6[indices],
        )
        drop_values[iteration] = (
            clean_values[iteration] - d6_values[iteration]
        )
    if not (
        np.isfinite(clean_values).all()
        and np.isfinite(d6_values).all()
        and np.isfinite(drop_values).all()
    ):
        raise RuntimeError("AUC bootstrap produced a non-finite value")
    alpha = 0.025
    quantile = lambda values: np.quantile(
        values,
        [alpha, 1.0 - alpha],
        method=quantile_method,
    )
    return {
        "clean": clean_values,
        "d6": d6_values,
        "drop": drop_values,
        "clean_ci": quantile(clean_values),
        "d6_ci": quantile(d6_values),
        "drop_ci": quantile(drop_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired video bootstrap for D0-D6 video-level AUC drop."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--quantile-method", default="linear")
    args = parser.parse_args()

    registry = load_yaml(args.registry)
    identity = read_identity(args.identity)
    dataset_specs = {item["key"]: item for item in registry["datasets"]}
    conditions = {item["id"]: item for item in registry["conditions"]}
    summary_rows = []
    distributions = {}

    for combination in registry["combinations"]:
        dataset_spec = dataset_specs[combination["dataset"]]
        display_name = dataset_spec["display_name"]
        rows = dataset_rows(identity, display_name)
        frame_labels = labels_from_rows(rows)
        _, groups, video_labels = full_video_groups(rows)
        clean_path = condition_path(
            args.raw_root,
            combination,
            conditions["D0_clean"],
            purpose="auc_score",
        )
        d6_path = condition_path(
            args.raw_root,
            combination,
            conditions["D6_combo30_50"],
            purpose="auc_score",
        )
        clean_data = load_npz_output(
            clean_path,
            frame_labels,
            int(combination["feature_dim"]),
            require_features=False,
        )
        d6_data = load_npz_output(
            d6_path,
            frame_labels,
            int(combination["feature_dim"]),
            require_features=False,
        )
        clean_video = video_scores(clean_data["scores"], groups)
        d6_video = video_scores(d6_data["scores"], groups)
        result = bootstrap_pair(
            video_labels,
            clean_video,
            d6_video,
            args.iterations,
            args.seed,
            args.quantile_method,
        )
        clean_point = float(roc_auc_score(video_labels, clean_video))
        d6_point = float(roc_auc_score(video_labels, d6_video))
        drop_point = clean_point - d6_point
        drop_ci = result["drop_ci"]
        key = (
            f"{combination['dataset']}__"
            f"{combination['detector'].lower().replace('-', '_')}"
        )
        distributions[f"{key}__clean_auc"] = result["clean"]
        distributions[f"{key}__d6_auc"] = result["d6"]
        distributions[f"{key}__auc_drop"] = result["drop"]
        summary_rows.append(
            {
                "dataset": display_name,
                "detector": combination["detector"],
                "n_videos": len(video_labels),
                "n_real_videos": int(np.count_nonzero(video_labels == 0)),
                "n_fake_videos": int(np.count_nonzero(video_labels == 1)),
                "n_frames": len(rows),
                "clean_video_auc": clean_point,
                "clean_video_auc_ci_low": result["clean_ci"][0],
                "clean_video_auc_ci_high": result["clean_ci"][1],
                "d6_video_auc": d6_point,
                "d6_video_auc_ci_low": result["d6_ci"][0],
                "d6_video_auc_ci_high": result["d6_ci"][1],
                "auc_drop": drop_point,
                "auc_drop_pp": drop_point * 100.0,
                "auc_drop_ci_low": drop_ci[0],
                "auc_drop_ci_high": drop_ci[1],
                "auc_drop_ci_low_pp": drop_ci[0] * 100.0,
                "auc_drop_ci_high_pp": drop_ci[1] * 100.0,
                "ci_excludes_zero": bool(
                    drop_ci[0] > 0.0 or drop_ci[1] < 0.0
                ),
                "bootstrap_n": args.iterations,
                "bootstrap_seed": args.seed,
                "identity_seed": registry["identity_seed"],
                "clean_npz": clean_path.name,
                "d6_npz": d6_path.name,
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "auc_bootstrap_summary.csv",
        summary_rows,
        FIELDS,
    )
    np.savez_compressed(
        output_dir / "auc_bootstrap_distributions.npz",
        **distributions,
    )


if __name__ == "__main__":
    main()
