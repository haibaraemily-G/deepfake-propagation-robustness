from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    EPSILON,
    condition_path,
    dataset_rows,
    fdr,
    labels_from_rows,
    load_npz_output,
    load_yaml,
    read_identity,
    safe_key,
    write_csv,
)


SUMMARY_FIELDS = [
    "dataset",
    "detector",
    "d0_fdr",
    "d6_fdr",
    "fdr_ratio",
    "ci_lower",
    "ci_upper",
    "valid_resamples",
]

QC_FIELDS = [
    "dataset",
    "detector",
    "nonfinite_d0",
    "nonfinite_d6",
    "nonfinite_ratio",
    "d0_nonpositive",
    "minimum_bootstrap_d0",
    "point_zero_variance_dimensions_d0",
    "point_zero_variance_dimensions_d6",
    "valid_resamples",
    "anomaly",
]


def identity_index(
    rows: list[dict],
) -> dict[int, dict[str, object]]:
    result = {}
    for label in (0, 1):
        grouped: dict[str, list[int]] = {}
        for row in rows:
            if int(row["label"]) == label:
                grouped.setdefault(str(row["video_id"]), []).append(
                    int(row["row_index"])
                )
        video_ids = sorted(grouped)
        result[label] = {
            "video_ids": video_ids,
            "row_indices": [
                np.asarray(grouped[video_id], dtype=np.int64)
                for video_id in video_ids
            ],
        }
    return result


def video_moments(
    features: np.ndarray,
    index: dict[int, dict[str, object]],
) -> dict[int, dict[str, np.ndarray]]:
    result = {}
    dimension = features.shape[1]
    for label in (0, 1):
        row_groups = index[label]["row_indices"]
        counts = np.empty(len(row_groups), dtype=np.int64)
        sums = np.empty((len(row_groups), dimension), dtype=np.float64)
        sumsq = np.empty((len(row_groups), dimension), dtype=np.float64)
        for group_index, row_indices in enumerate(row_groups):
            values = features[row_indices]
            counts[group_index] = values.shape[0]
            sums[group_index] = values.sum(axis=0, dtype=np.float64)
            sumsq[group_index] = np.einsum(
                "ij,ij->j",
                values,
                values,
                dtype=np.float64,
                optimize=True,
            )
        result[label] = {"n": counts, "sum": sums, "sumsq": sumsq}
    return result


def sampling_plans(
    dataset_order: list[str],
    counts: dict[str, tuple[int, int]],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    plans = {}
    for dataset_key in dataset_order:
        real_count, fake_count = counts[dataset_key]
        real_draws = rng.integers(
            0,
            real_count,
            size=(iterations, real_count),
            dtype=np.int32,
        )
        fake_draws = rng.integers(
            0,
            fake_count,
            size=(iterations, fake_count),
            dtype=np.int32,
        )
        real_multiplicity = np.vstack(
            [
                np.bincount(row, minlength=real_count)
                for row in real_draws
            ]
        ).astype(np.int32)
        fake_multiplicity = np.vstack(
            [
                np.bincount(row, minlength=fake_count)
                for row in fake_draws
            ]
        ).astype(np.int32)
        plans[dataset_key] = {
            "real_draws": real_draws,
            "fake_draws": fake_draws,
            "real_multiplicity": real_multiplicity,
            "fake_multiplicity": fake_multiplicity,
        }
    return plans


def fdr_from_moments(
    real_n: np.ndarray,
    real_sum: np.ndarray,
    real_sumsq: np.ndarray,
    fake_n: np.ndarray,
    fake_sum: np.ndarray,
    fake_sumsq: np.ndarray,
) -> np.ndarray:
    real_mean = real_sum / real_n[:, None]
    fake_mean = fake_sum / fake_n[:, None]
    real_var_raw = real_sumsq / real_n[:, None] - real_mean**2
    fake_var_raw = fake_sumsq / fake_n[:, None] - fake_mean**2
    minimum = float(
        min(real_var_raw.min(initial=0.0), fake_var_raw.min(initial=0.0))
    )
    if minimum < -1e-10:
        raise RuntimeError(f"Numerically invalid variance: {minimum}")
    real_var = np.maximum(real_var_raw, 0.0)
    fake_var = np.maximum(fake_var_raw, 0.0)
    per_dimension = (
        (real_mean - fake_mean) ** 2
        / (real_var + fake_var + EPSILON)
    )
    return per_dimension.mean(axis=1)


def bootstrap_pair(
    d0: dict[int, dict[str, np.ndarray]],
    d6: dict[int, dict[str, np.ndarray]],
    plan: dict[str, np.ndarray],
    chunk_size: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for label in (0, 1):
        if not np.array_equal(d0[label]["n"], d6[label]["n"]):
            raise ValueError("D0 and D6 frame counts differ within a video")
    iterations = plan["real_multiplicity"].shape[0]
    dimension = d0[0]["sum"].shape[1]
    combined = {
        label: np.concatenate(
            [
                d0[label]["sum"],
                d0[label]["sumsq"],
                d6[label]["sum"],
                d6[label]["sumsq"],
            ],
            axis=1,
        )
        for label in (0, 1)
    }
    d0_values = np.empty(iterations, dtype=np.float64)
    d6_values = np.empty(iterations, dtype=np.float64)

    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        real_weight = plan["real_multiplicity"][start:stop].astype(
            np.float64,
            copy=False,
        )
        fake_weight = plan["fake_multiplicity"][start:stop].astype(
            np.float64,
            copy=False,
        )
        real_n = real_weight @ d0[0]["n"].astype(np.float64)
        fake_n = fake_weight @ d0[1]["n"].astype(np.float64)
        real_moments = real_weight @ combined[0]
        fake_moments = fake_weight @ combined[1]
        d0_real_sum, d0_real_sumsq, d6_real_sum, d6_real_sumsq = np.split(
            real_moments,
            [dimension, 2 * dimension, 3 * dimension],
            axis=1,
        )
        d0_fake_sum, d0_fake_sumsq, d6_fake_sum, d6_fake_sumsq = np.split(
            fake_moments,
            [dimension, 2 * dimension, 3 * dimension],
            axis=1,
        )
        d0_values[start:stop] = fdr_from_moments(
            real_n,
            d0_real_sum,
            d0_real_sumsq,
            fake_n,
            d0_fake_sum,
            d0_fake_sumsq,
        )
        d6_values[start:stop] = fdr_from_moments(
            real_n,
            d6_real_sum,
            d6_real_sumsq,
            fake_n,
            d6_fake_sum,
            d6_fake_sumsq,
        )
    ratios = d6_values / d0_values
    return d0_values, d6_values, ratios


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified paired video bootstrap for D6/D0 FDR ratios."
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
    indexes = {}
    counts = {}
    for dataset in registry["datasets"]:
        rows = dataset_rows(identity, dataset["display_name"])
        index = identity_index(rows)
        indexes[dataset["key"]] = index
        counts[dataset["key"]] = (
            len(index[0]["video_ids"]),
            len(index[1]["video_ids"]),
        )
        for label, field in ((0, "expected_real_videos"), (1, "expected_fake_videos")):
            expected = dataset.get(field)
            if expected is not None and len(index[label]["video_ids"]) != int(expected):
                raise ValueError(
                    f"{dataset['display_name']} {field} mismatch"
                )
    dataset_order = [item["key"] for item in registry["datasets"]]
    plans = sampling_plans(
        dataset_order,
        counts,
        args.iterations,
        args.seed,
    )

    summary_rows = []
    qc_rows = []
    distributions = {}
    for combination in registry["combinations"]:
        dataset_spec = dataset_specs[combination["dataset"]]
        rows = dataset_rows(identity, dataset_spec["display_name"])
        labels = labels_from_rows(rows)
        index = indexes[combination["dataset"]]
        d0_path = condition_path(
            args.raw_root,
            combination,
            conditions["D0_clean"],
            purpose="feature",
        )
        d6_path = condition_path(
            args.raw_root,
            combination,
            conditions["D6_combo30_50"],
            purpose="feature",
        )
        d0_data = load_npz_output(
            d0_path,
            labels,
            int(combination["feature_dim"]),
            require_features=True,
        )
        d6_data = load_npz_output(
            d6_path,
            labels,
            int(combination["feature_dim"]),
            require_features=True,
        )
        d0_point, d0_zero = fdr(d0_data["features"], labels)
        d6_point, d6_zero = fdr(d6_data["features"], labels)
        ratio_point = d6_point / d0_point
        d0_values, d6_values, ratios = bootstrap_pair(
            video_moments(d0_data["features"], index),
            video_moments(d6_data["features"], index),
            plans[combination["dataset"]],
        )
        nonfinite_d0 = int(np.count_nonzero(~np.isfinite(d0_values)))
        nonfinite_d6 = int(np.count_nonzero(~np.isfinite(d6_values)))
        nonfinite_ratio = int(np.count_nonzero(~np.isfinite(ratios)))
        d0_nonpositive = int(np.count_nonzero(d0_values <= 0.0))
        if nonfinite_d0 or nonfinite_d6 or nonfinite_ratio or d0_nonpositive:
            raise RuntimeError(
                f"Invalid bootstrap replicate for "
                f"{dataset_spec['display_name']}/{combination['detector']}"
            )
        interval = np.quantile(
            ratios,
            [0.025, 0.975],
            method=args.quantile_method,
        )
        key = safe_key(
            combination["dataset"],
            combination["detector"],
        )
        distributions[f"{key}__d0_fdr"] = d0_values
        distributions[f"{key}__d6_fdr"] = d6_values
        distributions[f"{key}__ratio"] = ratios
        summary_rows.append(
            {
                "dataset": dataset_spec["display_name"],
                "detector": combination["detector"],
                "d0_fdr": d0_point,
                "d6_fdr": d6_point,
                "fdr_ratio": ratio_point,
                "ci_lower": interval[0],
                "ci_upper": interval[1],
                "valid_resamples": args.iterations,
            }
        )
        qc_rows.append(
            {
                "dataset": dataset_spec["display_name"],
                "detector": combination["detector"],
                "nonfinite_d0": nonfinite_d0,
                "nonfinite_d6": nonfinite_d6,
                "nonfinite_ratio": nonfinite_ratio,
                "d0_nonpositive": d0_nonpositive,
                "minimum_bootstrap_d0": float(d0_values.min()),
                "point_zero_variance_dimensions_d0": d0_zero,
                "point_zero_variance_dimensions_d6": d6_zero,
                "valid_resamples": args.iterations,
                "anomaly": "none",
            }
        )

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "fdr_bootstrap_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    write_csv(
        output_dir / "fdr_bootstrap_qc.csv",
        qc_rows,
        QC_FIELDS,
    )
    np.savez_compressed(
        output_dir / "fdr_bootstrap_distributions.npz",
        **distributions,
    )
    sampling_arrays = {}
    for dataset_key, plan in plans.items():
        for name, values in plan.items():
            sampling_arrays[f"{dataset_key}__{name}"] = values
    np.savez_compressed(
        output_dir / "bootstrap_sampling_indices.npz",
        **sampling_arrays,
    )


if __name__ == "__main__":
    main()
