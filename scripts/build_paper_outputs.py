from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import read_csv, write_csv


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


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["detector"]


def plot_auc_drop(rows: list[dict[str, str]], output: Path) -> None:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    detectors = list(dict.fromkeys(row["detector"] for row in rows))
    conditions = [
        "D1_jpeg50",
        "D2_jpeg30",
        "D3_resize50",
        "D4_resize35",
        "D5_combo50_50",
        "D6_combo30_50",
    ]
    lookup = {
        (row["dataset"], row["detector"], row["condition"]): float(
            row["auc_drop_pp"]
        )
        for row in rows
    }
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(13, 4.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(conditions))
    width = 0.8 / len(detectors)
    for axis, dataset in zip(axes, datasets):
        for detector_index, detector in enumerate(detectors):
            values = [
                lookup[(dataset, detector, condition)]
                for condition in conditions
            ]
            offset = (
                detector_index - (len(detectors) - 1) / 2.0
            ) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                label=detector,
            )
        axis.set_title(dataset)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["D1", "D2", "D3", "D4", "D5", "D6"]
        )
        axis.set_xlabel("Condition")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Frame-level AUC drop (percentage points)")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_fdr_ratio(rows: list[dict[str, str]], output: Path) -> None:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    detectors = list(dict.fromkeys(row["detector"] for row in rows))
    conditions = [
        "D1_jpeg50",
        "D2_jpeg30",
        "D3_resize50",
        "D4_resize35",
        "D5_combo50_50",
        "D6_combo30_50",
    ]
    lookup = {
        (row["dataset"], row["detector"], row["condition"]): float(
            row["fdr_ratio"]
        )
        for row in rows
    }
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(13, 4.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(conditions))
    for axis, dataset in zip(axes, datasets):
        for detector in detectors:
            axis.plot(
                x,
                [
                    lookup[(dataset, detector, condition)]
                    for condition in conditions
                ],
                marker="o",
                linewidth=1.6,
                label=detector,
            )
        axis.axhline(0.3, color="0.5", linestyle="--", linewidth=1)
        axis.axhline(0.8, color="0.5", linestyle=":", linewidth=1)
        axis.set_title(dataset)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["D1", "D2", "D3", "D4", "D5", "D6"]
        )
        axis.set_xlabel("Condition")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("FDR ratio relative to D0")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_auc_curves(rows: list[dict[str, str]], output: Path) -> None:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    detectors = list(dict.fromkeys(row["detector"] for row in rows))
    conditions = [
        "D0_clean",
        "D1_jpeg50",
        "D2_jpeg30",
        "D3_resize50",
        "D4_resize35",
        "D5_combo50_50",
        "D6_combo30_50",
    ]
    lookup = {
        (row["dataset"], row["detector"], row["condition"]): float(
            row["auc"]
        )
        for row in rows
    }
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(13, 4.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(conditions))
    for axis, dataset in zip(axes, datasets):
        for detector in detectors:
            axis.plot(
                x,
                [
                    lookup[(dataset, detector, condition)]
                    for condition in conditions
                ],
                marker="o",
                label=detector,
            )
        axis.set_title(dataset)
        axis.set_xticks(x)
        axis.set_xticklabels(
            ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
        )
        axis.set_xlabel("Condition")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Frame-level AUC")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the paper tables and main figures from analysis CSVs."
    )
    parser.add_argument("--performance", required=True)
    parser.add_argument("--diagnosis", required=True)
    parser.add_argument("--auc-bootstrap", required=True)
    parser.add_argument("--fdr-bootstrap", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    performance = read_csv(args.performance)
    diagnosis = read_csv(args.diagnosis)
    auc_bootstrap = read_csv(args.auc_bootstrap)
    fdr_bootstrap = read_csv(args.fdr_bootstrap)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    write_csv(
        output / "intermediate_d0_d6_performance.csv",
        performance,
        PERFORMANCE_FIELDS,
    )
    write_csv(
        output / "intermediate_d0_d6_diagnosis.csv",
        diagnosis,
        DIAGNOSIS_FIELDS,
    )

    performance_lookup = {
        (row["dataset"], row["detector"], row["condition"]): row
        for row in performance
    }
    diagnosis_lookup = {
        (row["dataset"], row["detector"], row["condition"]): row
        for row in diagnosis
    }
    auc_lookup = {key(row): row for row in auc_bootstrap}
    fdr_lookup = {key(row): row for row in fdr_bootstrap}
    manuscript_table1_rows = []
    for auc_row in auc_bootstrap:
        pair = key(auc_row)
        d0 = performance_lookup[pair + ("D0_clean",)]
        d6 = performance_lookup[pair + ("D6_combo30_50",)]
        diagnosis_d6 = diagnosis_lookup[pair + ("D6_combo30_50",)]
        fdr_row = fdr_lookup[pair]
        manuscript_table1_rows.append(
            {
                "dataset": pair[0],
                "detector": pair[1],
                "clean_frame_auc": d0["auc"],
                "d6_frame_auc": d6["auc"],
                "d6_frame_auc_drop_pp": d6["auc_drop_pp"],
                "clean_fdr": diagnosis_d6["fdr_clean"],
                "d6_fdr": diagnosis_d6["fdr_degraded"],
                "d6_fdr_ratio": diagnosis_d6["fdr_ratio"],
                "d6_fdr_ratio_ci_lower": fdr_row["ci_lower"],
                "d6_fdr_ratio_ci_upper": fdr_row["ci_upper"],
                "failure_type": diagnosis_d6["failure_type"],
            }
        )
    write_csv(
        output / "manuscript_table1_d6_frame_auc_fdr.csv",
        manuscript_table1_rows,
        [
            "dataset",
            "detector",
            "clean_frame_auc",
            "d6_frame_auc",
            "d6_frame_auc_drop_pp",
            "clean_fdr",
            "d6_fdr",
            "d6_fdr_ratio",
            "d6_fdr_ratio_ci_lower",
            "d6_fdr_ratio_ci_upper",
            "failure_type",
        ],
    )
    write_csv(
        output / "manuscript_table2_video_auc_bootstrap.csv",
        auc_bootstrap,
        list(auc_bootstrap[0]),
    )
    write_csv(
        output / "table_fdr_bootstrap.csv",
        fdr_bootstrap,
        list(fdr_bootstrap[0]),
    )
    plot_auc_drop(performance, output / "fig_main_auc_drop.png")
    plot_fdr_ratio(diagnosis, output / "fig_main_fdr_ratio.png")
    plot_auc_curves(performance, output / "fig_main_auc_curve_d0_d6.png")


if __name__ == "__main__":
    main()
