from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


CONDITIONS = [
    ("D0_clean", "d0_clean", 1.00),
    ("D1_jpeg50", "d1_jpeg50", 0.92),
    ("D2_jpeg30", "d2_jpeg30", 0.80),
    ("D3_resize50", "d3_resize50", 0.88),
    ("D4_resize35", "d4_resize35", 0.72),
    ("D5_combo50_50", "d5_combo50_50", 0.68),
    ("D6_combo30_50", "d6_combo30_50", 0.52),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small deterministic example without research data."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1024)

    rows = []
    labels = []
    for label, class_name in ((0, "real"), (1, "fake")):
        for video_number in range(4):
            video_id = f"{class_name}_{video_number:02d}"
            for frame_number in range(3):
                row_index = len(rows)
                rows.append(
                    {
                        "dataset": "Synthetic",
                        "row_index": row_index,
                        "label": label,
                        "class_name": class_name,
                        "video_id": video_id,
                        "frame_id": f"{frame_number:03d}",
                        "frame_path": (
                            f"frames/{class_name}/{video_id}/"
                            f"{frame_number:03d}.png"
                        ),
                    }
                )
                labels.append(label)
    labels_array = np.asarray(labels, dtype=np.int64)
    base_noise = rng.normal(0.0, 0.12, size=(len(rows), 2))

    for _, suffix, strength in CONDITIONS:
        sign = np.where(labels_array[:, None] == 1, 1.0, -1.0)
        features = base_noise + sign * np.asarray(
            [strength, strength * 0.65]
        )
        score_noise = rng.normal(0.0, 0.025, size=len(rows))
        scores = np.clip(
            0.5 + (labels_array * 2 - 1) * 0.42 * strength + score_noise,
            0.0,
            1.0,
        )
        np.savez(
            raw / f"synthetic_{suffix}.npz",
            scores=scores.astype(np.float32),
            labels=labels_array,
            features=features.astype(np.float32),
        )

    identity = root / "identity.jsonl"
    with identity.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    registry = {
        "version": 1,
        "identity_seed": 1024,
        "frame_num": 3,
        "datasets": [
            {
                "key": "synthetic",
                "display_name": "Synthetic",
                "expected_rows": len(rows),
                "expected_real_videos": 4,
                "expected_fake_videos": 4,
            }
        ],
        "conditions": [
            {"id": condition, "file_suffix": suffix}
            for condition, suffix, _ in CONDITIONS
        ],
        "combinations": [
            {
                "dataset": "synthetic",
                "detector": "SyntheticDetector",
                "feature_dim": 2,
                "file_template": "synthetic_{suffix}.npz",
            }
        ],
    }
    with (root / "registry.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(registry, handle, sort_keys=False)


if __name__ == "__main__":
    main()
