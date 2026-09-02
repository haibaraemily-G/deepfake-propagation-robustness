from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import numpy as np


LABELS = {
    "FF-SH": 1,
    "FF-F2F": 1,
    "FF-DF": 1,
    "FF-FS": 1,
    "FF-NT": 1,
    "FF-FH": 1,
    "FF-real": 0,
    "CelebDFv2_real": 0,
    "CelebDFv2_fake": 1,
}


def frame_number(path: str) -> int:
    normalized = path.replace("\\", "/")
    return int(PurePosixPath(normalized).stem)


def collect(
    manifest: Path,
    root_key: str,
    display_name: str,
    compression: str | None,
    frame_num: int,
) -> list[dict]:
    with manifest.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if root_key not in data:
        raise ValueError(f"{manifest} has no root {root_key!r}")
    rows = []
    for group in data[root_key].values():
        subset = group["test"]
        if compression is not None:
            subset = subset[compression]
        for manifest_video_name, video in subset.items():
            class_name = video["label"]
            if class_name not in LABELS:
                raise ValueError(f"Unsupported class label: {class_name}")
            paths = sorted(video["frames"], key=frame_number)
            if frame_num < len(paths):
                # This intentionally reproduces the evaluated DeepfakeBench
                # dataset behavior: retain the first frame_num sorted frames.
                paths = paths[:frame_num]
            video_id = f"{class_name}_{manifest_video_name}"
            for path in paths:
                normalized = str(path).replace("\\", "/")
                rows.append(
                    {
                        "dataset": display_name,
                        "row_index": -1,
                        "label": LABELS[class_name],
                        "class_name": class_name,
                        "video_id": video_id,
                        "frame_id": PurePosixPath(normalized).stem,
                        "frame_path": normalized,
                    }
                )
    return rows


def labels_sha256(rows: list[dict]) -> str:
    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )
    return hashlib.sha256(labels.tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct the evaluated feature-row identity order."
    )
    parser.add_argument("--ffpp-manifest", required=True)
    parser.add_argument("--celebdf-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-num", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    datasets = [
        collect(
            Path(args.ffpp_manifest),
            "FaceForensics++",
            "FF++ c23",
            "c23",
            args.frame_num,
        ),
        collect(
            Path(args.celebdf_manifest),
            "Celeb-DF-v2",
            "Celeb-DF-v2",
            None,
            args.frame_num,
        ),
    ]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    with destination.open("w", encoding="utf-8") as handle:
        for rows in datasets:
            random.Random(args.seed).shuffle(rows)
            for row_index, row in enumerate(rows):
                row["row_index"] = row_index
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            metadata[rows[0]["dataset"]] = {
                "rows": len(rows),
                "labels_sha256": labels_sha256(rows),
            }
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
