from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import load_yaml


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV JPEG decoding failed")
    return decoded


def resize_roundtrip(image: np.ndarray, scale: float) -> np.ndarray:
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    height, width = image.shape[:2]
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_condition(image: np.ndarray, condition: dict) -> np.ndarray:
    operation = condition["operation"]
    if operation == "clean":
        return np.asarray(image).copy()
    if operation == "jpeg":
        return jpeg_compress(image, int(condition["quality"]))
    if operation == "resize":
        return resize_roundtrip(image, float(condition["scale"]))
    if operation == "resize_then_jpeg":
        resized = resize_roundtrip(image, float(condition["scale"]))
        return jpeg_compress(resized, int(condition["quality"]))
    raise ValueError(f"Unknown operation: {operation}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply one frozen D0-D6 condition to an image."
    )
    parser.add_argument("--conditions", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_yaml(args.conditions)
    by_id = {item["id"]: item for item in config["conditions"]}
    if args.condition not in by_id:
        raise ValueError(f"Unknown condition {args.condition!r}")
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)
    output = apply_condition(image, by_id[args.condition])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), output):
        raise RuntimeError(f"Could not write {destination}")


if __name__ == "__main__":
    main()
