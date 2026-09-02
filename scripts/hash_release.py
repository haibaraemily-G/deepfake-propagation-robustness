from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_NAMES = {"MANIFEST_SHA256.txt"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a deterministic SHA256 manifest for release files."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="MANIFEST_SHA256.txt")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append(path)
    lines = [
        f"{digest(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(files)
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
