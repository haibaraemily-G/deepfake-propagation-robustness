from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def equal_value(left: str, right: str, tolerance: float) -> bool:
    if left == right:
        return True
    try:
        left_number = float(left)
        right_number = float(right)
    except ValueError:
        return False
    return math.isclose(
        left_number,
        right_number,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two CSV files.")
    parser.add_argument("--actual", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--columns",
        nargs="*",
        help="Compare only these columns; default compares common columns.",
    )
    args = parser.parse_args()
    actual = read(args.actual)
    expected = read(args.expected)
    if len(actual) != len(expected):
        raise SystemExit(
            f"Row count differs: {len(actual)} != {len(expected)}"
        )
    if not actual:
        raise SystemExit("CSV files are empty")
    columns = args.columns or [
        column for column in expected[0] if column in actual[0]
    ]
    errors = []
    for row_index, (left, right) in enumerate(zip(actual, expected), start=2):
        for column in columns:
            if not equal_value(
                left.get(column, ""),
                right.get(column, ""),
                args.tolerance,
            ):
                errors.append(
                    f"row {row_index}, {column}: "
                    f"{left.get(column)!r} != {right.get(column)!r}"
                )
                if len(errors) >= 20:
                    break
        if len(errors) >= 20:
            break
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"PASS: {len(actual)} rows, {len(columns)} columns, "
        f"absolute tolerance {args.tolerance:g}"
    )


if __name__ == "__main__":
    main()
