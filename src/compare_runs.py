from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _find_metrics_files(root: Path) -> List[Path]:
    return list(root.rglob("metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metrics.json files.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory that contains run subfolders.",
    )
    args = parser.parse_args()

    root = Path(args.runs_dir)
    files = _find_metrics_files(root)
    if not files:
        print(f"No metrics.json found in {root}")
        return

    rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            (
                path.parent.name,
                data.get("model"),
                data.get("best_val_f1_macro"),
                data.get("test_acc"),
                data.get("test_f1_macro"),
                data.get("test_f1_weighted"),
            )
        )

    rows.sort(key=lambda x: (x[1] or "", x[0]))

    print("run_dir\tmodel\tbest_val_f1_macro\ttest_acc\ttest_f1_macro\ttest_f1_weighted")
    for row in rows:
        print("\t".join("" if v is None else str(v) for v in row))


if __name__ == "__main__":
    main()
