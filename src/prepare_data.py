from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from sklearn.model_selection import train_test_split


def parse_color_from_name(path_in_zip: str) -> str | None:
    file_name = path_in_zip.rsplit("/", 1)[-1]
    parts = file_name.split("$$")
    if len(parts) < 4:
        return None
    return parts[3]


def read_records(zip_path: Path) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    counts: Counter = Counter()
    with ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            color = parse_color_from_name(name)
            if color is None:
                continue
            counts[color] += 1
            records.append({"rel_path": name, "color": color})
    return records, counts


def filter_records(records: list[dict], counts: Counter, min_samples: int) -> tuple[list[dict], list[str]]:
    valid_colors = sorted([c for c, n in counts.items() if c != "Unlisted" and n >= min_samples])
    valid_set = set(valid_colors)
    filtered: list[dict] = []
    color_to_label = {c: i for i, c in enumerate(valid_colors)}
    for row in records:
        color = row["color"]
        if color in valid_set:
            filtered.append({"rel_path": row["rel_path"], "color": color, "label": color_to_label[color]})
    return filtered, valid_colors


def stratified_split(records: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    y = [r["label"] for r in records]
    train_val, test = train_test_split(records, test_size=0.1, random_state=seed, stratify=y)
    y_train_val = [r["label"] for r in train_val]
    train, val = train_test_split(train_val, test_size=(1.0 / 9.0), random_state=seed, stratify=y_train_val)
    return train, val, test


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rel_path", "color", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_distribution(path: Path, rows: list[dict]) -> None:
    dist = Counter(r["color"] for r in rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["color", "count"])
        for color, count in sorted(dist.items(), key=lambda x: x[0]):
            writer.writerow([color, count])


def extract_filtered_images(zip_path: Path, image_root: Path, rows: list[dict]) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    need = [r["rel_path"] for r in rows]
    with ZipFile(zip_path, "r") as zf:
        for rel_path in need:
            dst = image_root / rel_path
            if dst.exists():
                continue
            zf.extract(rel_path, path=image_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-path", type=Path, default=Path("Confirmed_fronts.zip"))
    p.add_argument("--image-root", type=Path, default=Path("data/images"))
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--min-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-extract", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    records, full_counts = read_records(args.zip_path)
    filtered, classes = filter_records(records, full_counts, args.min_samples)

    if not filtered:
        raise RuntimeError("No records after filtering. Check dataset or min-samples value.")

    train_rows, val_rows, test_rows = stratified_split(filtered, args.seed)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.splits_dir / "train.csv", train_rows)
    write_csv(args.splits_dir / "val.csv", val_rows)
    write_csv(args.splits_dir / "test.csv", test_rows)

    write_distribution(args.splits_dir / "distribution_train.csv", train_rows)
    write_distribution(args.splits_dir / "distribution_val.csv", val_rows)
    write_distribution(args.splits_dir / "distribution_test.csv", test_rows)

    meta = {
        "seed": args.seed,
        "min_samples": args.min_samples,
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "total_images_before_filter": len(records),
        "total_images_after_filter": len(filtered),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "counts_before_filter": dict(full_counts),
    }
    with (args.splits_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.skip_extract:
        extract_filtered_images(args.zip_path, args.image_root, filtered)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
