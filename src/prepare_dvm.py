from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


COLOR_MAP = {"grey": "gray"}


def extract_label(path: Path, sep: str, index: int) -> Optional[str]:
    parts = path.stem.split(sep)
    if len(parts) <= index:
        return None
    label = parts[index].strip()
    if not label:
        return None
    label = label.lower()
    return COLOR_MAP.get(label, label)


def collect_images(root: Path, extensions: List[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in extensions}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]


def split_dataset(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")
    if val_size + test_size == 0.0:
        empty = df.iloc[0:0].copy()
        return df, empty, empty

    class_counts = df["label"].value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each class needs at least 2 samples for stratified split.")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=seed,
    )
    if test_size == 0:
        return train_df, temp_df, temp_df.iloc[0:0]

    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DVM confirmed_fronts dataset splits."
    )
    parser.add_argument("--image-dir", required=True, help="Folder with images.")
    parser.add_argument(
        "--output-dir",
        default="data/dvm_front",
        help="Output folder for train/val/test CSVs.",
    )
    parser.add_argument("--separator", default="$$")
    parser.add_argument("--label-index", type=int, default=3)
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png",
        help="Comma-separated list of image extensions.",
    )
    parser.add_argument(
        "--drop-label",
        action="append",
        default=["unlisted"],
        help="Labels to drop (can repeat).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=200,
        help="Drop color classes with fewer than this many images.",
    )
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    extensions = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
    drop_labels = {label.strip().lower() for label in args.drop_label if label}

    paths = collect_images(image_dir, extensions)
    records: List[Dict[str, str]] = []
    for path in paths:
        label = extract_label(path, args.separator, args.label_index)
        if label is None or label in drop_labels:
            continue
        records.append({"path": str(path), "label": label})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No images found. Check image dir and filename format.")

    counts = df["label"].value_counts()
    keep_labels = counts[counts >= args.min_count].index.tolist()
    df = df[df["label"].isin(keep_labels)]

    if df.empty:
        raise ValueError("No data left after filtering. Adjust min-count or drop-label.")

    label_list = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    df["label_id"] = df["label"].map(label2id)

    train_df, val_df, test_df = split_dataset(
        df, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df[["path", "label", "label_id"]].to_csv(train_path, index=False)
    val_df[["path", "label", "label_id"]].to_csv(val_path, index=False)
    test_df[["path", "label", "label_id"]].to_csv(test_path, index=False)

    label_map = {
        "label2id": label2id,
        "id2label": {str(v): k for k, v in label2id.items()},
        "num_classes": len(label2id),
        "total_images": int(len(df)),
        "class_counts": counts.loc[keep_labels].to_dict(),
    }
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=True)

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    print(f"Classes: {len(label2id)}")
    print(counts.loc[keep_labels])


if __name__ == "__main__":
    main()
