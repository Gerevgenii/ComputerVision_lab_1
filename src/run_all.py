from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

from src.evaluate import evaluate
from src.train import train


MODELS = ["custom_resnet18", "resnet18_pretrained", "resnet50_pretrained"]


def run_all(args: argparse.Namespace) -> None:
    experiment_root = Path(args.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name in MODELS:
        train_args = Namespace(
            model_name=model_name,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            image_root=args.image_root,
            output_dir=str(experiment_root),
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            patience=args.patience,
            seed=args.seed,
        )
        train(train_args)

        best_ckpt = experiment_root / model_name / "checkpoints" / "best.pt"
        eval_out = experiment_root / model_name / "test_eval"

        eval_args = Namespace(
            model_name=model_name,
            checkpoint=str(best_ckpt),
            csv=args.test_csv,
            meta_json=args.meta_json,
            image_root=args.image_root,
            output_dir=str(eval_out),
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
        )
        evaluate(eval_args)

        metrics = json.loads((eval_out / "metrics.json").read_text(encoding="utf-8"))
        results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "num_samples": metrics["num_samples"],
            }
        )

    results = sorted(results, key=lambda x: x["f1_macro"], reverse=True)

    summary_csv = experiment_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "f1_macro", "f1_weighted", "num_samples"])
        writer.writeheader()
        writer.writerows(results)

    summary_md = experiment_root / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("| model | accuracy | f1_macro | f1_weighted | num_samples |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                f"| {row['model']} | {row['accuracy']:.4f} | {row['f1_macro']:.4f} | {row['f1_weighted']:.4f} | {row['num_samples']} |\n"
            )

    print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=str, default="data/splits/train.csv")
    p.add_argument("--val-csv", type=str, default="data/splits/val.csv")
    p.add_argument("--test-csv", type=str, default="data/splits/test.csv")
    p.add_argument("--meta-json", type=str, default="data/splits/meta.json")
    p.add_argument("--image-root", type=str, default="data/images")
    p.add_argument("--experiment-root", type=str, default="experiments")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
