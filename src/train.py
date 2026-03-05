from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import CarColorDataset
from src.models import build_model, count_parameters, parameter_groups


MODEL_CHOICES = ["custom_resnet18", "resnet18_pretrained", "resnet50_pretrained"]
CSV_HEADER = [
    "epoch",
    "train_loss",
    "train_acc",
    "train_f1_macro",
    "train_f1_weighted",
    "val_loss",
    "val_acc",
    "val_f1_macro",
    "val_f1_weighted",
    "lr_group0",
    "lr_group1",
    "epoch_seconds",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(image_size: int):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_tf, eval_tf


def get_num_classes(train_csv: Path) -> int:
    df = pd.read_csv(train_csv)
    return int(df["label"].nunique())


def get_class_weights(train_csv: Path, num_classes: int) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()
    n = len(df)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)
        weights.append(n / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(
    train_csv: Path,
    val_csv: Path,
    image_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    train_tf, eval_tf = make_transforms(image_size)
    train_ds = CarColorDataset(train_csv, image_root, train_tf)
    val_ds = CarColorDataset(val_csv, image_root, eval_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        targets_cpu = targets
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets_cpu.tolist())

    avg_loss = total_loss / max(len(all_targets), 1)
    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = get_device()

    output_dir = Path(args.output_dir) / args.model_name
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    num_classes = get_num_classes(Path(args.train_csv))
    model = build_model(args.model_name, num_classes=num_classes).to(device)

    total_params, trainable_params = count_parameters(model)

    groups = parameter_groups(model, args.model_name, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = AdamW(groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    class_weights = get_class_weights(Path(args.train_csv), num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader, val_loader = build_loaders(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        image_root=Path(args.image_root),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    config = {
        "model_name": args.model_name,
        "seed": args.seed,
        "device": str(device),
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "image_root": args.image_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "num_classes": num_classes,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_path = output_dir / "train_log.csv"

    start_epoch = 1
    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resume_payload = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])

        start_epoch = int(resume_payload["epoch"]) + 1
        best_f1 = float(resume_payload.get("val_f1_macro", -1.0))
        best_epoch = int(resume_payload.get("epoch", -1))

        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if float(summary.get("best_val_f1_macro", -1.0)) > best_f1:
                best_f1 = float(summary["best_val_f1_macro"])
                best_epoch = int(summary["best_epoch"])

        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    if start_epoch > args.epochs:
        summary = {"best_epoch": best_epoch, "best_val_f1_macro": best_f1}
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("No training needed: start_epoch is greater than epochs.")
        print(json.dumps(summary, indent=2))
        return

    log_mode = "a" if args.resume_checkpoint else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w" or log_path.stat().st_size == 0:
            writer.writerow(CSV_HEADER)
            f.flush()

        for epoch in range(start_epoch, args.epochs + 1):
            start = time.time()

            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

            lr_group0 = optimizer.param_groups[0]["lr"]
            lr_group1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - start

            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["acc"],
                    train_metrics["f1_macro"],
                    train_metrics["f1_weighted"],
                    val_metrics["loss"],
                    val_metrics["acc"],
                    val_metrics["f1_macro"],
                    val_metrics["f1_weighted"],
                    lr_group0,
                    lr_group1,
                    epoch_time,
                ]
            )
            f.flush()

            ckpt_payload = {
                "epoch": epoch,
                "model_name": args.model_name,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_f1_macro": val_metrics["f1_macro"],
                "num_classes": num_classes,
                "config": config,
            }

            save_checkpoint(checkpoints_dir / f"epoch_{epoch:03d}.pt", ckpt_payload)

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(checkpoints_dir / "best.pt", ckpt_payload)
            else:
                no_improve += 1

            scheduler.step()

            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"train_f1_macro={train_metrics['f1_macro']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} "
                f"time={epoch_time:.1f}s"
            )

            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best_val_f1_macro={best_f1:.4f}")
                break

    summary = {"best_epoch": best_epoch, "best_val_f1_macro": best_f1}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, required=True, choices=MODEL_CHOICES)
    p.add_argument("--train-csv", type=str, default="data/splits/train.csv")
    p.add_argument("--val-csv", type=str, default="data/splits/val.csv")
    p.add_argument("--image-root", type=str, default="data/images")
    p.add_argument("--output-dir", type=str, default="experiments")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-checkpoint", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
