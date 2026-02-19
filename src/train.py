from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import models

from data import CarColorDataset, build_transforms
from models.scratch_resnet import resnet18_scratch


PATH_COLS = ["path", "image_path", "file", "filename"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col in df.columns:
            return col
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def load_split(
    csv_path: Path,
    label_map: Optional[Dict[str, int]] = None,
    is_train: bool = False,
) -> Tuple[List[str], List[int], Optional[Dict[str, int]]]:
    df = pd.read_csv(csv_path)

    path_col = _find_col(df, PATH_COLS)
    if path_col is None:
        raise ValueError(f"Could not find path column in {csv_path}")

    if "label_id" in df.columns:
        paths = df[path_col].astype(str).tolist()
        labels = df["label_id"].astype(int).tolist()
        if label_map is None and "label" in df.columns:
            pairs = df[["label", "label_id"]].drop_duplicates()
            label_map = {row["label"]: int(row["label_id"]) for _, row in pairs.iterrows()}
        return paths, labels, label_map

    if "label" not in df.columns:
        raise ValueError(f"Could not find label column in {csv_path}")

    if label_map is None:
        if not is_train:
            raise ValueError("Label map is required for non-train split.")
        labels_sorted = sorted(df["label"].astype(str).unique())
        label_map = {label: idx for idx, label in enumerate(labels_sorted)}

    labels = df["label"].map(label_map)
    if labels.isna().any():
        raise ValueError("Found labels missing from label map.")
    return df[path_col].astype(str).tolist(), labels.astype(int).tolist(), label_map


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "scratch_resnet":
        return resnet18_scratch(num_classes)

    def _build_torchvision(builder, weights_enum):
        if pretrained:
            try:
                weights = weights_enum.DEFAULT if weights_enum else None
                return builder(weights=weights)
            except (AttributeError, TypeError):
                return builder(pretrained=True)
        try:
            return builder(weights=None)
        except TypeError:
            return builder(pretrained=False)

    if name == "resnet18":
        model = _build_torchvision(
            models.resnet18, getattr(models, "ResNet18_Weights", None)
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50":
        model = _build_torchvision(
            models.resnet50, getattr(models, "ResNet50_Weights", None)
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")


def freeze_early_layers(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("layer1") or name.startswith("layer2"):
            param.requires_grad = False


def split_resnet_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler],
    num_classes: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    f1_macro = f1_score(
        all_labels,
        all_preds,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )
    return avg_loss, acc, f1_macro


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    num_classes: int,
) -> Tuple[float, float, float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = (np.array(all_preds) == np.array(all_labels)).mean() if total else 0.0
    f1_macro = f1_score(
        all_labels,
        all_preds,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )
    f1_weighted = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        labels=list(range(num_classes)),
        zero_division=0,
    )
    return avg_loss, acc, f1_macro, f1_weighted, all_preds, all_labels


def write_confusion_matrix(
    output_dir: Path, labels: List[int], preds: List[int], id2label: Dict[int, str]
) -> None:
    label_ids = sorted(id2label.keys())
    cm = confusion_matrix(labels, preds, labels=label_ids)
    names = [id2label[idx] for idx in label_ids]
    df = pd.DataFrame(cm, index=names, columns=names)
    df.to_csv(output_dir / "confusion_matrix.csv")


def get_lr_values(optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    if len(optimizer.param_groups) == 1:
        lr = float(optimizer.param_groups[0]["lr"])
        return lr, lr
    return (
        float(optimizer.param_groups[0]["lr"]),
        float(optimizer.param_groups[1]["lr"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train car color classifier on DVM.")
    parser.add_argument("--train-csv", required=True, help="Train split CSV.")
    parser.add_argument("--val-csv", required=True, help="Validation split CSV.")
    parser.add_argument("--test-csv", required=True, help="Test split CSV.")
    parser.add_argument(
        "--model",
        default="scratch_resnet",
        choices=["scratch_resnet", "resnet18", "resnet50"],
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use ImageNet pretrained weights for torchvision models.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for checkpoints and metrics.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv)

    label_map: Optional[Dict[str, int]] = None
    train_paths, train_labels, label_map = load_split(
        train_csv, label_map=label_map, is_train=True
    )
    val_paths, val_labels, label_map = load_split(
        val_csv, label_map=label_map, is_train=False
    )
    test_paths, test_labels, label_map = load_split(
        test_csv, label_map=label_map, is_train=False
    )

    num_classes = max(train_labels) + 1 if train_labels else 0
    if label_map:
        num_classes = max(num_classes, max(label_map.values()) + 1)

    if num_classes < 2:
        raise ValueError("Need at least 2 classes to train.")

    pretrained = args.pretrained
    if pretrained is None:
        pretrained = args.model != "scratch_resnet"

    model = build_model(args.model, num_classes=num_classes, pretrained=pretrained)
    if pretrained and args.model in {"resnet18", "resnet50"}:
        freeze_early_layers(model)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    train_tf, eval_tf = build_transforms(img_size=args.img_size)
    train_ds = CarColorDataset(train_paths, train_labels, transform=train_tf)
    val_ds = CarColorDataset(val_paths, val_labels, transform=eval_tf)
    test_ds = CarColorDataset(test_paths, test_labels, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    lr = args.lr
    if lr is None:
        lr = 1e-3 if args.model == "scratch_resnet" else 1e-4

    if pretrained and args.model in {"resnet18", "resnet50"}:
        backbone_params, head_params = split_resnet_params(model)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * args.backbone_lr_mult},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=args.weight_decay,
        )

    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(train_labels, num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output) if args.output else Path("runs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "train_f1_macro",
            "val_loss",
            "val_acc",
            "val_f1_macro",
            "val_f1_weighted",
            "lr_backbone",
            "lr_head",
            "epoch_seconds",
        ],
    )
    log_writer.writeheader()
    log_file.flush()

    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scaler, num_classes
        )
        val_loss, val_acc, val_f1, val_f1_weighted, _, _ = evaluate(
            model, val_loader, device, loss_fn, num_classes
        )
        scheduler.step()

        lr_backbone, lr_head = get_lr_values(optimizer)
        epoch_seconds = time.perf_counter() - start_time

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_f1_macro": float(train_f1),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_f1_macro": float(val_f1),
                "val_f1_weighted": float(val_f1_weighted),
                "lr_backbone": float(lr_backbone),
                "lr_head": float(lr_head),
                "epoch_seconds": float(epoch_seconds),
            }
        )

        log_writer.writerow(history[-1])
        log_file.flush()

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_f1": best_f1,
            },
            checkpoint_path,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    log_file.close()

    if (output_dir / "best.pt").is_file():
        model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))

    test_loss, test_acc, test_f1, test_f1_weighted, preds, labels = evaluate(
        model, test_loader, device, loss_fn, num_classes
    )

    id2label: Dict[int, str] = {}
    if label_map:
        id2label = {int(v): k for k, v in label_map.items()}
    if id2label:
        write_confusion_matrix(output_dir, labels, preds, id2label)

    metrics = {
        "model": args.model,
        "pretrained": bool(pretrained),
        "num_classes": int(num_classes),
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_f1_weighted": float(test_f1_weighted),
        "history": history,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=True)

    if label_map:
        with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"label2id": label_map, "id2label": {str(v): k for k, v in label_map.items()}},
                f,
                indent=2,
                ensure_ascii=True,
            )

    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
