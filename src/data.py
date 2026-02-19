from __future__ import annotations

from typing import Iterable, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CarColorDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[str],
        labels: Iterable[int],
        transform=None,
    ) -> None:
        self.paths: List[str] = list(paths)
        self.labels: List[int] = list(labels)
        self.transform = transform

        if len(self.paths) != len(self.labels):
            raise ValueError("paths and labels must have the same length")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]
        label = self.labels[idx]

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_tf, eval_tf
