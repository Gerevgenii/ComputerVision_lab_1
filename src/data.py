from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CarColorDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["rel_path"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
