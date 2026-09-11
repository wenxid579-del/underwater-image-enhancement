from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_pairs_manifest(path: Path) -> list[tuple[str, str]]:
    """读取包含 input 和 target 两列的成对图像清单。"""
    if not path.is_file():
        raise FileNotFoundError(f"没有找到数据清单：{path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"数据清单为空：{path}")
    required = {"input", "target"}
    if not required.issubset(rows[0]):
        raise ValueError(f"数据清单必须包含 input 和 target 两列：{path}")
    pairs = [(row["input"].strip(), row["target"].strip()) for row in rows]
    if any(not input_name or not target_name for input_name, target_name in pairs):
        raise ValueError(f"数据清单中存在空路径：{path}")
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"数据清单中存在重复记录：{path}")
    return pairs


def load_image_tensor(path: Path, image_size: int | None = None) -> torch.Tensor:
    """将 RGB 图像读取为取值范围在 0 到 1 之间的张量。"""
    with Image.open(path) as image_file:
        image = image_file.convert("RGB")
        if image_size is not None:
            image = image.resize(
                (image_size, image_size),
                Image.Resampling.BICUBIC,
            )
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """将 CHW 图像张量转换为可保存的 RGB 图像。"""
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _paired_augment(
    image: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        image = image.flip(-1)
        target = target.flip(-1)
    rotations = random.randrange(4)
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
        target = torch.rot90(target, rotations, dims=(-2, -1))
    return image, target


class PairedImageDataset(Dataset):
    """按照清单读取水下图像及其参考图像。"""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        self.manifest = manifest_path.resolve()
        self.image_size = image_size
        self.augment = augment
        self.pairs = read_pairs_manifest(self.manifest)

        missing = [
            relative_path
            for pair in self.pairs
            for relative_path in pair
            if not (self.root / relative_path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"共有 {len(missing)} 个图像文件缺失，前几项为：{missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        input_name, target_name = self.pairs[index]
        image = load_image_tensor(self.root / input_name, self.image_size)
        target = load_image_tensor(self.root / target_name, self.image_size)
        if self.augment:
            image, target = _paired_augment(image, target)
        return image, target, input_name
