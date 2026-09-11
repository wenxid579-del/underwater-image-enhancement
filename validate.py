from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedImageDataset
from losses import EnhancementLoss
from metrics import batch_psnr_ssim
from utils import (
    build_model,
    load_config,
    model_state_from_checkpoint,
    resolve_project_path,
    seed_everything,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 AquaLiteFormer 增强效果")
    parser.add_argument("--config", default="configs/uieb.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="待验证的权重文件")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def create_dataset(config: dict[str, Any], split: str) -> PairedImageDataset:
    data_config = config["data"]
    return PairedImageDataset(
        root=resolve_project_path(data_config["root"]),
        manifest=data_config[f"{split}_manifest"],
        image_size=int(data_config["image_size"]),
        augment=False,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    description: str = "验证",
    limit_batches: int | None = None,
) -> dict[str, float]:
    """计算数据集上的平均损失、PSNR 和 SSIM。"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0

    for batch_index, (images, targets, _) in enumerate(tqdm(loader, desc=description)):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            predictions = model(images)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"模型输出尺寸 {tuple(predictions.shape)} 与参考图像 "
                f"{tuple(targets.shape)} 不一致"
            )
        loss = criterion(predictions.float(), targets.float())
        psnr, ssim = batch_psnr_ssim(predictions, targets)
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_psnr += psnr.sum().item()
        total_ssim += ssim.sum().item()
        total_images += batch_size

    if total_images == 0:
        raise RuntimeError("数据加载器没有提供可验证的图像")
    return {
        "loss": total_loss / total_images,
        "psnr": total_psnr / total_images,
        "ssim": total_ssim / total_images,
        "images": float(total_images),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["train"]["seed"]))
    device = select_device(args.device)
    amp_enabled = bool(config["train"].get("amp", True) and device.type == "cuda")

    dataset = create_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config).to(device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"没有找到权重文件：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)

    loss_config = config["loss"]
    criterion = EnhancementLoss(
        ssim_weight=float(loss_config["ssim_weight"]),
        epsilon=float(loss_config["charbonnier_epsilon"]),
    ).to(device)
    result = evaluate_model(
        model,
        loader,
        criterion,
        device,
        amp_enabled,
        description=f"验证 {args.split}",
    )
    summary = {
        "split": args.split,
        "images": int(result["images"]),
        "loss": result["loss"],
        "psnr": result["psnr"],
        "ssim": result["ssim"],
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
