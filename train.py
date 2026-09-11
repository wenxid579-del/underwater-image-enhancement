from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedImageDataset
from losses import EnhancementLoss
from utils import (
    append_csv,
    build_model,
    load_config,
    model_state_from_checkpoint,
    resolve_output_dir,
    resolve_project_path,
    save_config,
    seed_everything,
    select_device,
    worker_init_fn,
)
from validate import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 AquaLiteFormer 水下图像增强模型")
    parser.add_argument("--config", default="configs/uieb.yaml")
    parser.add_argument("--epochs", type=int, help="覆盖配置文件中的训练轮数")
    parser.add_argument("--resume", type=Path, help="从训练检查点继续")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只运行少量批次，用于检查数据和训练流程",
    )
    return parser.parse_args()


def create_dataset(
    config: dict[str, Any],
    split: str,
    augment: bool,
) -> PairedImageDataset:
    data_config = config["data"]
    return PairedImageDataset(
        root=resolve_project_path(data_config["root"]),
        manifest=data_config[f"{split}_manifest"],
        image_size=int(data_config["image_size"]),
        augment=augment,
    )


def create_loader(
    dataset: PairedImageDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    use_cuda: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_psnr: float,
    config: dict[str, Any],
) -> None:
    """保存可恢复训练的完整检查点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("训练轮数必须大于零")
        config["train"]["epochs"] = args.epochs
    if args.smoke_test:
        config["train"]["epochs"] = 2

    train_config = config["train"]
    seed = int(train_config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    amp_enabled = bool(train_config.get("amp", True) and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_dataset = create_dataset(config, "train", augment=True)
    val_dataset = create_dataset(config, "val", augment=False)
    num_workers = int(train_config.get("num_workers", 0))
    train_loader = create_loader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        use_cuda=device.type == "cuda",
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        num_workers=num_workers,
        shuffle=False,
        seed=seed,
        use_cuda=device.type == "cuda",
    )

    model = build_model(config).to(device)
    loss_config = config["loss"]
    criterion = EnhancementLoss(
        ssim_weight=float(loss_config["ssim_weight"]),
        epsilon=float(loss_config["charbonnier_epsilon"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=(0.9, 0.999),
    )
    epochs = int(train_config["epochs"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_config.get("minimum_learning_rate", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    output_dir = resolve_output_dir(config)
    if args.smoke_test:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    metrics_path = output_dir / "metrics.csv"

    start_epoch = 1
    best_psnr = -math.inf
    if args.resume:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"没有找到训练检查点：{resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)
        if "optimizer_state" not in checkpoint or "scheduler_state" not in checkpoint:
            raise ValueError("该权重文件不包含继续训练所需的优化器状态")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", -math.inf))

    accumulation_steps = int(train_config.get("accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("梯度累积步数必须大于零")
    gradient_clip = float(train_config.get("gradient_clip", 0.0))
    limit_train_batches = 10 if args.smoke_test else None
    limit_val_batches = 5 if args.smoke_test else None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"模型参数量：{parameter_count:,}；设备：{device}；"
        f"训练图像：{len(train_dataset)}；验证图像：{len(val_dataset)}；"
        f"混合精度：{amp_enabled}"
    )

    if start_epoch > epochs:
        print(f"检查点已经完成第 {start_epoch - 1} 轮，无需继续训练")
        return

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_images = 0
        processed_batches = min(
            len(train_loader),
            limit_train_batches or len(train_loader),
        )
        progress = tqdm(train_loader, total=processed_batches, desc=f"训练第 {epoch} 轮")
        for batch_index, (images, targets, _) in enumerate(progress):
            if limit_train_batches is not None and batch_index >= limit_train_batches:
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
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"第 {epoch} 轮第 {batch_index + 1} 批出现非有限损失"
                )
            scaler.scale(loss / accumulation_steps).backward()

            should_update = (batch_index + 1) % accumulation_steps == 0
            is_last_batch = batch_index + 1 == processed_batches
            if should_update or is_last_batch:
                if gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_images, 1)
        validation = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled,
            description=f"验证第 {epoch} 轮",
            limit_batches=limit_val_batches,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        max_memory_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else 0.0
        )

        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            save_checkpoint(
                output_dir / "best_psnr.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                config,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_psnr,
            config,
        )
        append_csv(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{validation['loss']:.8f}",
                "val_psnr": f"{validation['psnr']:.6f}",
                "val_ssim": f"{validation['ssim']:.6f}",
                "learning_rate": f"{learning_rate:.10f}",
                "elapsed_seconds": f"{elapsed:.2f}",
                "max_memory_mb": f"{max_memory_mb:.1f}",
            },
        )
        print(
            f"第 {epoch} 轮：训练损失 {train_loss:.5f}，"
            f"验证损失 {validation['loss']:.5f}，"
            f"PSNR {validation['psnr']:.3f}，SSIM {validation['ssim']:.4f}，"
            f"最佳 PSNR {best_psnr:.3f}"
        )

    print(f"训练完成，最佳权重位于：{output_dir / 'best_psnr.pt'}")


if __name__ == "__main__":
    main()
