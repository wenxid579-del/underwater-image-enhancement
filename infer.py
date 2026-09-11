from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import IMAGE_SUFFIXES, load_image_tensor, tensor_to_image
from utils import (
    build_model,
    load_config,
    model_state_from_checkpoint,
    resolve_project_path,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 AquaLiteFormer 增强水下图像")
    parser.add_argument("--config", default="configs/uieb.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型权重文件")
    parser.add_argument("--input", type=Path, required=True, help="单张图像或图像目录")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument(
        "--size",
        type=int,
        help="推理前将图像缩放为指定的正方形尺寸，默认保留原尺寸",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def collect_inputs(path: Path) -> tuple[list[Path], Path]:
    """收集单张图像或目录中的全部支持格式图像。"""
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图像格式：{path.suffix}")
        return [path], path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"没有找到输入图像或目录：{path}")
    images = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"输入目录中没有支持的图像：{path}")
    return images, path


def main() -> None:
    args = parse_args()
    if args.size is not None and args.size <= 0:
        raise ValueError("推理尺寸必须大于零")
    config = load_config(args.config)
    device = select_device(args.device)
    amp_enabled = bool(config["train"].get("amp", True) and device.type == "cuda")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"没有找到权重文件：{checkpoint_path}")
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)
    model.eval()

    input_path = args.input.expanduser().resolve()
    input_files, input_root = collect_inputs(input_path)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for image_path in tqdm(input_files, desc="正在增强"):
            image = load_image_tensor(image_path, args.size).unsqueeze(0).to(device)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                prediction = model(image)
            relative_path = image_path.relative_to(input_root).with_suffix(".png")
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_image(prediction[0]).save(output_path)

    print(f"处理完成，共增强 {len(input_files)} 张图像，结果位于：{output_dir}")


if __name__ == "__main__":
    main()
