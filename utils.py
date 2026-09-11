from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

from models import AquaLiteFormer


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并检查顶层结构。"""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是映射结构：{config_path}")
    return config


def save_config(config: dict[str, Any], path: Path) -> None:
    """将本次实际使用的配置保存到输出目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def resolve_project_path(path: str | Path) -> Path:
    """将配置中的相对路径统一解释为相对于仓库根目录。"""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """为每个数据加载进程设置彼此独立且可复现的随机种子。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str) -> torch.device:
    """根据命令行设置和硬件状态选择计算设备。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA 设备")
    return torch.device(requested)


def build_model(config: dict[str, Any]) -> AquaLiteFormer:
    """按照配置创建 AquaLiteFormer。"""
    model_config = config.get("model", {})
    return AquaLiteFormer(
        base_channels=int(model_config.get("base_channels", 16)),
        depths=tuple(model_config.get("depths", (1, 2, 4, 2, 1))),
    )


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """生成当前随机种子对应的训练输出目录。"""
    output_root = resolve_project_path(
        config.get("experiment", {}).get("output_dir", "outputs")
    )
    seed = int(config["train"]["seed"])
    return output_root / "aqualiteformer" / f"seed_{seed}"


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """向训练日志追加一行，并在首次写入时创建表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def model_state_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    """兼容完整训练检查点和只含模型参数的权重文件。"""
    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件必须保存为字典结构")
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("权重文件中没有可用的模型参数")
    return state
