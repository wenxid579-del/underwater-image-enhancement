from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from dataset import IMAGE_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 UIEB 数据并生成可复现的训练、验证和测试清单"
    )
    parser.add_argument("--root", type=Path, default=Path("datasets/UIEB"))
    parser.add_argument("--raw-dir", default="raw", help="原始水下图像目录")
    parser.add_argument("--reference-dir", default="reference", help="参考图像目录")
    parser.add_argument("--challenging-dir", default="challenging", help="困难集图像目录")
    parser.add_argument("--split-dir", default="splits", help="数据清单输出目录")
    parser.add_argument("--seed", type=int, default=42, help="数据划分随机种子")
    parser.add_argument("--train-count", type=int, default=720)
    parser.add_argument("--val-count", type=int, default=80)
    parser.add_argument("--expected-pairs", type=int, default=890)
    parser.add_argument("--expected-challenging", type=int, default=60)
    parser.add_argument(
        "--skip-challenging",
        action="store_true",
        help="没有下载困难集时跳过对应检查",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的数据划分清单",
    )
    return parser.parse_args()


def collect_images(directory: Path) -> dict[str, Path]:
    """递归收集图像，并用不区分大小写的文件名建立索引。"""
    if not directory.is_dir():
        raise FileNotFoundError(f"没有找到图像目录：{directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.name.casefold()
        if key in indexed:
            raise ValueError(
                f"目录中存在同名图像，无法可靠配对：{indexed[key]} 和 {path}"
            )
        indexed[key] = path
    if not indexed:
        raise ValueError(f"目录中没有支持的图像：{directory}")
    return indexed


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"图像目录必须位于数据集根目录内：{path}") from error


def write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("input", "target"))
        writer.writerows(pairs)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    raw_images = collect_images(root / args.raw_dir)
    reference_images = collect_images(root / args.reference_dir)

    missing_reference = sorted(set(raw_images) - set(reference_images))
    missing_raw = sorted(set(reference_images) - set(raw_images))
    if missing_reference or missing_raw:
        raise ValueError(
            "原始图像与参考图像无法一一对应；"
            f"缺少参考图像：{missing_reference[:5]}，缺少原始图像：{missing_raw[:5]}"
        )

    pair_count = len(raw_images)
    if args.expected_pairs > 0 and pair_count != args.expected_pairs:
        raise ValueError(
            f"成对图像数量应为 {args.expected_pairs}，实际为 {pair_count}"
        )
    if args.train_count <= 0 or args.val_count <= 0:
        raise ValueError("训练集和验证集数量必须大于零")
    if args.train_count + args.val_count >= pair_count:
        raise ValueError("训练集与验证集数量之和必须小于成对图像总数")

    pairs = [
        (
            relative_to_root(raw_images[key], root),
            relative_to_root(reference_images[key], root),
        )
        for key in sorted(raw_images)
    ]
    random.Random(args.seed).shuffle(pairs)
    train_end = args.train_count
    val_end = train_end + args.val_count
    split_pairs = {
        "train.csv": pairs[:train_end],
        "val.csv": pairs[train_end:val_end],
        "test.csv": pairs[val_end:],
    }

    challenging_paths: list[str] = []
    if not args.skip_challenging:
        challenging_images = collect_images(root / args.challenging_dir)
        if (
            args.expected_challenging > 0
            and len(challenging_images) != args.expected_challenging
        ):
            raise ValueError(
                "困难集图像数量应为 "
                f"{args.expected_challenging}，实际为 {len(challenging_images)}"
            )
        challenging_paths = [
            relative_to_root(challenging_images[key], root)
            for key in sorted(challenging_images)
        ]

    split_dir = root / args.split_dir
    targets = [split_dir / name for name in split_pairs]
    targets.extend((split_dir / "metadata.json", split_dir / "challenging.txt"))
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"数据清单已经存在，请检查后使用 --overwrite：{existing[0]}"
        )

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in split_pairs.items():
        write_pairs(split_dir / name, rows)
    if challenging_paths:
        (split_dir / "challenging.txt").write_text(
            "\n".join(challenging_paths) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "dataset": "UIEB",
        "split_source": "本仓库使用固定随机种子生成，并非 UIEB 官方划分",
        "seed": args.seed,
        "train": len(split_pairs["train.csv"]),
        "validation": len(split_pairs["val.csv"]),
        "test": len(split_pairs["test.csv"]),
        "challenging": len(challenging_paths),
    }
    (split_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"数据清单已写入：{split_dir}")


if __name__ == "__main__":
    main()
