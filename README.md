# Underwater image enhancement

本仓库提供 AquaLiteFormer 水下图像增强网络的训练、验证和推理代码。项目以成对水下图像为训练数据，网络直接预测 RGB 残差，并将残差加到输入图像上得到增强结果。



## 主要内容

- 在 UIEB 成对数据上从头训练 AquaLiteFormer。
- 每轮训练后在验证集计算损失、PSNR 和 SSIM，并保存验证 PSNR 最好的权重。
- 使用独立脚本评估验证集或测试集。
- 对单张图片或整个目录执行增强，推理时不需要参考图像。
- 使用固定随机种子生成本仓库的数据划分，便于重复实验。

## 网络结构

AquaLiteFormer 采用三尺度编码器-解码器结构。默认通道数为 16、32 和 64，各阶段深度依次为 `(1, 2, 4, 2, 1)`，默认配置约有 29 万个可训练参数。

网络包含三个主要模块：

- `CVRA`：在通道转置注意力的值分支中串联局部细化和上下文细化，并按注意力头自适应融合两个尺度。
- `FDM-FFN`：使用局部均值分离低频与高频特征，再通过交叉门控调制两个频率分支。
- `DCAF`：从编码器特征中提取局部细节，并利用解码器特征进行交叉注意力校正。

CVRA 和 DCAF 的附加校正采用零初始化缩放参数，使网络在训练初期从较稳定的基础路径开始优化。最终输出采用全局残差形式：

```text
增强图像 = 输入图像 + 网络预测的校正量
```

## 目录说明

```text
underwater-image-enhancement/
├── configs/
│   └── uieb.yaml            # UIEB 训练配置
├── models/
│   ├── __init__.py
│   └── aqualiteformer.py    # AquaLiteFormer 网络
├── dataset.py               # 数据读取与成对增强
├── prepare_dataset.py       # 数据检查和划分
├── train.py                 # 训练及逐轮验证
├── validate.py              # 独立验证和测试
├── infer.py                 # 单图及目录推理
├── losses.py                # 训练损失
├── metrics.py               # PSNR 和 SSIM
├── utils.py                 # 配置、随机种子和公共函数
├── requirements.txt
└── LICENSE
```

## 环境准备

代码在以下环境中完成检查：

| 项目 | 版本 |
|---|---|
| Python | 3.10.18 |
| PyTorch | 2.5.1 |
| CUDA 构建版本 | 12.4 |
| NumPy | 2.2.6 |
| Pillow | 10.4.0 |
| PyYAML | 6.0.2 |

建议创建独立环境：

```powershell
conda create -n aquaformer python=3.10 -y
conda activate aquaformer
```

使用 NVIDIA 显卡时，先安装与本机驱动兼容的 PyTorch。下面是本项目测试过的 CUDA 12.4 版本：

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

只使用 CPU 时可以直接执行：

```powershell
pip install -r requirements.txt
```

## 下载 UIEB

UIEB 包含 890 对原始水下图像和参考图像，另有 60 张没有参考图像的困难样本。数据集官方规定仅用于学术和非商业用途，并明确禁止再次分发，因此本仓库不会提供数据副本、网盘转存或示例原图。

请从 [UIEB 官方页面](https://li-chongyi.github.io/proj_benchmark.html) 分别下载以下内容：

1. 原始水下图像 `Raws`
2. 参考图像 `References`
3. 困难样本 `Challenging Set`

官方页面列出的提取密码为：

| 内容 | 密码 |
|---|---|
| 原始图像 | `1234567` |
| 参考图像 | `8901234` |
| 困难样本 | `5678901` |

使用数据集时应同时引用 UIEB 论文，且数据集本身不受本仓库 MIT 许可证覆盖。

## 准备数据

下载并解压后，将三部分图像放到下面的目录。脚本会递归查找图片，因此压缩包自带的一层子目录可以保留。

```text
datasets/UIEB/
├── raw/          # 890 张原始水下图像
├── reference/    # 890 张参考图像
└── challenging/  # 60 张困难样本
```

执行数据检查和划分：

```powershell
python prepare_dataset.py --root datasets/UIEB
```

脚本会按文件名配对原始图像和参考图像，并检查缺失、重复和数量异常。默认使用随机种子 `42` 生成以下清单：

```text
datasets/UIEB/splits/
├── train.csv       # 720 对训练图像
├── val.csv         # 80 对验证图像
├── test.csv        # 90 对测试图像
├── challenging.txt
└── metadata.json
```

这是一套由本仓库生成的固定随机划分，不是 UIEB 官方训练测试划分。重新执行时如果清单已经存在，脚本会停止；确认需要重新划分后再增加 `--overwrite`。

如果只下载了成对数据，可以跳过困难样本检查：

```powershell
python prepare_dataset.py --root datasets/UIEB --skip-challenging
```

## 配置文件

默认配置位于 `configs/uieb.yaml`，相对路径均以仓库根目录为基准。

| 配置项 | 含义 |
|---|---|
| `data.root` | UIEB 数据根目录 |
| `data.image_size` | 训练和验证使用的正方形尺寸 |
| `model.base_channels` | 第一阶段基础通道数 |
| `model.depths` | 五个阶段各自的模块数量 |
| `train.epochs` | 训练轮数 |
| `train.batch_size` | 单次送入显存的图像数量 |
| `train.accumulation_steps` | 梯度累积步数 |
| `train.learning_rate` | AdamW 初始学习率 |
| `train.amp` | 使用 CUDA 时是否启用混合精度 |
| `loss.ssim_weight` | SSIM 损失权重 |

显存不足时优先减小 `train.batch_size`，再通过 `train.accumulation_steps` 保持有效批量大小。

## 检查训练流程

正式训练前建议先运行小规模检查。该命令只训练两轮，每轮最多读取 10 个训练批次，并最多验证 5 个批次：

```powershell
python train.py --config configs/uieb.yaml --smoke-test
```

检查权重写入：

```text
outputs/aqualiteformer/seed_42/smoke/
```

## 训练模型

使用默认配置训练：

```powershell
python train.py --config configs/uieb.yaml
```

训练过程中每轮执行一次验证，并在输出目录生成：

```text
outputs/aqualiteformer/seed_42/
├── best_psnr.pt  # 验证 PSNR 最好的完整检查点
├── last.pt       # 最近一轮的完整检查点
├── config.yaml   # 本次实际使用的配置
└── metrics.csv   # 每轮训练和验证记录
```

临时覆盖训练轮数：

```powershell
python train.py --config configs/uieb.yaml --epochs 50
```

从最近检查点继续训练：

```powershell
python train.py --config configs/uieb.yaml --resume outputs/aqualiteformer/seed_42/last.pt
```

恢复训练时应继续使用相同的模型结构和主要训练配置。不要加载来源不明的 PyTorch 权重文件。

## 验证和测试

在验证集上计算损失、PSNR 和 SSIM：

```powershell
python validate.py --config configs/uieb.yaml --checkpoint outputs/aqualiteformer/seed_42/best_psnr.pt --split val
```

在测试集上评估：

```powershell
python validate.py --config configs/uieb.yaml --checkpoint outputs/aqualiteformer/seed_42/best_psnr.pt --split test
```

验证脚本只在终端输出汇总指标，不保存逐图结果，也不会生成对比图。

## 图像推理

增强单张图像：

```powershell
python infer.py --config configs/uieb.yaml --checkpoint outputs/aqualiteformer/seed_42/best_psnr.pt --input path/to/image.jpg
```

增强整个目录：

```powershell
python infer.py --config configs/uieb.yaml --checkpoint outputs/aqualiteformer/seed_42/best_psnr.pt --input path/to/images --output-dir outputs/inference
```

推理默认保留输入分辨率，输出统一保存为 PNG。显存不足时可以指定推理尺寸：

```powershell
python infer.py --config configs/uieb.yaml --checkpoint outputs/aqualiteformer/seed_42/best_psnr.pt --input path/to/images --size 256
```

指定 `--size` 后，图像会直接缩放为对应的正方形尺寸，输出也保持该尺寸。



## 使用限制

- 仓库不提供预训练权重，首次使用需要自行训练。
- 训练和验证默认把图像缩放到 `256 × 256`，该设置可以在配置文件中修改。
- 推理可以处理不同尺寸的 RGB 图像，但大分辨率输入会增加显存占用。
- 仓库不附带实验结果，因此不对特定数据划分下的指标作承诺。

## 数据集引用

使用 UIEB 数据集时请引用：

```bibtex
@article{li2019underwater,
  title={An Underwater Image Enhancement Benchmark Dataset and Beyond},
  author={Li, Chongyi and Guo, Chunle and Ren, Wenqi and Cong, Runmin and Hou, Junhui and Kwong, Sam and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4376--4389},
  year={2019}
}
```

## 许可证

本仓库代码采用 MIT License，版权归 `wenxid579-del` 所有。UIEB 数据集使用其官方页面规定的独立条款，不随本仓库代码授权。
