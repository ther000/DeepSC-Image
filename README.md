# DeepSC-Image

基于 PyTorch 的图像语义通信实验工程，用于实现“语义编码器 -> 可微信道 -> 语义解码器”的端到端图像鲁棒传输流程。项目包含训练、评估、单图推理、Streamlit 可视化界面和传统 JPEG 基线，适合课程设计、毕业设计和小规模实验复现。

> 仓库不默认提供训练好的 checkpoint。未加载 checkpoint 时，推理和 GUI 会使用随机初始化模型，只能验证流程是否可运行，不能代表真实性能。

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 语义通信模型 | CNN 编码器/解码器，融合 SE 通道注意力和空间注意力 |
| 信道模型 | `none`、`awgn`、`rayleigh`，支持不同 SNR 条件 |
| 训练目标 | `MSE + (1 - SSIM)` 混合损失 |
| 评估指标 | PSNR、SSIM、推理延迟 |
| 传统基线 | PIL 内存 JPEG 编解码，并加入信道类退化 |
| 实验入口 | 训练、评估、单图推理、训练吞吐 benchmark、CPU smoke test |
| 可视化 | Streamlit 上传图片，对比原图、DeepSC 重构和 JPEG 基线 |

## 目录结构

```text
DeepSC-Image/
├── configs/                 # 训练、评估、GUI 配置
├── datasets/                # CIFAR-10、Kodak 或自定义图片数据
├── src/deepsc_image/        # 核心代码
├── tests/                   # Pytest 测试
├── outputs/                 # 训练和评估输出，通常不提交
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境安装

建议使用虚拟环境。项目 `device: auto` 会优先使用 CUDA，未检测到 GPU 时自动回退 CPU。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

快速检查：

```powershell
python -m deepsc_image.smoke_test
```

看到 `SMOKE_TEST_OK` 表示模型、信道、指标、JPEG 基线和最小训练步骤可正常运行。

## 数据准备

默认配置使用以下路径：

| 数据集 | 默认路径 | 用途 |
| --- | --- | --- |
| CIFAR-10 | `datasets/cifar10` | 训练 |
| Kodak | `datasets/kodak/kodim01.png` 至 `kodim24.png` | 评估 |

训练配置默认 `download: false`。如果本地没有 CIFAR-10，可将 `configs/train_cifar10_awgn.yaml` 或 `configs/train_cifar10_rayleigh.yaml` 中的 `download` 改为 `true`。

也可以使用普通图片文件夹，将配置中的 `dataset.name` 设为 `image_folder`，并修改 `dataset.root`。

## 常用命令

### 训练

```powershell
python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml
python -m deepsc_image.train --config configs/train_cifar10_rayleigh.yaml
```

快速连通性训练：

```powershell
python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml --epochs 1
```

交互式配置本次运行：

```powershell
python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml --interactive
```

`model.semantic_channels` 支持写成列表，例如 `[16, 32, 64]`，程序会分别训练并建立独立输出目录，便于比较压缩率。

### 评估

```powershell
python -m deepsc_image.evaluate `
  --config configs/eval_kodak.yaml `
  --checkpoint outputs/train_cifar10_awgn/<experiment_dir>/best_model.pth
```

常用覆盖参数：

```powershell
python -m deepsc_image.evaluate `
  --config configs/eval_kodak.yaml `
  --checkpoint path/to/best_model.pth `
  --channel awgn `
  --snr-db -5 0 5 10 15 20 `
  --monte-carlo-samples 5
```

评估结果会保存到 `outputs/eval_kodak/`，主要产物为 `metrics.json` 和指标曲线图。

### 单图推理

```powershell
python -m deepsc_image.infer `
  --input datasets\kodak\kodim24.png `
  --checkpoint outputs\train_cifar10_awgn\ts_05050955__ch_awgn__snr_0_5_10_15_20__sem_32__base_32__img_32__seed_42\best_model.pth `
  --channel awgn `
  --snr-db 0 `
  --output outputs/infer/deepsc.png `
  --baseline-output outputs/infer/jpeg.png
```

命令会输出 DeepSC 与 JPEG 基线的 PSNR、SSIM 和延迟，并保存两张重构图。

### GUI

```powershell
streamlit run src/deepsc_image/app.py
```

界面支持上传图片、选择 checkpoint、切换信道、调整 SNR，并展示原图、DeepSC 重构图、JPEG 基线图及指标。

### 训练吞吐 Benchmark

```powershell
python -m deepsc_image.benchmark_training `
  --config configs/train_cifar10_awgn.yaml `
  --epochs 1 `
  --warmup 1 `
  --repeat 2 `
  --max-batches 2 `
  --semantic-channels 16 `
  --output outputs/benchmark/baseline.json
```

该命令使用随机张量，不依赖真实数据集，适合比较 AMP、DataLoader 参数和模型规模对训练速度的影响。

## 配置说明

主要配置文件：

| 文件 | 说明 |
| --- | --- |
| `configs/train_cifar10_awgn.yaml` | CIFAR-10 + AWGN 训练配置 |
| `configs/train_cifar10_rayleigh.yaml` | CIFAR-10 + Rayleigh 训练配置 |
| `configs/eval_kodak.yaml` | Kodak 评估配置 |
| `configs/gui.yaml` | Streamlit 默认配置 |

常改字段：

| 字段 | 含义 |
| --- | --- |
| `dataset.root` | 数据集路径 |
| `dataset.image_size` | 输入图像尺寸 |
| `model.semantic_channels` | 语义通道数，影响压缩率和重构难度 |
| `channel.type` | 信道类型：`none`、`awgn`、`rayleigh` |
| `channel.train_snr_db` | 训练 SNR 列表 |
| `channel.snr_db` | 评估 SNR 列表 |
| `training.epochs` | 训练轮数 |
| `training.batch_size` | Batch 大小 |
| `training.output_dir` | 训练输出目录 |

## 训练产物

每次训练会在 `outputs/` 下创建带时间戳和关键参数的实验目录，例如：

```text
outputs/train_cifar10_awgn/ts_05041200__ch_awgn__snr_0_5_10_15_20__sem_16__base_32__img_32__seed_42/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `best_model.pth` | 训练损失最低的 checkpoint |
| `last_model.pth` | 最后一轮 checkpoint |
| `config.yaml` | 本次实际使用配置 |
| `history.csv` / `history.json` | 每轮 loss、PSNR、SSIM |
| `loss_curve.png` | 训练曲线 |
| `summary.json` | 最佳轮次、输出路径和带宽估计 |

当前模型两次 stride=2 下采样，带宽估算为：

```text
bandwidth_ratio = semantic_channels * ceil(H/4) * ceil(W/4) / (3 * H * W)
compression_ratio = 1 / bandwidth_ratio
```

## 代码模块

| 模块 | 作用 |
| --- | --- |
| `model.py` | DeepSC 图像编码器、信道连接和解码器 |
| `channels.py` | AWGN、Rayleigh 和无退化信道 |
| `losses.py` | MSE + SSIM 混合损失 |
| `metrics.py` | PSNR、SSIM |
| `baseline.py` | JPEG 基线 |
| `data.py` | CIFAR-10 和图片文件夹数据集 |
| `train.py` | 训练入口 |
| `evaluate.py` | 批量评估入口 |
| `infer.py` / `inference.py` | 单图推理逻辑 |
| `app.py` | Streamlit GUI |
| `benchmark_training.py` | 训练吞吐 benchmark |
| `smoke_test.py` | CPU 冒烟测试 |

## 注意事项

- 未加载 checkpoint 的结果只适合功能验证，不适合论文或答辩中的性能结论。
- 若要比较 DeepSC 与 JPEG，请保证数据集、图像尺寸、信道、SNR 和 checkpoint 条件一致。
- 当前 JPEG 基线是轻量实现，不等价于完整传统通信链路或 BPG 基线。
- `outputs/`、大数据集、checkpoint 等文件通常不应提交到 Git。
- 本项目的语义特征传输不等价于密码学安全，正式系统仍需结合加密和认证机制。
