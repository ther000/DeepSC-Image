# 基于 DeepSC 的图像鲁棒传输系统设计与实现

本项目是一个独立的 Python/PyTorch 工程，用于毕业设计题目“基于深度语义通信（DeepSC）的图像鲁棒传输系统设计与实现”。代码概念上参考端到端 JSCC/DeepSC 的“语义编码器 -> 可微信道 -> 语义解码器”范式，但采用新的包结构、卷积注意力模型、训练评估流程和 Streamlit GUI。

重要说明：仓库默认不包含训练好的 checkpoint。GUI 和推理 CLI 在没有 checkpoint 时会使用随机初始化模型，只能验证软件链路、界面和指标计算是否工作，不能代表 DeepSC 的真实性能。论文或答辩中的性能结论必须基于实际训练后的权重和完整评测日志。

## 功能覆盖

- 语义通信模型：`src/deepsc_image/model.py` 实现紧凑 CNN 编码器/解码器，并融合 SE 通道注意力和空间注意力。
- 可微信道：`none`、`awgn`、`rayleigh`，支持 `-5 dB ~ 20 dB` 等 SNR 评估范围。
- 损失函数：`MSE + (1 - SSIM)` 混合损失。
- 指标：PSNR、SSIM，供训练、评估、推理和 GUI 复用。
- 传统基线：PIL 内存 JPEG 编解码，并加入信道类退化，用于与 DeepSC 重构做可视化对比。
- 压缩率研究：训练配置支持 `semantic_channels` 列表扫参，并估算带宽/压缩比。
- Monte Carlo 评估：评估配置支持 `evaluation.monte_carlo_samples`，可在每个 SNR 下重复随机信道采样后取平均。
- CLI：训练、评估、单图推理、CPU smoke test 均支持 `python -m deepsc_image.<module>`。
- GUI：Streamlit 支持图片上传、信道选择、SNR 滑块、原图/DeepSC/JPEG 基线对比、PSNR/SSIM/延迟显示。
- 安全：checkpoint 加载限制为可信权重；GUI 上传图像有大小和像素数校验。

## 目录结构

```text
DeepSC-Image/
├── configs/
│   ├── train_cifar10_awgn.yaml
│   ├── train_cifar10_rayleigh.yaml
│   ├── eval_kodak.yaml
│   └── gui.yaml
├── datasets/
│   ├── cifar10/
│   └── kodak/
├── src/deepsc_image/
│   ├── app.py
│   ├── baseline.py
│   ├── channels.py
│   ├── data.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── inference.py
│   ├── interactive_cli.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── smoke_test.py
│   ├── train.py
│   └── utils.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 环境安装

建议使用虚拟环境。项目默认使用 `device: auto`，会优先使用可用的 NVIDIA GPU/CUDA；如果没有可用 GPU，则自动回退到 CPU。CPU 仍可运行 smoke test、推理和 GUI 功能验证。

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

执行过 `pip install -e .` 后，`deepsc_image` 会以可编辑模式安装到当前虚拟环境，后续在该虚拟环境中运行命令时不需要每次再设置 `$env:PYTHONPATH="src"`。如果没有执行可编辑安装，才需要临时设置 `PYTHONPATH`。

## CPU 冒烟测试

该测试不会下载数据集，会在随机张量上执行模型前向传播、三类信道、PSNR/SSIM、JPEG 基线和一次极小训练步骤。

```powershell
python -m deepsc_image.smoke_test
```

预期输出包含：

```text
SMOKE_TEST_OK loss=...
```

## 数据集位置

CIFAR-10 默认位置：

```text
datasets/cifar10/cifar-10-batches-py
```

Kodak 默认位置：

```text
datasets/kodak/kodim01.png
...
datasets/kodak/kodim24.png
```

当前配置默认 `download: false`，不会自动下载 CIFAR-10。如需自动下载，可把对应 YAML 中的 `download` 改为 `true`。

## 训练

配置驱动训练：

```powershell
python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml
python -m deepsc_image.train --config configs/train_cifar10_rayleigh.yaml
```

也可以在命令行交互式设置本次训练参数：

```powershell
python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml --interactive
```

交互模式会先读取 YAML，再以当前配置值作为默认值提示输入；直接回车保留默认值。交互结果只影响本次运行，不会改写 `configs/*.yaml`。训练输出目录中的 `config.yaml` 会记录本次实际使用的配置，便于复现实验。

快速连通性检查可临时减少训练轮数：

```powershell
python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml --epochs 1
```

训练参数说明：

| 参数 | YAML 位置 / CLI 参数 | 含义 | 建议取值 |
| --- | --- | --- | --- |
| 随机种子 | `seed` / `--seed` | 控制 PyTorch、NumPy 和 Python 随机性，便于复现实验。 | 默认 `42`；正式对比实验应固定。 |
| 计算设备 | `device` / `--device` | 训练使用的设备。`auto` 会优先使用 CUDA GPU，没有 GPU 时回退 CPU。 | 推荐 `auto`；也可显式设为 `cuda` 或 `cpu`。 |
| 数据集 | `dataset.name` / `--dataset` | 训练数据来源。`cifar10` 使用 CIFAR-10，`image_folder` 使用普通图片文件夹。 | 默认用 `cifar10`。 |
| 数据路径 | `dataset.root` / `--data-root` | 数据集根目录。CIFAR-10 默认读取 `datasets/cifar10/cifar-10-batches-py`。 | 本仓库默认 `datasets/cifar10`。 |
| 图像尺寸 | `dataset.image_size` / `--image-size` | 输入图像会 resize 到该尺寸后训练。尺寸越大越慢、显存占用越高。 | CIFAR-10 用 `32`；Kodak/自定义图片可按需求提高。 |
| 自动下载 | `dataset.download` / `--download` | 只对 CIFAR-10 生效，允许 torchvision 在缺失时下载数据。 | 本仓库已有数据，默认 `false`。 |
| 语义通道数 | `model.semantic_channels` / `--semantic-channels` | 编码器输出的语义特征通道数，决定传输符号数量和压缩率。越小压缩越强但重构更难；可写成列表做扫参。 | 正式训练常用 `16/32/64`。 |
| 基础通道数 | `model.base_channels` / `--base-channels` | CNN 主干宽度，影响模型容量、训练速度和显存。 | 正式训练默认 `32`。 |
| 信道类型 | `channel.type` / `--channel` | 训练时经过的信道模型：`none`、`awgn` 或 `rayleigh`。 | 论文实验重点用 `awgn` 和 `rayleigh`。 |
| 训练 SNR | `channel.train_snr_db` / `--snr-db` | 训练时随机抽取的 SNR 列表。列表越覆盖低中高 SNR，模型泛化越稳。 | 可用 `[0, 5, 10, 15, 20]`；低 SNR 鲁棒性可加入 `-5`。 |
| 训练轮数 | `training.epochs` / `--epochs` | 完整遍历训练集的次数。越大通常收敛越充分，但耗时更长。 | 连通性检查 `1`；正式训练建议至少几十轮并观察 Loss 曲线。 |
| Batch 大小 | `training.batch_size` / `--batch-size` | 每次优化使用的图片数量。越大训练更稳定但更占显存。 | CPU/小显存用 `8/16/32`；GPU 可用 `64` 或更高。 |
| 学习率 | `training.learning_rate` / `--learning-rate` | Adam 优化器步长，过大可能震荡，过小收敛慢。 | 默认 `0.001`。 |
| 权重衰减 | `training.weight_decay` / `--weight-decay` | L2 正则项，用于抑制过拟合。 | 默认 `0.0001`。 |
| SSIM 权重 | `training.ssim_weight` / `--ssim-weight` | 混合损失中 `(1 - SSIM)` 的权重 `alpha`，实际损失为 `(1 - alpha) * MSE + alpha * (1 - SSIM)`。 | 默认 `0.2`；更重视结构相似度可适当增大。 |
| 混合精度 | `training.amp.enabled` | CUDA 设备上启用 PyTorch AMP；CPU 下会安全降级为关闭。 | 默认 `false`；NVIDIA GPU 正式提速测试可改为 `true`。 |
| DataLoader worker | `training.dataloader.num_workers` | 数据加载子进程数。Windows/CPU 默认保持单进程以保证兼容。 | 默认 `0`；GPU 训练可从 `2/4` 试起。 |
| 固定页内存 | `training.dataloader.pin_memory` | CUDA 训练时可配合非阻塞拷贝降低 host-to-device 传输开销。 | 默认 `false`；CUDA benchmark 后再决定是否开启。 |
| 产物刷新频率 | `training.artifacts.*_every_epochs` | 分别控制 history、loss 曲线、checkpoint 的刷新间隔；最终 epoch 总会写出最终产物。 | 默认均为 `1`，保持每 epoch 刷新。 |
| 输出目录 | `training.output_dir` / `--output-dir` | 实验结果根目录，程序会在其下创建带时间戳和关键参数的子目录。 | 建议按信道或实验目的区分，如 `outputs/train_cifar10_awgn`。 |

`configs/train_cifar10_awgn.yaml` 演示了压缩率/语义通道扫参：

```yaml
model:
  semantic_channels: [16, 32, 64]
```

训练加速配置默认保持兼容；需要提速时可以在 YAML 中显式开启：

```yaml
model:
  # Benchmark 正式对比时建议用单个值；训练扫参仍可使用列表。
  semantic_channels: 16
training:
  amp:
    enabled: true
    dtype: float16
  dataloader:
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2
  artifacts:
    history_every_epochs: 1
    plot_every_epochs: 5
    checkpoint_every_epochs: 5
```

注意：`persistent_workers` 和 `prefetch_factor` 只会在 `num_workers > 0` 时生效；CPU 或非 CUDA 环境下 AMP 会记录为请求开启但实际关闭。`history_every_epochs`、`plot_every_epochs` 和 `checkpoint_every_epochs` 控制训练过程中的刷新频率，默认均为 `1`，因此默认仍保持每个 epoch 刷新；无论频率如何，最终 epoch 总会补写最终 history、loss 曲线和 checkpoint 产物。

训练吞吐 benchmark 可以用随机张量做有界对比，不会下载数据集：

```powershell
python -m deepsc_image.benchmark_training `
  --config configs/train_cifar10_awgn.yaml `
  --epochs 1 `
  --warmup 1 `
  --repeat 2 `
  --max-batches 2 `
  --semantic-channels 16 `
  --output outputs/benchmark/baseline.json

python -m deepsc_image.benchmark_training `
  --config configs/train_cifar10_awgn.yaml `
  --epochs 1 `
  --warmup 1 `
  --repeat 2 `
  --max-batches 2 `
  --semantic-channels 16 `
  --amp `
  --num-workers 4 `
  --pin-memory `
  --output outputs/benchmark/optimized.json
```

输出 JSON 会包含设备、PyTorch 版本、实际模型参数、AMP/DataLoader 实际配置、epoch 用时和 samples/sec 的 median/mean；CUDA 可用时还会记录峰值显存信息。benchmark 要求单个 `semantic_channels`，如果训练配置中是 `[16, 32, 64]` 这类列表，请用 `--semantic-channels` 明确指定本次测量的模型规模。

每次训练都会建立描述性实验目录，例如：

```text
outputs/train_cifar10_awgn/ts_05041200__ch_awgn__snr_0_5_10_15_20__sem_16__base_32__img_32__seed_42/
```

时间戳格式为 `MMDDhhmm`，例如 `05041200` 表示 05 月 04 日 12:00。当 `semantic_channels` 是列表时，`run_training()` 会为每个值建立一个子目录，并复用同一个时间戳，方便把同一轮扫参结果归组。

当前模型编码器有两次 stride=2 下采样，因此带宽估算为：

```text
bandwidth_ratio = semantic_channels * ceil(H/4) * ceil(W/4) / (3 * H * W)
compression_ratio = 1 / bandwidth_ratio
```

每个实验目录固定写入以下产物：

```text
best_model.pth
last_model.pth
config.yaml
history.csv
history.json
loss_curve.png
summary.json
```

说明：

- `best_model.pth`：训练损失最低的模型。当前没有验证集，因此 best 基于最低 `train_loss`。
- `last_model.pth`：最后一轮模型。
- `config.yaml`：本次有效配置，包含解析后的单个 `model.semantic_channels`、实际输出目录、时间戳和带宽估计。
- `history.csv` / `history.json`：每个 epoch 的 `epoch, train_loss, psnr, ssim, batches`，默认会在每个 epoch 结束后刷新；如果设置了 `training.artifacts.history_every_epochs`，则按该频率刷新，并在最终 epoch 补写最终结果。
- `loss_curve.png`：训练 Loss 曲线，默认会在每个 epoch 结束后刷新；如果设置了 `training.artifacts.plot_every_epochs`，则按该频率刷新，并在最终 epoch 补写最终曲线。
- `summary.json`：最佳轮次、最佳训练损失、最后一轮损失、checkpoint 文件名、输出目录和带宽估计，在训练结束后生成。

训练不再保存 `epoch_001.pth`、`epoch_002.pth` 这类逐 epoch checkpoint。评估、推理和 GUI 展示通常建议优先使用 `best_model.pth`。

## 评估

配置驱动评估：

```powershell
python -m deepsc_image.evaluate --config configs/eval_kodak.yaml --checkpoint outputs/train_cifar10_awgn/<experiment_dir>/best_model.pth
```

评估会按配置中的 SNR 列表输出 DeepSC 与 JPEG 基线的 PSNR/SSIM，并保存 `metrics.json`。提供 checkpoint 时，`output_dir` 会作为评估根目录，程序会在其下创建与训练实验目录和 checkpoint 文件名对应的子目录，例如 `outputs/eval_kodak/<experiment_dir>__ckpt_best_model/`，避免不同模型的评估结果互相覆盖。若未提供 checkpoint，程序会提示正在评估随机初始化模型，并直接写入配置中的 `output_dir`；这仅适合检查流程，不适合写入性能结论。

也可以在命令行交互式设置本次评估参数：

```powershell
python -m deepsc_image.evaluate --config configs/eval_kodak.yaml --interactive
```

交互模式同样只修改本次运行的内存配置，不会改写 YAML 文件。`--checkpoint`、`--snr-db`、`--monte-carlo-samples`、`--output-dir` 等参数也可以直接通过命令行覆盖。

`configs/eval_kodak.yaml` 中的 `evaluation.monte_carlo_samples` 控制每张图片、每个 SNR 的重复随机信道采样次数。`metrics.json` 会记录 `samples`、`monte_carlo_samples` 和总 `repetitions`，PSNR/SSIM 按所有重复结果平均。

## 单图推理

```powershell
python -m deepsc_image.infer `
  --input path/to/image.png `
  --checkpoint outputs/train_cifar10_awgn/<experiment_dir>/best_model.pth `
  --channel awgn `
  --snr-db 10 `
  --output outputs/infer/deepsc.png `
  --baseline-output outputs/infer/jpeg.png
```

没有 checkpoint 时也能运行，但会打印随机初始化模型的提示。

## Streamlit GUI

```powershell
streamlit run src/deepsc_image/app.py
```

GUI 功能：

1. 上传本地图像并自动 resize/归一化。
2. 选择 `none / awgn / rayleigh` 信道。
3. 使用 `-5 dB ~ 20 dB` SNR 滑块。
4. 同屏展示“发送原图 / DeepSC 重构图 / 传统 JPEG+信道基线”。
5. 自动显示 PSNR、SSIM 和 DeepSC 前向推理延迟。

`configs/gui.yaml` 中 `checkpoint: null` 表示默认随机模型。正式展示时建议在侧边栏选择训练好的 `best_model.pth`。

## 设计说明

### 模型

`DeepSCImageModel` 由 `SemanticEncoder`、可微信道和 `SemanticDecoder` 组成。编码器使用两次下采样提取低维语义特征，并加入 SE 注意力和空间注意力；编码后的语义张量进行平均功率归一化，便于在不同 SNR 下模拟发送符号。解码器用转置卷积逐级恢复空间分辨率，并通过 Sigmoid 输出 `[0, 1]` 图像。

### 信道

- `none`：用于无信道退化的上限/调试。
- `awgn`：按输入符号平均功率和 SNR 添加高斯噪声。
- `rayleigh`：加入平坦 Rayleigh 衰落、噪声和简化的完美信道均衡，保持端到端可微。

### 损失与指标

训练目标为：

```text
Loss = (1 - alpha) * MSE + alpha * (1 - SSIM)
```

默认 `alpha=0.2`。SSIM 与 PSNR 均在本项目中用 PyTorch 直接实现，避免依赖冷门包。

### JPEG/BPG 传统基线

本工程实现的是“JPEG + 信道类退化”的轻量传统基线：先用 PIL 在内存中 JPEG 压缩/解压，再按照所选信道和 SNR 加入噪声或衰落类视觉退化。任务书提到 JPEG/BPG，本项目优先选择无需额外系统依赖的 JPEG，保证 Windows/CPU 环境可直接运行。若后续需要更严格的传统通信基线，应在 `baseline.py` 中扩展为“编码 + 调制 + 信道 + 解调 + 解码”链路，或增加 BPG 编码器。

## 隐私与绿色通信分析要点

- 隐私与安全性：DeepSC 传输的是神经网络学习到的语义特征而非标准图像比特流。未知模型结构、权重和归一化方式时，窃听者直接从信道符号恢复可读图像的难度更高；但这不等价于密码学安全，正式系统仍应结合加密、认证和安全训练。
- 绿色通信：语义通信通过端到端学习压缩与抗噪特征，有机会在低 SNR 或带宽受限场景减少冗余编码、重传次数和传输时延，从而降低能耗。实际节能收益需要结合训练模型、信道统计、设备功耗和重传协议做实验量化。

## 任务书对应关系

| 任务要求 | 实现位置 |
| --- | --- |
| 卷积与注意力融合语义编码器/解码器 | `src/deepsc_image/model.py` |
| AWGN、Rayleigh 可微信道 | `src/deepsc_image/channels.py` |
| MSE+SSIM 混合损失 | `src/deepsc_image/losses.py` |
| PSNR/SSIM 自动计算 | `src/deepsc_image/metrics.py` |
| JPEG 传统基线 | `src/deepsc_image/baseline.py` |
| 训练、评估、推理 CLI | `train.py`, `evaluate.py`, `infer.py` |
| GUI 图片上传、SNR、对比展示 | `src/deepsc_image/app.py` |
| CIFAR-10/Kodak 配置入口 | `configs/*.yaml` |
| CPU smoke test | `src/deepsc_image/smoke_test.py` |

## 注意事项

- 本项目不会自动提交数据集、输出图、checkpoint 等大文件；这些目录已在 `.gitignore` 中排除。
- 任务书中部分 KPI 数值在解析文本中缺失，因此本文档不虚构具体阈值；只提供评估钩子和记录方式。
- 若要在论文中声明 DeepSC 优于基线，必须使用训练后的 checkpoint，并在相同数据集、SNR、信道和延迟测量条件下生成可复现实验结果。
