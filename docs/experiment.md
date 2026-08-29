# 语音增强实验步骤

## 实验内容

使用 ESPnet-SE 预训练模型增强 VCTK-DEMAND 单通道带噪语音，并通过增强音频、波形图、语谱图、PESQ 和 STOI 分析增强效果。

本实验使用 CPU 和 Docker Compose，不包含模型训练、GPU/CUDA 或多通道语音增强。

## 实验步骤

### 步骤 1：准备运行环境

在 Windows、macOS 或 Linux 上安装并启动 Docker Desktop。Python、PyTorch、ESPnet 和音频依赖全部安装在容器内，宿主机不需要安装 Python、CUDA 或 NVIDIA 工具。

安装完成后验证：

```bash
docker run hello-world
docker compose version
```

### 步骤 2：创建 Docker 运行环境

进入项目根目录并构建 CPU 镜像：

```bash
cd SpeechEnhanceLab
docker compose build
```

首次构建会下载基础镜像和 Python 依赖。未修改 `Dockerfile` 或 `requirements.txt` 时，后续构建会复用缓存。

### 步骤 3：快速运行实验

执行主流程：

```bash
docker compose run --rm app python src/main.py
```

每个主要步骤完成后，程序会询问是否继续：

```text
是否进入下一步“检查并获取预训练模型”？[yes/no]：
```

- 输入 `yes` 或 `y`：进入下一步。
- 输入 `no` 或 `n`：停止流程。

如需跳过确认并自动完成全部步骤：

```bash
docker compose run --rm app python src/main.py --yes
```

默认均匀选择 6 对音频。可通过参数修改数量：

```bash
docker compose run --rm app python src/main.py --limit 20
```

### 步骤 4：了解代码主要流程

主程序按以下顺序执行：

1. **检查并获取数据集**
   - 检查 `data/raw/` 中是否已有 VCTK-DEMAND WAV 文件。
   - 数据存在时跳过下载；缺失时自动下载、解压并删除压缩包。
   - 默认均匀选择 6 对干净/带噪音频，写入 `data/selected_pairs.json`。
2. **检查并获取预训练模型**
   - 使用模型 `wyz/vctk_bsrnn_small_noncausal`。
   - 检查模型配置和权重是否完整；已有缓存时跳过下载。
3. **执行增强与评价**
   - 将音频转换为 48 kHz 单声道。
   - 在 CPU 上执行 BSRNN 语音增强。
   - 保存增强音频并生成波形图、语谱图。
   - 重采样到 16 kHz 后计算 PESQ-WB 和 STOI。
   - 在终端显示进度、平均耗时和预计剩余时间。

### 步骤 5：查看实验结果

实验结果保存在：

```text
outputs/
├── audio/                   # 增强后的 WAV
├── figures/                 # 波形图与语谱图
└── metrics/
    ├── metrics.json         # 每条音频的 PESQ/STOI
    └── summary.txt          # 样本数、耗时和平均指标
```

完成后试听增强音频，查看对比图，并比较 `summary.txt` 中增强前后的平均 PESQ 和 STOI。
