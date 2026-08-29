# SpeechEnhanceLab

基于 **ESPnet-SE** 的单通道语音增强实验项目。项目使用预训练模型增强 VCTK-DEMAND 带噪语音，并生成增强音频、波形图、语谱图以及 PESQ/STOI 评价结果。

## 当前配置

| 项目 | 配置 |
| --- | --- |
| 数据集 | VCTK-DEMAND 成对测试语音 |
| 预训练模型 | `wyz/vctk_bsrnn_small_noncausal` |
| 输入 | 单通道 WAV |
| 模型采样率 | 48 kHz |
| 推理设备 | CPU |
| 评价指标 | PESQ-WB、STOI |
| 运行环境 | Docker Compose |
| 默认样本数 | 均匀选择 6 对音频 |

## 主流程

```text
检查/下载数据集
        ↓
检查/下载预训练模型
        ↓
音频预处理与语音增强
        ↓
生成波形图、语谱图和 PESQ/STOI 结果
```

数据集和模型已有完整缓存时会自动跳过下载。每个主要步骤完成后，可输入 `yes` 或 `no` 决定是否继续。

## 项目结构

```text
SpeechEnhanceLab/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── main.py              # 主流程入口
│   ├── prepare_data.py      # 数据下载与样本选择
│   ├── download_model.py    # 模型下载与缓存检查
│   ├── audio.py             # 音频预处理
│   ├── enhance.py           # ESPnet-SE 推理
│   ├── visualize.py         # 波形图与语谱图
│   └── evaluate.py          # PESQ/STOI 评价
├── data/                    # 数据集与选择清单
├── models/pretrained/       # 预训练模型缓存
├── outputs/                 # 音频、图像和指标结果
└── docs/experiment.md       # 环境搭建与实验步骤
```

## 快速运行

宿主机只需要安装并启动 Docker Desktop，然后在项目根目录执行：

```bash
docker compose build
docker compose run --rm app python src/main.py
```

程序会在步骤之间询问：

```text
是否进入下一步“检查并获取预训练模型”？[yes/no]：
```

输入 `yes` 继续，输入 `no` 停止。

如需跳过确认并自动完成全部步骤：

```bash
docker compose run --rm app python src/main.py --yes
```

临时修改处理数量：

```bash
docker compose run --rm app python src/main.py --limit 20
```

## 输出结果

```text
outputs/
├── audio/                   # 增强后的 WAV
├── figures/                 # 波形图与语谱图
└── metrics/
    ├── metrics.json         # 每条音频的 PESQ/STOI
    └── summary.txt          # 样本数、耗时和平均指标
```

模型在 48 kHz 下推理；计算 PESQ-WB 和 STOI 时统一重采样到 16 kHz。详细环境搭建和运行说明见 [docs/experiment.md](docs/experiment.md)。
