from __future__ import annotations

import io
import json
import logging
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import nltk
import torch

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated\..*",
    category=FutureWarning,
    module=r"espnet2\.enh\..*",
)

original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: False
try:
    with redirect_stdout(io.StringIO()):
        from espnet2.bin.enh_inference import SeparateSpeech
finally:
    nltk.download = original_nltk_download


class _DeprecatedEpsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != (
            "Eps is deprecated in si_snr loss, set clamp_db instead."
        )


def load_enhancer(model_info_path: Path) -> tuple[SeparateSpeech, int]:
    if not model_info_path.is_file():
        raise FileNotFoundError(
            f"未找到模型信息文件：{model_info_path}，请先运行 src/download_model.py。"
        )

    model_info: dict[str, Any] = json.loads(model_info_path.read_text(encoding="utf-8"))
    files = model_info.get("files", {})
    train_config = Path(files.get("train_config", ""))
    model_file = Path(files.get("model_file", ""))
    sample_rate = int(model_info.get("sample_rate", 0))

    if not train_config.is_file() or not model_file.is_file():
        raise FileNotFoundError("模型配置或权重文件不存在，请重新运行 src/download_model.py。")
    if sample_rate <= 0:
        raise ValueError("model_info.json 中的 sample_rate 无效。")

    eps_filter = _DeprecatedEpsFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(eps_filter)
    try:
        enhancer = SeparateSpeech(
            train_config=train_config,
            model_file=model_file,
            normalize_output_wav=True,
            device="cpu",
        )
    finally:
        root_logger.removeFilter(eps_filter)
    return enhancer, sample_rate


def enhance_audio(
    enhancer: SeparateSpeech,
    noisy_audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    if noisy_audio.ndim != 1 or noisy_audio.size == 0:
        raise ValueError("模型输入必须是一维非空单通道音频。")

    separated = enhancer(noisy_audio[None, :], fs=sample_rate)
    if not separated:
        raise RuntimeError("语音增强模型没有返回输出。")

    enhanced = separated[0]
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().numpy()
    enhanced = np.asarray(enhanced, dtype=np.float32).squeeze()
    if enhanced.ndim != 1:
        raise RuntimeError(f"模型返回了无法处理的形状：{enhanced.shape}")
    return np.clip(enhanced, -1.0, 1.0)
