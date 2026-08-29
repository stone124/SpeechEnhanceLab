from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"未找到音频文件：{path}")

    audio, source_rate = sf.read(path, always_2d=True, dtype="float32")
    mono_audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        mono_audio = librosa.resample(
            mono_audio,
            orig_sr=source_rate,
            target_sr=sample_rate,
        )
    return np.clip(np.asarray(mono_audio, dtype=np.float32), -1.0, 1.0)


def align_length(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    if not signals:
        raise ValueError("至少需要一段音频。")

    length = min(signal.shape[0] for signal in signals)
    if length == 0:
        raise ValueError("音频不能为空。")
    return tuple(signal[:length] for signal in signals)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate)
