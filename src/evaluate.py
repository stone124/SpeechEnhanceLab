from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi

METRIC_SAMPLE_RATE = 16000


def _prepare_for_metrics(
    signal: np.ndarray,
    source_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate != METRIC_SAMPLE_RATE:
        signal = librosa.resample(
            signal,
            orig_sr=source_sample_rate,
            target_sr=METRIC_SAMPLE_RATE,
        )
    return np.asarray(signal, dtype=np.float32)


def calculate_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
) -> dict[str, float | int]:
    clean_metric = _prepare_for_metrics(clean, sample_rate)
    noisy_metric = _prepare_for_metrics(noisy, sample_rate)
    enhanced_metric = _prepare_for_metrics(enhanced, sample_rate)
    length = min(clean_metric.size, noisy_metric.size, enhanced_metric.size)
    if length == 0:
        raise ValueError("无法评价空音频。")

    clean_metric = clean_metric[:length]
    noisy_metric = noisy_metric[:length]
    enhanced_metric = enhanced_metric[:length]
    return {
        "sample_rate": METRIC_SAMPLE_RATE,
        "pesq_before": float(pesq(METRIC_SAMPLE_RATE, clean_metric, noisy_metric, "wb")),
        "pesq_after": float(
            pesq(METRIC_SAMPLE_RATE, clean_metric, enhanced_metric, "wb")
        ),
        "stoi_before": float(
            stoi(clean_metric, noisy_metric, METRIC_SAMPLE_RATE, extended=False)
        ),
        "stoi_after": float(
            stoi(clean_metric, enhanced_metric, METRIC_SAMPLE_RATE, extended=False)
        ),
    }


def write_metrics(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
