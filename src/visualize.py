from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_waveform_figure(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    destination: Path,
) -> None:
    signals = (("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced))
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for axis, (title, signal) in zip(axes, signals):
        time = np.arange(signal.shape[0]) / sample_rate
        axis.plot(time, signal, linewidth=0.6)
        axis.set_title(title)
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def save_spectrogram_figure(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    destination: Path,
) -> None:
    signals = (("Clean", clean), ("Noisy", noisy), ("Enhanced", enhanced))
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    image = None
    for axis, (title, signal) in zip(axes, signals):
        spectrum = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal, n_fft=2048, hop_length=512)),
            ref=np.max,
        )
        duration = signal.shape[0] / sample_rate
        image = axis.imshow(
            spectrum,
            origin="lower",
            aspect="auto",
            extent=(0, duration, 0, sample_rate / 2),
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        axis.set_title(title)
        axis.set_ylabel("Frequency (Hz)")
    axes[-1].set_xlabel("Time (s)")
    if image is not None:
        figure.colorbar(image, ax=axes, label="Magnitude (dB)", pad=0.02)
    figure.subplots_adjust(hspace=0.32, right=0.88)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
