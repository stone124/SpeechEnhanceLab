from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from audio import align_length, load_audio, write_audio
from download_model import ensure_model
from enhance import enhance_audio, load_enhancer
from evaluate import calculate_metrics, write_metrics
from prepare_data import (
    DEFAULT_CLEAN_SOURCE,
    DEFAULT_NOISY_SOURCE,
    prepare_dataset,
)
from visualize import save_spectrogram_figure, save_waveform_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_FILE = PROJECT_ROOT / "data" / "selected_pairs.json"
DEFAULT_MODEL_INFO_FILE = PROJECT_ROOT / "models" / "pretrained" / "model_info.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"


def print_step(number: int, total: int, title: str) -> None:
    print(f"\n=== 步骤 {number}/{total}：{title} ===")


def confirm_next(next_step: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"自动继续进入下一步：{next_step}")
        return True
    while True:
        try:
            answer = input(f"是否进入下一步“{next_step}”？[yes/no]：").strip().lower()
        except EOFError as error:
            raise RuntimeError("无法读取确认输入；无人值守运行请添加 --yes。") from error
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("请输入 yes 或 no。")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def progress_line(
    completed: int,
    total: int,
    elapsed: float,
    current_filename: str | None = None,
) -> str:
    width = 14
    filled = round(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    line = f"[{bar}] {completed}/{total} {completed / total:.0%}"
    if completed:
        average = elapsed / completed
        remaining = average * (total - completed)
        line += (
            f" | avg {average:.1f}s"
            f" | used {format_duration(elapsed)}"
            f" | ETA {format_duration(remaining)}"
        )
    if current_filename is not None:
        line += f" | {current_filename}"
    return line


def show_progress(line: str) -> None:
    print(f"\r\033[2K{line}", end="", flush=True)


def resolve_project_file(path_value: str) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"路径必须位于项目目录内：{path_value}")
    return resolved


def load_pairs(selection_file: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not selection_file.is_file():
        raise FileNotFoundError(
            f"未找到音频选择清单：{selection_file}，请先运行 src/prepare_data.py。"
        )

    manifest: dict[str, Any] = json.loads(selection_file.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("音频选择清单中没有有效的 pairs。")

    required_keys = {"filename", "clean_path", "noisy_path"}
    for pair in pairs:
        if not isinstance(pair, dict) or not required_keys.issubset(pair):
            raise ValueError("音频选择清单中的配对记录格式无效。")
    selection = manifest.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    return pairs, selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行音频预处理、ESPnet-SE 增强、可视化和 PESQ/STOI 评价。"
    )
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION_FILE)
    parser.add_argument("--model-info", type=Path, default=DEFAULT_MODEL_INFO_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean-source", type=Path, default=DEFAULT_CLEAN_SOURCE)
    parser.add_argument("--noisy-source", type=Path, default=DEFAULT_NOISY_SOURCE)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument(
        "--selection",
        choices=("first", "uniform"),
        default="uniform",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过步骤间确认，自动运行完整流程。",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    pairs, selection = load_pairs(args.selection_file)
    enhancer, sample_rate = load_enhancer(args.model_info)
    audio_dir = args.output_dir / "audio"
    figures_dir = args.output_dir / "figures"
    metrics_file = args.output_dir / "metrics" / "metrics.json"
    metric_results: list[dict[str, object]] = []
    total = len(pairs)
    started_at = time.perf_counter()
    print(f"共需处理 {total} 对音频，模型采样率：{sample_rate} Hz")

    for index, pair in enumerate(pairs, start=1):
        filename = pair["filename"]
        if Path(filename).name != filename:
            raise ValueError(f"无效的音频文件名：{filename}")

        clean_path = resolve_project_file(pair["clean_path"])
        noisy_path = resolve_project_file(pair["noisy_path"])
        clean = load_audio(clean_path, sample_rate)
        noisy = load_audio(noisy_path, sample_rate)
        clean, noisy = align_length(clean, noisy)

        show_progress(
            progress_line(
                index - 1,
                total,
                time.perf_counter() - started_at,
                filename,
            )
        )
        enhanced = enhance_audio(enhancer, noisy, sample_rate)
        clean, noisy, enhanced = align_length(clean, noisy, enhanced)

        stem = Path(filename).stem
        enhanced_path = audio_dir / f"{stem}_enhanced.wav"
        write_audio(enhanced_path, enhanced, sample_rate)
        save_waveform_figure(
            clean,
            noisy,
            enhanced,
            sample_rate,
            figures_dir / f"{stem}_waveform.png",
        )
        save_spectrogram_figure(
            clean,
            noisy,
            enhanced,
            sample_rate,
            figures_dir / f"{stem}_spectrogram.png",
        )

        metrics = calculate_metrics(clean, noisy, enhanced, sample_rate)
        metric_results.append({"filename": filename, **metrics})
        show_progress(progress_line(index, total, time.perf_counter() - started_at))

    write_metrics(metrics_file, metric_results)
    elapsed = time.perf_counter() - started_at
    print()

    pesq_before = sum(float(result["pesq_before"]) for result in metric_results) / total
    pesq_after = sum(float(result["pesq_after"]) for result in metric_results) / total
    stoi_before = sum(float(result["stoi_before"]) for result in metric_results) / total
    stoi_after = sum(float(result["stoi_after"]) for result in metric_results) / total
    pesq_improved = sum(
        float(result["pesq_after"]) > float(result["pesq_before"])
        for result in metric_results
    )
    stoi_improved = sum(
        float(result["stoi_after"]) > float(result["stoi_before"])
        for result in metric_results
    )
    speakers = sorted({result["filename"].split("_", 1)[0] for result in metric_results})
    available_count = int(selection.get("available_count", total))
    strategy = str(selection.get("strategy", "unknown"))
    summary = "\n".join(
        (
            "SpeechEnhanceLab 实验总结",
            f"本次选择：{total}/{available_count} 对音频",
            f"选择方式：{strategy}",
            f"覆盖说话人：{'、'.join(speakers)}",
            f"总耗时：{format_duration(elapsed)}",
            f"平均耗时：{elapsed / total:.2f} 秒/条",
            f"平均 PESQ：{pesq_before:.4f} -> {pesq_after:.4f}",
            f"平均 STOI：{stoi_before:.4f} -> {stoi_after:.4f}",
            f"PESQ 提高：{pesq_improved}/{total}",
            f"STOI 提高：{stoi_improved}/{total}",
        )
    )
    summary_file = args.output_dir / "metrics" / "summary.txt"
    summary_file.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"增强音频目录：{audio_dir}")
    print(f"可视化目录：{figures_dir}")
    print(f"评价结果：{metrics_file}")
    print(f"实验总结：{summary_file}")


def main() -> None:
    args = parse_args()
    total_steps = 3
    print("SpeechEnhanceLab 单通道语音增强实验")

    print_step(1, total_steps, "检查并获取 VCTK-DEMAND 数据集")
    prepare_dataset(
        clean_source=args.clean_source,
        noisy_source=args.noisy_source,
        limit=args.limit,
        selection=args.selection,
        selection_file=args.selection_file,
    )
    print("数据集步骤完成。")
    if not confirm_next("检查并获取预训练模型", args.yes):
        print("流程已停止。")
        return

    print_step(2, total_steps, "检查并获取 ESPnet-SE 预训练模型")
    ensure_model(
        model_cache_dir=args.model_info.parent,
        model_info_file=args.model_info,
    )
    print("模型步骤完成。")
    if not confirm_next("执行语音增强、可视化与指标评价", args.yes):
        print("流程已停止。")
        return

    print_step(3, total_steps, "执行语音增强、可视化与指标评价")
    run_experiment(args)
    print("\n全部步骤已完成。")


if __name__ == "__main__":
    main()
