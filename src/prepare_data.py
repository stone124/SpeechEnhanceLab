from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CLEAN_SOURCE = RAW_DATA_DIR / "clean_testset_wav"
DEFAULT_NOISY_SOURCE = RAW_DATA_DIR / "noisy_testset_wav"
DEFAULT_SELECTION_FILE = PROJECT_ROOT / "data" / "selected_pairs.json"
CLEAN_TESTSET_URL = "https://datashare.ed.ac.uk/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/download"
NOISY_TESTSET_URL = "https://datashare.ed.ac.uk/bitstreams/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/download"


def wav_files(source_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(source_dir.rglob("*.wav")):
        if path.name in files:
            raise ValueError(f"重复的 WAV 文件名：{path.name}")
        files[path.name] = path
    return files


def download_file(url: str, destination: Path) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        },
    )
    print(f"正在下载：{destination.name}")
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                filled = int(percent // 5)
                bar = "#" * filled + "-" * (20 - filled)
                status = f"\r[{bar}] {percent:5.1f}% {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB"
            else:
                status = f"\r已下载 {downloaded / 1024 / 1024:.1f} MB"
            print(status, end="", file=sys.stdout, flush=True)
    print()
    if not is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError(f"官方下载地址未返回有效 ZIP 文件：{destination.name}")


def extract_archive(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = (root / member.filename).resolve()
            if not member_path.is_relative_to(root):
                raise ValueError(f"压缩包包含不安全路径：{member.filename}")
        zip_file.extractall(root)


def download_testsets(clean_source: Path, noisy_source: Path) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = (
        (clean_source, CLEAN_TESTSET_URL, "clean_testset_wav.zip"),
        (noisy_source, NOISY_TESTSET_URL, "noisy_testset_wav.zip"),
    )
    for source_dir, url, archive_name in datasets:
        archive = RAW_DATA_DIR / archive_name
        existing_files = wav_files(source_dir) if source_dir.is_dir() else {}
        if existing_files:
            archive.unlink(missing_ok=True)
            print(f"数据集已存在，跳过下载：{source_dir}（{len(existing_files)} 个 WAV）")
            continue
        download_file(url, archive)
        print(f"正在解压：{archive.name}")
        extract_archive(archive, RAW_DATA_DIR)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"解压后未找到目录：{source_dir}")
        archive.unlink()
        print(f"已删除压缩包：{archive.name}")


def stored_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 VCTK-DEMAND 的干净和带噪目录生成成对音频选择清单。"
    )
    parser.add_argument("--clean-source", type=Path, default=DEFAULT_CLEAN_SOURCE)
    parser.add_argument("--noisy-source", type=Path, default=DEFAULT_NOISY_SOURCE)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument(
        "--selection",
        choices=("first", "uniform"),
        default="first",
        help="first 按文件名取前 N 条；uniform 从完整测试集中均匀取样。",
    )
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION_FILE)
    return parser.parse_args()


def prepare_dataset(
    clean_source: Path = DEFAULT_CLEAN_SOURCE,
    noisy_source: Path = DEFAULT_NOISY_SOURCE,
    limit: int = 6,
    selection: str = "first",
    selection_file: Path = DEFAULT_SELECTION_FILE,
) -> Path:
    if limit < 1:
        raise ValueError("--limit 必须至少为 1。")
    if selection not in {"first", "uniform"}:
        raise ValueError("selection 必须是 first 或 uniform。")
    if clean_source == DEFAULT_CLEAN_SOURCE and noisy_source == DEFAULT_NOISY_SOURCE:
        download_testsets(clean_source, noisy_source)
    if not clean_source.is_dir() or not noisy_source.is_dir():
        raise ValueError("--clean-source 和 --noisy-source 必须是已解压的数据集目录。")

    clean_files = wav_files(clean_source)
    noisy_files = wav_files(noisy_source)
    matched_names = sorted(clean_files.keys() & noisy_files.keys())
    if not matched_names:
        raise FileNotFoundError(
            "未找到成对 WAV 文件。请手动解压到 "
            "data/raw/clean_testset_wav/ 和 data/raw/noisy_testset_wav/。"
        )
    if len(matched_names) < limit:
        raise ValueError(
            f"只找到 {len(matched_names)} 对同名 WAV 文件，少于要求的 {limit} 对。"
        )

    if selection == "uniform" and limit > 1:
        selected_names = [
            matched_names[index * (len(matched_names) - 1) // (limit - 1)]
            for index in range(limit)
        ]
    else:
        selected_names = matched_names[:limit]

    pairs = [
        {
            "filename": name,
            "clean_path": stored_path(clean_files[name]),
            "noisy_path": stored_path(noisy_files[name]),
        }
        for name in selected_names
    ]
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    selection_file.write_text(
        json.dumps(
            {
                "selection": {
                    "strategy": selection,
                    "selected_count": len(pairs),
                    "available_count": len(matched_names),
                },
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已生成选择清单：{selection_file}")
    print(f"选择方式：{selection}，已选择 {len(pairs)}/{len(matched_names)} 对音频")
    return selection_file


def main() -> None:
    args = parse_args()
    prepare_dataset(
        clean_source=args.clean_source,
        noisy_source=args.noisy_source,
        limit=args.limit,
        selection=args.selection,
        selection_file=args.selection_file,
    )


if __name__ == "__main__":
    main()
