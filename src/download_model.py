from __future__ import annotations

import json
from pathlib import Path

from espnet_model_zoo.downloader import ModelDownloader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "pretrained"
MODEL_INFO_FILE = MODEL_CACHE_DIR / "model_info.json"
MODEL_TAG = "wyz/vctk_bsrnn_small_noncausal"
SAMPLE_RATE = 48000


def model_is_ready(model_info_file: Path = MODEL_INFO_FILE) -> bool:
    if not model_info_file.is_file():
        return False
    try:
        model_info = json.loads(model_info_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if not isinstance(model_info, dict):
        return False
    files = model_info.get("files", {})
    if not isinstance(files, dict):
        return False
    return (
        model_info.get("model_tag") == MODEL_TAG
        and model_info.get("sample_rate") == SAMPLE_RATE
        and Path(files.get("train_config", "")).is_file()
        and Path(files.get("model_file", "")).is_file()
    )


def ensure_model(
    model_cache_dir: Path = MODEL_CACHE_DIR,
    model_info_file: Path = MODEL_INFO_FILE,
) -> Path:
    if model_is_ready(model_info_file):
        print(f"预训练模型已存在，跳过下载：{MODEL_TAG}")
        print(f"模型信息：{model_info_file}")
        return model_info_file

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"未找到完整模型缓存，开始下载：{MODEL_TAG}")
    downloader = ModelDownloader(str(model_cache_dir))
    files = downloader.download_and_unpack(MODEL_TAG)
    model_info = {
        "model_tag": MODEL_TAG,
        "sample_rate": SAMPLE_RATE,
        "files": {name: str(path) for name, path in files.items()},
    }
    model_info_file.write_text(
        json.dumps(model_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"模型已就绪：{MODEL_TAG}")
    print(f"模型信息：{model_info_file}")
    return model_info_file


def main() -> None:
    ensure_model()


if __name__ == "__main__":
    main()
