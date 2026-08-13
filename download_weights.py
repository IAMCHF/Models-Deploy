#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载模型权重脚本
- 镜像站优先（hf-mirror.com），官方源兜底（huggingface.co）
- 支持从环境变量 HTTP_PROXY / HTTPS_PROXY 读取代理
- 下载完成后计算 weights/ 总大小，>=10GB 自动标注大模型
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path

# ============================================================
# 镜像站优先：设定环境变量（huggingface_hub 自动读取）
# ============================================================
MIRROR_ENDPOINT = "https://hf-mirror.com"
OFFICIAL_ENDPOINT = "https://huggingface.co"

os.environ.setdefault("HF_ENDPOINT", MIRROR_ENDPOINT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_weights")

# ============================================================
# 大模型阈值（10GB）
# ============================================================
LARGE_MODEL_THRESHOLD_BYTES = 10 * 1024 ** 3  # 10 GB

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"


def get_dir_size(path: Path) -> int:
    """递归计算目录总大小（字节）"""
    total = 0
    if not path.exists():
        return 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def download_model(model_id: str, folder_name: str, force: bool = False) -> dict:
    """
    下载单个模型权重到 models/{folder_name}/weights/
    返回下载结果信息 dict
    """
    weights_dir = MODELS_DIR / folder_name / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_root = MODELS_DIR / folder_name

    if force:
        logger.info("强制重新下载，清空 weights 目录: %s", weights_dir)
        shutil.rmtree(weights_dir, ignore_errors=True)
        weights_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    result = {
        "model_id": model_id,
        "folder_name": folder_name,
        "success": False,
        "endpoint_used": None,
        "size_bytes": 0,
        "size_gb": 0.0,
        "is_large": False,
        "error": None,
    }

    # --------------------------------------------------------
    # 镜像站优先
    # --------------------------------------------------------
    for endpoint_name, endpoint_url in [("镜像站", MIRROR_ENDPOINT), ("官方源", OFFICIAL_ENDPOINT)]:
        os.environ["HF_ENDPOINT"] = endpoint_url
        try:
            logger.info("[%s] 正在通过 %s 下载: %s", folder_name, endpoint_name, model_id)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(weights_dir),
                resume_download=True,
                max_workers=4,
            )
            result["success"] = True
            result["endpoint_used"] = endpoint_url
            if endpoint_name == "官方源":
                logger.warning("[%s] 镜像站不可用，已切换至官方源", folder_name)
            logger.info("[%s] 下载完成", folder_name)
            break
        except Exception as e:
            logger.warning("[%s] %s 下载失败: %s", folder_name, endpoint_name, e)
            result["error"] = str(e)
            continue

    # --------------------------------------------------------
    # 下载失败处理
    # --------------------------------------------------------
    if not result["success"]:
        logger.error("[%s] 镜像站和官方源均不可用，下载失败！请检查网络或代理设置。", folder_name)
        return result

    # --------------------------------------------------------
    # 计算权重大小，大模型标注
    # --------------------------------------------------------
    size_bytes = get_dir_size(weights_dir)
    size_gb = size_bytes / (1024 ** 3)
    result["size_bytes"] = size_bytes
    result["size_gb"] = round(size_gb, 2)
    result["is_large"] = size_bytes >= LARGE_MODEL_THRESHOLD_BYTES

    if result["is_large"]:
        # 创建 .large_model 标记文件
        (model_root / ".large_model").write_text(
            f"model_id: {model_id}\nsize_gb: {size_gb:.2f}\n", encoding="utf-8"
        )
        logger.warning("[%s] 权重 %.2f GB >= 10GB，已标记为大模型，内网容器需扩容存储！", folder_name, size_gb)
    else:
        logger.info("[%s] 权重大小: %.2f GB", folder_name, size_gb)

    return result


def load_models_list(list_path: Path) -> list:
    """从 models_list.json 加载模型清单"""
    if not list_path.exists():
        logger.error("模型清单文件不存在: %s", list_path)
        sys.exit(1)
    with open(list_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", data) if isinstance(data, dict) else data


def main():
    parser = argparse.ArgumentParser(description="批量下载 HuggingFace 模型权重（镜像站优先）")
    parser.add_argument(
        "--list", type=str, default=str(ROOT_DIR / "models_list.json"),
        help="模型清单 JSON 文件路径（默认 models_list.json）",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="仅下载指定 folder_name（逗号分隔）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载（清空已有 weights）",
    )
    args = parser.parse_args()

    models = load_models_list(Path(args.list))
    logger.info("共加载 %d 个模型", len(models))

    if args.only:
        only_set = set(args.only.split(","))
        models = [m for m in models if m["folder_name"] in only_set]
        logger.info("筛选后剩 %d 个模型", len(models))

    results = []
    large_models = []

    for i, model in enumerate(models, 1):
        model_id = model["model_id"]
        folder_name = model["folder_name"]
        logger.info("=" * 60)
        logger.info("(%d/%d) 模型: %s -> %s", i, len(models), model_id, folder_name)
        logger.info("=" * 60)

        result = download_model(model_id, folder_name, force=args.force)
        results.append(result)
        if result["is_large"]:
            large_models.append(result)

    # --------------------------------------------------------
    # 汇总报告
    # --------------------------------------------------------
    logger.info("=" * 60)
    logger.info("下载汇总报告")
    logger.info("=" * 60)
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    logger.info("成功: %d / 失败: %d / 总计: %d", success_count, fail_count, len(results))

    if large_models:
        logger.warning("大模型（>=10GB）共 %d 个，内网容器需扩容存储:", len(large_models))
        # 生成 large_models_list.txt
        list_path = ROOT_DIR / "large_models_list.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("# 大模型清单（权重 >= 10GB）\n")
            f.write("# 格式: folder_name | model_id | size_gb\n")
            for r in large_models:
                line = f"{r['folder_name']} | {r['model_id']} | {r['size_gb']} GB"
                f.write(line + "\n")
                logger.warning("  %s", line)
        logger.info("大模型清单已写入: %s", list_path)

    # 写入下载结果 JSON
    report_path = ROOT_DIR / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("详细下载报告已写入: %s", report_path)

    if fail_count > 0:
        logger.error("有 %d 个模型下载失败，请检查日志后重试。", fail_count)
        sys.exit(1)


if __name__ == "__main__":
    main()
