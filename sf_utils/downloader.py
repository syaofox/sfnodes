from pathlib import Path
import re
import shutil

import requests
from tqdm import tqdm
from .logger import get_logger

logger = get_logger(__name__)

# 连接/读超时：无 timeout 时挂起的连接会无限阻塞执行线程。
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 120

# HuggingFace resolve URL：https://huggingface.co/<repo_id>/resolve/<rev>/<path...>
# filepath 可含子目录（如 antelopev2/1k3d68.onnx）。
_HF_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)$"
)


def parse_hf_url(url):
    """解析 HF resolve URL → (repo_id, revision, filepath)；非 HF URL 返回 None。"""
    if not isinstance(url, str):
        return None
    m = _HF_RESOLVE_RE.match(url)
    if not m:
        return None
    return m.group("repo"), m.group("rev"), m.group("path")


def _download_hf(url, save_loc, model_name):
    """HF resolve URL → huggingface_hub.hf_hub_download（官方缓存/etag 校验/断点
    续传/并发安全）→ 复制到约定路径 save_loc/model_name（落盘契约与 requests
    路径一致，调用方零改动）。返回 True/False；非 HF URL 返回 None 由调用方
    决定走 requests 兜底。"""
    parsed = parse_hf_url(url)
    if parsed is None:
        return None
    repo_id, rev, filepath = parsed
    try:
        from huggingface_hub import hf_hub_download
        # 不带 local_dir：文件落在 HF 官方缓存（~/.cache/huggingface/hub/），
        # 避免 local_dir 保留子目录结构破坏 save_loc/model_name 拼接，也避开
        # local_dir_use_symlinks 在新旧 huggingface_hub 的签名差异（rfmsr 踩过）。
        cached = hf_hub_download(repo_id=repo_id, filename=filepath, revision=rev)
    except Exception as exc:
        logger.error(f"模型下载失败(HF): {model_name} ({url}), 错误: {exc}")
        return False
    target = save_loc / model_name
    try:
        shutil.copy2(cached, target)
    except OSError as exc:
        logger.error(f"模型复制失败: {model_name}, 错误: {exc}")
        return False
    logger.info(f"模型下载完成: {model_name}")
    return True


def download_model(model_url, save_loc, model_name):
    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)

    if (save_loc / model_name).is_file():
        return True

    # HF resolve URL → huggingface_hub（统一官方下载器）。HF 失败不静默回退
    # requests（同一网络下 requests 也大概率失败，静默回退难排查）。
    hf_result = _download_hf(model_url, save_loc, model_name)
    if hf_result is not None:
        return hf_result

    # 非 HF URL：requests 流式兜底（当前无使用方，为未来 Civitai 等预留）。
    logger.info(f"正在下载模型: {model_name}")
    tmp_path = save_loc / (model_name + ".part")
    try:
        response = requests.get(
            model_url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)
        )
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        with (
            tmp_path.open("wb") as file,
            tqdm(
                desc="下载中",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        # 原子替换：下载中途失败不会留下被 is_file() 误判为"已下载"的半成品
        tmp_path.replace(save_loc / model_name)
        logger.info(f"模型下载完成: {model_name}")
        return True
    except requests.exceptions.RequestException as err:
        logger.error(f"模型下载失败: {model_name}, 错误: {err}")
        logger.info(f"请从以下链接手动下载: {model_url}")
        logger.info(f"并将其放置在: {save_loc}")
    except Exception as e:
        logger.error(f"发生意外错误: {e}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return False
