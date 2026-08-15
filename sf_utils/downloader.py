from pathlib import Path

import requests
from tqdm import tqdm
from .logger import get_logger

logger = get_logger(__name__)

# 连接/读超时：无 timeout 时挂起的连接会无限阻塞执行线程。
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 120


def download_model(model_url, save_loc, model_name):
    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)

    if (save_loc / model_name).is_file():
        return True

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
