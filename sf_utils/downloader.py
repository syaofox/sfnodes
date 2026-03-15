from pathlib import Path

import requests
from tqdm import tqdm
from .logger import get_logger

logger = get_logger(__name__)


def download_model(model_url, save_loc, model_name):
    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)

    if not (save_loc / model_name).is_file():
        logger.info(f"正在下载模型: {model_name}")
        response = requests.get(model_url, stream=True)
        try:
            if response.status_code == 200:
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte

                with (
                    (save_loc / model_name).open("wb") as file,
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
                logger.info(f"模型下载完成: {model_name}")
            else:
                logger.warning(
                    f"模型下载失败: {model_name}, 状态码: {response.status_code}"
                )

        except requests.exceptions.RequestException as err:
            logger.error(f"模型下载失败: {model_name}, 错误: {err}")
            logger.info(f"请从以下链接手动下载: {model_url}")
            logger.info(f"并将其放置在: {save_loc}")
            return False
        except Exception as e:
            logger.error(f"发生意外错误: {e}")
            return False

    return True
