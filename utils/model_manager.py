from pathlib import Path
import folder_paths
from .downloader import download_model


class ModelManager:
    def __init__(self, models):
        self.models = models

    def get_model_info(self, model_name):
        """获取指定模型的信息"""
        if model_name not in self.models:
            raise ValueError(
                f"未知的模型名称: {model_name}，可用模型有: {', '.join(self.models.keys())}"
            )
        return self.models[model_name]

    def get_model_path(self, model_name, sub_dir="models"):
        """获取模型路径，如果不存在会自动下载"""
        model_info = self.get_model_info(model_name)
        save_loc = Path(folder_paths.models_dir) / "sfnodes" / sub_dir
        save_loc.mkdir(parents=True, exist_ok=True)

        model_url = model_info["url"]
        filename = model_info["filename"]

        # 下载模型（如果尚未下载）
        download_model(model_url, save_loc, filename)

        return str(save_loc / filename)

    def get_model_description(self, model_name):
        """获取模型描述"""
        model_info = self.get_model_info(model_name)
        return model_info.get("description", "无可用描述")
