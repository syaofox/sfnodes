import torch
import comfy.model_management
from nodes import MAX_RESOLUTION

_CATEGORY = "sfnodes/latent"

# 定义宽高比和对应的预设分辨率
ASPECT_RATIOS = {
    "4:3": ["1024x768", "1280x960", "1600x1200", "2048x1536"],
    "3:4": ["768x1024", "960x1280", "1200x1600", "1536x2048"],
    "16:9": ["1280x720", "1920x1080", "2560x1440", "3840x2160"],
    "9:16": ["720x1280", "1080x1920", "1440x2560", "2160x3840"],
    "1:1": ["512x512", "768x768", "1024x1024", "1536x1536", "2048x2048"]
}


class EmptyLatentByAspectRatio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        # 获取所有宽高比选项
        aspect_ratio_options = list(ASPECT_RATIOS.keys())
        
        # 获取默认宽高比（4:3）的分辨率选项
        default_ratio = "4:3"
        resolution_options = ASPECT_RATIOS[default_ratio]
        
        # 合并所有分辨率选项（用于JavaScript动态更新）
        all_resolutions = []
        for resolutions in ASPECT_RATIOS.values():
            all_resolutions.extend(resolutions)
        # 去重并保持顺序
        all_resolutions = list(dict.fromkeys(all_resolutions))
        
        return {
            "required": {
                "aspect_ratio": (aspect_ratio_options, {"default": default_ratio}),
                "resolution": (all_resolutions, {"default": resolution_options[0]}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "批次大小"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "width", "height")
    FUNCTION = "generate"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据宽高比生成空潜在图像"

    def generate(self, aspect_ratio, resolution, batch_size=1):
        # 解析分辨率字符串 "1024x768" -> (1024, 768)
        width, height = resolution.split("x")
        width = int(width)
        height = int(height)
        
        # 创建空潜在图像
        # 标准SD模型使用4通道，下采样8倍
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        
        return ({"samples": latent}, width, height)
