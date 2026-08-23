"""SFLoadDiffusionModel —— Load Diffusion Model 强化版（信息面板）。

加载行为与 ComfyUI 原生 UNETLoader 对齐：unet_name 取自 diffusion_models
目录类型（覆盖 models/diffusion_models 与旧 models/unet），weight_dtype 四档
fp8 选项；执行直接委托原生 UNETLoader.load_unet（函数内 import），保证与
官方行为零漂移，不内联副本其权重映射。

强化点全部在前端：节点上有 i 信息图标（web/sf_load_diffusion_model.js），
点击打开 SF LoRA Stack 同款浮动信息面板——safetensors 头部元数据（架构/
config）、文件大小、打开即自动 Civitai 匹配、用户自定义描述/预览图/
sample 图。数据域与 LoRA 完全隔离：info 组装在 sf_utils/diffusion_routes.py
（dmodel_info），查询/描述/预览等同构路由由 lora_routes 以 /api/sfnodes/
dmodel/* 别名提供（存储换 dmodels.json + previews_model/）。
"""
import folder_paths

from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"

# 与原生 UNETLoader.INPUT_TYPES 保持一致的 weight_dtype 档位。
_WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


class SFLoadDiffusionModel:
    DESCRIPTION = (
        "加载扩散模型（UNet/DiT 权重），行为与官方 Load Diffusion Model 一致，"
        "额外提供信息面板：点击节点上的 i 图标可查看该模型的架构信息"
        "（safetensors 头部 config）、文件大小；打开面板时会自动按文件指纹在 "
        "Civitai 匹配模型页（首次需计算大文件哈希，可能较慢；结果缓存在模型旁，"
        "之后离线秒开），并可编写自定义描述、设置预览图、管理 sample 图。"
        "这些数据保存在用户目录，与 LoRA 的数据互不影响。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),
                              {"tooltip": "要加载的扩散模型（models/diffusion_models 或旧 models/unet 目录）。"}),
                "weight_dtype": (_WEIGHT_DTYPES,
                                 {"advanced": True,
                                  "tooltip": "权重精度。default=按文件存储精度加载；fp8_e4m3fn/e5m2=以对应 fp8 格式载入（省显存）；fp8_e4m3fn_fast 额外启用快速 fp8 矩阵乘。"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("加载的扩散模型，接入采样器/KSampler 等。",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, unet_name, weight_dtype):
        # 委托原生实现：weight_dtype -> model_options 映射与 get_full_path_or_raise
        # 全在官方代码里，官方将来调整此处自动跟随。函数内 import 最安全
        # （AGENTS.md：运行时符号函数内 import）。
        from nodes import UNETLoader

        return UNETLoader().load_unet(unet_name, weight_dtype)


# 导入以触发 dmodel_info 路由注册（lora_routes 同款先例，见本模块尾注释）
from ...sf_utils import diffusion_routes  # noqa: F401, E402
