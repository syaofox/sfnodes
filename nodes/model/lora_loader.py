import folder_paths

from nodes import LoraLoader as NativeLoraLoader

_CATEGORY = "sfnodes/model"


class LoraLoader(NativeLoraLoader):
    """原生 LoraLoader 的等价实现，附带 LoRA 信息展示/编辑能力（前端信息图标）。"""
    DESCRIPTION = "将 LoRA 应用到扩散模型与 CLIP（MODEL+CLIP），行为与原生 LoraLoader 一致；节点上的信息图标可查看/编辑 LoRA 元数据与自定义备注。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "要应用 LoRA 的扩散模型"}),
                "clip": ("CLIP", {"tooltip": "要应用 LoRA 的 CLIP 模型"}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA 文件名"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "扩散模型修改强度，可为负值"}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "CLIP 模型修改强度，可为负值"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = _CATEGORY
