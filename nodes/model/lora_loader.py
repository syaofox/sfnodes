import os

import folder_paths

from nodes import LoraLoader as NativeLoraLoader

from ...sf_utils.lora_notes import get_merged_metadata

_CATEGORY = "sfnodes/model"


class LoraLoader(NativeLoraLoader):
    """原生 LoraLoader 的等价实现，附带 LoRA 信息展示/编辑能力（前端信息图标）。"""
    DESCRIPTION = "将 LoRA 应用到扩散模型与 CLIP（MODEL+CLIP），行为与原生 LoraLoader 一致；同时输出该 LoRA 的触发词（Trigger Words）、描述（Description）与文件名（不含路径和扩展名）；节点上的信息图标可查看/编辑 LoRA 元数据与自定义备注。"

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

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "trigger_words", "description", "lora_stem")
    OUTPUT_TOOLTIPS = ("修改后的扩散模型", "修改后的 CLIP 模型", "LoRA 触发词", "LoRA 描述", "LoRA 文件名（不含路径和扩展名）")
    FUNCTION = "load_lora"
    CATEGORY = _CATEGORY

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        result = super().load_lora(model, clip, lora_name, strength_model, strength_clip)
        meta = get_merged_metadata(lora_name)
        stem = os.path.splitext(os.path.basename(lora_name))[0]
        return result + (meta.get("trigger_words", ""), meta.get("description", ""), stem)
