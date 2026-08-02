import folder_paths

from nodes import LoraLoader

from ...sf_utils.lora_notes import get_merged_metadata

_CATEGORY = "sfnodes/model"


class LoraLoaderModelOnly(LoraLoader):
    """原生 LoraLoaderModelOnly 的等价实现，附带 LoRA 信息展示/编辑能力（前端信息图标）。"""
    DESCRIPTION = "仅将 LoRA 应用到扩散模型（MODEL），行为与原生 LoraLoaderModelOnly 一致；同时输出该 LoRA 的触发词（Trigger Words）与描述（Description）；节点上的信息图标可查看/编辑 LoRA 元数据与自定义备注。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "trigger_words", "description")
    OUTPUT_TOOLTIPS = ("修改后的扩散模型", "LoRA 触发词", "LoRA 描述")
    FUNCTION = "load_lora_model_only"
    CATEGORY = _CATEGORY

    def load_lora_model_only(self, model, lora_name, strength_model):
        meta = get_merged_metadata(lora_name, "loras")
        return (
            super().load_lora(model, None, lora_name, strength_model, 0)[0],
            meta.get("trigger_words", ""),
            meta.get("description", ""),
        )
