import folder_paths

from nodes import LoraLoader

_CATEGORY = "sfnodes/model"


class LoraLoaderModelOnly(LoraLoader):
    """原生 LoraLoaderModelOnly 的等价实现，附带 LoRA 信息展示/编辑能力（前端信息图标）。"""
    DESCRIPTION = "仅将 LoRA 应用到扩散模型（MODEL），行为与原生 LoraLoaderModelOnly 一致；节点上的信息图标可查看/编辑 LoRA 元数据与自定义备注。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_lora_model_only"
    CATEGORY = _CATEGORY

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)
