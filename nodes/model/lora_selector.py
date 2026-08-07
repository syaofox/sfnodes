import os

import folder_paths

_CATEGORY = "sfnodes/model"


class LoraSelector:
    """参考 ControlNetPreprocessorSelector 的 LoRA 选择器：下拉选择 LoRA，输出文件名供 SFLoraLoader/SFLoraLoaderModelOnly 的 lora_name（Convert to input 后）连接使用。"""
    DESCRIPTION = "下拉选择一个 LoRA 并输出其文件名（含子目录路径与扩展名）与不含路径/扩展名的名称；将输出连接到 SFLoraLoader / SFLoraLoaderModelOnly 的 lora_name 输入（需在目标节点上右键 lora_name → Convert to input）即可在画布上动态切换 LoRA。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "选择要输出的 LoRA"}),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING")
    RETURN_NAMES = ("lora_name", "lora_stem")
    OUTPUT_TOOLTIPS = ("LoRA 文件名（含子目录路径与扩展名），可连接 SFLoraLoader/SFLoraLoaderModelOnly 的 lora_name 输入", "LoRA 文件名（不含路径和扩展名）")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, lora_name):
        stem = os.path.splitext(os.path.basename(lora_name))[0]
        return (lora_name, stem)
