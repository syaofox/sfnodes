from ...sf_utils import lora_presets
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"

PRESET_TYPE = "SF_LORA_PRESET"


class PowerLoraPreset:
    """选择已保存的 LoRA 预设并输出，供 SF Power Lora Loader 使用"""
    DESCRIPTION = (
        "选择已保存的 LoRA 预设（含顺序与强度），输出到 "
        "SF Power Lora Loader 的 preset 输入（连接后预设优先）"
    )

    @classmethod
    def INPUT_TYPES(cls):
        names = sorted(lora_presets._load_presets().keys())
        return {
            "required": {
                "preset": (
                    ["None"] + names,
                    {
                        "default": "None",
                        "tooltip": "选择预设；None 表示不使用预设",
                    },
                ),
            },
        }

    RETURN_TYPES = (PRESET_TYPE, "STRING")
    RETURN_NAMES = ("preset", "preset_name")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, preset):
        if not preset or preset == "None":
            return (None, "")
        data = lora_presets._load_presets().get(preset)
        return (data, preset)
