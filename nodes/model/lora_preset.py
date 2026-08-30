from ...sf_utils import lora_presets
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"

PRESET_TYPE = "SF_LORA_PRESET"


class SFLoraPreset:
    """选择已保存的 LoRA 预设并输出，供 SFLoraStack 使用"""
    DESCRIPTION = (
        "选择已保存的 LoRA 预设（含顺序、强度与正向提示词），输出到 "
        "SFLoraStack 的 preset 输入（连接后预设优先）；positive 为预设保存的"
        "正向提示词（与 triggers 分离，不自动拼接），可直连 CLIP 文本编码。"
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

    RETURN_TYPES = (PRESET_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("preset", "preset_name", "positive")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, preset):
        if not preset or preset == "None":
            return (None, "", "")
        data = lora_presets._load_presets().get(preset)
        if not isinstance(data, dict):
            return (data, preset, "")
        pos = data.get("positive", "")
        if not isinstance(pos, str):
            pos = ""
        return (data, preset, pos)
