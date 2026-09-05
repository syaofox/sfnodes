from ...sf_utils.common import AnyType
from ...sf_utils.string import pad_number_text

any_type = AnyType("*")

_CATEGORY = "sfnodes/text"


class SFAnyToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "", "tooltip": "拼接在文本前的前缀"}),
                "suffix": ("STRING", {"default": "", "tooltip": "拼接在文本后的后缀"}),
                "pad_digits": ("INT", {"default": 2, "min": 0, "max": 8, "tooltip": "纯整数文本补0到指定位数（0=不补），浮点数与非数字不补"}),
            },
            "optional": {
                "value": (any_type, {"tooltip": "任意值，转为文本"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "接收任意值转为文本输出，支持前缀/后缀拼接（默认空），纯整数可补0到指定位数（默认2，0=不补），浮点数与非数字原样，None 输出空字符串"

    def execute(self, prefix, suffix, pad_digits, value=None):
        text = "" if value is None else str(value)
        text = pad_number_text(text, pad_digits)
        return (prefix + text + suffix,)
