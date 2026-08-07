from ...sf_utils.common import AnyType

any_type = AnyType("*")

_CATEGORY = "sfnodes/text"


class SFAnyToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING", {"default": "", "tooltip": "拼接在文本前的前缀"}),
                "suffix": ("STRING", {"default": "", "tooltip": "拼接在文本后的后缀"}),
            },
            "optional": {
                "value": (any_type, {"tooltip": "任意值，转为文本"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "接收任意值转为文本输出，支持前缀/后缀拼接（默认空），None 输出空字符串"

    def execute(self, prefix, suffix, value=None):
        text = "" if value is None else str(value)
        return (prefix + text + suffix,)
