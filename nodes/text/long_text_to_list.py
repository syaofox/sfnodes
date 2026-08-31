from comfy.comfy_types.node_typing import IO

from ...sf_utils.string import split_text

_CATEGORY = "sfnodes/text"


class SFLongTextToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "default": "", "tooltip": "待分割的长文本"}),
                "delimiter": (IO.STRING, {"multiline": False, "default": "\\n", "tooltip": "分隔符；输入 \\n 表示换行、\\t 表示制表符，右键可 Convert to Input 连线输入"}),
                "i": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip": "要取的下标（0 起）"}),
                "filter_empty": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false", "tooltip": "过滤空项（去空白后为空的行不计入列表与长度）"}),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING, "INT")
    RETURN_NAMES = ("text_at_i", "list", "count")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "按分隔符将长文本分割为列表，输出指定下标的文本、完整列表与长度；复刻 ComfyUI_Lam LongTextToList（分隔符支持 \\n/\\t 转义、空分隔符与越界不崩、空项过滤）"

    def execute(self, text, delimiter, i, filter_empty=True):
        # split_text 处理 None/"" 守卫、\\n/\\t 转义与空项过滤（见 sf_utils/string.py）
        parts = split_text(text, delimiter, filter_empty=bool(filter_empty))
        count = len(parts)
        if 0 <= i < count:
            picked = parts[i]
        else:
            picked = ""
            if count > 0:
                print(f"[SFLongTextToList] i={i} 越界 count={count}，返回空字符串")
        return (picked, parts, count)
