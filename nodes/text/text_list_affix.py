from comfy.comfy_types.node_typing import IO

from ...sf_utils.string import affix_list

_CATEGORY = "sfnodes/text"


class SFTextListAffix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend_text": (IO.STRING, {"multiline": False, "default": "", "tooltip": "前缀；支持 \\n 转换行、\\t 转制表"}),
                "append_text": (IO.STRING, {"multiline": False, "default": "", "tooltip": "后缀；支持 \\n 转换行、\\t 转制表"}),
                "filter_empty": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false", "tooltip": "过滤空项（去空白后为空的项不输出）"}),
            },
            "optional": {
                "text_list": (IO.STRING, {"forceInput": True, "tooltip": "输入字符串列表（可接 SFLongTextToList / SFPromptList 等的 list 输出）；未连接时输出空列表"}),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "为字符串列表的每项添加前缀/后缀，支持 \\n/\\t 转义与空项过滤；输入为列表（可接任意 STRING 列表），输出为同长度列表"

    def execute(self, prepend_text="", append_text="", filter_empty=True, text_list=None):
        # INPUT_IS_LIST：单值输入被包裹为单元素列表
        if isinstance(prepend_text, (list, tuple)):
            prepend_text = prepend_text[0] if prepend_text else ""
        if isinstance(append_text, (list, tuple)):
            append_text = append_text[0] if append_text else ""
        if isinstance(filter_empty, (list, tuple)):
            filter_empty = filter_empty[0] if filter_empty else True
        # text_list：INPUT_IS_LIST 下为列表或 [None]（未连接）
        # 统一为列表，供 affix_list 处理
        items = text_list
        # ComfyUI 在 INPUT_IS_LIST 下，未连接的 optional 会传 [None] 或 None
        if items is None:
            items = []
        elif isinstance(items, (list, tuple)):
            # 保留原列表形态，affix_list 会处理单层包裹
            pass
        else:
            items = [items]
        # 处理 forceInput 未连接时 [None] 的情况：affix_list 已过滤空项
        result = affix_list(items, prepend_text or "", append_text or "", bool(filter_empty))
        return (result,)
