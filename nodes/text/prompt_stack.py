"""SFPromptStack —— 动态 Prompt 列表节点（类似 SFPromptList，但行动态添加）。

前端驱动（Vue Compat #9，同 SFLoraStack）：行数据（每条 prompt 的开关与文本）
存浏览器 node.properties.promptStackState，由 graphToPrompt 钩子注入隐藏
PromptStackState 输入。键名与状态形状（rows/enabled/label/text）对齐 Pixaroma
PromptStack——sf_utils/prompt_reader.py 的 _pix_prompt_stack_extract 可直接
恢复本节点生成的图。

Python 职责很小：解析 rows，只取 enabled 且去首尾空白后非空的行，输出
prompt（全局前后缀包裹）与 body_text（原文）两个列表。
"""
import json

_CATEGORY = "sfnodes/text"


class SFPromptStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend_text": ("STRING", {"multiline": False, "default": "", "tooltip": "添加到每行前面的文本"}),
                "append_text": ("STRING", {"multiline": False, "default": "", "tooltip": "添加到每行后面的文本"}),
            },
            "hidden": {"PromptStackState": ("STRING", {"default": "{}"})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "body_text")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "动态添加任意多个 prompt 条目，每条可独立开关；开着的行（去首尾空白后"
        "非空）按顺序输出为 prompt 列表（可加全局前后缀）与 body_text 原文列表。"
    )

    def execute(self, prepend_text="", append_text="", PromptStackState="{}"):
        try:
            state = json.loads(PromptStackState or "{}")
        except (ValueError, TypeError):
            state = {}
        rows = state.get("rows") if isinstance(state, dict) else None
        if not isinstance(rows, list):
            rows = []

        prompt_list = []
        body_list = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get("enabled"):
                continue
            text = row.get("text")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            body_list.append(text)
            prompt_list.append(prepend_text + text + append_text)

        return (prompt_list, body_list)
