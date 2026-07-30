from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"
_MAX_TEXT_SLOTS = 16


class SFTextConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_TEXT_SLOTS + 1):
            optional[f"text_{i}"] = (
                IO.STRING,
                {"forceInput": True, "tooltip": f"第 {i} 段输入文本"},
            )
        return {
            "required": {
                "delimiter": (
                    IO.STRING,
                    {"default": ", ", "tooltip": "分隔符，输入 \\n 表示换行"},
                ),
                "clean_whitespace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "true",
                        "label_off": "false",
                        "tooltip": "是否去除每段文本的首尾空白",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "将多段文本用指定分隔符连接，输入端口随连接自动增减，支持去除首尾空白"
    )

    def execute(self, delimiter, clean_whitespace, **kwargs):
        if delimiter in ("\n", "\\n"):
            delimiter = "\n"
        text_inputs = []
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, str) and v:
                text = v.strip() if clean_whitespace else v
                if text:
                    text_inputs.append(text)
        return (delimiter.join(text_inputs),)
