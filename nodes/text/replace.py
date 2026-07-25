from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"

_MAX_REPLACE_SLOTS = 20


class SFTextReplace:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_REPLACE_SLOTS + 1):
            optional[f"replace_{i}"] = (IO.STRING, {"multiline": False, "default": ""})
        return {
            "required": {
                "template": (IO.STRING, {"multiline": True, "default": "", "tooltip": "包含占位符 {1} {2} ... 的模板文本"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将模板文本中的 {1} {2} 等占位符替换为指定文本"

    def execute(self, template, **kwargs):
        result = template
        for i in range(1, _MAX_REPLACE_SLOTS + 1):
            replacement = kwargs.get(f"replace_{i}", "") or ""
            if replacement:
                result = result.replace(f"{{{i}}}", replacement)
        return (result,)
