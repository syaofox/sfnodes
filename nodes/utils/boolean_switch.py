_CATEGORY = "sfnodes/utils"


class SFBooleanSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "复刻孤海布尔开关：Canvas 大开关（单击切换 + 双击改标签），单 BOOLEAN 输出直通"

    def execute(self, value):
        return (value,)
