import torch

from ..sf_utils.common import AnyType

any_type = AnyType("*")
lazy_options = {"lazy": True}
MAX_FLOW_NUM = 20

_CATEGORY = "sfnodes/logic"


class IfElse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "boolean": ("BOOLEAN",),
                "on_true": (any_type, lazy_options),
                "on_false": (any_type, lazy_options),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据布尔值选择输出，条件为真时输出 on_true，否则输出 on_false"

    def check_lazy_status(self, boolean=True, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    def execute(self, *args, **kwargs):
        return (kwargs['on_true'] if kwargs['boolean'] else kwargs['on_false'],)


class AnythingIndexSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {}
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["value%d" % i] = (any_type, lazy_options)
        return inputs

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "index_switch"
    CATEGORY = _CATEGORY
    DESCRIPTION = "根据索引从多个输入中选择一个输出"

    def check_lazy_status(self, index, **kwargs):
        key = "value%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "value%d" % index
        return (kwargs[key],)


class IsMaskEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "判断遮罩是否全黑，是则返回 True，否则返回 False"

    def execute(self, mask):
        if mask is None:
            return (True,)
        return (torch.all(mask == 0).item(),)
