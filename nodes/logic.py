import torch

from ..sf_utils.common import AnyType

any_type = AnyType("*")
lazy_options = {"lazy": True}
MAX_FLOW_NUM = 20

_CATEGORY = "sfnodes/logic"


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


class AnyPack:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {},
            "optional": {}
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ("SF_PACK",)
    RETURN_NAMES = ("pack",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将多个输入按位置打包为一条线，配合 SF Any Unpack 使用，减少工作流连线"

    def execute(self, **kwargs):
        values = [kwargs.get("value%d" % i) for i in range(MAX_FLOW_NUM)]
        return (values,)


class AnyUnpack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pack": ("SF_PACK",),
            },
        }

    RETURN_TYPES = tuple(any_type for _ in range(MAX_FLOW_NUM))
    RETURN_NAMES = tuple("out%d" % i for i in range(MAX_FLOW_NUM))
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "解包 SF Any Pack 打包的数据，按位置还原为多条输出线"

    def execute(self, pack):
        if pack is None:
            values = [None] * MAX_FLOW_NUM
        else:
            values = list(pack)
            values.extend([None] * (MAX_FLOW_NUM - len(values)))
        return tuple(values[:MAX_FLOW_NUM])


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
