from ..sf_utils.common import AnyType

any_type = AnyType("*")
lazy_options = {"lazy": True}
MAX_FLOW_NUM = 20

_CATEGORY = "sfnodes/logic"


class SFIfElse:
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

    def check_lazy_status(self, boolean=True, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    def execute(self, *args, **kwargs):
        return (kwargs['on_true'] if kwargs['boolean'] else kwargs['on_false'],)


class SFAnythingIndexSwitch:
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

    def check_lazy_status(self, index, **kwargs):
        key = "value%d" % index
        if kwargs.get(key, None) is None:
            return [key]

    def index_switch(self, index, **kwargs):
        key = "value%d" % index
        return (kwargs[key],)
