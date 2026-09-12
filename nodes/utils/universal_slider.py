from ...sf_utils.common import AnyType

_CATEGORY = "sfnodes/utils"

any = AnyType("*")


class SFUniversalSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": -999999,
                        "max": 999999,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            },
            "hidden": {
                "output_type": (["float", "int"], {"default": "float"}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "复刻孤海万能滑条：通用数值滑条，动态范围（最小/最大/步长/整数-浮点切换经右键设置面板调整），单 any 输出随 output_type 输出真类型（float/int）"

    def execute(self, value, output_type="float"):
        processed_value = round(value, 10)
        if output_type == "int":
            return (int(round(processed_value)),)
        return (float(processed_value),)

    @classmethod
    def IS_CHANGED(cls, value, output_type="float"):
        processed_value = round(float(value), 10)
        if output_type == "int":
            return int(round(processed_value))
        return processed_value
