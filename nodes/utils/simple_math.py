import ast
import math
import operator as op

from ...sf_utils.common import AnyType
from ...sf_utils.logger import get_logger

_CATEGORY = "sfnodes/utils"

logger = get_logger(__name__)

any = AnyType("*")


class SFNumber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "number_type": (
                    ["FLOAT", "INT", "PERCENT"],
                    {"default": "FLOAT"},
                ),
                "value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("int", "float")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "输出数值，支持 INT / FLOAT / PERCENT 三种类型"

    def execute(self, number_type, value):
        if number_type == "INT":
            v = round(value)
            return (int(v), float(v))
        elif number_type == "PERCENT":
            v = max(0.0, min(1.0, value))
            return (int(v), v)
        else:
            return (int(value), value)



class SimpleMathSlider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "display": "slider",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
                "min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.001,
                    },
                ),
                "max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.001,
                    },
                ),
                "rounding": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "通过滑块调节浮点数值，可设置范围和精度"

    def execute(self, value, min, max, rounding):
        value = min + value * (max - min)

        if rounding > 0:
            value = round(value, rounding)

        return (
            value,
            int(value),
        )


class SimpleMathSliderLowRes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (
                    "INT",
                    {"display": "slider", "default": 5, "min": 0, "max": 10, "step": 1},
                ),
                "min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.001,
                    },
                ),
                "max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.001,
                    },
                ),
                "rounding": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
    )
    RETURN_NAMES = (
        "float",
        "int",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "低精度滑块，值乘以 0.1 后映射到范围，用于粗调"

    def execute(self, value, min, max, rounding):
        value = 0.1 * value
        value = min + value * (max - min)
        if rounding > 0:
            value = round(value, rounding)

        return (value, int(value))


class SimpleMathBoolean:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("boolean", "int")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "输出布尔值及其对应的整数（0/1）"

    def execute(self, value):
        return (
            value,
            int(value),
        )


_VARIABLE_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_MAX_VARIABLES = len(_VARIABLE_LETTERS)


class SimpleMath:
    @classmethod
    def INPUT_TYPES(s):
        optional = {}
        for ch in _VARIABLE_LETTERS:
            optional[ch] = (any, {"default": 0.0})
        return {
            "required": {
                "value": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "自由表达式计算，支持四则运算和变量 a~z，输入端口随连接自动增减，输出整数和浮点数"

    def execute(self, value, **kwargs):
        vars = {"h": 0.0, "w": 0.0}
        for k, v in kwargs.items():
            if v is None:
                continue
            if hasattr(v, "shape"):
                v = list(v.shape)
            if isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    pass
            vars[k] = v

        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.Mod: op.mod,
            ast.Eq: op.eq,
            ast.NotEq: op.ne,
            ast.Lt: op.lt,
            ast.LtE: op.le,
            ast.Gt: op.gt,
            ast.GtE: op.ge,
            ast.And: lambda x, y: x and y,
            ast.Or: lambda x, y: x or y,
            ast.Not: op.not_,
        }

        op_functions = {
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "len": len,
        }

        def eval_(node):
            if isinstance(node, ast.Constant):
                # node.n 是 value 的旧别名：3.13 起 deprecated、3.14 移除。
                return node.value
            if isinstance(node, ast.Name):
                return vars.get(node.id, 0.0)
            if isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            if isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_(node.operand))
            if isinstance(node, ast.Compare):
                left = eval_(node.left)
                for op_, comparator in zip(node.ops, node.comparators):
                    if not operators[type(op_)](left, eval_(comparator)):
                        return 0
                return 1
            if isinstance(node, ast.BoolOp):
                values = [eval_(v) for v in node.values]
                return operators[type(node.op)](*values)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in op_functions:
                    args = [eval_(arg) for arg in node.args]
                    return op_functions[node.func.id](*args)
            if isinstance(node, ast.Subscript):
                val = eval_(node.value)
                if isinstance(node.slice, ast.Constant):
                    return val[node.slice.value]
                return 0
            return 0

        try:
            result = eval_(ast.parse(value, mode="eval").body)
        except (SyntaxError, ZeroDivisionError, KeyError, TypeError, AttributeError) as e:
            logger.warning(f"[SimpleMath] 表达式求值失败: {value!r} -> {type(e).__name__}: {e}")
            return (0, 0.0)
        if not isinstance(result, (int, float)):
            # 字符串常量/字符串变量等非数值结果（如 "abc"）不能进 isnan/round
            return (0, 0.0)
        if math.isnan(result):
            result = 0.0
        return (round(result), result)


class SimpleMathCondition:
    @classmethod
    def INPUT_TYPES(s):
        optional = {}
        for ch in _VARIABLE_LETTERS:
            optional[ch] = (any, {"default": 0.0})
        return {
            "required": {
                "evaluate": (any, {"default": 0}),
                "on_true": ("STRING", {"multiline": False, "default": ""}),
                "on_false": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "条件表达式计算，根据 evaluate 的真假选择 on_true 或 on_false 表达式，支持变量 a~z"

    def execute(self, evaluate, on_true, on_false, **kwargs):
        expression = on_true if evaluate else on_false
        return SimpleMath().execute(expression, **kwargs)





class SimpleComparison:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (any, {"default": 0}),
                "b": (any, {"default": 0}),
                "comparison": (["==", "!=", "<", "<=", ">", ">="],),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "比较两个值，支持 ==、!=、<、<=、>、>= 运算符"

    def execute(self, a, b, comparison):
        if comparison == "==":
            return (a == b,)
        elif comparison == "!=":
            return (a != b,)
        elif comparison == "<":
            return (a < b,)
        elif comparison == "<=":
            return (a <= b,)
        elif comparison == ">":
            return (a > b,)
        elif comparison == ">=":
            return (a >= b,)


class BatchCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch": (any, {}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "获取 batch 的数量（第一维大小）"

    def execute(self, batch):
        count = 0
        if hasattr(batch, "shape"):
            count = batch.shape[0]
        elif isinstance(batch, dict) and "samples" in batch:
            count = batch["samples"].shape[0]
        elif isinstance(batch, list) or isinstance(batch, dict):
            count = len(batch)

        return (count,)
