import math
from ...sf_utils.common import AnyType

_CATEGORY = "sfnodes/utils"

any = AnyType("*")


class SimpleMathFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.05,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value):
        return (float(value),)


class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value):
        return (round(float(value), 2),)


class SimpleMathPercent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value):
        return (float(value),)


class SimpleMathInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (
                    "INT",
                    {
                        "default": 0,
                        "min": -0xFFFFFFFFFFFFFFFF,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value):
        return (int(value),)


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
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value, min, max, rounding):
        value = 0.1 * value
        value = min + value * (max - min)
        if rounding > 0:
            value = round(value, rounding)

        return (value,)


class SimpleMathBoolean:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value):
        return (
            value,
            int(value),
        )


class SimpleMath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "a": (any, {"default": 0.0}),
                "b": (any, {"default": 0.0}),
                "c": (any, {"default": 0.0}),
            },
            "required": {
                "value": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value, a=0.0, b=0.0, c=0.0, d=0.0):
        import ast
        import operator as op

        h, w = 0.0, 0.0
        if hasattr(a, "shape"):
            a = list(a.shape)
        if hasattr(b, "shape"):
            b = list(b.shape)
        if hasattr(c, "shape"):
            c = list(c.shape)
        if hasattr(d, "shape"):
            d = list(d.shape)

        if isinstance(a, str):
            a = float(a)
        if isinstance(b, str):
            b = float(b)
        if isinstance(c, str):
            c = float(c)
        if isinstance(d, str):
            d = float(d)

        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Pow: op.pow,
            # ast.BitXor: op.xor,
            # ast.BitOr: op.or_,
            # ast.BitAnd: op.and_,
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
            if isinstance(node, ast.Num):  # number
                return node.n
            elif isinstance(node, ast.Name):  # variable
                if node.id == "a":
                    return a
                if node.id == "b":
                    return b
                if node.id == "c":
                    return c
                if node.id == "d":
                    return d
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                return operators[type(node.op)](eval_(node.operand))
            elif isinstance(node, ast.Compare):  # comparison operators
                left = eval_(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    if not operators[type(op)](left, eval_(comparator)):
                        return 0
                return 1
            elif isinstance(node, ast.BoolOp):  # boolean operators (And, Or)
                values = [eval_(value) for value in node.values]
                return operators[type(node.op)](*values)
            elif isinstance(node, ast.Call):  # custom function
                if node.func.id in op_functions:
                    args = [eval_(arg) for arg in node.args]
                    return op_functions[node.func.id](*args)
            elif isinstance(node, ast.Subscript):  # indexing or slicing
                value = eval_(node.value)
                if isinstance(node.slice, ast.Constant):
                    return value[node.slice.value]
                else:
                    return 0
            else:
                return 0

        result = eval_(ast.parse(value, mode="eval").body)

        if math.isnan(result):
            result = 0.0

        return (
            round(result),
            result,
        )


class SimpleMathDual:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "a": (any, {"default": 0.0}),
                "b": (any, {"default": 0.0}),
                "c": (any, {"default": 0.0}),
                "d": (any, {"default": 0.0}),
            },
            "required": {
                "value_1": ("STRING", {"multiline": False, "default": ""}),
                "value_2": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = ("int_1", "float_1", "int_2", "float_2")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, value_1, value_2, a=0.0, b=0.0, c=0.0, d=0.0):
        return SimpleMath().execute(value_1, a, b, c, d) + SimpleMath().execute(
            value_2, a, b, c, d
        )


class SimpleMathCondition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "a": (any, {"default": 0.0}),
                "b": (any, {"default": 0.0}),
                "c": (any, {"default": 0.0}),
            },
            "required": {
                "evaluate": (any, {"default": 0}),
                "on_true": ("STRING", {"multiline": False, "default": ""}),
                "on_false": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    def execute(self, evaluate, on_true, on_false, a=0.0, b=0.0, c=0.0):
        return SimpleMath().execute(on_true if evaluate else on_false, a, b, c)


class SimpleCondition:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "evaluate": (any, {"default": 0}),
                "on_true": (any, {"default": 0}),
            },
            "optional": {
                "on_false": (any, {"default": None}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("result",)
    FUNCTION = "execute"

    CATEGORY = _CATEGORY

    def execute(self, evaluate, on_true, on_false=None):
        from comfy_execution.graph import ExecutionBlocker

        if not evaluate:
            return (on_false if on_false is not None else ExecutionBlocker(None),)

        return (on_true,)


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


class ConsoleDebug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (any, {}),
            },
            "optional": {
                "prefix": ("STRING", {"multiline": False, "default": "Value:"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def execute(self, value, prefix):
        return (None,)


class DebugTensorShape:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (any, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    def execute(self, tensor):
        shapes = []

        def tensorShape(tensor):
            if isinstance(tensor, dict):
                for k in tensor:
                    tensorShape(tensor[k])
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensorShape(tensor[i])
            elif hasattr(tensor, "shape"):
                shapes.append(list(tensor.shape))

        tensorShape(tensor)

        return (None,)


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

    def execute(self, batch):
        count = 0
        if hasattr(batch, "shape"):
            count = batch.shape[0]
        elif isinstance(batch, dict) and "samples" in batch:
            count = batch["samples"].shape[0]
        elif isinstance(batch, list) or isinstance(batch, dict):
            count = len(batch)

        return (count,)
