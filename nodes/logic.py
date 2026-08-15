import os

import torch

import comfy.utils
from comfy_execution.graph_utils import GraphBuilder, ExecutionBlocker, is_link

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


COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
    "a > 0": lambda a, b: a > 0,
    "a <= 0": lambda a, b: a <= 0,
    "b > 0": lambda a, b: b > 0,
    "b <= 0": lambda a, b: b <= 0,
}


class SFMathInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "整数运算：按 operation 对 a、b 执行加减乘除模幂，输出整数结果。循环节点内部依赖此节点"

    def execute(self, a, b, operation):
        # b 默认 0（不接线时）：divide/modulo 除零直接崩会拖垮整个 run（循环
        # 节点内部依赖本节点），回退 0 并告警。power 负指数产出 float、0**-1
        # 抛 ZeroDivisionError，统一 int() 化并兜底。
        if operation == "divide":
            if b == 0:
                print("[SFMathInt] divide by zero - returning 0")
                return (0,)
            return (a // b,)
        if operation == "modulo":
            if b == 0:
                print("[SFMathInt] modulo by zero - returning 0")
                return (0,)
            return (a % b,)
        if operation == "power":
            try:
                return (int(a ** b),)
            except (ZeroDivisionError, ValueError, OverflowError):
                return (0,)
        ops = {
            "add": lambda: a + b,
            "subtract": lambda: a - b,
            "multiply": lambda: a * b,
        }
        return (ops[operation](),)


class SFCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comparison": (list(COMPARE_FUNCTIONS.keys()), {"default": "a == b"}),
            },
            "optional": {
                "a": (any_type, {"default": 0}),
                "b": (any_type, {"default": 0}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "比较运算：按 comparison 规则比较 a、b，输出布尔结果。循环节点内部依赖此节点"

    def execute(self, a=0, b=0, comparison="a == b"):
        return (COMPARE_FUNCTIONS[comparison](a, b),)


class SFWhileLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = ("FLOW_CONTROL",) + tuple(any_type for _ in range(MAX_FLOW_NUM))
    RETURN_NAMES = ("flow",) + tuple("value%d" % i for i in range(MAX_FLOW_NUM))
    FUNCTION = "while_loop_open"
    CATEGORY = _CATEGORY
    DESCRIPTION = "While 循环起始节点：condition 为真时输出初始值并执行循环体，为假时输出被阻断，配合 SF While Loop End 使用"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get("initial_value%d" % i, None) if condition else ExecutionBlocker(None))
        return tuple(["stub"] + values)


class SFWhileLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {},
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (any_type,)
        return inputs

    RETURN_TYPES = tuple(any_type for _ in range(MAX_FLOW_NUM))
    RETURN_NAMES = tuple("value%d" % i for i in range(MAX_FLOW_NUM))
    FUNCTION = "while_loop_close"
    CATEGORY = _CATEGORY
    DESCRIPTION = "While 循环结束节点：condition 为真时重建并重跑循环体，为假时输出当前 value0-19"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ["SFForLoopEnd", "SFWhileLoopEnd"]:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id, links in output_nodes.items():
                for v in links:
                    id = v[0]
                    if id in parent_ids and display_id == id and output_id not in upstream[parent_id]:
                        if "." in parent_id:
                            arr = parent_id.split(".")
                            arr[len(arr) - 1] = output_id
                            upstream[parent_id].append(".".join(arr))
                        else:
                            upstream[parent_id].append(output_id)
                        break

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow, condition, dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, "OUTPUT_NODE") and class_def.OUTPUT_NODE == True:
                for k, v in node["inputs"].items():
                    if is_link(v):
                        output_nodes.setdefault(id, []).append(v)

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = tuple(my_clone.out(x) for x in range(MAX_FLOW_NUM))
        return {
            "result": result,
            "expand": graph.finalize(),
        }


class SFForLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: (any_type,) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "initial_value0": (any_type,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL", "INT") + tuple(any_type for _ in range(1, MAX_FLOW_NUM))
    RETURN_NAMES = ("flow", "index") + tuple("value%d" % i for i in range(1, MAX_FLOW_NUM))
    FUNCTION = "for_loop_start"
    CATEGORY = _CATEGORY
    DESCRIPTION = "For 循环起始节点：按 total 次数循环，index 输出当前迭代下标，value1-19 传递循环状态，配合 SF For Loop End 使用"

    def for_loop_start(self, total, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        i = 0
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        initial_values = {("initial_value%d" % num): kwargs.get("initial_value%d" % num, None) for num in
                          range(1, MAX_FLOW_NUM)}
        graph.node("SFWhileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }


class SFForLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = tuple(any_type for _ in range(1, MAX_FLOW_NUM))
    RETURN_NAMES = tuple("value%d" % i for i in range(1, MAX_FLOW_NUM))
    FUNCTION = "for_loop_end"
    CATEGORY = _CATEGORY
    DESCRIPTION = "For 循环结束节点：接收 SF For Loop Start 的 flow 与循环体末态，未达 total 时自动重建循环体继续迭代，结束后输出最终 value1-19"

    def for_loop_end(self, flow, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        while_open = flow[0]
        forstart_node = dynprompt.get_node(while_open)
        if forstart_node is None or forstart_node.get("class_type") != "SFForLoopStart":
            raise Exception("SF For Loop End 的 flow 输入必须连接 SF For Loop Start 的 flow 输出")
        total = forstart_node["inputs"]["total"]

        sub = graph.node("SFMathInt", operation="add", a=[while_open, 1], b=1)
        cond = graph.node("SFCompare", a=sub.out(0), b=total, comparison="a < b")
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in
                        range(1, MAX_FLOW_NUM)}
        while_close = graph.node(
            "SFWhileLoopEnd",
            flow=flow,
            condition=cond.out(0),
            initial_value0=sub.out(0),
            **input_values,
        )
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }


class SFBatchAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_1": (any_type,),
                "any_2": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("batch",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "合并两个任意输入：图像/潜空间按 batch 维拼接，字符串/数字/列表/元组按顺序合并，常用于循环结果累积"

    def latent_batch(self, latent_1, latent_2):
        samples_out = latent_1.copy()
        s1 = latent_1["samples"]
        s2 = latent_2["samples"]
        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = latent_1.get("batch_index", list(range(s1.shape[0]))) + \
                                     latent_2.get("batch_index", list(range(s2.shape[0])))
        return samples_out

    def execute(self, any_1, any_2):
        if isinstance(any_1, torch.Tensor) and isinstance(any_2, torch.Tensor):
            # 两端都是 Tensor 才走张量分支；一端为 None 时由末尾的 None 直通
            # 处理（None 不是 Tensor，原 or 条件会让 str+Tensor 误入本分支，
            # 对 str 调 .shape 抛 AttributeError）。
            if any_1.shape[1:] != any_2.shape[1:]:
                any_2 = comfy.utils.common_upscale(any_2.movedim(-1, 1), any_1.shape[2], any_1.shape[1],
                                                   "bilinear", "center").movedim(1, -1)
            return (torch.cat((any_1, any_2), 0),)
        elif isinstance(any_1, (str, float, int)):
            if any_2 is None:
                return (any_1,)
            elif isinstance(any_2, tuple):
                return (any_2 + (any_1,),)
            elif isinstance(any_2, list):
                return (any_2 + [any_1],)
            return ([any_1, any_2],)
        elif isinstance(any_2, (str, float, int)):
            if any_1 is None:
                return (any_2,)
            elif isinstance(any_1, tuple):
                return (any_1 + (any_2,),)
            elif isinstance(any_1, list):
                return (any_1 + [any_2],)
            return ([any_2, any_1],)
        elif isinstance(any_1, dict) and "samples" in any_1:
            if any_2 is None:
                return (any_1,)
            if isinstance(any_2, dict) and "samples" in any_2:
                return (self.latent_batch(any_1, any_2),)
        elif isinstance(any_2, dict) and "samples" in any_2:
            if any_1 is None:
                return (any_2,)
            if isinstance(any_1, dict) and "samples" in any_1:
                return (self.latent_batch(any_2, any_1),)
        if any_1 is None:
            return (any_2,)
        if any_2 is None:
            return (any_1,)
        try:
            return (any_1 + any_2,)
        except TypeError:
            # 非同型（tensor+list、两个无 samples 的 dict 等）不能直接相加，
            # 兜底包成列表而不是崩溃。
            return ([any_1, any_2],)


class ComboSelector:
    """通用下拉选择器：下拉选项在连接到目标节点的 combo 输入（Convert to input 后）时自动同步为目标选项列表。"""
    DESCRIPTION = "通用下拉选择器：将输出连接到任意节点的 combo 输入（先在目标节点右键 combo → Convert to input），下拉选项即自动同步为该 combo 的选项列表；同时输出 value_stem（去掉路径与扩展名的值，非文件名时原样返回）。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    [""],
                    {
                        "default": "",
                        "tooltip": "下拉选择的值；选项在连接到目标节点的 combo 输入（Convert to input 后）时自动同步为目标选项列表",
                    },
                ),
            }
        }

    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("value", "value_stem")
    OUTPUT_TOOLTIPS = ("选中的值（任意类型，可连接任何 combo 输入）", "去掉路径与扩展名的值（非文件名时原样返回）")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # value 选项由前端根据连接目标动态重建，会超出 INPUT_TYPES 的静态初始列表（[""]），
        # 跳过默认的 "Value not in list" 校验
        return True

    def execute(self, value):
        stem = os.path.splitext(os.path.basename(str(value)))[0]
        return (value, stem)
