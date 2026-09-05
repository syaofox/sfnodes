# SFMathInt / SFBatchAnything / 循环 LoopEnd 后端逻辑测试（H7/H8）：
#  - SFMathInt divide/modulo 除零回退 0；power 负指数/0**-1 兜底
#  - SFBatchAnything 标量+张量、两个非 samples dict 不再崩溃
#  - SFForLoopEnd/SFWhileLoopEnd OUTPUT_NODE 声明（悬空循环可执行）；
#    SFWhileLoopEnd._collect_output_nodes 跳过 SFForLoopEnd 防克隆嵌套展开
# mock：torch / comfy.utils / comfy_execution.graph_utils / nodes（仅执行 execute 路径）
# 运行：python tests/test_logic.py
import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock torch ──
class _T:
    """最小 tensor 形状对象：只支持 isinstance / shape / 报错的加法。"""
    def __init__(self, shape, device="cpu"):
        self.shape = tuple(shape)
        self.device = device
    def movedim(self, *a):
        return self
    def __add__(self, other):
        raise TypeError(f"unsupported operand for +: Tensor and {type(other).__name__}")
    def __radd__(self, other):
        raise TypeError(f"unsupported operand for +: {type(other).__name__} and Tensor")

torch = types.ModuleType("torch")
torch.Tensor = _T
torch.cat = lambda tensors, dim=0: ("cat", [t.shape for t in tensors])
sys.modules["torch"] = torch

# ── mock comfy.utils ──
comfy = types.ModuleType("comfy")
comfy.utils = types.ModuleType("comfy.utils")
comfy.utils.common_upscale = lambda *a, **k: ("upscaled",)
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = comfy.utils

# ── mock comfy_execution.graph_utils ──
ce = types.ModuleType("comfy_execution")
cg = types.ModuleType("comfy_execution.graph_utils")
cg.GraphBuilder = type("GraphBuilder", (), {})
cg.ExecutionBlocker = lambda v: ("blocker", v)
cg.is_link = lambda v: isinstance(v, list) and len(v) == 2
sys.modules["comfy_execution"] = ce
sys.modules["comfy_execution.graph_utils"] = cg

# ── 加载 nodes/logic.py ──
for name, path in [("sfnodes", "."), ("sfnodes.sf_utils", "sf_utils"), ("sfnodes.nodes", "nodes"), ("sfnodes.nodes.utils", "nodes/utils")]:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(root, path)]
        sys.modules[name] = m
spec = importlib.util.spec_from_file_location("sfnodes.nodes.logic", os.path.join(root, "nodes", "logic.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["sfnodes.nodes.logic"] = mod
spec.loader.exec_module(mod)

# ── SFMathInt ──
mi = mod.SFMathInt()
check("divide 正常", mi.execute(10, 2, "divide")[0] == 5)
check("divide 除零回退 0", mi.execute(10, 0, "divide")[0] == 0)
check("modulo 正常", mi.execute(10, 3, "modulo")[0] == 1)
check("modulo 除零回退 0", mi.execute(10, 0, "modulo")[0] == 0)
check("add 正常", mi.execute(1, 2, "add")[0] == 3)
check("power 正常", mi.execute(2, 8, "power")[0] == 256)
check("power 负指数 int 化", isinstance(mi.execute(2, -1, "power")[0], int))
check("power 0**-1 兜底 0", mi.execute(0, -1, "power")[0] == 0)

# ── SFBatchAnything ──
ba = mod.SFBatchAnything()
check("tensor+tensor 拼接", ba.execute(_T((1, 2, 2, 3)), _T((1, 2, 2, 3)))[0][0] == "cat")
check("tensor+None 直通", ba.execute(_T((1, 2, 2, 3)), None)[0] is not None)
check("None+tensor 直通", ba.execute(None, _T((1, 2, 2, 3)))[0] is not None)
r = ba.execute(_T((1, 2, 2, 3)), "x")
check("tensor+str 不崩 -> [str, tensor]", isinstance(r[0], list) and r[0][0] == "x")
r = ba.execute(_T((1, 2, 2, 3)), [1, 2])
check("tensor+list 不崩 -> [tensor, list]", isinstance(r[0], list) and r[0][1] == [1, 2])
r = ba.execute({"a": 1}, {"b": 2})
check("两 dict 无 samples 不崩 -> [d1, d2]", isinstance(r[0], list) and r[0][0] == {"a": 1} and r[0][1] == {"b": 2})
r = ba.execute("x", "y")
check("str+str 合并 [x, y]", r[0] == ["x", "y"])
r = ba.execute(1, 2)
check("int+int 合并 [1, 2]", r[0] == [1, 2])

# ── 悬空循环可执行（OUTPUT_NODE）+ 重建收集排除 SFForLoopEnd ──
# 执行器只从 OUTPUT_NODE 节点反向入队：LoopEnd 输出悬空时若无 OUTPUT_NODE
# 则整个循环静默不跑（platform.md §1）。两 LoopEnd 均须声明 OUTPUT_NODE。
check("SFForLoopEnd.OUTPUT_NODE", getattr(mod.SFForLoopEnd, "OUTPUT_NODE", None) is True)
check("SFWhileLoopEnd.OUTPUT_NODE", getattr(mod.SFWhileLoopEnd, "OUTPUT_NODE", None) is True)

# _collect_output_nodes 需要 nodes.NODE_CLASS_MAPPINGS（运行时由 ComfyUI 提供）
mock_nodes = types.ModuleType("nodes")
mock_nodes.NODE_CLASS_MAPPINGS = {
    "SaveImage": type("SaveImage", (), {"OUTPUT_NODE": True}),
    "PreviewImage": type("PreviewImage", (), {"OUTPUT_NODE": True}),
    "SFForLoopStart": mod.SFForLoopStart,
    "SFForLoopEnd": mod.SFForLoopEnd,
    "SFWhileLoopStart": mod.SFWhileLoopStart,
    "SFWhileLoopEnd": mod.SFWhileLoopEnd,
    "SFMathInt": mod.SFMathInt,
}
sys.modules["nodes"] = mock_nodes

prompts = {
    "1": {"class_type": "SaveImage", "inputs": {"image": ["2", 0]}},
    "2": {"class_type": "SFForLoopEnd", "inputs": {"flow": ["0", 0], "initial_value1": ["3", 0]}},
    "3": {"class_type": "SFMathInt", "inputs": {"a": 1, "b": 1, "operation": "add"}},
    "4": {"class_type": "SFForLoopStart", "inputs": {"total": 3}},
    "5": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
}
collected = mod.SFWhileLoopEnd()._collect_output_nodes(prompts)
check("收集 OUTPUT_NODE 节点", set(collected.keys()) == {"1", "5"})
check("收集内容为链接输入", collected["1"] == [["2", 0]])

# while_loop_close condition=True 重建：contained 不得包含 SFForLoopEnd（"2"），
# 否则重建图克隆 LoopEnd 再次 expand → 嵌套错误展开。
class _FakeBuilderNode:
    def __init__(self, g, nid):
        self.g, self.nid = g, nid
    def set_input(self, k, v):
        self.g.nodes[self.nid]["inputs"][k] = v
    def set_override_display_id(self, did):
        self.g.nodes[self.nid]["override_display_id"] = did
    def out(self, slot):
        return [self.nid, slot]

class _FakeBuilder:
    def __init__(self):
        self.nodes = {}
    def node(self, class_type, nid=None):
        self.nodes[nid] = {"class_type": class_type, "inputs": {}}
        return _FakeBuilderNode(self, nid)
    def lookup_node(self, nid):
        return _FakeBuilderNode(self, nid)
    def finalize(self):
        return self.nodes

class _FakeDynPrompt:
    def __init__(self, nodes_):
        self.nodes = nodes_
    def get_node(self, nid):
        return self.nodes.get(nid)
    def get_display_node_id(self, nid):
        return nid
    def get_original_prompt(self):
        return self.nodes

mod.GraphBuilder = _FakeBuilder
wle = mod.SFWhileLoopEnd()
loop_prompts = {
    "0": {"class_type": "SFForLoopStart", "inputs": {"total": 3}},
    "1": {"class_type": "SFMathInt", "inputs": {"a": ["0", 1], "b": 1, "operation": "add"}},
    "2": {"class_type": "SFWhileLoopEnd", "inputs": {
        "flow": ["0", 0], "condition": ["1", 0], "initial_value1": ["1", 0]}},
    "3": {"class_type": "SFForLoopEnd", "inputs": {"flow": ["0", 0], "initial_value1": ["1", 0]}},
}
ret = wle.while_loop_close(
    flow=["0", 0], condition=True,
    dynprompt=_FakeDynPrompt(loop_prompts), unique_id="2",
    initial_value1=("v",),
)
check("重建返回 expand", isinstance(ret, dict) and "expand" in ret)
check("重建图不含 SFForLoopEnd 克隆", "3" not in ret["expand"])
check("重建图含循环体/起始/自身克隆", set(ret["expand"].keys()) == {"0", "1", "Recurse"})
check("Recurse 输出作为结果 link", ret["result"][0] == ["Recurse", 0])

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
