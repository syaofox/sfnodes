# SFMathInt / SFBatchAnything 后端逻辑测试（H7/H8）：
#  - SFMathInt divide/modulo 除零回退 0；power 负指数/0**-1 兜底
#  - SFBatchAnything 标量+张量、两个非 samples dict 不再崩溃
# mock：torch / comfy.utils / comfy_execution.graph_utils（仅执行 execute 路径）
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

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
