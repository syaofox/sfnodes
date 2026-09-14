# SFMaskBatch 后端逻辑测试（Node/Python 直接运行：python tests/test_mask_batch.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（mask_1..16 全 optional、required 空）、根 __init__.py 注册键一致
#   - execute（FakeTensor + mock torch.cat）：
#     数字序拼接（mask_10 不排到 mask_2 前）、None 槽跳过、2D 遮罩升 3D、
#     尺寸不一致抛错、全空抛错
import ast
import importlib.util
import os
import sys
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


class FakeTensor:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.ndim

    def unsqueeze(self, dim):
        return FakeTensor(np.expand_dims(self.data, dim))


torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.cat = lambda seq, dim=0: FakeTensor(np.concatenate(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
sys.modules["torch"] = torch

for pkg, rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"), ("sfnodes.nodes.image", "nodes/image")]:
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(root, rel)]
    sys.modules[pkg] = m

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.batch", os.path.join(root, "nodes", "image", "batch.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
SFMaskBatch = mod.SFMaskBatch

# ── 1. 结构 ──
check("CATEGORY", SFMaskBatch.CATEGORY == "sfnodes/image")
check("FUNCTION", SFMaskBatch.FUNCTION == "execute")
check("RETURN_TYPES", SFMaskBatch.RETURN_TYPES == ("MASK",))
check("RETURN_NAMES", SFMaskBatch.RETURN_NAMES == ("mask",))
check("DESCRIPTION 存在", isinstance(getattr(SFMaskBatch, "DESCRIPTION", None), str)
      and SFMaskBatch.DESCRIPTION.strip() != "")
schema = SFMaskBatch.INPUT_TYPES()
check("required 为空", schema["required"] == {})
check("optional mask_1..16", set(schema["optional"].keys()) == {f"mask_{i}" for i in range(1, 17)})
check("optional 全为 MASK", all(v[0] == "MASK" for v in schema["optional"].values()))

with open(os.path.join(root, "__init__.py"), encoding="utf-8") as f:
    init_src = f.read()
tree = ast.parse(init_src)
mapping_keys = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"):
                if isinstance(node.value, ast.Dict):
                    mapping_keys[t.id] = {ast.literal_eval(k) for k in node.value.keys if k is not None}
check("__init__ 注册 SFMaskBatch 双字典一致",
      "SFMaskBatch" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFMaskBatch" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. execute ──
node = SFMaskBatch()


def t(n, h=4, w=4):
    a = np.zeros((n, h, w), dtype=np.float32)
    a[:] = n  # 用帧数当标记，验证拼接顺序
    return FakeTensor(a)


# 数字序拼接：mask_10 必须排在 mask_2 之后（不是字典序）
out = node.execute(mask_2=t(2), mask_10=t(10), mask_1=t(1))
check("总帧数", out[0].shape == (13, 4, 4))
d = out[0].data
check("顺序 1→2→10", d[0].mean() == 1 and d[1].mean() == 2 and d[3].mean() == 10)

# None 槽跳过（分段数 < 端口数）
out = node.execute(mask_1=t(1), mask_2=None, mask_3=t(3))
check("None 槽跳过", out[0].shape == (4, 4, 4) and out[0].data[1].mean() == 3)

# 2D 遮罩升 3D
out = node.execute(mask_1=FakeTensor(np.zeros((4, 4), dtype=np.float32)), mask_2=t(1))
check("2D 升 3D", out[0].shape == (2, 4, 4))

# 尺寸不一致抛错
try:
    node.execute(mask_1=t(1, 4, 4), mask_2=t(1, 8, 8))
    check("尺寸不一致抛错", False)
except ValueError:
    check("尺寸不一致抛错", True)

# 全空抛错
try:
    node.execute(mask_1=None)
    check("全空抛错", False)
except ValueError:
    check("全空抛错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
