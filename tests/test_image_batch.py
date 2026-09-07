# SFImageBatch 后端逻辑测试（Node/Python 直接运行：python tests/test_image_batch.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（image_1..16 全 optional、required 空）、根 __init__.py 注册键一致
#   - execute 集成（FakeTensor + mock torch.cat）：
#     多路收集按键排序、dim=0 拼接批次、None 槽跳过、
#     尺寸不一致抛错列出槽名、全空输入抛错
# mock：torch（numpy 本机真实可用）
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

# ── FakeTensor：numpy 代理 ──
class FakeTensor:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

# ── mock torch ──
torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.cat = lambda seq, dim=0: FakeTensor(np.concatenate(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
sys.modules["torch"] = torch

# ── 注册 sfnodes 包结构，使节点的相对导入可解析 ──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3

def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

mod = load(os.path.join(root, "nodes", "image", "batch.py"), "sfnodes.nodes.image.batch")
SFImageBatch = mod.SFImageBatch

# ── 1. 结构 ──
check("CATEGORY", SFImageBatch.CATEGORY == "sfnodes/image")
check("FUNCTION", SFImageBatch.FUNCTION == "execute")
check("RETURN_TYPES", SFImageBatch.RETURN_TYPES == ("IMAGE",))
check("RETURN_NAMES", SFImageBatch.RETURN_NAMES == ("image",))
check("DESCRIPTION 存在", isinstance(getattr(SFImageBatch, "DESCRIPTION", None), str)
      and SFImageBatch.DESCRIPTION.strip() != "")

schema = SFImageBatch.INPUT_TYPES()
check("required 为空", schema["required"] == {})
optional = schema["optional"]
expected_keys = {f"image_{i}" for i in range(1, 17)}
check("optional image_1..16", set(optional.keys()) == expected_keys)
check("optional 全为 IMAGE", all(v[0] == "IMAGE" for v in optional.values()))

# 根 __init__.py 注册键一致（AST 解析两个字典）
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
check("__init__ 注册 SFImageBatch 双字典一致",
      "SFImageBatch" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFImageBatch" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. execute 集成 ──
node = SFImageBatch()

def t(*shape):
    return FakeTensor(np.zeros(shape, dtype=np.float32))

# 多路收集按键排序（image_2 与 image_10 乱序传入，仍按编号顺序拼接）
a, b, c = t(1, 4, 4, 3), t(2, 4, 4, 3), t(3, 4, 4, 3)
out = node.execute(image_2=b, image_10=c, image_1=a)
check("乱序按键排序收集", out[0].shape == (6, 4, 4, 3))
check("批次内容顺序 image_1→image_2→image_10",
      out[0].data[0:1].shape == (1, 4, 4, 3)
      and out[0].data[1:3].shape == (2, 4, 4, 3)
      and np.array_equal(out[0].data, np.concatenate([a.data, b.data, c.data], axis=0)))

# None 槽跳过（ComfyUI 未连接的 optional 传 None）
out = node.execute(image_1=a, image_2=None, image_3=b)
check("None 槽跳过", out[0].shape == (3, 4, 4, 3))

# 尺寸不一致抛错并列出槽名
try:
    node.execute(image_1=t(1, 4, 4, 3), image_2=t(1, 8, 8, 3))
    check("尺寸不一致抛错", False)
except ValueError as e:
    check("尺寸不一致抛错", "image_2" in str(e))
    check("尺寸一致批次维不同仍通过", True)

# 全空输入抛错
try:
    node.execute(image_1=None, image_2=None)
    check("全空输入抛错", False)
except ValueError as e:
    check("全空输入抛错", True)

# 通道数不一致抛错
try:
    node.execute(image_1=t(1, 4, 4, 3), image_2=t(1, 4, 4, 4))
    check("通道数不一致抛错", False)
except ValueError:
    check("通道数不一致抛错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("ALL PASSED")
