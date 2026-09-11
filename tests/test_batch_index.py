# SFImageBatchIndex 后端逻辑测试（Node/Python 直接运行：python tests/test_batch_index.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（index min -1 / -1 取尾帧 tooltip）、根 __init__.py 注册键一致
#   - execute 集成（FakeTensor numpy 代理，无 torch 依赖）：
#     正常索引、-1 取尾帧、越界抛错、负值（非 -1）抛错、ndim 非法抛错
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


# ── FakeTensor：numpy 代理（支持 shape/ndim/切片）──
class FakeTensor:
    device = "cpu"

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __getitem__(self, key):
        return FakeTensor(self.data[key])

    def numpy(self):
        return self.data


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


mod = load(os.path.join(root, "nodes", "image", "batch_index.py"), "sfnodes.nodes.image.batch_index")
SFImageBatchIndex = mod.SFImageBatchIndex

# ── 1. 结构 ──
check("CATEGORY", SFImageBatchIndex.CATEGORY == "sfnodes/image")
check("FUNCTION", SFImageBatchIndex.FUNCTION == "execute")
check("RETURN_TYPES", SFImageBatchIndex.RETURN_TYPES == ("IMAGE",))
check("RETURN_NAMES", SFImageBatchIndex.RETURN_NAMES == ("image",))
check("DESCRIPTION 存在", isinstance(getattr(SFImageBatchIndex, "DESCRIPTION", None), str)
      and SFImageBatchIndex.DESCRIPTION.strip() != "")

schema = SFImageBatchIndex.INPUT_TYPES()
check("index min -1", schema["required"]["index"][1].get("min") == -1)

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
check("__init__ 注册 SFImageBatchIndex 双字典一致",
      "SFImageBatchIndex" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFImageBatchIndex" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. execute 集成 ──
node = SFImageBatchIndex()
data = np.arange(5 * 2 * 2 * 3, dtype=np.float32).reshape(5, 2, 2, 3)
images = FakeTensor(data)

out = node.execute(images, 0)
check("首帧", np.array_equal(out[0].numpy(), data[0:1]))
out = node.execute(images, 4)
check("尾帧正索引", np.array_equal(out[0].numpy(), data[4:5]))
out = node.execute(images, -1)
check("-1 取尾帧", np.array_equal(out[0].numpy(), data[4:5]))

try:
    node.execute(images, 5)
    check("越界抛错", False)
except ValueError:
    check("越界抛错", True)
try:
    node.execute(images, -2)
    check("负值非-1抛错", False)
except ValueError:
    check("负值非-1抛错", True)
try:
    node.execute(FakeTensor(np.zeros((2, 2, 3), dtype=np.float32)), 0)
    check("ndim 非法抛错", False)
except ValueError:
    check("ndim 非法抛错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("ALL PASSED")
