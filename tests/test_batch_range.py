# SFImageBatchRange 后端逻辑测试（Node/Python 直接运行：python tests/test_batch_range.py）
# 覆盖：
#   - 结构：CATEGORY、RETURN_TYPES/NAMES、FUNCTION、DESCRIPTION、
#     INPUT_TYPES（start_index min -1 / num_frames min 1、images+mask 双 optional）、
#     根 __init__.py 注册键一致
#   - _resolve_range 纯函数：正常切片、尾部截断、-1 取尾部、-1 不足截断、
#     start 越界抛错、负值（非 -1）抛错
#   - execute 集成（FakeTensor，无 torch 依赖，切片走 numpy）：
#     IMAGE/MASK 双路独立切片、单路 None 透传、双空抛错、ndim 非法抛错
# mock：无（纯 numpy FakeTensor 实现切片语义）
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


mod = load(os.path.join(root, "nodes", "image", "batch_range.py"), "sfnodes.nodes.image.batch_range")
SFImageBatchRange = mod.SFImageBatchRange
_resolve_range = mod._resolve_range

# ── 1. 结构 ──
check("CATEGORY", SFImageBatchRange.CATEGORY == "sfnodes/image")
check("FUNCTION", SFImageBatchRange.FUNCTION == "execute")
check("RETURN_TYPES", SFImageBatchRange.RETURN_TYPES == ("IMAGE", "MASK"))
check("RETURN_NAMES", SFImageBatchRange.RETURN_NAMES == ("images", "masks"))
check("DESCRIPTION 存在", isinstance(getattr(SFImageBatchRange, "DESCRIPTION", None), str)
      and SFImageBatchRange.DESCRIPTION.strip() != "")

schema = SFImageBatchRange.INPUT_TYPES()
check("start_index min -1", schema["required"]["start_index"][1].get("min") == -1)
check("num_frames min 1", schema["required"]["num_frames"][1].get("min") == 1)
check("num_frames default 1", schema["required"]["num_frames"][1].get("default") == 1)
check("images optional IMAGE", schema["optional"]["images"][0] == "IMAGE")
check("masks optional MASK", schema["optional"]["masks"][0] == "MASK")

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
check("__init__ 注册 SFImageBatchRange 双字典一致",
      "SFImageBatchRange" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFImageBatchRange" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. _resolve_range 纯函数 ──
check("正常切片", _resolve_range(5, 1, 2) == (1, 3))
check("尾部截断", _resolve_range(5, 4, 3) == (4, 5))
check("-1 取尾部", _resolve_range(5, -1, 2) == (3, 5))
check("-1 不足取全", _resolve_range(2, -1, 5) == (0, 2))
try:
    _resolve_range(5, 5, 1)
    check("start 越界抛错", False)
except ValueError:
    check("start 越界抛错", True)
try:
    _resolve_range(5, -2, 1)
    check("负值非-1抛错", False)
except ValueError:
    check("负值非-1抛错", True)

# ── 3. execute 集成 ──
node = SFImageBatchRange()


def img(n):
    data = np.arange(n * 2 * 2 * 3, dtype=np.float32).reshape(n, 2, 2, 3)
    return FakeTensor(data)


def msk(n):
    data = np.arange(n * 2 * 2, dtype=np.float32).reshape(n, 2, 2)
    return FakeTensor(data)


images, masks = img(5), msk(5)

# 双路独立切片
out_img, out_msk = node.execute(1, 2, images=images, masks=masks)
check("双路 IMAGE 切片", np.array_equal(out_img.numpy(), images.numpy()[1:3]))
check("双路 MASK 切片", np.array_equal(out_msk.numpy(), masks.numpy()[1:3]))

# 单路 None 透传
out_img, out_msk = node.execute(0, 1, images=images, masks=None)
check("单 IMAGE 路 MASK 透传 None", out_msk is None and np.array_equal(out_img.numpy(), images.numpy()[0:1]))
out_img, out_msk = node.execute(0, 1, images=None, masks=masks)
check("单 MASK 路 IMAGE 透传 None", out_img is None and np.array_equal(out_msk.numpy(), masks.numpy()[0:1]))

# -1 尾取 + 截断
out_img, _ = node.execute(-1, 2, images=images)
check("-1 尾取 IMAGE", np.array_equal(out_img.numpy(), images.numpy()[3:5]))
out_img, _ = node.execute(4, 10, images=images)
check("尾部截断 IMAGE", np.array_equal(out_img.numpy(), images.numpy()[4:5]))

# 双空抛错
try:
    node.execute(0, 1)
    check("双空抛错", False)
except ValueError:
    check("双空抛错", True)

# ndim 非法抛错
try:
    node.execute(0, 1, images=FakeTensor(np.zeros((2, 2, 3), dtype=np.float32)))
    check("IMAGE ndim 非法抛错", False)
except ValueError:
    check("IMAGE ndim 非法抛错", True)
try:
    node.execute(0, 1, masks=FakeTensor(np.zeros((5,), dtype=np.float32)))
    check("MASK ndim 非法抛错", False)
except ValueError:
    check("MASK ndim 非法抛错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} 项 -> {failures}")
    sys.exit(1)
print("ALL PASSED")
