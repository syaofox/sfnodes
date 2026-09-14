# SFMaskCache 后端逻辑测试（Node/Python 直接运行：python tests/test_mask_cache.py）
# 覆盖：
#   - 结构：CATEGORY/RETURN_TYPES/NAMES/FUNCTION/DESCRIPTION、INPUT_TYPES
#     （name combo / force / lazy masks / signature forceInput / source IMAGE）、
#     VALIDATE_INPUTS=True、根 __init__.py 注册键一致
#   - 纯函数（tempdir + 内存 safetensors/torch 桩）：
#     clean_name / cache_paths / quantize_masks / source_signature /
#     save→load 往返 / cache_hit 签名与源键 / list_cache_names / read_meta
#   - lazy 决策与执行：无 masks→[]、未命中→["masks"]、命中→[]、force→["masks"]；
#     execute 读取路径/保存路径/缓存缺失报错
import ast
import importlib.util
import os
import sys
import tempfile
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


# 注册 sfnodes 包结构，使节点相对导入（from ...sf_utils.disk_state import）
# 可解析（test_save_image_exact.py 同款）。
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.mask", "nodes/mask"),
                   ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules[_pkg] = _m

_spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.mask.mask_cache",
    os.path.join(root, "nodes", "mask", "mask_cache.py"),
)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)
SFMaskCache = mod.SFMaskCache

tmpdir = tempfile.mkdtemp(prefix="sf_mask_cache_test_")
mod.cache_dir = lambda: tmpdir


class FakeSF:
    def __init__(self):
        self.files = {}

    def save_file(self, tensors, path, metadata=None):
        self.files[path] = tensors
        with open(path, "wb") as f:  # 占位文件，让 isfile 路径检查成立
            f.write(b"")

    def load_file(self, path):
        return self.files[path]


fake_sf = FakeSF()
fake_torch = types.SimpleNamespace(from_numpy=lambda a: a)

_orig_save = mod.save_mask_cache
_orig_load = mod.load_mask_cache


def save_stub(name, masks, signature="", source_sig=""):
    return _orig_save(name, masks, signature, source_sig, torch=fake_torch, sf=fake_sf, write_preview=False)


def load_stub(name):
    return _orig_load(name, torch=fake_torch, sf=fake_sf)


mod.save_mask_cache = save_stub
mod.load_mask_cache = load_stub

# ── 1. 结构 ──
check("CATEGORY", SFMaskCache.CATEGORY == "sfnodes/mask")
check("FUNCTION", SFMaskCache.FUNCTION == "execute")
check("RETURN_TYPES", SFMaskCache.RETURN_TYPES == ("MASK",))
check("RETURN_NAMES", SFMaskCache.RETURN_NAMES == ("mask",))
check("DESCRIPTION 存在", isinstance(getattr(SFMaskCache, "DESCRIPTION", None), str)
      and SFMaskCache.DESCRIPTION.strip() != "")
check("VALIDATE_INPUTS True", SFMaskCache.VALIDATE_INPUTS(name="whatever") is True)

schema = SFMaskCache.INPUT_TYPES()
check("name required", "name" in schema["required"] and isinstance(schema["required"]["name"][0], list))
check("force required BOOLEAN", schema["required"]["force"][0] == "BOOLEAN")
check("masks optional MASK", schema["optional"]["masks"][0] == "MASK")
check("masks lazy", schema["optional"]["masks"][1].get("lazy") is True)
check("signature forceInput", schema["optional"]["signature"][1].get("forceInput") is True)
check("source optional IMAGE", schema["optional"]["source"][0] == "IMAGE")

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
check("__init__ 注册 SFMaskCache 双字典一致",
      "SFMaskCache" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFMaskCache" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. 纯函数 ──
check("clean_name 正常", mod.clean_name("drive") == "drive")
check("clean_name 去路径", mod.clean_name("a/b") == "a_b")
check("clean_name 拒绝穿越", mod.clean_name("..") == "")
check("clean_name 空", mod.clean_name("   ") == "")

paths = mod.cache_paths("drive")
check("cache_paths 后缀", paths[0].endswith(".safetensors") and paths[1].endswith(".json")
      and paths[2].endswith(".png"))
check("cache_paths 非法返回 None", mod.cache_paths("..") is None)

m2 = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
q2 = mod.quantize_masks(m2)
check("quantize 2D→3D", q2.shape == (1, 2, 2))
check("quantize 值", q2[0, 0, 1] == 255 and q2[0, 1, 0] == 128)

m3 = np.zeros((3, 4, 5), dtype=np.float32)
m3[1] = 1.0
q3 = mod.quantize_masks(m3)
check("quantize 3D shape", q3.shape == (3, 4, 5) and q3.dtype == np.uint8)

src_a = np.zeros((2, 8, 8, 3), dtype=np.float32)
src_b = src_a.copy()
src_b[1, 0, 0, 0] = 1.0
sa1 = mod.source_signature(src_a)
sa2 = mod.source_signature(src_a)
sb = mod.source_signature(src_b)
check("source_signature 确定性", sa1 == sa2 and sa1 != "")
check("source_signature 区分源", sa1 != sb)
check("source_signature None", mod.source_signature(None) == "")

# save → load 往返
mod.save_mask_cache("drive", m3, signature="sig1", source_sig="src1")
loaded = mod.load_mask_cache("drive")
check("save/load 往返 shape", loaded.shape == (3, 4, 5))
check("save/load 往返 值", np.allclose(loaded, mod.quantize_masks(m3).astype(np.float32) / 255.0))
check("list_cache_names", "drive" in mod.list_cache_names())
meta = mod.read_meta("drive")
check("read_meta 字段", meta["n_frames"] == 3 and meta["height"] == 4 and meta["width"] == 5
      and meta["signature"] == "sig1" and meta["source"] == "src1")
check("cache_hit 命中", mod.cache_hit("drive", "sig1", "src1") is True)
check("cache_hit 签名不符", mod.cache_hit("drive", "other", "src1") is False)
check("cache_hit 源不符", mod.cache_hit("drive", "sig1", "other") is False)
check("cache_hit 缺失", mod.cache_hit("nope") is False)

# ── 3. lazy 决策与 execute ──
node = SFMaskCache()
mod.save_mask_cache("hitone", m3, signature="sig1", source_sig="")
check("check_lazy 无 masks→[]", node.check_lazy_status("hitone", **{}) == [])
check("check_lazy 未命中→masks",
      node.check_lazy_status("missing", False, "", None, **{"masks": None}) == ["masks"])
check("check_lazy 命中→[]",
      node.check_lazy_status("hitone", False, "sig1", None, **{"masks": None}) == [])
check("check_lazy 签名不符→masks",
      node.check_lazy_status("hitone", False, "other", None, **{"masks": None}) == ["masks"])
check("check_lazy force→masks",
      node.check_lazy_status("hitone", True, "sig1", None, **{"masks": None}) == ["masks"])

# execute 保存路径
res = node.execute("newone", False, "", None, m3)
check("execute 保存返回 masks", isinstance(res, tuple) and len(res) == 1 and res[0].shape == (3, 4, 5))
check("execute 保存已落盘", "newone" in mod.list_cache_names())
# execute 读取路径（masks=None，且命中）
res2 = node.execute("newone", False, "", None, None)
check("execute 读取返回 masks", res2[0].shape == (3, 4, 5))
# 缓存缺失 + 无 masks → 报错
try:
    node.execute("does_not_exist", False, "", None, None)
    check("execute 缓存缺失报错", False)
except RuntimeError:
    check("execute 缓存缺失报错", True)
# 非法名 → ValueError
try:
    node.execute("..", False, "", None, None)
    check("execute 非法名报错", False)
except ValueError:
    check("execute 非法名报错", True)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
