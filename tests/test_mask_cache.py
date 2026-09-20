# SFMaskCache 后端逻辑测试（Node/Python 直接运行：python tests/test_mask_cache.py）
# 覆盖：
#   - 结构：CATEGORY/RETURN_TYPES/NAMES/FUNCTION/DESCRIPTION、INPUT_TYPES
#     （name combo / force / source_key required forceInput / lazy masks /
#     name_text 文本覆盖 forceInput；无 source/signature）、VALIDATE_INPUTS=True、
#     根 __init__.py 注册键一致
#   - 纯函数（tempdir + 内存 safetensors/torch 桩）：
#     clean_name / cache_paths / quantize_masks / required_source_key（空串报错）/
#     save→load 往返 / cache_hit 单键 / list_cache_names / read_meta / 原子写 /
#     旧 meta source 字段兼容读
#   - lazy 决策与执行：无 masks→[]、未命中→["masks"]、命中→[]、force→["masks"]；
#     execute 读取路径/保存路径/缓存缺失报错/空键报错、命中带 masks 不重写
import ast
import importlib.util
import json
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
        self.save_paths = []

    @staticmethod
    def _final(path):
        # 原子写临时名 <final>.<pid>.<tid>.tmp 归一化回最终路径
        return path.rsplit(".", 3)[0] if path.endswith(".tmp") else path

    def save_file(self, tensors, path, metadata=None):
        self.save_paths.append(path)
        self.files[self._final(path)] = tensors
        with open(path, "wb") as f:  # 占位文件（原子临时路径），供 os.replace 搬移
            f.write(b"")

    def load_file(self, path):
        return self.files[path]


fake_sf = FakeSF()
fake_torch = types.SimpleNamespace(from_numpy=lambda a: a)

_orig_save = mod.save_mask_cache
_orig_load = mod.load_mask_cache


def save_stub(name, masks, source_key="", **kwargs):
    return _orig_save(name, masks, source_key, torch=fake_torch, sf=fake_sf, write_preview=False)


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
check("source_key required STRING forceInput",
      schema["required"]["source_key"][0] == "STRING"
      and schema["required"]["source_key"][1].get("forceInput") is True)
check("masks optional lazy", schema["optional"]["masks"][0] == "MASK"
      and schema["optional"]["masks"][1].get("lazy") is True)
check("name_text optional STRING forceInput",
      schema["optional"]["name_text"][0] == "STRING"
      and schema["optional"]["name_text"][1].get("forceInput") is True
      and schema["optional"]["name_text"][1].get("default") == "")
check("source/signature 已移除", "source" not in schema["optional"] and "signature" not in schema["optional"])

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

check("required_source_key 文本", mod.cache_store.required_source_key("a.mp4|n=1") == "a.mp4|n=1")
check("required_source_key 列表取首个非空",
      mod.cache_store.required_source_key(["", "k1", "k2"]) == "k1")
try:
    mod.cache_store.required_source_key("   ")
    check("required_source_key 空串报错", False)
except ValueError:
    check("required_source_key 空串报错", True)

m2 = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
q2 = mod.quantize_masks(m2)
check("quantize 2D→3D", q2.shape == (1, 2, 2))
check("quantize 值", q2[0, 0, 1] == 255 and q2[0, 1, 0] == 128)

m3 = np.zeros((3, 4, 5), dtype=np.float32)
m3[1] = 1.0
q3 = mod.quantize_masks(m3)
check("quantize 3D shape", q3.shape == (3, 4, 5) and q3.dtype == np.uint8)

# save → load 往返
mod.save_mask_cache("drive", m3, source_key="src1")
loaded = mod.load_mask_cache("drive")
check("save/load 往返 shape", loaded.shape == (3, 4, 5))
check("save/load 往返 值", np.allclose(loaded, mod.quantize_masks(m3).astype(np.float32) / 255.0))
check("list_cache_names", "drive" in mod.list_cache_names())
meta = mod.read_meta("drive")
check("read_meta 字段", meta["n_frames"] == 3 and meta["height"] == 4 and meta["width"] == 5
      and meta["source_key"] == "src1" and "signature" not in meta)
check("cache_hit 命中", mod.cache_hit("drive", "src1") is True)
check("cache_hit 键不符", mod.cache_hit("drive", "other") is False)
check("cache_hit 缺失", mod.cache_hit("nope") is False)

# 旧 meta（§111 前）回退读 source 字段
legacy = json.load(open(mod.cache_paths("drive")[1], encoding="utf-8"))
legacy.pop("source_key")
legacy["source"] = "oldhash"
mod.cache_store.write_meta(tmpdir, "drive", legacy)
check("旧 meta source 字段兼容读", mod.cache_hit("drive", "oldhash") is True
      and mod.cache_hit("drive", "src1") is False)
mod.save_mask_cache("drive", m3, source_key="src1")  # 恢复新格式

# ── 3. lazy 决策与 execute ──
node = SFMaskCache()
mod.save_mask_cache("hitone", m3, source_key="key1")
check("check_lazy 无 masks→[]", node.check_lazy_status("hitone", source_key="key1", **{}) == [])
check("check_lazy 未命中→masks",
      node.check_lazy_status("missing", source_key="key1", **{"masks": None}) == ["masks"])
check("check_lazy 命中→[]",
      node.check_lazy_status("hitone", source_key="key1", **{"masks": None}) == [])
check("check_lazy 键不符→masks",
      node.check_lazy_status("hitone", source_key="other", **{"masks": None}) == ["masks"])
check("check_lazy force→masks",
      node.check_lazy_status("hitone", True, source_key="key1", **{"masks": None}) == ["masks"])
try:
    node.check_lazy_status("hitone", source_key="", **{"masks": None})
    check("check_lazy 空 key 报错", False)
except ValueError:
    check("check_lazy 空 key 报错", True)

# name_text 文本覆盖：非空优先（下拉仅作回退）
check("_resolve_name 文本优先", mod._resolve_name("combo", "text") == "text")
check("_resolve_name 空白回退", mod._resolve_name("combo", "   ") == "combo")
check("_resolve_name 列表取首个非空", mod._resolve_name("combo", ["", "only"]) == "only")
check("_resolve_name 非字符串回退", mod._resolve_name("combo", 123) == "combo")
mod.save_mask_cache("textname", m3, source_key="k")
check("check_lazy name_text 命中→[]",
      node.check_lazy_status("combo", name_text="textname", source_key="k",
                             **{"masks": None}) == [])
check("check_lazy name_text 未命中→masks",
      node.check_lazy_status("combo", name_text="missing", source_key="k",
                             **{"masks": None}) == ["masks"])
res_text = node.execute("combo", masks=m3, name_text="textwrite", source_key="k")
check("execute name_text 覆盖写入", "textwrite" in mod.list_cache_names() and res_text[0].shape == (3, 4, 5))
res_read = node.execute("combo", name_text="textwrite", source_key="k")
check("execute name_text 覆盖读取", res_read[0].shape == (3, 4, 5))
try:
    node.execute("", masks=m3, name_text="   ", source_key="k")
    check("execute 文本空白且下拉为空报错", False)
except ValueError:
    check("execute 文本空白且下拉为空报错", True)
try:
    node.execute("newone", masks=m3, source_key="")
    check("execute 空 key 报错", False)
except ValueError:
    check("execute 空 key 报错", True)

# execute 保存路径
res = node.execute("newone", masks=m3, source_key="k")
check("execute 保存返回 masks", isinstance(res, tuple) and len(res) == 1 and res[0].shape == (3, 4, 5))
check("execute 保存已落盘", "newone" in mod.list_cache_names())
# execute 读取路径（masks=None，且命中）
res2 = node.execute("newone", source_key="k")
check("execute 读取返回 masks", res2[0].shape == (3, 4, 5))
# 缓存缺失 + 无 masks → 报错
try:
    node.execute("does_not_exist", source_key="k")
    check("execute 缓存缺失报错", False)
except RuntimeError:
    check("execute 缓存缺失报错", True)
# 非法名 → ValueError
try:
    node.execute("..", source_key="k")
    check("execute 非法名报错", False)
except ValueError:
    check("execute 非法名报错", True)

# 命中带 masks 不重写 + 原子写（P0 写盘守卫）
res_new = node.execute("knew", masks=m3, source_key="path/w.mp4|cap=32")
check("execute 写 meta 用 source_key", mod.read_meta("knew")["source_key"] == "path/w.mp4|cap=32")
check("cache_hit 用 source_key",
      mod.cache_hit("knew", "path/w.mp4|cap=32") is True
      and mod.cache_hit("knew", "other") is False)
check("原子写：save_file 先写 .tmp", fake_sf.save_paths[-1].endswith(".tmp"))
check("原子写：临时文件无残留", not [p for p in os.listdir(tmpdir) if p.endswith(".tmp")])
check("原子写：最终文件存在", os.path.isfile(mod.cache_paths("knew")[0]))

_key_calls = len(fake_sf.save_paths)
res_key = node.execute("knew", masks=m3, source_key="path/w.mp4|cap=32")
check("execute 命中跳过重写（守卫）",
      res_key[0] is m3 and len(fake_sf.save_paths) == _key_calls)
node.execute("knew", True, masks=m3, source_key="path/w.mp4|cap=32")
check("execute force 仍强制重写", len(fake_sf.save_paths) == _key_calls + 1)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
