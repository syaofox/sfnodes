# SFTrackDataCache 后端逻辑测试（Node/Python 直接运行：python tests/test_track_data_cache.py）
# 覆盖：
#   - 结构/注册：CATEGORY/RETURN_TYPES/NAMES/FUNCTION/DESCRIPTION、INPUT_TYPES
#     （name combo / force / lazy track_data / signature forceInput / source IMAGE）、
#     VALIDATE_INPUTS=True、根 __init__.py 双字典键一致
#   - 纯函数（tempdir + 内存 safetensors/torch 桩）：clean_name / cache_paths /
#     cache_hit / list / save→load 往返（多对象 packed + scores + n_frames +
#     orig_size 保留）/ packed=None 分支 / 非法输入报错
#   - lazy 决策与执行：无 track_data→[]、未命中→["track_data"]、命中→[]、force→；
#     execute 读/写/缺失报错
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


# 注册 sfnodes 包结构，使节点相对导入（from ...sf_utils import cache_store）
# 可解析（test_save_image_exact.py 同款）。
for _pkg, _rel in [("sfnodes", "."), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.image", "nodes/image"),
                   ("sfnodes.sf_utils", "sf_utils")]:
    _m = types.ModuleType(_pkg)
    _m.__path__ = [os.path.join(root, _rel)]
    sys.modules[_pkg] = _m

_spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.track_data_cache",
    os.path.join(root, "nodes", "image", "track_data_cache.py"),
)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)
SFTrackDataCache = mod.SFTrackDataCache

tmpdir = tempfile.mkdtemp(prefix="sf_track_cache_test_")
mod.cache_dir = lambda: tmpdir


class FakeSF:
    def __init__(self):
        self.files = {}

    def save_file(self, tensors, path, metadata=None):
        self.files[path] = tensors
        with open(path, "wb") as f:
            f.write(b"")

    def load_file(self, path):
        return self.files[path]


fake_sf = FakeSF()
fake_torch = types.SimpleNamespace(from_numpy=lambda a: a)

_orig_save = mod.save_track_cache
_orig_load = mod.load_track_cache


def save_stub(name, track_data, signature="", source_sig=""):
    return _orig_save(name, track_data, signature, source_sig,
                      torch=fake_torch, sf=fake_sf, write_preview=False)


def load_stub(name):
    return _orig_load(name, torch=fake_torch, sf=fake_sf)


mod.save_track_cache = save_stub
mod.load_track_cache = load_stub

# ── 1. 结构 ──
check("CATEGORY", SFTrackDataCache.CATEGORY == "sfnodes/image")
check("FUNCTION", SFTrackDataCache.FUNCTION == "execute")
check("RETURN_TYPES", SFTrackDataCache.RETURN_TYPES == ("SAM3_TRACK_DATA",))
check("RETURN_NAMES", SFTrackDataCache.RETURN_NAMES == ("track_data",))
check("DESCRIPTION 存在", isinstance(getattr(SFTrackDataCache, "DESCRIPTION", None), str)
      and SFTrackDataCache.DESCRIPTION.strip() != "")
check("VALIDATE_INPUTS True", SFTrackDataCache.VALIDATE_INPUTS(name="x") is True)

schema = SFTrackDataCache.INPUT_TYPES()
check("name required", "name" in schema["required"] and isinstance(schema["required"]["name"][0], list))
check("force required BOOLEAN", schema["required"]["force"][0] == "BOOLEAN")
check("track_data optional", schema["optional"]["track_data"][0] == "SAM3_TRACK_DATA")
check("track_data lazy", schema["optional"]["track_data"][1].get("lazy") is True)
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
check("__init__ 注册 SFTrackDataCache 双字典一致",
      "SFTrackDataCache" in mapping_keys.get("NODE_CLASS_MAPPINGS", set())
      and "SFTrackDataCache" in mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))
check("__init__ 字典键集合完全一致",
      mapping_keys.get("NODE_CLASS_MAPPINGS", set()) == mapping_keys.get("NODE_DISPLAY_NAME_MAPPINGS", set()))

# ── 2. 纯函数 + 存取往返 ──
check("clean_name 正常", mod.clean_name("sam3drive") == "sam3drive")
check("clean_name 拒绝穿越", mod.clean_name("..") == "")
check("cache_paths 后缀", mod.cache_paths("x")[0].endswith(".safetensors"))
check("cache_paths 非法 None", mod.cache_paths("..") is None)

packed = np.array([[[[1, 2, 3, 4]], [[255, 0, 255, 0]]],
                   [[[5, 6, 7, 8]], [[0, 255, 0, 255]]]], dtype=np.uint8)  # [T=2,N=2,H=1,Wp=4]
td = {"packed_masks": packed, "n_frames": 2, "scores": [0.9, 0.42], "orig_size": (4, 32)}
mod.save_track_cache("sam3drive", td, signature="person", source_sig="srcA")
loaded = mod.load_track_cache("sam3drive")
check("往返 packed shape", loaded["packed_masks"].shape == (2, 2, 1, 4))
check("往返 packed 内容", np.array_equal(loaded["packed_masks"], packed))
check("往返 scores", loaded["scores"] == [0.9, 0.42])
check("往返 n_frames", loaded["n_frames"] == 2)
check("往返 orig_size", loaded["orig_size"] == (4, 32))
meta = mod.read_meta("sam3drive")
check("meta num_objects/尺寸", meta["num_objects"] == 2 and meta["height"] == 4 and meta["width"] == 32)
check("list_cache_names", "sam3drive" in mod.list_cache_names())
check("cache_hit 命中", mod.cache_hit("sam3drive", "person", "srcA") is True)
check("cache_hit 签名不符", mod.cache_hit("sam3drive", "cat", "srcA") is False)
check("cache_hit 缺失", mod.cache_hit("nope") is False)

# packed=None（无对象）分支
td_empty = {"packed_masks": None, "n_frames": 3, "scores": [], "orig_size": (5, 6)}
mod.save_track_cache("emptyobj", td_empty)
le = mod.load_track_cache("emptyobj")
check("packed None 保留", le["packed_masks"] is None)
check("packed None n_frames", le["n_frames"] == 3)
check("packed None scores 空", le["scores"] == [])
check("packed None orig_size", le["orig_size"] == (5, 6))

# 非法输入
try:
    mod.save_track_cache("bad", {"not": "track"})
    check("非法 track_data 报错", False)
except ValueError:
    check("非法 track_data 报错", True)

# ── 3. lazy 决策与 execute ──
mod.save_track_cache("hitname", td, signature="person", source_sig="")
node = SFTrackDataCache()
check("check_lazy 无 track_data→[]", node.check_lazy_status("hitname", **{}) == [])
check("check_lazy 未命中→track_data",
      node.check_lazy_status("missing", False, "", None, **{"track_data": None}) == ["track_data"])
check("check_lazy 命中→[]",
      node.check_lazy_status("hitname", False, "person", None, **{"track_data": None}) == [])
check("check_lazy 签名不符→track_data",
      node.check_lazy_status("hitname", False, "other", None, **{"track_data": None}) == ["track_data"])
check("check_lazy force→track_data",
      node.check_lazy_status("hitname", True, "person", None, **{"track_data": None}) == ["track_data"])

res = node.execute("brandnew", False, "", None, td)
check("execute 保存返回 dict", isinstance(res, tuple) and res[0]["packed_masks"].shape == (2, 2, 1, 4))
check("execute 保存已落盘", "brandnew" in mod.list_cache_names())
res2 = node.execute("brandnew", False, "", None, None)
check("execute 读取返回 dict", res2[0]["packed_masks"].shape == (2, 2, 1, 4))
try:
    node.execute("does_not_exist", False, "", None, None)
    check("execute 缓存缺失报错", False)
except RuntimeError:
    check("execute 缓存缺失报错", True)
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
