# SFTrackDataCache 后端逻辑测试（Node/Python 直接运行：python tests/test_track_data_cache.py）
# 覆盖：
#   - 结构/注册：CATEGORY/RETURN_TYPES/NAMES/FUNCTION/DESCRIPTION、INPUT_TYPES
#     （name combo / force / source_key required forceInput / lazy track_data /
#     name_text 文本覆盖 forceInput；无 source/signature）、VALIDATE_INPUTS=True、
#     根 __init__.py 双字典键一致
#   - 纯函数（tempdir + 内存 safetensors/torch 桩）：clean_name / cache_paths /
#     cache_hit 单键 / required_source_key（空串报错）/ list / save→load 往返
#     （多对象 packed + scores + n_frames + orig_size 保留）/ packed=None 分支 /
#     非法输入报错 / 原子写（.tmp + replace）/ 旧 meta source 字段兼容读
#   - lazy 决策与执行：无 track_data→[]、未命中→["track_data"]、命中→[]、force→；
#     execute 读/写/缺失报错/空键报错、命中带 track_data 不重写（写盘守卫）
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
        self.save_paths = []

    @staticmethod
    def _final(path):
        # 原子写临时名 <final>.<pid>.<tid>.tmp 归一化回最终路径
        return path.rsplit(".", 3)[0] if path.endswith(".tmp") else path

    def save_file(self, tensors, path, metadata=None):
        self.save_paths.append(path)
        self.files[self._final(path)] = tensors
        with open(path, "wb") as f:  # 真实写传入路径（原子临时文件），供 os.replace 搬移
            f.write(b"")

    def load_file(self, path):
        return self.files[path]


fake_sf = FakeSF()
fake_torch = types.SimpleNamespace(from_numpy=lambda a: a)

_orig_save = mod.save_track_cache
_orig_load = mod.load_track_cache


def save_stub(name, track_data, source_key="", **kwargs):
    return _orig_save(name, track_data, source_key,
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
check("source_key required STRING forceInput",
      schema["required"]["source_key"][0] == "STRING"
      and schema["required"]["source_key"][1].get("forceInput") is True)
check("track_data optional lazy", schema["optional"]["track_data"][0] == "SAM3_TRACK_DATA"
      and schema["optional"]["track_data"][1].get("lazy") is True)
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

check("required_source_key 文本", mod.cache_store.required_source_key("a.mp4|n=1") == "a.mp4|n=1")
check("required_source_key 列表取首个非空",
      mod.cache_store.required_source_key(["", "k1", "k2"]) == "k1")
try:
    mod.cache_store.required_source_key("   ")
    check("required_source_key 空串报错", False)
except ValueError:
    check("required_source_key 空串报错", True)

packed = np.array([[[[1, 2, 3, 4]], [[255, 0, 255, 0]]],
                   [[[5, 6, 7, 8]], [[0, 255, 0, 255]]]], dtype=np.uint8)  # [T=2,N=2,H=1,Wp=4]
td = {"packed_masks": packed, "n_frames": 2, "scores": [0.9, 0.42], "orig_size": (4, 32)}
mod.save_track_cache("sam3drive", td, source_key="srcA")
loaded = mod.load_track_cache("sam3drive")
check("往返 packed shape", loaded["packed_masks"].shape == (2, 2, 1, 4))
check("往返 packed 内容", np.array_equal(loaded["packed_masks"], packed))
check("往返 scores", loaded["scores"] == [0.9, 0.42])
check("往返 n_frames", loaded["n_frames"] == 2)
check("往返 orig_size", loaded["orig_size"] == (4, 32))
meta = mod.read_meta("sam3drive")
check("meta num_objects/尺寸", meta["num_objects"] == 2 and meta["height"] == 4 and meta["width"] == 32)
check("meta 存 source_key", meta["source_key"] == "srcA" and "signature" not in meta)
check("list_cache_names", "sam3drive" in mod.list_cache_names())
check("cache_hit 命中", mod.cache_hit("sam3drive", "srcA") is True)
check("cache_hit 键不符", mod.cache_hit("sam3drive", "other") is False)
check("cache_hit 缺失", mod.cache_hit("nope") is False)

# 旧 meta（§111 前）回退读 source 字段
legacy = json.load(open(mod.cache_paths("sam3drive")[1], encoding="utf-8"))
legacy.pop("source_key")
legacy["source"] = "oldhash"
mod.cache_store.write_meta(tmpdir, "sam3drive", legacy)
check("旧 meta source 字段兼容读", mod.cache_hit("sam3drive", "oldhash") is True
      and mod.cache_hit("sam3drive", "srcA") is False)
mod.save_track_cache("sam3drive", td, source_key="srcA")  # 恢复新格式

# packed=None（无对象）分支
td_empty = {"packed_masks": None, "n_frames": 3, "scores": [], "orig_size": (5, 6)}
mod.save_track_cache("emptyobj", td_empty, source_key="k")
le = mod.load_track_cache("emptyobj")
check("packed None 保留", le["packed_masks"] is None)
check("packed None n_frames", le["n_frames"] == 3)
check("packed None scores 空", le["scores"] == [])
check("packed None orig_size", le["orig_size"] == (5, 6))

# 非法输入
try:
    mod.save_track_cache("bad", {"not": "track"}, source_key="k")
    check("非法 track_data 报错", False)
except ValueError:
    check("非法 track_data 报错", True)

# ── 3. lazy 决策与 execute ──
mod.save_track_cache("hitname", td, source_key="key1")
node = SFTrackDataCache()
check("check_lazy 无 track_data→[]", node.check_lazy_status("hitname", source_key="key1", **{}) == [])
check("check_lazy 未命中→track_data",
      node.check_lazy_status("missing", source_key="key1", **{"track_data": None}) == ["track_data"])
check("check_lazy 命中→[]",
      node.check_lazy_status("hitname", source_key="key1", **{"track_data": None}) == [])
check("check_lazy 键不符→track_data",
      node.check_lazy_status("hitname", source_key="other", **{"track_data": None}) == ["track_data"])
check("check_lazy force→track_data",
      node.check_lazy_status("hitname", True, source_key="key1", **{"track_data": None}) == ["track_data"])
try:
    node.check_lazy_status("hitname", source_key="", **{"track_data": None})
    check("check_lazy 空 key 报错", False)
except ValueError:
    check("check_lazy 空 key 报错", True)

# name_text 文本覆盖：非空优先（下拉仅作回退）
check("_resolve_name 文本优先", mod._resolve_name("combo", "text") == "text")
check("_resolve_name 空白回退", mod._resolve_name("combo", "   ") == "combo")
check("_resolve_name 列表取首个非空", mod._resolve_name("combo", ["", "first", "second"]) == "first")
check("_resolve_name 非字符串回退", mod._resolve_name("combo", 123) == "combo")
check("_resolve_name 全空", mod._resolve_name("", "") == "")
mod.save_track_cache("textname", td, source_key="k")
check("check_lazy name_text 命中→[]",
      node.check_lazy_status("combo", name_text="textname", source_key="k",
                             **{"track_data": None}) == [])
check("check_lazy name_text 未命中→track_data",
      node.check_lazy_status("combo", name_text="missing", source_key="k",
                             **{"track_data": None}) == ["track_data"])
res_text = node.execute("combo", track_data=td, name_text="textwrite", source_key="k")
check("execute name_text 覆盖写入", "textwrite" in mod.list_cache_names() and res_text[0] is td)
res_read = node.execute("combo", name_text="textwrite", source_key="k")
check("execute name_text 覆盖读取", res_read[0]["packed_masks"].shape == (2, 2, 1, 4))
try:
    node.execute("", track_data=td, name_text="   ", source_key="k")
    check("execute 文本空白且下拉为空报错", False)
except ValueError:
    check("execute 文本空白且下拉为空报错", True)
try:
    node.execute("brandnew", track_data=td, source_key="")
    check("execute 空 key 报错", False)
except ValueError:
    check("execute 空 key 报错", True)

res = node.execute("brandnew", track_data=td, source_key="k")
check("execute 保存返回 dict", isinstance(res, tuple) and res[0]["packed_masks"].shape == (2, 2, 1, 4))
check("execute 保存已落盘", "brandnew" in mod.list_cache_names())
res2 = node.execute("brandnew", source_key="k")
check("execute 读取返回 dict", res2[0]["packed_masks"].shape == (2, 2, 1, 4))
try:
    node.execute("does_not_exist", source_key="k")
    check("execute 缓存缺失报错", False)
except RuntimeError:
    check("execute 缓存缺失报错", True)
try:
    node.execute("..", source_key="k")
    check("execute 非法名报错", False)
except ValueError:
    check("execute 非法名报错", True)

# 命中带 track_data 不重写 + 原子写（P0 写盘守卫）
res_new = node.execute("knew", track_data=td, source_key="path/w.mp4|cap=32")
check("execute 写 meta 用 source_key", mod.read_meta("knew")["source_key"] == "path/w.mp4|cap=32")
check("cache_hit 用 source_key",
      mod.cache_hit("knew", "path/w.mp4|cap=32") is True
      and mod.cache_hit("knew", "other") is False)
check("原子写：save_file 先写 .tmp", fake_sf.save_paths[-1].endswith(".tmp"))
check("原子写：临时文件无残留", not [p for p in os.listdir(tmpdir) if p.endswith(".tmp")])
check("原子写：最终文件存在", os.path.isfile(mod.cache_paths("knew")[0]))

_key_calls = len(fake_sf.save_paths)
res_key = node.execute("knew", track_data=td, source_key="path/w.mp4|cap=32")
check("execute 命中跳过重写（守卫）",
      res_key[0] is td and len(fake_sf.save_paths) == _key_calls)
node.execute("knew", True, track_data=td, source_key="path/w.mp4|cap=32")
check("execute force 仍强制重写", len(fake_sf.save_paths) == _key_calls + 1)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    sys.exit(1)
print("ALL PASS")
