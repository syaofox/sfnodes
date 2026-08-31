"""SFLongTextToList mock tests — 复刻 ComfyUI_Lam LongTextToList。"""
import importlib.util
import sys
import types

# stub comfy
pkg_comfy = types.ModuleType("comfy")
pkg_ct = types.ModuleType("comfy.comfy_types")
pkg_nt = types.ModuleType("comfy.comfy_types.node_typing")
class _IO:
    STRING = "STRING"
    INT = "INT"
pkg_nt.IO = _IO
sys.modules["comfy"] = pkg_comfy
sys.modules["comfy.comfy_types"] = pkg_ct
sys.modules["comfy.comfy_types.node_typing"] = pkg_nt

# load sf_utils/string
spec_u = importlib.util.spec_from_file_location("sf_utils_string", "sf_utils/string.py")
mod_u = importlib.util.module_from_spec(spec_u)
spec_u.loader.exec_module(mod_u)
split_text = mod_u.split_text

# load node
spec = importlib.util.spec_from_file_location("long_text_to_list", "nodes/text/long_text_to_list.py")
mod = importlib.util.module_from_spec(spec)
# inject sf_utils.string for relative import
sys.modules["sf_utils.string"] = mod_u
# ensure package parents exist for relative import resolution
# mock ...sf_utils.string via direct load — node imports via ...sf_utils.string, so put in sys.modules under expected name
# Create synthetic parent packages if needed
import os
# simpler: load via exec without relative — patch file read
# fallback: exec source with split_text already available
import pathlib
src = pathlib.Path("nodes/text/long_text_to_list.py").read_text()
# replace relative import with direct binding
src = src.replace("from ...sf_utils.string import split_text", "")
src = src.replace("from comfy.comfy_types.node_typing import IO", "")
exec_globals = {"__name__": "long_text_to_list", "split_text": split_text, "IO": _IO, "_IO": _IO}
exec(src, exec_globals)
SFLongTextToList = exec_globals["SFLongTextToList"]

def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")

node = SFLongTextToList()

# 1. 换行分隔（默认 \\n 转 \n）
picked, lst, count = node.execute("a\nb\nc", "\\n", 0)
assert_eq(picked, "a", "nl 0")
assert_eq(lst, ["a", "b", "c"], "nl list")
assert_eq(count, 3, "nl count")

picked, lst, count = node.execute("a\nb\nc", "\\n", 1)
assert_eq(picked, "b", "nl 1")

picked, lst, count = node.execute("a\nb\nc", "\n", 2)
assert_eq(picked, "c", "real nl 2")

# 2. 逗号
picked, lst, count = node.execute("x,y,z", ",", 1)
assert_eq(picked, "y", "comma")
assert_eq(count, 3)

# 3. 空分隔符 -> 单元素
picked, lst, count = node.execute("hello", "", 0)
assert_eq(lst, ["hello"], "empty delim list")
assert_eq(count, 1)
assert_eq(picked, "hello")

picked, lst, count = node.execute("", "", 0)
assert_eq(lst, [], "empty both list")
assert_eq(count, 0)
assert_eq(picked, "", "empty both picked")

# 4. 越界
picked, lst, count = node.execute("a,b", ",", 5)
assert_eq(picked, "", "oob picked")
assert_eq(count, 2)

picked, lst, count = node.execute("a,b", ",", 0)
assert_eq(picked, "a", "oob sanity")

# 5. \\t 制表
picked, lst, count = node.execute("a\tb\tc", "\\t", 1)
assert_eq(picked, "b", "tab")

# 6. None delimiter -> 同空分隔符
picked, lst, count = node.execute("hello world", None, 0)
assert_eq(lst, ["hello world"])

# 7. 纯 split_text 直接
assert_eq(split_text("a,b,c", ","), ["a", "b", "c"])
assert_eq(split_text("a\nb\nc", "\\n"), ["a", "b", "c"])
assert_eq(split_text("a\nb\nc", "\n"), ["a", "b", "c"])
assert_eq(split_text("", ","), [""])
assert_eq(split_text(None, ","), [""])

# 8. OUTPUT_IS_LIST 契约
assert_eq(SFLongTextToList.OUTPUT_IS_LIST, (False, True, False), "output_is_list")
assert_eq(SFLongTextToList.RETURN_TYPES, ("STRING", "STRING", "INT"))
assert_eq(SFLongTextToList.RETURN_NAMES, ("text_at_i", "list", "count"))

# 9. CATEGORY / DESCRIPTION
assert_eq(SFLongTextToList.CATEGORY, "sfnodes/text")
assert SFLongTextToList.DESCRIPTION

print("test_long_text_to_list: 21 assertions passed")
