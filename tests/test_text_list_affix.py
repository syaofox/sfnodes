"""SFTextListAffix mock tests."""
import importlib.util
import sys
import types

# stub comfy
pkg_comfy = types.ModuleType("comfy")
pkg_ct = types.ModuleType("comfy.comfy_types")
pkg_nt = types.ModuleType("comfy.comfy_types.node_typing")
class _IO:
    STRING = "STRING"
pkg_nt.IO = _IO
sys.modules["comfy"] = pkg_comfy
sys.modules["comfy.comfy_types"] = pkg_ct
sys.modules["comfy.comfy_types.node_typing"] = pkg_nt

# load sf_utils/string
spec_u = importlib.util.spec_from_file_location("sf_utils_string", "sf_utils/string.py")
mod_u = importlib.util.module_from_spec(spec_u)
spec_u.loader.exec_module(mod_u)
affix_list = mod_u.affix_list
sys.modules["sf_utils.string"] = mod_u
for n in ['sf_utils','nodes','nodes.text']:
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)

# load node via exec (avoid relative import complexity)
import pathlib
src = pathlib.Path("nodes/text/text_list_affix.py").read_text().replace("from ...sf_utils.string import affix_list","").replace("from comfy.comfy_types.node_typing import IO","")
g={'IO':_IO,'affix_list':affix_list}
exec(src, g)
SFTextListAffix = g['SFTextListAffix']

def assert_eq(a,b,msg=""):
    if a!=b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")

node = SFTextListAffix()

# ---- pure affix_list tests ----
assert_eq(affix_list(["a","b","c"], "pre-", "-suf", False), ["pre-a-suf","pre-b-suf","pre-c-suf"], "basic affix")
assert_eq(affix_list(["a","", "b"], "P", "S", True), ["PaS","PbS"], "filter_empty True skips empty")
assert_eq(affix_list(["a","", "b"], "P", "S", False), ["PaS","PS","PbS"], "filter_empty False keeps")
assert_eq(affix_list(["a"," b ",""], "(", ")", True), ["(a)", "( b )"], "strip filter")
assert_eq(affix_list(None, "a","b", True), [], "None input")
assert_eq(affix_list([], "a","b", True), [], "empty list")
# escape
assert_eq(affix_list(["x"], "\\n", "\\t", False), ["\nx\t"], "escape n/t")
assert_eq(affix_list(["x"], "a\\nb", "c\\td", False), ["a\nbxc\td"], "escape in middle")
# whitespace only with filter
assert_eq(affix_list(["   ","x"], "p","s", True), ["pxs"], "whitespace filtered")

# ---- node execute with INPUT_IS_LIST handling ----
# Simulate ComfyUI INPUT_IS_LIST wrapping: prepend etc arrive as [value]
# text_list direct (simulated non-wrapped call) – node should unwrap
# When INPUT_IS_LIST True, text_list is list, prepend is [str], etc.
# We call execute with wrapped forms as ComfyUI would
res, = node.execute(text_list=["a","b","c"], prepend_text=["pre-"], append_text=["-suf"], filter_empty=[False])
assert_eq(res, ["pre-a-suf","pre-b-suf","pre-c-suf"], "node unwrapped prepend/append")

# filter_empty handling via node
res, = node.execute(text_list=["a","","b"], prepend_text=[""], append_text=[""], filter_empty=[True])
assert_eq(res, ["a","b"], "node filter_empty True")

res, = node.execute(text_list=["a","","b"], prepend_text=[""], append_text=[""], filter_empty=[False])
assert_eq(res, ["a","","b"], "node filter_empty False")

# escape via node
res, = node.execute(text_list=["x"], prepend_text=["\\n"], append_text=["\\t"], filter_empty=[False])
assert_eq(res, ["\nx\t"], "node escape")

# None / empty handling
res, = node.execute(text_list=None, prepend_text=["p"], append_text=["s"], filter_empty=[True])
assert_eq(res, [], "node None -> []")
res, = node.execute(text_list=[], prepend_text=["p"], append_text=["s"], filter_empty=[True])
assert_eq(res, [], "node []")

# single non-list string (if upstream not list) – INPUT_IS_LIST still wraps as ["single"]
res, = node.execute(text_list="single", prepend_text=["["], append_text=["]"], filter_empty=[False])
assert_eq(res, ["[single]"], "single string")

# check metadata
assert_eq(SFTextListAffix.INPUT_IS_LIST, True)
assert_eq(SFTextListAffix.OUTPUT_IS_LIST, (True,))
assert_eq(SFTextListAffix.RETURN_TYPES, ("STRING",))
assert_eq(SFTextListAffix.RETURN_NAMES, ("list",))
assert SFTextListAffix.CATEGORY == "sfnodes/text"
assert SFTextListAffix.DESCRIPTION

print("test_text_list_affix: 18 assertions passed")
