"""SFSpreadsheetOutputList mock tests — 复刻 ComfyUI-outputlists-combiner SpreadsheetOutputList（仅文本表格）。

黄金用例值与上游 pandas 实现逐例对照过（容器内跑上游原码），除两处刻意加固外完全一致：
① base64 文本解码（上游会当 CSV 解析出空/垃圾）；② 行内 `#` 注释不剥离（仅行首，README 语义）。
"""
import base64
import importlib.util
import pathlib
import sys
import types

# stub comfy
pkg_comfy = types.ModuleType("comfy")
pkg_ct = types.ModuleType("comfy.comfy_types")
pkg_nt = types.ModuleType("comfy.comfy_types.node_typing")


class _IO:
    STRING = "STRING"
    INT = "INT"
    BOOLEAN = "BOOLEAN"


pkg_nt.IO = _IO
sys.modules["comfy"] = pkg_comfy
sys.modules["comfy.comfy_types"] = pkg_ct
sys.modules["comfy.comfy_types.node_typing"] = pkg_nt

# load sf_utils/spreadsheet (pure stdlib)
spec_u = importlib.util.spec_from_file_location("sf_utils_spreadsheet", "sf_utils/spreadsheet.py")
mod_u = importlib.util.module_from_spec(spec_u)
spec_u.loader.exec_module(mod_u)

# load node via exec (avoid relative import complexity)
src = pathlib.Path("nodes/text/spreadsheet_outputlist.py").read_text()
src = src.replace("from comfy.comfy_types.node_typing import IO", "")
src = src.replace("from ...sf_utils.spreadsheet import build_outputs, decode_separator, parse_table", "")
g = {
    "IO": _IO,
    "_IO": _IO,
    "build_outputs": mod_u.build_outputs,
    "decode_separator": mod_u.decode_separator,
    "parse_table": mod_u.parse_table,
}
exec(src, g)
SFSpreadsheetOutputList = g["SFSpreadsheetOutputList"]


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg}: {a!r} != {b!r}")


node = SFSpreadsheetOutputList()

CSV = "name,val\nalpha,1\nbeta,2"
MD = "| head_a | head_b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"


def run(data, selectors="", sep=",", topdown=True, headers=1, nth=-1):
    return node.execute(rows_and_cols=selectors, separator=sep, is_topdown=topdown,
                        num_headers=headers, select_nth=nth, string_or_base64=data)


# ── 1. 列名 ↔ 下标 ──
assert_eq(mod_u.column_to_index("A"), 0, "A")
assert_eq(mod_u.column_to_index("B"), 1, "B")
assert_eq(mod_u.column_to_index("Z"), 25, "Z")
assert_eq(mod_u.column_to_index("AA"), 26, "AA")
assert_eq(mod_u.column_to_index("AZ"), 51, "AZ")
assert_eq(mod_u.column_to_index("BA"), 52, "BA")
assert_eq(mod_u.column_to_index("ZZZZ"), None, "超 65536 列")
assert_eq(mod_u.column_to_index("a"), None, "小写非法")
assert_eq(mod_u.column_to_index(""), None, "空非法")
assert_eq(mod_u.column_to_index("ABCDE"), None, "超 4 位非法")
assert_eq(mod_u.column_to_name(0), "A", "0")
assert_eq(mod_u.column_to_name(25), "Z", "25")
assert_eq(mod_u.column_to_name(26), "AA", "26")
assert_eq(mod_u.column_to_name(701), "ZZ", "701")
assert_eq(mod_u.column_to_name(702), "AAA", "702")
for i in range(1000):
    assert_eq(mod_u.column_to_index(mod_u.column_to_name(i)), i, f"roundtrip {i}")

# ── 2. 分隔符转义解码 ──
assert_eq(mod_u.decode_separator(None), ",", "None 回退逗号")
assert_eq(mod_u.decode_separator("\\t"), "\t", "\\t")
assert_eq(mod_u.decode_separator("\\n"), "\n", "\\n")
assert_eq(mod_u.decode_separator("\\\\"), "\\", "\\\\")
assert_eq(mod_u.decode_separator("，"), "，", "中文分隔符保留")
assert_eq(mod_u.decode_separator("\\"), "\\", "尾随反斜杠回退")
assert_eq(mod_u.decode_separator(""), "", "空串")

# ── 3. parse_table ──
assert_eq(mod_u.parse_table("", ","), None, "空文本")
assert_eq(mod_u.parse_table("a,b", ""), None, "空分隔符")
assert_eq(mod_u.parse_table("a,b\n1,2", ","), [["a", "b"], ["1", "2"]], "基础")
assert_eq(mod_u.parse_table("a,b\n1", ","), None, "行宽不一致（pandas ParserError）")
assert_eq(mod_u.parse_table('a,"b,c"\n1,2', ","), [["a", "b,c"], ["1", "2"]], "引号内逗号")
assert_eq(mod_u.parse_table('a,"l1\nl2"', ","), [["a", "l1\nl2"]], "引号内换行")
assert_eq(mod_u.parse_table("# c\na,b\n\n  # d\n1,2", ","), [["a", "b"], ["1", "2"]], "注释/空行")
assert_eq(mod_u.parse_table("a::b\n1::2", "::"), [["a", "b"], ["1", "2"]], "多字符分隔符")
assert_eq(mod_u.parse_table("a,b # note\n1,2", ","), [["a", "b # note"], ["1", "2"]], "行内 # 不剥离（行首才注释）")
assert_eq(mod_u.parse_table("\ufeffa,b\n1,2", ","), [["a", "b"], ["1", "2"]], "BOM")

# ── 4. selectors 解析（转义/去空）──
assert_eq(mod_u.parse_selectors("", ","), [], "空")
assert_eq(mod_u.parse_selectors("  ", ","), [], "全空白")
assert_eq(mod_u.parse_selectors("a, b ,c", ","), ["a", "b", "c"], "strip")
assert_eq(mod_u.parse_selectors("a\\,b,c", ","), ["a,b", "c"], "转义逗号")
assert_eq(mod_u.parse_selectors("a;b", ";"), ["a", "b"], "分号")

# ── 5. top-down 黄金用例（与上游一致）──
out = run(CSV)
assert_eq(out[0], 2, "count")
assert_eq(out[1], [
    {"name": "alpha", "A": "alpha", "val": "1", "B": "1"},
    {"name": "beta", "A": "beta", "val": "2", "B": "2"},
], "values_dict 别名")
assert_eq(out[2], [["alpha", "1"], ["beta", "2"]], "values_list")
assert_eq(out[3], ["alpha", "beta"], "item_a")
assert_eq(out[4], ["1", "2"], "item_b")
assert_eq(out[5], [], "item_c")
assert_eq(out[6], [], "item_d")

# 选择器：表头名 / 列名 / 顺序
out = run(CSV, selectors="val")
assert_eq(out[3], ["1", "2"], "按表头名选")
assert_eq(out[4], [], "单选择器 item_b 空")
out = run(CSV, selectors="B,name")
assert_eq((out[3], out[4]), (["1", "2"], ["alpha", "beta"]), "选择器顺序保留")
out = run(CSV, selectors="A")
assert_eq(out[1], [{"A": "alpha", "name": "alpha"}, {"A": "beta", "name": "beta"}], "列名字典键")

# select_nth
out = run(CSV, nth=0)
assert_eq((out[0], out[3]), (1, ["alpha"]), "nth=0")
out = run(CSV, nth=5)
assert_eq(out, (0, [], [], [], [], [], []), "nth 越界全空")
out = run(CSV, nth=-1)
assert_eq(out[0], 2, "nth=-1 全部")

# num_headers
out = run(CSV, headers=0)
assert_eq(out[3], ["name", "alpha", "beta"], "num_headers=0 含表头行")
out = run(CSV, headers=50)
assert_eq(out, (0, [], [], [], [], [], []), "num_headers 超行数")

# 未知选择器
out = run(CSV, selectors="zz")
assert_eq(out, (0, [], [], [], [], [], []), "未知选择器全空")

# ── 6. left-to-right 黄金用例 ──
out = run(CSV, topdown=False)
assert_eq(out[0], 1, "ltr count=1（只有 val 列）")
assert_eq(out[1], [{"1": "val", "name": "val", "2": "1", "alpha": "1", "3": "2", "beta": "2"}], "ltr 行头别名")
assert_eq((out[3], out[4], out[5]), (["val"], ["1"], ["2"]), "ltr item_a..c")
out = run(CSV, topdown=False, selectors="alpha")
assert_eq((out[0], out[3]), (1, ["1"]), "ltr 按行头选 alpha")
out = run(CSV, topdown=False, selectors="val")
assert_eq(out, (0, [], [], [], [], [], []), "ltr 列名不是行选择器")
out = run(CSV, topdown=False, headers=0)
assert_eq(out[3], ["name", "val"], "ltr num_headers=0")

# ── 7. Markdown 表（含 --- 分隔行）──
out = run(MD, sep="|")
assert_eq(out[0], 3, "md count=3（--- 行是数据行）")
assert_eq(out[3], ["", "", ""], "md A 列空")
assert_eq(out[4], ["---", " 1 ", " 3 "], "md head_a 列")
assert_eq(out[5], ["---", " 2 ", " 4 "], "md head_b 列")
assert_eq(out[6], ["", "", ""], "md D 列空")
out = run(MD, selectors="head_b", sep="|")
assert_eq(out[3], ["---", " 2 ", " 4 "], "md 按表头选")
out = run("| h1 | h2 |\n|:---|---:|\n| 1 | 2 |", sep="|")
assert_eq(out[0], 2, "md 对齐行头跳过 + 作为数据行")
assert_eq(out[4], [":---", " 1 "], "md 对齐行数据")

# ── 8. TSV（转义分隔符）──
out = run("h1\th2\n1\t2", sep="\\t")
assert_eq(out[0], 1, "tsv count")
assert_eq((out[3], out[4]), (["1"], ["2"]), "tsv 值")

# ── 9. 多字符分隔符 / 引号换行 ──
out = run("a::b::c\n1::2::3", sep="::", headers=0)
assert_eq(out[0], 2, "多字符分隔符")
out = run('name,"note"\nalpha,"l1\nl2"\nbeta,x')
assert_eq(out[4], ["l1\nl2", "x"], "引号内换行值")

# ── 10. base64 与 Office 二进制守卫 ──
b64 = base64.b64encode(CSV.encode()).decode()
out = run(b64)
assert_eq(out[0], 2, "base64 文本解码")
assert_eq(out[3], ["alpha", "beta"], "base64 值")
for payload in (b"PK\x03\x04" + b"x" * 20, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"x" * 20):
    try:
        run(base64.b64encode(payload).decode())
        raise AssertionError("Office 二进制应抛 ValueError")
    except ValueError as e:
        assert "不支持 Excel/ODS" in str(e), "错误信息"

# ── 11. 空输入 / 空表 ──
assert_eq(run(""), (0, [], [], [], [], [], []), "空输入")
assert_eq(run("   \n  "), (0, [], [], [], [], [], []), "全空白输入")
assert_eq(run("name,val"), (0, [], [], [], [], [], []), "只有表头无数据")

# ── 12. 契约 ──
assert_eq(SFSpreadsheetOutputList.RETURN_TYPES, ("INT", "DICT", "ARRAY", "STRING", "STRING", "STRING", "STRING"), "RETURN_TYPES")
assert_eq(SFSpreadsheetOutputList.RETURN_NAMES, ("count", "values_dict", "values_list", "item_a", "item_b", "item_c", "item_d"), "RETURN_NAMES")
assert_eq(SFSpreadsheetOutputList.OUTPUT_IS_LIST, (False, True, True, True, True, True, True), "OUTPUT_IS_LIST")
assert_eq(SFSpreadsheetOutputList.FUNCTION, "execute", "FUNCTION")
assert_eq(SFSpreadsheetOutputList.CATEGORY, "sfnodes/text", "CATEGORY")
assert SFSpreadsheetOutputList.DESCRIPTION, "DESCRIPTION 必填"
inputs = SFSpreadsheetOutputList.INPUT_TYPES()["required"]
assert_eq(list(inputs), ["rows_and_cols", "separator", "is_topdown", "num_headers", "select_nth", "string_or_base64"], "输入键（对齐上游）")

print("test_spreadsheet_outputlist: all assertions passed")
