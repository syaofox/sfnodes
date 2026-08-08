# SFTextFindReplace 后端逻辑测试（Python 直接运行：python tests/test_find_replace.py）
# 覆盖：INPUT_TYPES 结构、apply() 返回值（ui 预览 + result）、_apply_rules 的
# literal/whole-word/regex/tidy/ReDoS 防护/非法正则警告、畸形状态容错、预览截断
import importlib.util
import json
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.find_replace",
    os.path.join(root, "nodes", "text", "find_replace.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

node = mod.SFTextFindReplace()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("OUTPUT_NODE", node.OUTPUT_NODE is True)

it = node.INPUT_TYPES()
check("required 含 text", "text" in it["required"])
check("text forceInput", it["required"]["text"][1].get("forceInput") is True)
check("INPUT_TYPES 含 hidden FindReplaceState", "FindReplaceState" in it["hidden"])
check("返回类型 text", node.RETURN_TYPES == ("STRING",) and node.RETURN_NAMES == ("text",))
check("FUNCTION = apply", node.FUNCTION == "apply")

# ---- 无规则：透传（tidy 默认开） ----
r = node.apply(text="hello  world")
check("无规则透传", r["result"] == ("hello world",))  # tidy 折叠双空格

# ---- literal 替换 ----
state = json.dumps({"rules": [{"enabled": True, "find": "cat", "replace": "dog"}], "tidy": False})
r = node.apply(text="a cat", FindReplaceState=state)
check("literal 替换", r["result"] == ("a dog",))

# ---- 空 replace = 删除 ----
state = json.dumps({"rules": [{"enabled": True, "find": "bad", "replace": ""}], "tidy": True})
r = node.apply(text="good bad words", FindReplaceState=state)
check("空 replace 删除", r["result"] == ("good words",))

# ---- 大小写：默认忽略，caseSensitive 精确 ----
state = json.dumps({"rules": [{"enabled": True, "find": "hello", "replace": "hi"}], "tidy": False})
r = node.apply(text="Hello world", FindReplaceState=state)
check("默认忽略大小写", r["result"] == ("hi world",))
state = json.dumps({"rules": [{"enabled": True, "find": "hello", "replace": "hi"}], "caseSensitive": True, "tidy": False})
r = node.apply(text="Hello world", FindReplaceState=state)
check("caseSensitive 不命中", r["result"] == ("Hello world",))

# ---- whole word ----
state = json.dumps({"rules": [{"enabled": True, "find": "art", "replace": "X"}], "wholeWord": True, "tidy": False})
r = node.apply(text="art artist heart", FindReplaceState=state)
check("whole word 只命中整词", r["result"] == ("X artist heart",))
state = json.dumps({"rules": [{"enabled": True, "find": "art", "replace": "X"}], "wholeWord": False, "tidy": False})
r = node.apply(text="art artist", FindReplaceState=state)
check("非 whole word 命中子串", r["result"] == ("X Xist",))

# ---- regex 模式 ----
state = json.dumps({"rules": [{"enabled": True, "find": r"\d+", "replace": "N"}], "regex": True, "tidy": False})
r = node.apply(text="a 3 b 42", FindReplaceState=state)
check("regex \\d+ 替换", r["result"] == ("a N b N",))
state = json.dumps({"rules": [{"enabled": True, "find": r"(\w+) (\w+)", "replace": r"\2 \1"}], "regex": True, "tidy": False})
r = node.apply(text="hello world", FindReplaceState=state)
check("regex 反向引用 \\2 \\1", r["result"] == ("world hello",))

# ---- literal 模式下反斜杠不当反向引用 ----
state = json.dumps({"rules": [{"enabled": True, "find": "x", "replace": r"\1"}], "tidy": False})
r = node.apply(text="x", FindReplaceState=state)
check("literal 模式 \\1 为字面文本", r["result"] == ("\\1",))

# ---- 多条规则按序应用 ----
state = json.dumps({"rules": [
    {"enabled": True, "find": "a", "replace": "b"},
    {"enabled": True, "find": "b", "replace": "c"},
], "tidy": False})
r = node.apply(text="aaa", FindReplaceState=state)
check("规则按序应用", r["result"] == ("ccc",))

# ---- 禁用的规则跳过 ----
state = json.dumps({"rules": [
    {"enabled": False, "find": "a", "replace": "X"},
    {"enabled": True, "find": "a", "replace": "Y"},
], "tidy": False})
r = node.apply(text="aaa", FindReplaceState=state)
check("禁用规则跳过", r["result"] == ("YYY",))

# ---- ReDoS 防护：嵌套无界量词警告 + 跳过 ----
state = json.dumps({"rules": [{"enabled": True, "find": r"(a+)+", "replace": "X"}], "regex": True, "tidy": False})
r = node.apply(text="aaaa", FindReplaceState=state)
check("嵌套量词 (a+)+ 跳过", r["result"] == ("aaaa",))
check("嵌套量词警告", any("catastrophically slow" in w for w in r["ui"]["sf_find_replace"][0]["warnings"]))
state = json.dumps({"rules": [{"enabled": True, "find": r"(a*)*", "replace": "X"}], "regex": True, "tidy": False})
r = node.apply(text="aaaa", FindReplaceState=state)
check("嵌套量词 (a*)* 跳过", r["result"] == ("aaaa",))
state = json.dumps({"rules": [{"enabled": True, "find": r"(a+){2}b", "replace": "X"}], "regex": True, "tidy": False})
r = node.apply(text="aab", FindReplaceState=state)
check("有界量词 (a+){2}b 命中（无嵌套）", r["result"] == ("X",))
check("无嵌套量词无警告", r["ui"]["sf_find_replace"][0]["warnings"] == [])
state = json.dumps({"rules": [{"enabled": True, "find": r"[()]+", "replace": "X"}], "regex": True, "tidy": False})
r = node.apply(text="(((", FindReplaceState=state)
check("字符类内括号/量词不误报", r["result"] == ("X",))

# ---- 非法正则警告 ----
state = json.dumps({"rules": [{"enabled": True, "find": "(", "replace": "X"}], "regex": True, "tidy": False})
r = node.apply(text="abc", FindReplaceState=state)
check("非法正则保留原文", r["result"] == ("abc",))
check("非法正则警告", any("invalid regex" in w for w in r["ui"]["sf_find_replace"][0]["warnings"]))

# ---- tidy ----
state = json.dumps({"rules": [{"enabled": True, "find": "x", "replace": ""}], "tidy": True})
r = node.apply(text="a  x , ,  b,", FindReplaceState=state)
check("tidy 折叠空格与逗号", r["result"] == ("a, b",))
state = json.dumps({"rules": [{"enabled": True, "find": "x", "replace": ""}], "tidy": False})
r = node.apply(text="a  x b", FindReplaceState=state)
check("tidy 关闭保留原样", r["result"] == ("a   b",))  # 删除 x 留下的三空格不动
check("tidy 默认开", node.apply(text="a  b")["result"] == ("a b",))

# ---- 非字符串输入转字符串 ----
check("text 数字转字符串", node.apply(text=123)["result"] == ("123",))
check("text None 转空串", node.apply(text=None)["result"] == ("",))
r = node.apply(text=["a", "b"], FindReplaceState=json.dumps({"tidy": False}))
check("text 列表转字符串", r["result"] == ("['a', 'b']",))

# ---- 畸形状态容错 ----
check("非法 JSON 容错", node.apply(text="t", FindReplaceState="not json{{{")["result"] == ("t",))
check("非对象 JSON 容错", node.apply(text="t", FindReplaceState="[1,2]")["result"] == ("t",))
check("None state 容错", node.apply(text="t", FindReplaceState=None)["result"] == ("t",))
check("数字 state 容错", node.apply(text="t", FindReplaceState=123)["result"] == ("t",))
state = json.dumps({"rules": [{"enabled": True, "find": 123, "replace": "X"}], "tidy": False})
r = node.apply(text="abc", FindReplaceState=state)
check("find 非字符串忽略", r["result"] == ("abc",) and r["ui"]["sf_find_replace"][0]["warnings"] == [])
state = json.dumps({"rules": [{"enabled": True, "find": "a", "replace": ["x"]}], "tidy": False})
check("replace 非字符串强转空", node.apply(text="aaa", FindReplaceState=state)["result"] == ("",))
state = json.dumps({"rules": [{"enabled": True, "find": "a", "replace": "b"}], "tidy": False})
check("rules 非列表忽略", node.apply(text="aa", FindReplaceState='{"rules": "junk"}')["result"] == ("aa",))

# ---- 预览 ui 形状 + 截断 ----
r = node.apply(text="cat", FindReplaceState=state)
ui = r["ui"]["sf_find_replace"][0]
check("ui 键 sf_find_replace", "sf_find_replace" in r["ui"])
check("ui input/output 样本", ui["input"] == "cat" and ui["output"] == "cbt" and ui["truncated"] is False)
long = "x" * 5000
r = node.apply(text=long, FindReplaceState='{"tidy": false}')
ui = r["ui"]["sf_find_replace"][0]
check("预览样本截断 4000", len(ui["input"]) == 4000 and ui["truncated"] is True)
check("实际输出全长", len(r["result"][0]) == 5000)
r = node.apply(text="y" * 5000, FindReplaceState=json.dumps({"rules": [{"enabled": True, "find": "y", "replace": "yy"}], "tidy": False}))
ui = r["ui"]["sf_find_replace"][0]
check("输出侧截断", ui["truncated"] is True)

# ---- 中文/Unicode ----
state = json.dumps({"rules": [{"enabled": True, "find": "水彩", "replace": "油画"}], "tidy": False})
r = node.apply(text="画水彩画", FindReplaceState=state)
check("中文 literal 替换", r["result"] == ("画油画画",))
state = json.dumps({"rules": [{"enabled": True, "find": "k", "replace": "K"}], "tidy": False})
r = node.apply(text="Kelvin", FindReplaceState=state)
check("Unicode 大小写折叠（Kelvin 符号）", r["result"] == ("Kelvin",))  # \u212a 折叠为 k

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
