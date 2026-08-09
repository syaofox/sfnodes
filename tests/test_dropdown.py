# SFValueDropdown 后端逻辑测试（Python 直接运行：python tests/test_dropdown.py）
# 覆盖：
#   - 结构：ValueDropdown 类、CATEGORY、DESCRIPTION、INPUT_TYPES（hidden
#     DropdownState）、RETURN_TYPES（ANY）、FUNCTION、注册键
#   - 数字语法 _as_number（JS/Python 双端契约：拒 0x10/1_0/Infinity，收 5./.5/1e3）
#   - readable（text 恒真；bool 单词表；数值钳制 -1e12..1e12 之外不可读）
#   - coerce_value（text 拼写 / bool 单词表 / int half-away / float / 钳制 /
#     fallback 永不抛）
#   - parse_state 畸形容错（非 dict 行丢弃、index 归一）
#   - selected_value 双形状（lean 优先 + full 回退）、空列表/越界 fallback
import importlib.util
import json
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils.common import AnyType）──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.text"); pkg3.__path__ = [os.path.join(root, "nodes", "text")]; sys.modules["sfnodes.nodes.text"] = pkg3

# 纯逻辑模块（无相对导入，直接加载）
spec_utils = importlib.util.spec_from_file_location(
    "sf_utils_dropdown",
    os.path.join(root, "sf_utils", "dropdown.py"),
)
utils = importlib.util.module_from_spec(spec_utils)
sys.modules[spec_utils.name] = utils
spec_utils.loader.exec_module(utils)

# 节点类（含相对导入）
spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.dropdown_value",
    os.path.join(root, "nodes", "text", "dropdown_value.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# ── 结构 ──
node = mod.ValueDropdown()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
it = node.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("hidden DropdownState", it["hidden"]["DropdownState"][1]["default"] == "{}")
check("RETURN_TYPES 为 ANY", node.RETURN_TYPES == ("*",) and node.RETURN_NAMES == ("value",))
check("FUNCTION = run", node.FUNCTION == "run")
check("注册键", mod.NODE_CLASS_MAPPINGS == {"SFValueDropdown": mod.ValueDropdown})
check("显示名键", mod.NODE_DISPLAY_NAME_MAPPINGS == {"SFValueDropdown": "SF Value Dropdown"})

# ── 数字语法 ──
num = utils._as_number
check("as_number 5", num("5") == 5.0)
check("as_number 5.", num("5.") == 5.0)
check("as_number .5", num(".5") == 0.5)
check("as_number 5.5", num("5.5") == 5.5)
check("as_number +5", num("+5") == 5.0)
check("as_number -3", num("-3") == -3.0)
check("as_number 1e3", num("1e3") == 1000.0)
check("as_number 1E3", num("1E3") == 1000.0)
check("as_number -1e3", num("-1e3") == -1000.0)
check("as_number 拒 0x10", num("0x10") is None)
check("as_number 拒 0b1", num("0b1") is None)
check("as_number 拒 1_0", num("1_0") is None)
check("as_number 拒 1,024", num("1,024") is None)
check("as_number 拒 1024px", num("1024px") is None)
check("as_number 拒 abc", num("abc") is None)
check("as_number 拒 Infinity", num("Infinity") is None)
check("as_number 拒 NaN", num("NaN") is None)
check("as_number 拒 空串", num("") is None)
check("as_number 拒 全空格", num("   ") is None)
check("as_number BOM 前缀数字", num("\ufeff5") == 5.0)
check("as_number bool True", num(True) == 1.0)
check("as_number int 大数", num(10**20) == 1e20)
check("as_number 拒非有限 float", num(float("inf")) is None)
check("as_number 拒其他类型", num(None) is None and num([1]) is None)

# ── readable ──
r = utils.readable
check("readable text 恒真", r("anything", "text") is True and r(123, "text") is True and r(None, "text") is True)
check("readable int 好", r("1024", "int") is True)
check("readable int 坏", r("abc", "int") is False)
check("readable int 超钳制", r("1e308", "int") is False)
check("readable int 负超钳制", r("-1e20", "int") is False)
check("readable float 好", r("0.35", "float") is True)
check("readable bool 单词", r("YES", "bool") is True and r("off", "bool") is True)
check("readable bool 数字", r("0", "bool") is True and r("3", "bool") is True)
check("readable bool 坏", r("maybe", "bool") is False)
check("readable bool 超钳制仍读", r("1e308", "bool") is True)
check("readable 未知类型回退 text", r("x", "bogus") is True)
check("readable 别名类型", r("1024", "integer") is True and r("true", "boolean") is True)

# ── coerce_value ──
cv = utils.coerce_value
check("coerce text 原样", cv("hi", "text") == "hi")
check("coerce text None -> 空", cv(None, "text") == "")
check("coerce text bool -> 拼写", cv(True, "text") == "true" and cv(False, "text") == "false")
check("coerce text 整 float 去 .0", cv(2.0, "text") == "2")
check("coerce text 非整 float", cv(2.5, "text") == "2.5")
check("coerce int 取整", cv("2.5", "int") == 3 and cv("-2.5", "int") == -3)
check("coerce int 钳制", cv("1e308", "int") == 10**12)
check("coerce int 坏 -> 0", cv("abc", "int") == 0)
check("coerce int 大整数字符串", cv("400" + "0" * 300, "int") == 10**12)
check("coerce float 好", cv("0.35", "float") == 0.35)
check("coerce float 坏 -> 0.0", cv("abc", "float") == 0.0)
check("coerce float 钳制", cv("1e20", "float") == 1e12)
check("coerce bool 单词", cv("yes", "bool") is True and cv("No", "bool") is False)
check("coerce bool 数字", cv("0", "bool") is False and cv("3", "bool") is True)
check("coerce bool 坏 -> False", cv("maybe", "bool") is False)
check("coerce bool 真值保持", cv(True, "bool") is True)

# ── parse_state 畸形容错 ──
ps = utils.parse_state
check("parse_state 空", ps(None) == {"type": "text", "index": 0, "options": []})
check("parse_state 非 JSON", ps("{{{") == {"type": "text", "index": 0, "options": []})
check("parse_state 非 dict JSON", ps("[1,2]") == {"type": "text", "index": 0, "options": []})
st = ps(json.dumps({
    "type": "float",
    "index": 1.9,
    "options": [
        {"name": "a", "value": "1"},
        "junk",
        None,
        {"name": 5, "value": "2"},
        {"name": "b"},
    ],
}))
check("parse_state 归一", st["type"] == "float" and st["index"] == 1 and len(st["options"]) == 3)
check("parse_state 非 dict 行丢弃", all(isinstance(o, dict) for o in st["options"]))
check("parse_state 坏名字 -> 空串", st["options"][1]["name"] == "")
check("parse_state 缺失 value -> None", st["options"][2]["value"] is None)
check("parse_state index 越界不裁剪", utils.parse_state('{"index": 99, "options": []}')["index"] == 99)

# ── selected_value（lean/full 双形状）──
sv = utils.selected_value
check("lean text", sv('{"type": "text", "value": "warm"}') == "warm")
check("lean int", sv('{"type": "int", "value": "42"}') == 42)
check("lean float", sv('{"type": "float", "value": "0.5"}') == 0.5)
check("lean bool", sv('{"type": "bool", "value": "yes"}') is True)
check("lean 空字符串值保持", sv('{"type": "text", "value": ""}') == "")
check("lean 数字 0 值保持", sv('{"type": "int", "value": "0"}') == 0)
check("full 形状", sv(json.dumps({
    "type": "text", "index": 1,
    "options": [{"name": "a", "value": "one"}, {"name": "b", "value": "two"}],
})) == "two")
check("full 空列表 fallback", sv('{"type": "int", "index": 0, "options": []}') == 0)
check("full 越界 fallback", sv('{"type": "text", "index": 5, "options": [{"name": "a", "value": "x"}]}') == "")
check("full 类型归一回退 text", sv('{"type": "bogus", "index": 0, "options": [{"name": "a", "value": "x"}]}') == "x")
check("selected_value 非 dict -> fallback", sv("junk") == "")
check("selected_value dict 直接接受", sv({"type": "bool", "value": "n"}) is False)
check("selected_value 深嵌套 JSON 不炸", isinstance(sv('{"type": "text", "value": ' + "[" * 2000 + "]" * 2000 + "}"), str))
check("parse_state 超深 JSON 不炸", ps("[" * 100000 + "]" * 100000) == {"type": "text", "index": 0, "options": []})
check("selected_value 大整数字符串钳制", sv('{"type": "float", "value": "9' + "9" * 300 + '"}') == 1e12)

if failures:
    print(f"\n{len(failures)} FAILED")
    sys.exit(1)
print("\nALL PASS")
