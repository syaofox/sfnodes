# SFTextPreset 后端逻辑测试（Node/Python 直接运行：python tests/test_text_preset.py）
# 覆盖：INPUT_TYPES 结构、execute 的预设查找/容错
import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

comfy = types.ModuleType("comfy")
node_typing = types.ModuleType("comfy.comfy_types")
node_typing_module = types.ModuleType("comfy.comfy_types.node_typing")
class IO:
    STRING = "STRING"
node_typing_module.IO = IO
comfy.comfy_types = node_typing
comfy.comfy_types.node_typing = node_typing_module
sys.modules["comfy"] = comfy
sys.modules["comfy.comfy_types"] = node_typing
sys.modules["comfy.comfy_types.node_typing"] = node_typing_module

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.text_preset",
    os.path.join(root, "nodes", "text", "text_preset.py"),
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

node = mod.SFTextPreset()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)

it = node.INPUT_TYPES()
required = it["required"]
check("INPUT_TYPES 含 preset", "preset" in required)
check("preset 初始选项为空占位", required["preset"][0] == [""])
check("INPUT_TYPES 含 presets_json", "presets_json" in required)
check("presets_json 隐藏", required["presets_json"][1].get("display") == "hidden")
check("presets_json 默认空数组", required["presets_json"][1].get("default") == "[]")
check("返回类型 text+preset_name", node.RETURN_TYPES == ("STRING", "STRING") and node.RETURN_NAMES == ("text", "preset_name"))
check("VALIDATE_INPUTS 跳过 combo 校验", node.VALIDATE_INPUTS(preset="a", presets_json="[]") is True)

json_data = '[{"name": "A", "text": "hello"}, {"name": "B", "text": "world"}]'
t, n = node.execute("B", json_data)
check("按名命中返回文本", t == "world" and n == "B")
t, n = node.execute("A", json_data)
check("命中第一个预设", t == "hello" and n == "A")
t, n = node.execute("Nope", json_data)
check("未命中返回空文本", t == "" and n == "Nope")
t, n = node.execute("", json_data)
check("空选择返回空文本", t == "" and n == "")
t, n = node.execute("B", "")
check("空 presets_json 容错", t == "" and n == "B")
t, n = node.execute("B", None)
check("None presets_json 容错", t == "" and n == "B")
t, n = node.execute("B", "not json{{{")
check("非法 JSON 容错", t == "" and n == "B")
t, n = node.execute("B", '{"not": "list"}')
check("非数组 JSON 容错", t == "" and n == "B")
t, n = node.execute("B", '[{"name": 123, "text": 456}]')
check("name 非字符串时按 str 比较", t == "" and n == "B")
t, n = node.execute(123, '[{"name": "123", "text": "num"}]')
check("preset 非字符串输入", t == "num" and n == "123")
t, n = node.execute("B", '[{"name": "B", "text": 456}]')
check("text 非字符串转字符串", t == "456" and n == "B")
t, n = node.execute("B", '[{"name": "B"}]')
check("缺 text 字段输出空", t == "" and n == "B")

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
