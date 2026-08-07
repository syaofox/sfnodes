# SFPauseText 后端逻辑测试（Node/Python 直接运行：python tests/test_pause_text.py）
# 覆盖：INPUT_TYPES 结构、run() 三模式（continue/pause/pass × 有线/无线）与容错
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.pause_text",
    os.path.join(root, "nodes", "text", "pause_text.py"),
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

node = mod.SFPauseText()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
check("OUTPUT_NODE", node.OUTPUT_NODE is True)

it = node.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("INPUT_TYPES 含 text", "text" in it["optional"])
check("text forceInput", it["optional"]["text"][1].get("forceInput") is True)
check("INPUT_TYPES 含 hidden PauseState", "PauseState" in it["hidden"])
check("返回类型 text", node.RETURN_TYPES == ("STRING",) and node.RETURN_NAMES == ("text",))
check("FUNCTION = run", node.FUNCTION == "run")

# continue：有线被剪（text=None），输出编辑文本
r = node.run(text=None, PauseState='{"mode": "continue", "text": "edited words"}')
check("continue 输出编辑文本", r == {"result": ("edited words",)})

# continue：模式缺省（空 state）回退 pause；无线 -> 盒子文本
r = node.run(text=None, PauseState="")
check("空 state 无线输出盒子空文本", r == {"result": ("",)})

# pause 有线：透传 + emit
r = node.run(text="model text", PauseState='{"mode": "pause", "text": "box"}')
check("pause 有线透传并 emit", r["result"] == ("model text",) and r["ui"]["sf_pause_text"] == ["model text"])

# pass 有线：同 pause 行为（透传 + emit）
r = node.run(text="model text", PauseState='{"mode": "pass", "text": "box"}')
check("pass 有线透传并 emit", r["result"] == ("model text",) and "ui" in r)

# pause 无线：保留盒子文本，不 emit
r = node.run(text=None, PauseState='{"mode": "pause", "text": "hand typed"}')
check("pause 无线保留盒子且不 emit", r == {"result": ("hand typed",)})

# 模式未知回退 pause
r = node.run(text="m", PauseState='{"mode": "bogus", "text": "b"}')
check("未知模式回退 pause（有线透传）", r["result"] == ("m",))

# 容错
r = node.run(text=None, PauseState="not json{{{")
check("非法 JSON 容错", r == {"result": ("",)})
r = node.run(text=None, PauseState="[1,2]")
check("非对象 JSON 容错", r == {"result": ("",)})
r = node.run(text=None, PauseState=None)
check("None state 容错", r == {"result": ("",)})
r = node.run(text=123, PauseState='{"mode": "pause", "text": "x"}')
check("text 非字符串转字符串", r["result"] == ("123",))
r = node.run(text=None, PauseState='{"mode": "continue", "text": 456}')
check("box_text 非字符串转字符串", r == {"result": ("456",)})
r = node.run(text=["a"], PauseState='{"mode": "pause", "text": "x"}')
check("text 为列表转字符串", r["result"] == ("['a']",))

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
