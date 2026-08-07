# SFPromptTags 后端逻辑测试（Node/Python 直接运行：python tests/test_prompt_tags.py）
# 覆盖：INPUT_TYPES 结构、run() 的状态解析/拼接分支（order/sep/空白/list 输入/非法 state/超长 sep）
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.prompt_tags",
    os.path.join(root, "nodes", "text", "prompt_tags.py"),
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

node = mod.SFPromptTags()
check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)

it = node.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("INPUT_TYPES 含 text_in", "text_in" in it["optional"])
check("text_in forceInput", it["optional"]["text_in"][1].get("forceInput") is True)
check("INPUT_TYPES 含 hidden PromptState", "PromptState" in it["hidden"])
check("PromptState 默认空对象", it["hidden"]["PromptState"][1].get("default") == "{}")
check("返回类型 text", node.RETURN_TYPES == ("STRING",) and node.RETURN_NAMES == ("text",))
check("FUNCTION = run", node.FUNCTION == "run")

# 仅输入框内容
t = node.run(PromptState='{"text": "hello", "order": "mine", "sep": ", "}')
check("无 text_in 输出输入框内容", t == ("hello",))
t = node.run(text_in=None, PromptState="{}")
check("空 state 且无 text_in 输出空", t == ("",))
t = node.run(text_in="", PromptState="{}")
check("空 text_in 且空 state 输出空", t == ("",))

# 仅 text_in
t = node.run(text_in="wired", PromptState="{}")
check("空输入框时输出 text_in", t == ("wired",))

# 拼接：order mine
t = node.run(text_in="wired", PromptState='{"text": "mine", "order": "mine", "sep": ", "}')
check("mine 在前拼接", t == ("mine, wired",))
# 拼接：order wired
t = node.run(text_in="wired", PromptState='{"text": "mine", "order": "wired", "sep": ", "}')
check("wired 在前拼接", t == ("wired, mine",))
# 自定义分隔符
t = node.run(text_in="wired", PromptState='{"text": "mine", "order": "mine", "sep": " | "}')
check("自定义分隔符", t == ("mine | wired",))
# 任一侧空白 -> 丢弃分隔符
t = node.run(text_in="wired", PromptState='{"text": "  ", "order": "mine", "sep": ", "}')
check("mine 侧空白只输出 text_in", t == ("wired",))
t = node.run(text_in="   ", PromptState='{"text": "mine", "order": "wired", "sep": ", "}')
check("text_in 侧空白只输出 mine", t == ("mine",))

# state 解析容错
t = node.run(text_in="wired", PromptState="not json{{{")
check("非法 JSON 容错（空文本拼接 text_in）", t == ("wired",))
t = node.run(text_in="wired", PromptState="[1,2]")
check("非对象 JSON 容错", t == ("wired",))
t = node.run(text_in="wired", PromptState="null")
check("null JSON 容错", t == ("wired",))
t = node.run(text_in="wired", PromptState=None)
check("None state 容错", t == ("wired",))
t = node.run(text_in="wired", PromptState='{"text": "mine", "order": "bogus", "sep": ", "}')
check("非法 order 回退 mine", t == ("mine, wired",))
t = node.run(text_in="wired", PromptState='{"text": "mine", "order": "wired", "sep": 123}')
check("非字符串 sep 回退默认", t == ("wired, mine",))
t = node.run(text_in="wired", PromptState='{"text": "mine", "order": "mine", "sep": "' + "x" * 30 + '"}')
check("超长 sep 截断为默认", t == ("mine, wired",))
t = node.run(text_in="wired", PromptState='{"text": 123, "order": "mine", "sep": ", "}')
check("text 非字符串转空", t == ("wired",))

# text_in 容错
t = node.run(text_in=["wired"], PromptState='{"text": "mine", "order": "mine", "sep": ", "}')
check("text_in 为单元素列表取首项", t == ("mine, wired",))
t = node.run(text_in=[], PromptState='{"text": "mine", "order": "mine", "sep": ", "}')
check("text_in 为空列表当空", t == ("mine",))
t = node.run(text_in=123, PromptState='{"text": "mine", "order": "mine", "sep": ", "}')
check("text_in 非字符串转空", t == ("mine",))
t = node.run(text_in=["a", "b"], PromptState='{"text": "mine", "order": "mine", "sep": ", "}')
check("text_in 多元素列表取首项", t == ("mine, a",))

print("\nFAILURES:", len(failures))
sys.exit(1 if failures else 0)
