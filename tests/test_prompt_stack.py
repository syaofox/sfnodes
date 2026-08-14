# SFPromptStack 后端测试（Python 直接运行：python tests/test_prompt_stack.py）
# 覆盖：
#   - 类结构：CATEGORY / DESCRIPTION / INPUT_TYPES（prepend/append + 隐藏
#     PromptStackState）/ RETURN_TYPES / FUNCTION / OUTPUT_IS_LIST
#   - execute：开关过滤、空白行过滤、前后缀、顺序、坏 JSON / 空状态兜底
import importlib.util
import json
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

spec = importlib.util.spec_from_file_location(
    "prompt_stack", os.path.join(root, "nodes", "text", "prompt_stack.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

Node = mod.SFPromptStack
node = Node()

check("CATEGORY", Node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(Node.DESCRIPTION, str) and Node.DESCRIPTION)

it = Node.INPUT_TYPES()
check("required.prepend_text", "prepend_text" in it["required"])
check("required.append_text", "append_text" in it["required"])
check("hidden.PromptStackState", "PromptStackState" in it.get("hidden", {}))
check("RETURN_TYPES", Node.RETURN_TYPES == ("STRING", "STRING"))
check("RETURN_NAMES", Node.RETURN_NAMES == ("prompt", "body_text"))
check("OUTPUT_IS_LIST", Node.OUTPUT_IS_LIST == (True, True))
check("FUNCTION", Node.FUNCTION == "execute")

def st(rows):
    return json.dumps({"version": 1, "rows": rows})

# 基础：开着的行按顺序输出
p, b = node.execute(PromptStackState=st([
    {"enabled": True, "text": "alpha"},
    {"enabled": True, "text": "beta"},
]))
check("基础输出", p == ["alpha", "beta"] and b == ["alpha", "beta"])

# 关闭的行过滤
p, b = node.execute(PromptStackState=st([
    {"enabled": True, "text": "alpha"},
    {"enabled": False, "text": "skipped"},
    {"enabled": True, "text": "beta"},
]))
check("关闭行过滤", p == ["alpha", "beta"])

# 空白行过滤（strip 后为空）
p, b = node.execute(PromptStackState=st([
    {"enabled": True, "text": "  alpha  "},
    {"enabled": True, "text": "   "},
    {"enabled": True, "text": ""},
]))
check("空白行过滤+strip", p == ["alpha"] and b == ["alpha"])

# 前后缀
p, b = node.execute(prepend_text="PRE ", append_text=" POST", PromptStackState=st([
    {"enabled": True, "text": "alpha"},
    {"enabled": True, "text": "beta"},
]))
check("前后缀", p == ["PRE alpha POST", "PRE beta POST"] and b == ["alpha", "beta"])

# 全部关闭 → 空列表
p, b = node.execute(PromptStackState=st([
    {"enabled": False, "text": "a"},
]))
check("全部关闭空输出", p == [] and b == [])

# 空状态 / 坏 JSON / 非 dict / 非 list 兜底
p, b = node.execute(PromptStackState="{}")
check("空状态兜底", p == [] and b == [])
p, b = node.execute(PromptStackState="not-json")
check("坏 JSON 兜底", p == [] and b == [])
p, b = node.execute(PromptStackState="[]")
check("非 dict 兜底", p == [] and b == [])
p, b = node.execute(PromptStackState=st("nope"))
check("rows 非 list 兜底", p == [] and b == [])
p, b = node.execute()
check("缺省参数兜底", p == [] and b == [])

# 行内非文本字段跳过
p, b = node.execute(PromptStackState=st([
    {"enabled": True, "text": 123},
    {"enabled": True, "text": None},
]))
check("非文本行跳过", p == [] and b == [])

# 顺序保持（含关闭行穿插）
p, b = node.execute(PromptStackState=st([
    {"enabled": True, "text": "one"},
    {"enabled": False, "text": "x"},
    {"enabled": True, "text": "two"},
    {"enabled": True, "text": "three"},
]))
check("顺序保持", p == ["one", "two", "three"])

if failures:
    print(f"\n{failures.length if False else len(failures)} FAILED")
    sys.exit(1)
print("\nALL PASS")
