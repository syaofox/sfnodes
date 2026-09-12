# SFIgnoreGroups 后端测试（复刻孤海忽略多组）：
#  - 类结构：空 required / 空 RETURN_TYPES / OUTPUT_NODE / 空执行
# 运行：python3 tests/test_ignore_groups.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

mod = L.load_node("nodes/utils/ignore_groups.py")
Node = mod.SFIgnoreGroups
node = Node()

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


check("CATEGORY", Node.CATEGORY == "sfnodes/utils")
check("DESCRIPTION 存在", isinstance(Node.DESCRIPTION, str) and Node.DESCRIPTION)
check("INPUT_TYPES 空 required", Node.INPUT_TYPES() == {"required": {}})
check("RETURN_TYPES 空（纯面板 OUTPUT_NODE）", Node.RETURN_TYPES == ())
check("OUTPUT_NODE 真", Node.OUTPUT_NODE is True)
check("FUNCTION execute", Node.FUNCTION == "execute")
check("execute 返空", node.execute() == ())

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASSED")
