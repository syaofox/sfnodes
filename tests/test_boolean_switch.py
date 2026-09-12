# SFBooleanSwitch 后端测试（复刻孤海布尔开关）：
#  - 类结构：CATEGORY/DESCRIPTION/单 BOOLEAN 输出/RETURN_NAMES 静态 value
#  - execute：True/False 直通（原版 1:1，default True）
# 运行：python3 tests/test_boolean_switch.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

mod = L.load_node("nodes/utils/boolean_switch.py")
Node = mod.SFBooleanSwitch
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
check("RETURN_TYPES 单 BOOLEAN", Node.RETURN_TYPES == ("BOOLEAN",))
check("RETURN_NAMES 静态 value", Node.RETURN_NAMES == ("value",))
check("FUNCTION execute", Node.FUNCTION == "execute")
types = Node.INPUT_TYPES()
check("required value BOOLEAN", types["required"]["value"][0] == "BOOLEAN")
check("value default True（原版行为）", types["required"]["value"][1]["default"] is True)
check("True 直通", node.execute(True) == (True,))
check("False 直通", node.execute(False) == (False,))
check("输出真 bool 类型", type(node.execute(True)[0]) is bool)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASSED")
