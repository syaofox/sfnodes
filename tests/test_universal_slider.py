# SFUniversalSlider 后端测试（复刻孤海万能滑条）：
#  - 类结构：CATEGORY/DESCRIPTION/RETURN_NAMES 静态 value（前端槽名动态 int/float）
#  - execute：float 直通 / int 取整，round(...,10) 精度（原版 1:1）
#  - IS_CHANGED：同 execute 逻辑回显
# 运行：python3 tests/test_universal_slider.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

mod = L.load_node("nodes/utils/universal_slider.py")
Node = mod.SFUniversalSlider
node = Node()

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── 类结构 ──
check("CATEGORY", Node.CATEGORY == "sfnodes/utils")
check("DESCRIPTION 存在", isinstance(Node.DESCRIPTION, str) and Node.DESCRIPTION)
check("RETURN_TYPES any", Node.RETURN_TYPES == ("*",))
check("RETURN_NAMES 静态 value", Node.RETURN_NAMES == ("value",))
check("FUNCTION execute", Node.FUNCTION == "execute")
types = Node.INPUT_TYPES()
check("required value FLOAT", types["required"]["value"][0] == "FLOAT")
check("value default 0.75", types["required"]["value"][1]["default"] == 0.75)
check("hidden output_type 两档", types["hidden"]["output_type"][0] == ["float", "int"])

# ── execute：float 档 ──
check("float 0.75 直通", node.execute(0.75) == (0.75,))
check("float 缺省 output_type 直通", node.execute(0.75, "float") == (0.75,))
r = node.execute(0.123456789012345)
check("float round 10 位精度", r == (round(0.123456789012345, 10),))

# ── execute：int 档 ──
check("int 1.5→2", node.execute(1.5, "int") == (2,))
check("int 1.4→1", node.execute(1.4, "int") == (1,))
check("int 输出真 int 类型", type(node.execute(1.5, "int")[0]) is int)
check("float 输出真 float 类型", type(node.execute(1.5, "float")[0]) is float)

# ── IS_CHANGED：同逻辑回显 ──
check("IS_CHANGED float", Node.IS_CHANGED(0.75) == 0.75)
check("IS_CHANGED int", Node.IS_CHANGED(1.5, "int") == 2)
check("IS_CHANGED 字符串输入不崩", Node.IS_CHANGED("0.5") == 0.5)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASSED")
