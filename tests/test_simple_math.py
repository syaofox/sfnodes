# SimpleMath 表达式求值防御测试（H6）：
#  - 字符串常量/字符串变量不再崩溃（isnan 类型校验）
#  - 语法错误/除零/未注册运算符兜底回退 (0, 0.0)
#  - ast.Constant 用 .value（3.13 deprecated / 3.14 removed 兼容）
# 运行：python tests/test_simple_math.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

sm = L.load_node("nodes/utils/simple_math.py")
Node = sm.SimpleMath
node = Node()

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# 类结构
check("CATEGORY", Node.CATEGORY == "sfnodes/utils")
check("DESCRIPTION 存在", isinstance(Node.DESCRIPTION, str) and Node.DESCRIPTION)

# 正常表达式
check("1+2*3 -> 7", node.execute("1+2*3")[1] == 7.0)
check("(1+2)*3 -> 9", node.execute("(1+2)*3")[1] == 9.0)
# 变量
check("a+b 变量", node.execute("a+b", a=1.5, b=2.5)[1] == 4.0)
# 裸名 abc（未连接变量）-> 0
r = node.execute("abc")
check("裸名 abc -> (0, 0.0)", r == (0, 0.0))
# 字符串常量 "abc" 不再崩（isnan 类型校验）
r = node.execute('"abc"')
check('字符串常量 "abc" -> (0, 0.0) 不崩', r == (0, 0.0))
# 字符串变量不再崩
r = node.execute("a", a="hello")
check('字符串变量 a="hello" -> (0, 0.0) 不崩', r == (0, 0.0))
# 除零不再崩
r = node.execute("1/0")
check("1/0 -> (0, 0.0) 不崩", r == (0, 0.0))
# 模零不再崩
r = node.execute("5%0")
check("5%0 -> (0, 0.0) 不崩", r == (0, 0.0))
# 语法错误不再崩
r = node.execute("2**")
check("2** 语法错误 -> (0, 0.0) 不崩", r == (0, 0.0))
# 未注册运算符不再崩
r = node.execute("2@3")
check("2@3 未注册运算符 -> (0, 0.0) 不崩", r == (0, 0.0))
# 字符串与数字比较不再崩
r = node.execute("a < b", a="x", b=1)
check("a<b 类型不匹配 -> (0, 0.0) 不崩", r == (0, 0.0))
# NaN 结果兜底（0**-1 之类）—— 0**-1 在 3.14 上经 Constant 兜底
r = node.execute("0**-1")
check("0**-1 -> (0, 0.0) 不崩", r == (0, 0.0))

# SimpleMathCondition 委托路径不崩
cond = sm.SimpleMathCondition()
check("condition on_true 合法", cond.execute(True, "1+1", "2/0")[1] == 2.0)
r = cond.execute(False, "1+1", "2/0")
check("condition on_false 除零不崩", r == (0, 0.0))

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
