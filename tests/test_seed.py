# SFSeed -2/-3 继承语义测试（M17）：
#  -2 输出 上次+1；-3 输出 上次-1；首次/无上次时随机起点；固定值也更新上次
# 运行：python tests/test_seed.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sf_loader as L

seed_mod = L.load_node("nodes/utils/seed.py")
Node = seed_mod.SFSeed

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

check("CATEGORY", Node.CATEGORY == "sfnodes/utils")

n = Node()
r1 = n.execute(-2)
check("首次 -2 是随机 int", isinstance(r1["result"][0], int))
r2 = n.execute(-2)
check("-2 继承上次+1", r2["result"][0] == r1["result"][0] + 1)
r3 = n.execute(-3)
check("-3 继承上次-1", r3["result"][0] == r2["result"][0] - 1)
r4 = n.execute(42)
check("固定值输出 42", r4["result"][0] == 42)
r5 = n.execute(-2)
check("-2 继承固定值+1", r5["result"][0] == 43)
# -1 仍随机（不受继承链影响）
r6 = n.execute(-1)
check("-1 随机", isinstance(r6["result"][0], int))
# 新实例：-2 重新随机起点
n2 = Node()
r7 = n2.execute(-2)
check("新实例 -2 随机起点", isinstance(r7["result"][0], int))

if failures:
    print(f"\n{failures}")
    sys.exit(1)
print("\nALL PASS")
