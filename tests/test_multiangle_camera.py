# SFMultiangleCamera 后端逻辑测试（Python 直接运行：python tests/test_multiangle_camera.py）
# 覆盖：
#   - 结构：SFMultiangleCamera 类、CATEGORY、DESCRIPTION、INPUT_TYPES、
#     RETURN_TYPES/RETURN_NAMES、FUNCTION
#   - 三个下拉选项组（8/4/3）与默认值
#   - 96 组合全覆盖、add_sks 开关
import importlib.util
import os
import random
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

comfy = types.ModuleType("comfy")
node_typing = types.ModuleType("comfy.comfy_types")
node_typing_module = types.ModuleType("comfy.comfy_types.node_typing")
class IO:
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
node_typing_module.IO = IO
comfy.comfy_types = node_typing
comfy.comfy_types.node_typing = node_typing_module
sys.modules["comfy"] = comfy
sys.modules["comfy.comfy_types"] = node_typing
sys.modules["comfy.comfy_types.node_typing"] = node_typing_module

pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.text"); pkg3.__path__ = [os.path.join(root, "nodes", "text")]; sys.modules["sfnodes.nodes.text"] = pkg3

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.text.multiangle_camera",
    os.path.join(root, "nodes", "text", "multiangle_camera.py"),
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

node = mod.SFMultiangleCamera()

check("CATEGORY", node.CATEGORY == "sfnodes/text")
check("DESCRIPTION 存在", isinstance(node.DESCRIPTION, str) and len(node.DESCRIPTION) > 0)
it = node.INPUT_TYPES()
check("required 四输入", sorted(it["required"]) == ["add_sks", "distance", "horizontal_direction", "vertical_direction"])
check("水平方向选项带随机（8+1）", it["required"]["horizontal_direction"][0] == [mod._RANDOM] + mod.H_DIRECTIONS)
check("水平方向默认 front view", it["required"]["horizontal_direction"][1]["default"] == "front view")
check("垂直角度选项带随机（4+1）", it["required"]["vertical_direction"][0] == [mod._RANDOM] + mod.V_DIRECTIONS)
check("垂直角度默认 eye-level shot", it["required"]["vertical_direction"][1]["default"] == "eye-level shot")
check("景别选项带随机（3+1）", it["required"]["distance"][0] == [mod._RANDOM] + mod.DISTANCES)
check("景别默认 medium shot", it["required"]["distance"][1]["default"] == "medium shot")
check("add_sks 默认 True", it["required"]["add_sks"][1]["default"] is True)
check("RETURN_TYPES 为 STRING", node.RETURN_TYPES == (IO.STRING,) and node.RETURN_NAMES == ("prompt",))
check("FUNCTION = execute", node.FUNCTION == "execute")

combos = set()
for hd in mod.H_DIRECTIONS:
    for vd in mod.V_DIRECTIONS:
        for d in mod.DISTANCES:
            combos.add((hd, vd, d))
check("96 组合全覆盖（8x4x3 各不相同）", len(combos) == 96)

check("add_sks=True 带前缀", node.execute("front view", "eye-level shot", "medium shot", True) == ("<sks> front view eye-level shot medium shot",))
check("add_sks=False 无前缀", node.execute("front view", "eye-level shot", "medium shot", False) == ("front view eye-level shot medium shot",))
check("拼接顺序 h v d", node.execute("back view", "high-angle shot", "close-up", False) == ("back view high-angle shot close-up",))

rng_state = random.getstate()
random.seed(42)
h_r = mod._resolve_choice(mod._RANDOM, mod.H_DIRECTIONS)
random.setstate(rng_state)
check("随机水平方向落在 8 选项内", h_r in mod.H_DIRECTIONS)
check("随机不改变确定性选项", mod._resolve_choice("front view", mod.H_DIRECTIONS) == "front view")
out = node.execute(mod._RANDOM, "eye-level shot", "medium shot", False)[0]
check("执行时随机维度被替换（其余保留）", out in {f"{h} eye-level shot medium shot" for h in mod.H_DIRECTIONS})
check("IS_CHANGED 确定性模式返回固定值", node.IS_CHANGED("front view", "eye-level shot", "medium shot")
      == ("front view", "eye-level shot", "medium shot"))
c1 = node.IS_CHANGED(mod._RANDOM, "eye-level shot", "medium shot")
c2 = node.IS_CHANGED(mod._RANDOM, "eye-level shot", "medium shot")
check("IS_CHANGED 随机模式每次不同", isinstance(c1, float) and c1 != c2)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASS")
