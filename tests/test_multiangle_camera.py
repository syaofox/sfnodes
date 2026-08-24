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
check("required 七输入", sorted(it["required"]) == ["add_sks", "distance", "horizontal_direction", "ordered", "prefix", "suffix", "vertical_direction"])
check("水平方向选项带随机（8+1）", it["required"]["horizontal_direction"][0] == [mod._RANDOM] + mod.H_DIRECTIONS)
check("水平方向默认 front view", it["required"]["horizontal_direction"][1]["default"] == "front view")
check("垂直角度选项带随机（4+1）", it["required"]["vertical_direction"][0] == [mod._RANDOM] + mod.V_DIRECTIONS)
check("垂直角度默认 eye-level shot", it["required"]["vertical_direction"][1]["default"] == "eye-level shot")
check("景别选项带随机（3+1）", it["required"]["distance"][0] == [mod._RANDOM] + mod.DISTANCES)
check("景别默认 medium shot", it["required"]["distance"][1]["default"] == "medium shot")
check("add_sks 默认 True", it["required"]["add_sks"][1]["default"] is True)
check("prefix 默认空", it["required"]["prefix"][1]["default"] == "")
check("suffix 默认空", it["required"]["suffix"][1]["default"] == "")
check("ordered 默认 False（随机打乱）", it["required"]["ordered"][1]["default"] is False)
check("RETURN_TYPES 为两个 STRING", node.RETURN_TYPES == (IO.STRING, IO.STRING) and node.RETURN_NAMES == ("prompt", "combinations"))
check("OUTPUT_IS_LIST（combinations 为列表）", node.OUTPUT_IS_LIST == (False, True))
check("FUNCTION = execute", node.FUNCTION == "execute")

combos = set()
for hd in mod.H_DIRECTIONS:
    for vd in mod.V_DIRECTIONS:
        for d in mod.DISTANCES:
            combos.add((hd, vd, d))
check("96 组合全覆盖（8x4x3 各不相同）", len(combos) == 96)

check("add_sks=True 带前缀", node.execute("front view", "eye-level shot", "medium shot", True)[0] == "<sks> front view eye-level shot medium shot")
check("add_sks=False 无前缀", node.execute("front view", "eye-level shot", "medium shot", False)[0] == "front view eye-level shot medium shot")
check("拼接顺序 h v d", node.execute("back view", "high-angle shot", "close-up", False)[0] == "back view high-angle shot close-up")
check("前后缀包裹 prompt", node.execute("front view", "eye-level shot", "medium shot", True, "P ", " S")[0] == "P <sks> front view eye-level shot medium shot S")
check("前后缀空时输出不变", node.execute("front view", "eye-level shot", "medium shot", False, "", "")[0] == "front view eye-level shot medium shot")

combos_on = node.execute("front view", "eye-level shot", "medium shot", True, ordered=True)[1]
combos_off = node.execute("front view", "eye-level shot", "medium shot", False, ordered=True)[1]
check("combinations 为 96 条", len(combos_off) == 96 and len(combos_on) == 96)
check("combinations 元素各不相同", len(set(combos_off)) == 96)
check("combinations 覆盖全部组合", set(combos_off) == {f"{h} {v} {d}" for h in mod.H_DIRECTIONS for v in mod.V_DIRECTIONS for d in mod.DISTANCES})
check("combinations 与输入选择无关（恒定全量）",
      node.execute("back view", "high-angle shot", "close-up", False, ordered=True)[1] == combos_off)
check("add_sks=True 每条带 <sks> 前缀", all(c.startswith("<sks> ") for c in combos_on) and all(not c.startswith("<sks> ") for c in combos_off))
check("ordered=True 首条顺序 h→v→d", combos_off[0] == "front view low-angle shot wide shot")
combos_pfx = node.execute("front view", "eye-level shot", "medium shot", True, "P ", " S", ordered=True)[1]
check("ordered=True 每条带前后缀", combos_pfx == [f"P <sks> {h} {v} {d} S" for h in mod.H_DIRECTIONS for v in mod.V_DIRECTIONS for d in mod.DISTANCES])

shuf1 = node.execute("front view", "eye-level shot", "medium shot", False)[1]
shuf2 = node.execute("front view", "eye-level shot", "medium shot", False)[1]
check("默认（ordered=False）为全量组合", set(shuf1) == {f"{h} {v} {d}" for h in mod.H_DIRECTIONS for v in mod.V_DIRECTIONS for d in mod.DISTANCES})
check("默认每次顺序不同（随机打乱）", shuf1 != shuf2)

rng_state = random.getstate()
random.seed(42)
h_r = mod._resolve_choice(mod._RANDOM, mod.H_DIRECTIONS)
random.setstate(rng_state)
check("随机水平方向落在 8 选项内", h_r in mod.H_DIRECTIONS)
check("随机不改变确定性选项", mod._resolve_choice("front view", mod.H_DIRECTIONS) == "front view")
out = node.execute(mod._RANDOM, "eye-level shot", "medium shot", False)[0]
check("执行时随机维度被替换（其余保留）", out in {f"{h} eye-level shot medium shot" for h in mod.H_DIRECTIONS})
check("IS_CHANGED 固定顺序模式返回固定值", node.IS_CHANGED("front view", "eye-level shot", "medium shot", ordered=True)
      == ("front view", "eye-level shot", "medium shot", "", ""))
check("IS_CHANGED 含前后缀变化", node.IS_CHANGED("front view", "eye-level shot", "medium shot", True, "P ", " S", ordered=True)
      == ("front view", "eye-level shot", "medium shot", "P ", " S"))
c1 = node.IS_CHANGED(mod._RANDOM, "eye-level shot", "medium shot", ordered=True)
c2 = node.IS_CHANGED(mod._RANDOM, "eye-level shot", "medium shot", ordered=True)
check("IS_CHANGED 随机模式每次不同", isinstance(c1, float) and c1 != c2)
d1 = node.IS_CHANGED("front view", "eye-level shot", "medium shot")
d2 = node.IS_CHANGED("front view", "eye-level shot", "medium shot")
check("IS_CHANGED 打乱模式每次不同（触发重排）", isinstance(d1, float) and d1 != d2)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASS")
