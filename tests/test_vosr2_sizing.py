# VOSR2 目标尺寸解析纯逻辑测试（无 torch 依赖）：
#  - 四模式（倍率 / 总像素 / 长边 / 短边）数学与宽高比保持
#  - 1 MP = 1024×1024 binary-MP 约定
#  - 钳制：超上限等比缩小、低于下限逐维抬升
#  - 参数校验 / coerce / 非法源尺寸
# 运行：python tests/test_vosr2_sizing.py
import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond, extra=""):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name} {extra}")

# 包骨架（sizing.py 有相对导入 ....sf_utils.resize_engine）
for name, path in [("sfnodes", "."), ("sfnodes.sf_utils", "sf_utils"), ("sfnodes.nodes", "nodes"),
                   ("sfnodes.nodes.model", "nodes/model"), ("sfnodes.nodes.model.vosr2", "nodes/model/vosr2")]:
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(root, path)]
    sys.modules[name] = m

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.model.vosr2.sizing", os.path.join(root, "nodes/model/vosr2/sizing.py")
)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)

# ── scale 模式 ──
w, h, clamped = S.TargetSizeSpec(mode="scale", scale=4.0).resolve(256, 256)
check("scale 4x → 1024", (w, h) == (1024, 1024) and not clamped)
w, h, _ = S.TargetSizeSpec(mode="scale", scale=1.5).resolve(100, 50)
check("scale 1.5x 非整数倍", (w, h) == (150, 75))
w, h, _ = S.TargetSizeSpec(mode="scale", scale=0.5).resolve(100, 50)
check("scale 允许缩小", (w, h) == (50, 25))
w, h, _ = S.TargetSizeSpec(mode="scale", scale=0.05).resolve(2000, 1000)
check("scale 0.05x", (w, h) == (100, 50))

# ── total pixels 模式（binary-MP：1.00 = 1024×1024）──
w, h, clamped = S.TargetSizeSpec(mode="total pixels", total_pixels=1.0).resolve(512, 512)
check("1MP @512 方图 → 1024", (w, h) == (1024, 1024) and not clamped)
w, h, _ = S.TargetSizeSpec(mode="total pixels", total_pixels=1.0).resolve(1024, 512)
check("1MP @2:1 → 1448x724", (w, h) == (1448, 724))
w, h, _ = S.TargetSizeSpec(mode="total pixels", total_pixels=0.25).resolve(1024, 1024)
check("0.25MP 缩小", (w, h) == (512, 512))
# 约定核对：1MP 方图总像素 = 1024*1024
check("1MP = 1048576 像素", 1024 * 1024 == 1048576)

# ── 长边 / 短边 ──
w, h, _ = S.TargetSizeSpec(mode="longer dimension", longer_size=1024).resolve(512, 256)
check("长边 1024", (w, h) == (1024, 512))
w, h, _ = S.TargetSizeSpec(mode="longer dimension", longer_size=1024).resolve(256, 512)
check("长边 1024 竖图", (w, h) == (512, 1024))
w, h, _ = S.TargetSizeSpec(mode="shorter dimension", shorter_size=512).resolve(1024, 256)
check("短边 512", (w, h) == (2048, 512))

# ── 宽高比保持 ──
for mode, kwargs, src in [
    ("total pixels", {"total_pixels": 0.5}, (3000, 2000)),
    ("longer dimension", {"longer_size": 1024}, (3000, 2000)),
    ("shorter dimension", {"shorter_size": 720}, (3000, 2000)),
]:
    w, h, _ = S.TargetSizeSpec(mode=mode, **kwargs).resolve(*src)
    src_ratio = src[0] / src[1]
    check(f"{mode} 保持宽高比", abs(w / h - src_ratio) < 0.01, f"{w}x{h}")

# ── 钳制 ──
w, h, clamped = S.TargetSizeSpec(mode="scale", scale=16.0).resolve(1024, 1024)
check("超上限等比缩到 8192", (w, h) == (8192, 8192) and clamped)
w, h, clamped = S.TargetSizeSpec(mode="scale", scale=16.0).resolve(2048, 1024)
check("超上限保持宽高比", (w, h) == (8192, 4096) and clamped)
w, h, clamped = S.TargetSizeSpec(mode="scale", scale=0.05).resolve(100, 100)
check("低于下限抬到 16", (w, h) == (16, 16) and clamped)
w, h, _ = S.TargetSizeSpec(mode="total pixels", total_pixels=64.0).resolve(8192, 8192)
check("64MP 方图 = 8192 不钳制", (w, h) == (8192, 8192))

# ── 校验 ──
check("默认合法", S.TargetSizeSpec().validate() is None)
check("非法 mode", S.TargetSizeSpec(mode="width & height").validate() is not None)
check("scale 越界", S.TargetSizeSpec(scale=32.0).validate() is not None)
check("total_pixels 越界", S.TargetSizeSpec(total_pixels=0.001).validate() is not None)
check("长边越界", S.TargetSizeSpec(longer_size=4).validate() is not None)

# ── coerce / 非法源 ──
check("coerce 数字 → 倍率", S.coerce_target_size(4).mode == "scale" and S.coerce_target_size(4).scale == 4.0)
check("coerce 浮点", S.coerce_target_size(2.5).scale == 2.5)
check("coerce 实例透传", S.coerce_target_size(S.TargetSizeSpec()) == S.TargetSizeSpec())
try:
    S.coerce_target_size("4x")
    check("coerce 非法类型报错", False)
except TypeError:
    check("coerce 非法类型报错", True)
try:
    S.TargetSizeSpec(mode="scale", scale=2.0).resolve(0, 100)
    check("非法源尺寸报错", False)
except ValueError:
    check("非法源尺寸报错", True)

# ── describe ──
check("describe scale", "×4.00" in S.TargetSizeSpec(mode="scale", scale=4.0).describe())
check("describe MP", "1.00MP" in S.TargetSizeSpec(mode="total pixels").describe())
check("describe 长边", "1024px" in S.TargetSizeSpec(mode="longer dimension").describe())

if failures:
    print(f"\n{len(failures)} 项失败: {failures}")
    sys.exit(1)
print("\n全部通过")
