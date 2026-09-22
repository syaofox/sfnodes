# VOSR2 档位设置纯逻辑测试（无 torch 依赖）：
#  - manual/speed 档位批量解析、0=自动、batch_override 覆盖
#  - tile_strategy 三态与尺寸联动
#  - normalize_settings 容错（None/dict/实例/非法类型）
#  - validate_settings 枚举与范围
# 运行：python tests/test_vosr2_settings.py
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

spec = importlib.util.spec_from_file_location(
    "vosr2_settings", os.path.join(root, "nodes/model/vosr2/settings.py")
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# ── 默认值 ──
d = S.default_settings()
check("默认 manual", d.quality_profile == "manual")
check("默认 auto 显存策略", d.memory_policy == "auto")
check("默认 auto 分块策略", d.tile_strategy == "auto")
check("manual 批量=1", (
    d.resolved_dit_tile_batch() == 1 and d.resolved_dino_batch() == 1
    and d.resolved_image_batch() == 1 and d.resolved_frame_batch() == 1
))

# ── speed 档位 ──
sp = S.VOSR2Settings(quality_profile="speed")
check("speed DiT 批量>1", sp.resolved_dit_tile_batch() > 1)
check("speed 帧批量>1", sp.resolved_frame_batch() > 1)

# ── 显式值优先于档位 ──
custom = S.VOSR2Settings(quality_profile="speed", dit_tile_batch=2)
check("显式批量覆盖档位", custom.resolved_dit_tile_batch() == 2)
check("未显式字段仍走档位", custom.resolved_dino_batch() == 4)

# ── batch_override ──
ov = S.VOSR2Settings(quality_profile="manual", image_batch=1, frame_batch=1, batch_override=6)
check("override 覆盖 image_batch", ov.resolved_image_batch() == 6)
check("override 覆盖 frame_batch", ov.resolved_frame_batch() == 6)
check("override=0 不生效", d.resolved_image_batch() == 1)

# ── 非法档位回退 manual（不抛错）──
weird = S.VOSR2Settings(quality_profile="turbo")
check("非法档位回退", weird.resolved_dit_tile_batch() == 1)

# ── tile_strategy ──
check("auto 小图不分块", not d.use_tiling(512, 512, 512))
check("auto 大图分块", d.use_tiling(1024, 768, 512))
check("auto tile_size=0 不分块", not d.use_tiling(4096, 4096, 0))
check("tiled 强制分块", S.VOSR2Settings(tile_strategy="tiled").use_tiling(64, 64, 0))
check("full_frame 强制不分块", not S.VOSR2Settings(tile_strategy="full_frame").use_tiling(4096, 4096, 512))

# ── normalize_settings ──
check("None → 默认", S.normalize_settings(None) == d)
check("实例原样", S.normalize_settings(d) is d)
merged = S.normalize_settings({"quality_profile": "speed", "unknown_key": 1})
check("dict 只取已知字段", merged.quality_profile == "speed")
try:
    S.normalize_settings(42)
    check("非法类型报错", False)
except TypeError:
    check("非法类型报错", True)

# ── with_overrides ──
patched = S.with_overrides(None, memory_policy="staged", torch_compile=True, cache_refresh=None)
check("覆盖字段生效", patched.memory_policy == "staged" and patched.torch_compile)
check("None 值忽略", patched.cache_refresh == 0)
check("原设置不变（frozen）", d.memory_policy == "auto")

# ── validate_settings ──
check("默认合法", S.validate_settings(d) is None)
check("非法档位报错", S.validate_settings(S.VOSR2Settings(quality_profile="x")) is not None)
check("非法阈值报错", S.validate_settings(S.VOSR2Settings(cache_threshold=1.5)) is not None)
check("负批量报错", S.validate_settings(S.VOSR2Settings(image_batch=-1)) is not None)

# ── describe 不抛错 ──
check("describe 可读", "profile=" in d.describe())

if failures:
    print(f"\n{len(failures)} 项失败: {failures}")
    sys.exit(1)
print("\n全部通过")
