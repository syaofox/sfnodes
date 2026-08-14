# SFCanvasSizePreset 后端逻辑测试（Python 直接运行：python tests/test_canvas_size.py）
# 覆盖：
#   - 全部预设分辨率 16 整除（DiT patch 约束，各模型官方规格）
#   - 宽高比标签与实际数值一致（约分验证）
#   - _parse_resolution 解析 + 分组头/非法值兜底
#   - execute 输出（width/height/resolution/aspect_ratio）
#   - INPUT_TYPES 结构（model/resolution、默认值）
#   - VALIDATE_INPUTS 恒 True（动态 combo 兜底）
#   - API handler 响应结构（mock aiohttp，模块顶层 import 不炸）
import asyncio
import importlib.util
import os
import sys
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock aiohttp（本机无运行时依赖，仅需 web.json_response 形状）──
fake_web = types.ModuleType("aiohttp")
class _FakeWeb:
    @staticmethod
    def json_response(payload, **kw):
        return payload
fake_web.web = _FakeWeb()
sys.modules["aiohttp"] = fake_web

# ── mock server（ComfyUI 运行时提供；捕获路由注册的 handler 供断言）──
handlers = {}
class _FakeRoutes:
    def get(self, path):
        def deco(fn):
            handlers[path] = fn
            return fn
        return deco
fake_server = types.ModuleType("server")
fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=_FakeRoutes()))
sys.modules["server"] = fake_server

spec = importlib.util.spec_from_file_location(
    "canvas_size",
    os.path.join(root, "nodes", "utils", "canvas_size.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

PRESETS = mod.PRESETS
RESOLUTION_VALUES = mod.RESOLUTION_VALUES
MODELS = mod.MODELS

# ── 1. 全表整除 + 比例一致 ──
# 全部条目均 16 整除（官方尺寸/算法输出，含 Hunyuan 480p 的 848x480 与
# LTX 的 1280x736/1376x768）
RATIO_OK = 0
for model, tiers in PRESETS.items():
    for tier, items in tiers.items():
        for ratio, w, h in items:
            check(f"{model}/{tier}/{w}x{h}: 16 整除", w % 16 == 0 and h % 16 == 0)
            # 比例容差校验：官方近似标签（如 Qwen 16:9=1664x928 实际约分
            # 52:29）允许 <1% 偏差；精确约分标签（53:30/40:23 等）偏差为 0
            rw, rh = map(int, ratio.split(":"))
            expect = rw / rh
            check(f"{model}/{tier}/{w}x{h}: 比例 {ratio}",
                  abs((w / h) - expect) / expect < 0.01)
            RATIO_OK += 1
print(f"--- 共 {RATIO_OK} 个分辨率条目 ---")

# ── 2. 平铺选项结构（分组头 + 默认模型列表）──
for model in MODELS:
    if model.startswith("--") and model.endswith("--"):
        continue  # 分组头无 resolution 表
    values = RESOLUTION_VALUES[model]
    headers = [v for v in values if v.startswith("--") and v.endswith("--")]
    n_tiers = len(PRESETS[model])
    check(f"{model}: {n_tiers} 个分组头", len(headers) == n_tiers)
    n_items = len(values) - len(headers)
    n_expected = sum(len(items) for items in PRESETS[model].values())
    check(f"{model}: {n_items} 个分辨率项", n_items == n_expected)
check("model 下拉含 2 个分组头", len([m for m in MODELS if m.startswith("--")]) == 2)
check("分组头为 Image/Video", MODELS[0] == "-- Image --" and "-- Video --" in MODELS)
check("PRESETS 键全部在 MODELS 中", set(PRESETS.keys()) <= set(MODELS))
check("默认模型为 Z-Image (Turbo)", mod.DEFAULT_MODEL == "Z-Image (Turbo)")
check("Qwen 官方表 7 项", len(PRESETS["Qwen-Image (2512)"]["Official"]) == 7)
check("HunyuanVideo 1.5 两档", set(PRESETS["HunyuanVideo 1.5"].keys()) == {"720p", "480p"})
check("Hunyuan 480p 官方 bucket 值", PRESETS["HunyuanVideo 1.5"]["480p"][0] == ("53:30", 848, 480))
check("LTX-2.5 两档", set(PRESETS["LTX-2.5"].keys()) == {"0.9MP", "1K"})
check("LTX 0.9MP 模板输出值", PRESETS["LTX-2.5"]["0.9MP"][0] == ("40:23", 1280, 736))
check("LTX 1K 模板输出值", PRESETS["LTX-2.5"]["1K"][1] == ("43:24", 1376, 768))

# ── 3. 解析兜底 ──
w, h, r = mod._parse_resolution("1024x1024 (1:1)")
check("解析 1024x1024 (1:1)", (w, h, r) == (1024, 1024, "1:1"))
w, h, r = mod._parse_resolution("1280x704 (20:11)")
check("解析 1280x704 (20:11)", (w, h, r) == (1280, 704, "20:11"))
w, h, r = mod._parse_resolution("--1MP--")
check("分组头兜底回默认", (w, h, r) == (1024, 1024, "1:1"))
w, h, r = mod._parse_resolution("")
check("空值兜底回默认", (w, h, r) == (1024, 1024, "1:1"))

# ── 4. execute 输出 ──
node = mod.CanvasSizePreset()
out = node.execute("Z-Image (Turbo)", "1152x896 (9:7)")
check("execute 四元组", isinstance(out, tuple) and len(out) == 4)
check("width=1152", out[0] == 1152)
check("height=896", out[1] == 896)
check("resolution 透传", out[2] == "1152x896 (9:7)")
check("aspect_ratio=1.285714", out[3] == round(1152 / 896, 6))
out = node.execute("Wan2.2 T2V", "832x480 (16:9)")
check("Wan2.2 480p aspect", out[3] == round(832 / 480, 6))

# ── 5. INPUT_TYPES / VALIDATE_INPUTS ──
it = mod.CanvasSizePreset.INPUT_TYPES()
check("model combo 含分组头与全部模型", it["required"]["model"][0] == MODELS)
check("resolution 静态列表为默认模型表", it["required"]["resolution"][0] == RESOLUTION_VALUES[mod.DEFAULT_MODEL])
check("默认分辨率为 1024x1024 (1:1)", it["required"]["resolution"][1]["default"] == "1024x1024 (1:1)")
check("VALIDATE_INPUTS 恒 True", mod.CanvasSizePreset.VALIDATE_INPUTS(foo="任意值") is True)
check("RETURN_TYPES", mod.CanvasSizePreset.RETURN_TYPES == ("INT", "INT", "STRING", "FLOAT"))
check("CATEGORY", mod.CanvasSizePreset.CATEGORY == "sfnodes/utils")
check("DESCRIPTION 非空", bool(mod.CanvasSizePreset.DESCRIPTION))

# ── 6. API handler 响应结构（mock aiohttp/server 已注入，模块导入即注册路由）──
check("路由已注册", "/api/sfnodes/canvas_size_presets" in handlers)
async def _call():
    return await handlers["/api/sfnodes/canvas_size_presets"](None)
payload = asyncio.run(_call())
check("API payload 含 models", payload["models"] == MODELS)
check("API payload values 与常量一致", payload["values"] == RESOLUTION_VALUES)

print()
if failures:
    print(f"FAILED: {len(failures)} 项")
    sys.exit(1)
print("ALL PASS")
