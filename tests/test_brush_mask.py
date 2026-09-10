# SFImageBrushMask 后端逻辑测试（Node/Python 直接运行：python3 tests/test_brush_mask.py）
# 覆盖：
#   - 结构：类、CATEGORY、RETURN_TYPES/NAMES（5 输出）、DESCRIPTION、hidden 声明
#   - 纯逻辑 sf_utils/brush_mask.py：parse_strokes 三格式兼容/越界丢弃、
#     parse_state_strokes、build_brush_data 往返、rasterize brush/erase/空笔
#   - execute()：磁盘源图 + 笔触遮罩语义（二值）+ 缺源退化 + filename 透传
#   - IS_CHANGED：笔触变化键变化、预览字段（opacity/color）不进键、源文件 mtime
# mock：torch（numpy 代理）/ aiohttp / folder_paths；同链路加载
# nodes/image/crop.py（brush_mask 复用其 _safe_join）。
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []


def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")


# ── mock torch（numpy 代理，支持 [None,] 与 .shape）──
torch = types.ModuleType("torch")
torch.float32 = np.float32
torch.Tensor = type("Tensor", (), {})
torch.zeros = lambda shape, **k: np.zeros(shape, dtype=np.float32)
torch.ones = lambda shape, **k: np.ones(shape, dtype=np.float32)
torch.from_numpy = lambda a: np.asarray(a, dtype=np.float32)
torch.clamp = lambda a, lo, hi: np.clip(np.asarray(a), lo, hi)
torch.nn = types.ModuleType("torch.nn")
torch.nn.functional = types.ModuleType("torch.nn.functional")
torch.nn.functional.interpolate = lambda *a, **k: None
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional

# ── mock aiohttp ──
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.ModuleType("aiohttp.web")
aiohttp.web.json_response = lambda *a, **k: types.SimpleNamespace(status=200, body=a)
aiohttp.web.Response = types.SimpleNamespace
aiohttp.web.Request = types.SimpleNamespace
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

# ── mock folder_paths ──
tmp_in = tempfile.mkdtemp(prefix="sf_brush_input_")
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: tmp_in
fp.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
sys.modules["folder_paths"] = fp

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils.x import ...）──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3
pkg_u = types.ModuleType("sfnodes.sf_utils"); pkg_u.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg_u


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# 先加载 crop.py（brush_mask 复用 _safe_join），再加载纯逻辑与节点
_load("sfnodes.nodes.image.crop", os.path.join(root, "nodes", "image", "crop.py"))
pure = _load("sfnodes.sf_utils.brush_mask", os.path.join(root, "sf_utils", "brush_mask.py"))
mod = _load("sfnodes.nodes.image.brush_mask", os.path.join(root, "nodes", "image", "brush_mask.py"))

# ── 结构断言 ──
check("SFImageBrushMask 已加载", hasattr(mod, "SFImageBrushMask"))
check("CATEGORY", mod.SFImageBrushMask.CATEGORY == "sfnodes/image")
check("DESCRIPTION", isinstance(mod.SFImageBrushMask.DESCRIPTION, str) and len(mod.SFImageBrushMask.DESCRIPTION) > 0)
check("非 OUTPUT_NODE", getattr(mod.SFImageBrushMask, "OUTPUT_NODE", False) is False)
check("RETURN_TYPES", mod.SFImageBrushMask.RETURN_TYPES == ("IMAGE", "MASK", "INT", "INT", "STRING"))
check("RETURN_NAMES", mod.SFImageBrushMask.RETURN_NAMES == ("image", "mask", "width", "height", "filename"))

it = mod.SFImageBrushMask.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("hidden 声明 SFBrushMaskJson", it["hidden"]["SFBrushMaskJson"][0] == "STRING")
check("hidden 默认 {}", it["hidden"]["SFBrushMaskJson"][1]["default"] == "{}")

# 注册键一致性（根 __init__.py 文本检查，避免全包导入）
init_src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
check("NODE_CLASS_MAPPINGS 注册", '"SFImageBrushMask": SFImageBrushMask' in init_src)
check("NODE_DISPLAY_NAME_MAPPINGS 注册", '"SFImageBrushMask": "SF Image Brush Mask"' in init_src)
check("导入语句存在", "from .nodes.image.brush_mask import SFImageBrushMask" in init_src)

# ── _parse_state / _lean_key ──
check("_parse_state 合法 json", mod._parse_state('{"src_w": 4}') == {"src_w": 4})
check("_parse_state 非法 json", mod._parse_state("not-json") == {})
check("_parse_state 空串", mod._parse_state("") == {})
check("_parse_state None", mod._parse_state(None) == {})
check("_parse_state dict 直通", mod._parse_state({"a": 1}) == {"a": 1})

# ── parse_strokes：三格式兼容 ──
W = H = 100
s = pure.parse_strokes("brush:20:1.0:10,10;20,20", W, H)
check("新无色格式", len(s) == 1 and s[0]["mode"] == "brush" and s[0]["size"] == 20
      and s[0]["points"] == [(10, 10), (20, 20)])
s = pure.parse_strokes("erase:30:0.5:255,0,0:5,5;6,6", W, H, default_size=80)
check("新带色格式（色被忽略）", len(s) == 1 and s[0]["mode"] == "erase" and s[0]["size"] == 30
      and s[0]["points"] == [(5, 5), (6, 6)])
s = pure.parse_strokes("brush:7,7;8,8", W, H)
check("旧 mode:points 格式", len(s) == 1 and s[0]["points"] == [(7, 7), (8, 8)])
s = pure.parse_strokes("9,9;10,10", W, H)
check("裸点列格式", len(s) == 1 and s[0]["mode"] == "brush" and s[0]["points"] == [(9, 9), (10, 10)])
s = pure.parse_strokes("brush:20:1.0:90,90;200,200;-5,0", W, H)
check("越界点丢弃", len(s) == 1 and s[0]["points"] == [(90, 90)])
check("空串", pure.parse_strokes("", W, H) == [])
check("多笔分隔", len(pure.parse_strokes("brush:10:1.0:1,1|erase:10:1.0:2,2", W, H)) == 2)
check("空笔被丢弃", pure.parse_strokes("brush:10:1.0:200,200", W, H) == [])

# ── parse_state_strokes / build_brush_data 往返 ──
state = {"src_w": 100, "src_h": 100, "brush_size": 20,
         "strokes": [{"mode": "brush", "size": 20, "points": [[10, 10], [500, 500]]},
                     {"mode": "nope", "size": "x", "points": [[5, 5]]}]}
ps = pure.parse_state_strokes(state)
check("state 越界裁剪", ps[0]["points"] == [(10, 10)])
check("state 非法 mode/size 兜底", len(ps) == 2 and ps[1]["mode"] == "brush" and ps[1]["size"] == 20)
wire = pure.build_brush_data([{"mode": "brush", "size": 20, "points": [(10, 10), (20, 20)]}])
check("build_brush_data 形状", wire == "brush:20:1.0:10,10;20,20")
check("往返一致", pure.parse_strokes(wire, W, H)[0]["points"] == [(10, 10), (20, 20)])

# ── rasterize_strokes ──
m = pure.rasterize_strokes([], 8, 8)
check("空笔全黑", m.shape == (8, 8) and np.allclose(m, 0.0))
m = pure.rasterize_strokes([{"mode": "brush", "size": 3, "points": [(4, 4)]}], 9, 9)
check("单点印章中心白", m[4, 4] == 1.0)
check("单点印章远处黑", m[0, 0] == 0.0 and m.dtype == np.float32)
m = pure.rasterize_strokes([{"mode": "brush", "size": 3, "points": [(1, 4), (7, 4)]}], 9, 9)
check("线段连续", all(m[4, x] == 1.0 for x in range(1, 8)))
m = pure.rasterize_strokes([
    {"mode": "brush", "size": 5, "points": [(4, 4)]},
    {"mode": "erase", "size": 5, "points": [(4, 4)]},
], 9, 9)
check("erase 清零", m[4, 4] == 0.0 and np.allclose(m, 0.0))

# ── execute()：磁盘源 + 缺源退化 ──
from PIL import Image

_crop_dir = os.path.join(tmp_in, "sfnodes_crop")
os.makedirs(_crop_dir, exist_ok=True)
Image.new("RGB", (6, 5), (255, 0, 0)).save(os.path.join(_crop_dir, "crop_src_brushx.png"), "PNG")

node = mod.SFImageBrushMask()
payload = json.dumps({
    "src_path": "sfnodes_crop/crop_src_brushx.png",
    "src_w": 6, "src_h": 5, "brush_size": 3,
    "strokes": [{"mode": "brush", "size": 3, "points": [[2, 2]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFBrushMaskJson=payload)
check("execute 尺寸取源图", (w, h) == (6, 5))
check("execute 图形状 [1,5,6,3]", np.asarray(img_t).shape == (1, 5, 6, 3))
check("execute 源图红像素", np.allclose(np.asarray(img_t)[0, 0, 0, 0], 1.0))
check("execute 遮罩形状 [1,5,6]", np.asarray(mask_t).shape == (1, 5, 6))
check("execute 笔触处白", np.asarray(mask_t)[0, 2, 2] == 1.0)
check("execute 远处黑", np.asarray(mask_t)[0, 0, 0] == 0.0)
check("execute filename=src_path", fname == "sfnodes_crop/crop_src_brushx.png")

img_t, mask_t, w, h, fname = node.execute(SFBrushMaskJson=json.dumps({"src_path": ""}))
check("无源默认 512", (w, h) == (512, 512) and np.asarray(img_t).shape == (1, 512, 512, 3))
check("无源遮罩全黑", np.allclose(np.asarray(mask_t), 0.0))
check("无源 filename 空串", fname == "")

img_t, mask_t, w, h, fname = node.execute()
check("空状态默认尺寸", (w, h) == (512, 512))
check("缺源 filename 透传", node.execute(
    SFBrushMaskJson=json.dumps({"src_path": "sfnodes_crop/missing.png"}))[4] == "sfnodes_crop/missing.png")

# ── IS_CHANGED：笔触进键、预览字段不进键 ──
key_a = mod.SFImageBrushMask.IS_CHANGED(SFBrushMaskJson=payload)
key_b = mod.SFImageBrushMask.IS_CHANGED(SFBrushMaskJson=json.dumps({
    "src_path": "sfnodes_crop/crop_src_brushx.png",
    "src_w": 6, "src_h": 5, "brush_size": 3,
    "strokes": [{"mode": "brush", "size": 3, "points": [[3, 3]]}],
}))
check("IS_CHANGED 笔触变化键变化", key_a != key_b)
key_prev = mod.SFImageBrushMask.IS_CHANGED(SFBrushMaskJson=json.dumps({
    "src_path": "sfnodes_crop/crop_src_brushx.png",
    "src_w": 6, "src_h": 5, "brush_size": 3,
    "strokes": [{"mode": "brush", "size": 3, "points": [[2, 2]]}],
    "brush_opacity": 0.9, "brush_color": "0,0,255", "eraser_color": "0,255,0",
}))
check("IS_CHANGED 预览字段不进键", key_a == key_prev)
check("IS_CHANGED 无源返回状态键", "|" in mod.SFImageBrushMask.IS_CHANGED(SFBrushMaskJson="{}"))

# ── 结果 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
