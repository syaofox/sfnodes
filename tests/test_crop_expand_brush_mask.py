# SFImageCropExpandBrushMask 后端逻辑测试（Node/Python 直接运行：python3 tests/test_crop_expand_brush.py）
# 覆盖：
#   - 结构：类、CATEGORY、RETURN_TYPES/NAMES、DESCRIPTION、hidden 声明、非 OUTPUT_NODE
#   - 注册键（根 __init__.py 文本断言）
#   - 纯函数：_compose_expand(overlay=) 联合语义 / _clamp_crop / _state_key
#   - execute()：磁盘源 + 笔触联合（brush/erase/fill/框外裁掉）+ 缺源退化
#   - IS_CHANGED：源文件 mtime/size、crop/fill/strokes 进键、预览字段不进键
# mock：torch / aiohttp / folder_paths（numpy/PIL 本机真实可用）；同链路加载
# nodes/image/crop.py（_safe_join / load_src_rgb）→ crop_expand.py → 本节点。
import importlib.util
import json
import os
import sys
import tempfile
import types

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock torch（含 nn.functional）──
torch = types.ModuleType("torch")
torch.float32 = "float32"
torch.Tensor = type("Tensor", (), {})
torch.zeros = lambda *a, **k: "zeros"
torch.ones = lambda *a, **k: "ones"
torch.from_numpy = lambda a: a  # numpy 数组 [None,] 合法，mock 保持轻量
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
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

# ── mock folder_paths ──
tmp_in = tempfile.mkdtemp(prefix="sf_cebm_input_")
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: tmp_in
fp.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
sys.modules["folder_paths"] = fp

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils.brush_mask import ...）──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3
pkg_u = types.ModuleType("sfnodes.sf_utils"); pkg_u.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg_u


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(root, "nodes", "image", filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# 先加载 crop.py（_safe_join / load_src_rgb），再 crop_expand.py（合成纯函数）
_load("sfnodes.nodes.image.crop", "crop.py")
_load("sfnodes.nodes.image.crop_expand", "crop_expand.py")
mod = _load("sfnodes.nodes.image.crop_expand_brush_mask", "crop_expand_brush_mask.py")

# ── 结构断言 ──
check("SFImageCropExpandBrushMask 已加载", hasattr(mod, "SFImageCropExpandBrushMask"))
check("CATEGORY", mod.SFImageCropExpandBrushMask.CATEGORY == "sfnodes/image")
check("DESCRIPTION", isinstance(mod.SFImageCropExpandBrushMask.DESCRIPTION, str)
      and len(mod.SFImageCropExpandBrushMask.DESCRIPTION) > 0)
check("非 OUTPUT_NODE", getattr(mod.SFImageCropExpandBrushMask, "OUTPUT_NODE", False) is False)
check("RETURN_TYPES", mod.SFImageCropExpandBrushMask.RETURN_TYPES == ("IMAGE", "MASK", "INT", "INT", "STRING"))
check("RETURN_NAMES", mod.SFImageCropExpandBrushMask.RETURN_NAMES == ("image", "mask", "width", "height", "filename"))

it = mod.SFImageCropExpandBrushMask.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("hidden 声明 SFCropExpandBrushMaskJson", it["hidden"]["SFCropExpandBrushMaskJson"][0] == "STRING")
check("hidden 默认 {}", it["hidden"]["SFCropExpandBrushMaskJson"][1]["default"] == "{}")

# 注册键一致性（根 __init__.py 文本检查，避免全包导入）
init_src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
check("NODE_CLASS_MAPPINGS 注册", '"SFImageCropExpandBrushMask": SFImageCropExpandBrushMask' in init_src)
check("NODE_DISPLAY_NAME_MAPPINGS 注册", '"SFImageCropExpandBrushMask": "SF Image Crop Expand Brush Mask"' in init_src)
check("导入语句存在", "from .nodes.image.crop_expand_brush_mask import SFImageCropExpandBrushMask" in init_src)

# ── _clamp_crop（自 crop_expand 复用）──
check("_clamp_crop 默认值", mod._clamp_crop({}) == (0, 0, 512, 512))
check("_clamp_crop 超限钳制", mod._clamp_crop(
    {"crop_x": -99999, "crop_y": 99999, "crop_w": 99999, "crop_h": 0}) == (-4096, 4096, 8192, 1))

# ── _compose_expand(overlay=)：扩展区 ∪ 笔触 ──
import numpy as np

src = np.zeros((4, 4, 3), dtype=np.float32)
src[:, :, 0] = 1.0  # 全红 4×4 源图

# 无 overlay（默认）行为不变
img, mask = mod._compose_expand(src, -2, -2, 8, 8, (0, 0, 255))
check("无 overlay 扩展遮罩白", np.allclose(mask[0:2, :], 1.0) and np.allclose(mask[:, 6:8], 1.0))
check("无 overlay 交集遮罩黑", np.allclose(mask[2:6, 2:6], 0.0))

# overlay 在交集内并入（画布坐标 = 源图坐标 + 2）
overlay = np.zeros((4, 4), dtype=np.float32)
overlay[1, 1] = 1.0
img, mask = mod._compose_expand(src, -2, -2, 8, 8, (0, 0, 255), overlay)
check("overlay 点并入遮罩", mask[3, 3] == 1.0)
check("overlay 其余交集仍黑", mask[2:6, 2:6].sum() == 1.0)
check("overlay 不动扩展区", np.allclose(mask[0:2, :], 1.0))

# 无源图时 overlay 忽略（纯填充 + 全白）
img, mask = mod._compose_expand(None, 0, 0, 4, 4, (128, 128, 128), overlay)
check("无源图 overlay 忽略", np.allclose(mask, 1.0))

# overlay 尺寸不匹配 → 防御性忽略（不抛错）
img, mask = mod._compose_expand(src, 0, 0, 4, 4, (0, 0, 0), np.zeros((2, 2), dtype=np.float32))
check("overlay 尺寸不匹配忽略", np.allclose(mask, 0.0))

# include_ext=False（§113 Ext 开关）：扩展区不进遮罩，overlay 并入不变
img, mask = mod._compose_expand(src, -2, -2, 8, 8, (0, 0, 255), overlay, include_ext=False)
check("include_ext=False 扩展区黑", np.allclose(mask[0:2, :], 0.0) and np.allclose(mask[:, 6:8], 0.0))
check("include_ext=False overlay 点仍并入", mask[3, 3] == 1.0 and mask[2:6, 2:6].sum() == 1.0)
img, mask = mod._compose_expand(None, 0, 0, 4, 4, (128, 128, 128), overlay, include_ext=False)
check("include_ext=False 无源全黑（overlay 忽略）", np.allclose(mask, 0.0))

# ── _state_key ──
key_a = mod._state_key({"crop_x": 1, "crop_y": 2, "crop_w": 3, "crop_h": 4, "fill_color": "#000000",
                        "strokes": [], "brush_size": 80})
key_b = mod._state_key({"crop_x": 1, "crop_y": 2, "crop_w": 3, "crop_h": 4, "fill_color": "#000000",
                        "strokes": [{"mode": "brush", "size": 2, "points": [[0, 0]]}], "brush_size": 80})
key_c = mod._state_key({"crop_x": 1, "crop_y": 2, "crop_w": 3, "crop_h": 4, "fill_color": "#ffffff",
                        "strokes": [], "brush_size": 80})
key_ext = mod._state_key({"crop_x": 1, "crop_y": 2, "crop_w": 3, "crop_h": 4, "fill_color": "#000000",
                          "strokes": [], "brush_size": 80, "include_ext": False})
check("_state_key 笔触变化", key_a != key_b)
check("_state_key 填充色变化", key_a != key_c)
check("_state_key Ext 切换（缺省 true vs false）", key_a != key_ext)

# ── execute()：磁盘源 + 笔触联合 ──
from PIL import Image

_crop_dir = os.path.join(tmp_in, "sfnodes_crop")
os.makedirs(_crop_dir, exist_ok=True)
src_img_path = os.path.join(_crop_dir, "cebm_src_testx.png")
Image.new("RGB", (4, 4), (255, 0, 0)).save(src_img_path, "PNG")
SRC_PATH = "sfnodes_crop/cebm_src_testx.png"

node = mod.SFImageCropExpandBrushMask()

# brush 笔触落在交集内 → 该点变白；erase 再擦回黑
state = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff",
    "brush_size": 2,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state)
m = np.asarray(mask_t)
check("execute 输出尺寸", (w, h) == (8, 8) and m.shape == (1, 8, 8))
check("execute 交集红像素", np.allclose(np.asarray(img_t)[0][4, 4, 0], 1.0))
check("execute brush 点并入（画布 4,4）", m[0, 4, 4] == 1.0)
# size 2 → 半径 1（后端 max(1, size//2)）：十字 5 像素，画布 (3..5, 3..5)
check("execute brush 仅覆盖印章像素", m[0, 2:6, 2:6].sum() == 5.0)
check("execute 扩展区仍白", np.allclose(m[0, 0:2, :], 1.0))
check("execute filename=src_path", fname == SRC_PATH)

# erase 笔触擦回（同一列表按画序）
state_erase = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff",
    "brush_size": 2,
    "strokes": [
        {"mode": "brush", "size": 2, "points": [[2, 2]]},
        {"mode": "erase", "size": 2, "points": [[2, 2]]},
    ],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_erase)
check("execute erase 擦除交集笔触", np.allclose(np.asarray(mask_t)[0, 2:6, 2:6], 0.0))
check("execute erase 不动扩展区", np.allclose(np.asarray(mask_t)[0, 0:2, :], 1.0))

# fill 笔触覆盖全图 → 交集整块变白
state_fill = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#000000",
    "brush_size": 80,
    "strokes": [{"mode": "fill", "size": 0, "points": [[0, 0], [3, 0], [3, 3], [0, 3]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_fill)
check("execute fill 覆盖交集", np.allclose(np.asarray(mask_t), 1.0))

# fill_erase 从已有 fill 中打洞（Eraser 模式识别结果；画序在后）
state_fill_erase = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#000000",
    "brush_size": 80,
    "strokes": [
        {"mode": "fill", "size": 0, "points": [[0, 0], [3, 0], [3, 3], [0, 3]]},
        {"mode": "fill_erase", "size": 0, "points": [[1, 1], [2, 1], [2, 2], [1, 2]]},
    ],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_fill_erase)
mf = np.asarray(mask_t)
check("execute fill_erase 打洞（画布 3,3 黑）", mf[0, 3, 3] == 0.0)
check("execute fill_erase 洞外仍白（画布 2,2）", mf[0, 2, 2] == 1.0)
check("execute fill_erase 不动扩展区", np.allclose(mf[0, 0:2, :], 1.0))

# 笔触落在裁剪框外 → 被裁掉（crop 只含左上 2×2）
state_clip = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": 0, "crop_y": 0, "crop_w": 2, "crop_h": 2,
    "fill_color": "#000000",
    "brush_size": 2,
    "strokes": [{"mode": "brush", "size": 2, "points": [[3, 3]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_clip)
check("execute 框外笔触裁掉", np.allclose(np.asarray(mask_t), 0.0))

# 缺源图退化：笔触忽略 + 纯填充 + 全白
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=json.dumps({
    "src_path": "sfnodes_crop/missing.png",
    "crop_x": 0, "crop_y": 0, "crop_w": 6, "crop_h": 5,
    "strokes": [{"mode": "brush", "size": 2, "points": [[1, 1]]}],
}))
check("缺源缺口尺寸", (w, h) == (6, 5) and np.asarray(mask_t).shape == (1, 5, 6))
check("缺源遮罩全白", np.allclose(np.asarray(mask_t), 1.0))
check("缺源 filename 仍输出 src_path", fname == "sfnodes_crop/missing.png")

# 空状态：默认 512 黑画布 + 全白
img_t, mask_t, w, h, fname = node.execute()
check("空状态默认尺寸", (w, h) == (512, 512))
check("空状态 masked 全白", np.allclose(np.asarray(mask_t), 1.0))
check("空状态 filename 空串", fname == "")

# ── IS_CHANGED ──
state_base = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
})
key_a = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=state_base)
key_b = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=json.dumps({
    "src_path": SRC_PATH,
    "crop_x": 0, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
}))
check("IS_CHANGED 裁剪变化键变化", key_a != key_b)
key_c = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "strokes": [{"mode": "brush", "size": 2, "points": [[1, 1]]}],
}))
check("IS_CHANGED 笔触变化键变化", key_a != key_c)
key_d = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
    # 预览/交互字段：不进键
    "brush_opacity": 0.9, "brush_color": "255,0,0", "brush_mode": "erase",
    "aspect_ratio": "16:9", "custom_w": 3, "custom_h": 4,
}))
check("IS_CHANGED 预览字段不进键", key_a == key_d)
key_e = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson="{}")
check("IS_CHANGED 无源返回状态键", key_e == "0:0:512:512:|" + "|||80|[]|inv=0|ext=1")

# 反选：合体节点 = 扩展区 ∪ (1 - 笔触)，扩展区不被反选取消
state_inv = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "invert": True,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_inv)
mi = np.asarray(mask_t)
check("反选：交集内非笔触点变白", mi[0, 2, 3] == 1.0)
check("反选：笔触点变黑", mi[0, 4, 4] == 0.0)
check("反选：扩展区仍白", np.allclose(mi[0, 0:2, :], 1.0))
key_inv = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=state_inv)
check("反选进 IS_CHANGED 键", key_inv != key_a)

# ── Ext 开关（§113）：OFF 时 mask 只含笔触层，扩展区黑 ──
state_noext = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "include_ext": False,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_noext)
mn = np.asarray(mask_t)
check("Ext OFF：扩展区黑", np.allclose(mn[0, 0:2, :], 0.0) and np.allclose(mn[0, :, 6:8], 0.0))
check("Ext OFF：笔触点仍白（画布 4,4）", mn[0, 4, 4] == 1.0)
check("Ext OFF：交集非笔触黑（画布 2,2）", mn[0, 2, 2] == 0.0)
check("Ext OFF：输出图像不变（交集红像素）", np.allclose(np.asarray(img_t)[0][4, 4, 0], 1.0))

# Ext OFF + 反选：mask = 1 - 笔触（与画笔节点同语义）
state_noext_inv = json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2,
    "include_ext": False, "invert": True,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandBrushMaskJson=state_noext_inv)
mni = np.asarray(mask_t)
check("Ext OFF+反选：笔触点黑", mni[0, 4, 4] == 0.0)
check("Ext OFF+反选：非笔触交集白", mni[0, 2, 2] == 1.0)
check("Ext OFF+反选：扩展区仍黑", np.allclose(mni[0, 0:2, :], 0.0))

# Ext 进 IS_CHANGED 键；缺省与显式 true 同键（存量工作流不误重跑）
key_noext = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=state_noext)
key_true = mod.SFImageCropExpandBrushMask.IS_CHANGED(SFCropExpandBrushMaskJson=json.dumps({
    "src_path": SRC_PATH,
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff", "brush_size": 2, "include_ext": True,
    "strokes": [{"mode": "brush", "size": 2, "points": [[2, 2]]}],
}))
check("Ext 切换进 IS_CHANGED 键", key_noext != key_a)
check("Ext 缺省 ≡ 显式 true", key_true == key_a)

# 结果
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
