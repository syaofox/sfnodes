# SFImageCropExpand 后端逻辑测试（Node/Python 直接运行：python tests/test_crop_expand.py）
# 覆盖：
#   - 结构：类、CATEGORY、RETURN_TYPES/NAMES、DESCRIPTION、hidden 声明、非 OUTPUT_NODE
#   - 纯函数：_parse_state、_clamp_crop、_compose_expand（numpy 真实合成）
#   - common._parse_fill_color（提升后的公共实现）
#   - execute()：磁盘源图交集贴回 + mask 语义（白=扩展）+ 缺源退化
#   - IS_CHANGED：含源文件 mtime/size、状态变化键变化
# mock：torch / aiohttp / folder_paths（numpy/PIL 本机真实可用）；同链路加载
# nodes/image/crop.py（crop_expand 复用其 _safe_join）。
import importlib.util
import io
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
tmp_in = tempfile.mkdtemp(prefix="sf_cropx_input_")
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: tmp_in
fp.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
sys.modules["folder_paths"] = fp

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils.common import ...）──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(root, "nodes", "image", filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# 先加载 crop.py（crop_expand 复用 _safe_join），再加载 crop_expand.py
_load("sfnodes.nodes.image.crop", "crop.py")
mod = _load("sfnodes.nodes.image.crop_expand", "crop_expand.py")

# ── 结构断言 ──
check("SFImageCropExpand 已加载", hasattr(mod, "SFImageCropExpand"))
check("CATEGORY", mod.SFImageCropExpand.CATEGORY == "sfnodes/image")
check("DESCRIPTION", isinstance(mod.SFImageCropExpand.DESCRIPTION, str) and len(mod.SFImageCropExpand.DESCRIPTION) > 0)
check("非 OUTPUT_NODE", getattr(mod.SFImageCropExpand, "OUTPUT_NODE", False) is False)
check("RETURN_TYPES", mod.SFImageCropExpand.RETURN_TYPES == ("IMAGE", "MASK", "INT", "INT", "STRING"))
check("RETURN_NAMES", mod.SFImageCropExpand.RETURN_NAMES == ("image", "mask", "width", "height", "filename"))

it = mod.SFImageCropExpand.INPUT_TYPES()
check("required 为空", it["required"] == {})
check("optional 为空", it.get("optional", {}) == {})
check("hidden 声明 SFCropExpandJson", it["hidden"]["SFCropExpandJson"][0] == "STRING")
check("hidden 默认 {}", it["hidden"]["SFCropExpandJson"][1]["default"] == "{}")

# 注册键一致性（根 __init__.py 文本检查，避免全包导入）
init_src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
check("NODE_CLASS_MAPPINGS 注册", '"SFImageCropExpand": SFImageCropExpand' in init_src)
check("NODE_DISPLAY_NAME_MAPPINGS 注册", '"SFImageCropExpand": "SF Image Crop Expand"' in init_src)
check("导入语句存在", "from .nodes.image.crop_expand import SFImageCropExpand" in init_src)

# ── _parse_fill_color（sf_utils/common.py 公共实现，masks.py 同源）──
pkg_u = types.ModuleType("sfnodes.sf_utils"); pkg_u.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg_u
spec_u = importlib.util.spec_from_file_location(
    "sfnodes.sf_utils.common", os.path.join(root, "sf_utils", "common.py"))
common = importlib.util.module_from_spec(spec_u)
sys.modules[spec_u.name] = common
spec_u.loader.exec_module(common)

check("hex 解析", common._parse_fill_color("#ff8000") == (255, 128, 0))
check("hex 无井号", common._parse_fill_color("000000") == (0, 0, 0))
check("元组直通", common._parse_fill_color([10, 20, 30]) == (10, 20, 30))
check("masks.py 同源导入", "from ...sf_utils.common import _parse_fill_color" in
      open(os.path.join(root, "nodes", "mask", "masks.py"), encoding="utf-8").read())

# ── _parse_state ──
check("_parse_state 合法 json", mod._parse_state('{"crop_x": 5}') == {"crop_x": 5})
check("_parse_state 非法 json", mod._parse_state("not-json") == {})
check("_parse_state 空串", mod._parse_state("") == {})
check("_parse_state None", mod._parse_state(None) == {})
check("_parse_state dict 直通", mod._parse_state({"a": 1}) == {"a": 1})
check("_parse_state 非对象 json", mod._parse_state("[1,2]") == {})

# ── _clamp_crop ──
check("_clamp_crop 默认值", mod._clamp_crop({}) == (0, 0, 512, 512))
check("_clamp_crop 正常", mod._clamp_crop({"crop_x": -10, "crop_y": 20, "crop_w": 100, "crop_h": 200}) == (-10, 20, 100, 200))
check("_clamp_crop 超限钳制", mod._clamp_crop({"crop_x": -99999, "crop_y": 99999, "crop_w": 99999, "crop_h": 0}) == (-4096, 4096, 8192, 1))
check("_clamp_crop 非法值兜底", mod._clamp_crop({"crop_x": "abc", "crop_w": None}) == (0, 0, 512, 512))

# ── _compose_expand（numpy 真实合成）──
import numpy as np

src = np.zeros((4, 4, 3), dtype=np.float32)
src[:, :, 0] = 1.0  # 全红 4×4 源图

# 完全在源图内：全 0 遮罩，像素等于源图
img, mask = mod._compose_expand(src, 1, 1, 2, 2, (0, 255, 0))
check("框内像素等于源图", np.allclose(img[:, :, 0], 1.0))
check("框内遮罩全黑", np.allclose(mask, 0.0))

# 出界扩展：(-2,-2,8,8)，交集 4×4 贴回，四周扩展
img, mask = mod._compose_expand(src, -2, -2, 8, 8, (0, 0, 255))
check("扩展画布尺寸", img.shape == (8, 8, 3) and mask.shape == (8, 8))
check("交集红像素", np.allclose(img[2:6, 2:6, 0], 1.0))
check("交集遮罩黑", np.allclose(mask[2:6, 2:6], 0.0))
check("扩展区填充色", np.allclose(img[0:2, :, 2], 1.0) and np.allclose(img[0:2, :, 0], 0.0))
check("扩展区遮罩白", np.allclose(mask[0:2, :], 1.0) and np.allclose(mask[:, 6:8], 1.0))

# 无源图：纯填充 + 全白遮罩
img, mask = mod._compose_expand(None, 0, 0, 4, 4, (128, 128, 128))
check("无源纯填充", np.allclose(img, 128.0 / 255.0))
check("无源遮罩全白", np.allclose(mask, 1.0))

# 完全出界（不相交）：纯填充 + 全白遮罩
img, mask = mod._compose_expand(src, 100, 100, 8, 8, (0, 0, 255))
check("不相交遮罩全白", np.allclose(mask, 1.0))

# ── execute()：磁盘源 + 缺源退化 ──
from PIL import Image

_crop_dir = os.path.join(tmp_in, "sfnodes_crop")
os.makedirs(_crop_dir, exist_ok=True)
src_img = Image.new("RGB", (4, 4), (255, 0, 0))
src_img.save(os.path.join(_crop_dir, "crop_src_testx.png"), "PNG")

node = mod.SFImageCropExpand()
state = json.dumps({
    "src_path": "sfnodes_crop/crop_src_testx.png",
    "crop_x": -2, "crop_y": -2, "crop_w": 8, "crop_h": 8,
    "fill_color": "#0000ff",
})
img_t, mask_t, w, h, fname = node.execute(SFCropExpandJson=state)
check("execute 输出宽度", (w, h) == (8, 8))
check("execute 图形状 [1,8,8,3]", img_t.shape == (1, 8, 8, 3))
check("execute 遮罩形状 [1,8,8]", mask_t.shape == (1, 8, 8))
check("execute 交集红", np.allclose(np.asarray(img_t)[2:6, 2:6, 0], 1.0))
check("execute 扩展遮罩白", np.allclose(np.asarray(mask_t)[:, 0:2, :], 1.0))
check("execute 交集遮罩黑", np.allclose(np.asarray(mask_t)[2:6, 2:6], 0.0))
check("execute filename=src_path", fname == "sfnodes_crop/crop_src_testx.png")

img_t, mask_t, w, h, fname = node.execute(SFCropExpandJson=json.dumps({
    "src_path": "sfnodes_crop/missing.png", "crop_x": 0, "crop_y": 0, "crop_w": 6, "crop_h": 5,
    "fill_color": "#000000",
}))
check("缺源退化画布尺寸", (w, h) == (6, 5) and np.asarray(img_t).shape == (1, 5, 6, 3))
check("缺源遮罩全白", np.allclose(np.asarray(mask_t), 1.0))
check("缺源 filename 仍输出 src_path", fname == "sfnodes_crop/missing.png")

# 空状态：默认 512 黑画布
img_t, mask_t, w, h, fname = node.execute()
check("空状态默认尺寸", (w, h) == (512, 512) and np.asarray(img_t).shape == (1, 512, 512, 3))
check("空状态默认黑", np.allclose(np.asarray(img_t), 0.0))
check("空状态 filename 空串", fname == "")

# 非法 fill_color 兜底黑
img_t, mask_t, w, h, fname = node.execute(SFCropExpandJson=json.dumps({"fill_color": "zzz", "crop_w": 3, "crop_h": 3}))
check("非法颜色兜底黑", np.allclose(np.asarray(img_t), 0.0))

# ── IS_CHANGED ──
key_a = mod.SFImageCropExpand.IS_CHANGED(SFCropExpandJson=state)
check("IS_CHANGED 含 mtime", "mtime" not in key_a and ":" in key_a)
key_b = mod.SFImageCropExpand.IS_CHANGED(SFCropExpandJson=json.dumps({
    "src_path": "sfnodes_crop/crop_src_testx.png",
    "crop_x": -1, "crop_y": -2, "crop_w": 8, "crop_h": 8, "fill_color": "#0000ff",
}))
check("IS_CHANGED 状态变化键变化", key_a != key_b)
key_c = mod.SFImageCropExpand.IS_CHANGED(SFCropExpandJson="{}")
check("IS_CHANGED 无源返回状态键", key_c == "0:0:512:512:")

# ── 结果 ──
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")