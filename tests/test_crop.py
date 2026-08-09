# SFImageCrop / SFImageUncrop 后端逻辑测试（Node/Python 直接运行：python tests/test_crop.py）
# 覆盖：
#   - 结构：两个类、CATEGORY、RETURN_TYPES/NAMES、注册键、OUTPUT_NODE
#   - 纯函数：_sanitize_id（路径穿越清洗）、_safe_join（绝对/UNC/.. 拒绝）、
#     _decode_image（dataURL 解码）、_rect_from_meta（绝对像素 + 越界 clamp）
#   - _CropOptionalInputs：image/mask 具体类型 + 其余 any_type
# mock：torch / aiohttp / folder_paths（numpy/PIL 本机真实可用）
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
tmp_in = tempfile.mkdtemp(prefix="sf_crop_input_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_input_directory = lambda: tmp_in
folder_paths.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
sys.modules["folder_paths"] = folder_paths

# ── 注册 sfnodes 包结构（相对导入 from ...sf_utils.common import AnyType）──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.image"); pkg3.__path__ = [os.path.join(root, "nodes", "image")]; sys.modules["sfnodes.nodes.image"] = pkg3

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.crop",
    os.path.join(root, "nodes", "image", "crop.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# ── 结构断言 ──
check("SFImageCrop 已加载", hasattr(mod, "SFImageCrop"))
check("SFImageUncrop 已加载", hasattr(mod, "SFImageUncrop"))
check("CATEGORY", mod.SFImageCrop.CATEGORY == "sfnodes/image" and mod.SFImageUncrop.CATEGORY == "sfnodes/image")
check("Crop DESCRIPTION", isinstance(mod.SFImageCrop.DESCRIPTION, str) and len(mod.SFImageCrop.DESCRIPTION) > 0)
check("Crop OUTPUT_NODE", mod.SFImageCrop.OUTPUT_NODE is True)
check("Uncrop 非 OUTPUT_NODE", getattr(mod.SFImageUncrop, "OUTPUT_NODE", False) is False)
check("Crop RETURN_TYPES", mod.SFImageCrop.RETURN_TYPES == ("IMAGE", "MASK", "SF_CROP_INFO", "INT", "INT"))
check("Uncrop RETURN_TYPES", mod.SFImageUncrop.RETURN_TYPES == ("IMAGE", "MASK", "SF_CROP_INFO"))
check("SF_CROP_INFO 常量", mod.SF_CROP_INFO == "SF_CROP_INFO")

it = mod.SFImageCrop.INPUT_TYPES()
check("Crop required 为空", it["required"] == {})
check("Crop hidden 声明 SFCropJson", it["hidden"]["SFCropJson"][0] == "STRING")
check("Crop optional 含 image", "image" in it["optional"])
check("Crop image 类型 IMAGE", it["optional"]["image"][0] == "IMAGE")
check("Crop image tooltip 中文", "接入上游 IMAGE" in it["optional"]["image"][1]["tooltip"])
check("Crop mask 类型 MASK", it["optional"]["mask"][0] == "MASK")
check("Crop 任意键回退 any_type", it["optional"]["AnythingElse"][0] == "*")

itu = mod.SFImageUncrop.INPUT_TYPES()
check("Uncrop required 含 image", itu["required"]["image"][0] == "IMAGE")
check("Uncrop optional 含 crop_info", itu["optional"]["crop_info"][0] == "SF_CROP_INFO")
check("Uncrop feather 默认 0", itu["optional"]["feather"][1]["default"] == 0)

# ── 纯函数 ──
check("_sanitize_id 清洗路径穿越", mod._sanitize_id("../../etc/passwd", "fallback") == "etcpasswd")
check("_sanitize_id 保留字与连字符", mod._sanitize_id("a-b_c.1", "fb") == "a-b_c1")
check("_sanitize_id 空回退", mod._sanitize_id(None, "fb") == "fb")

_crop_dir = os.path.join(tmp_in, "sfnodes_crop")
os.makedirs(_crop_dir, exist_ok=True)
open(os.path.join(_crop_dir, "crop_src_x.png"), "w").close()

check("_safe_join 合法相对路径", mod._safe_join("crop_src_x.png") is not None)
check("_safe_join 子目录前缀兼容", mod._safe_join("sfnodes_crop/crop_src_x.png") is not None)
check("_safe_join 子目录前缀反斜杠", mod._safe_join("sfnodes_crop\\crop_src_x.png") is not None)
check("_safe_join 不存在的文件", mod._safe_join("nope.png") is None)
check("_safe_join 绝对路径拒绝", mod._safe_join("/etc/passwd") is None)
check("_safe_join .. 穿越拒绝", mod._safe_join("../../x.png") is None)
check("_safe_join UNC 拒绝", mod._safe_join("\\\\host\\share\\x.png") is None)
check("_safe_join 空拒绝", mod._safe_join("") is None)

import base64
from PIL import Image
import io
buf = io.BytesIO()
Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, "PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
dec = mod._decode_image(f"data:image/png;base64,{b64}")
check("_decode_image dataURL", dec is not None and dec.size == (4, 4))
check("_decode_image 裸 base64", mod._decode_image(b64).size == (4, 4))
check("_decode_image 非法", mod._decode_image("not-base64!!") is None)

node = mod.SFImageCrop()
rect = node._rect_from_meta({"crop_x": 10, "crop_y": 20, "crop_w": 100, "crop_h": 50}, 200, 200)
check("_rect_from_meta 正常", rect == (10, 20, 110, 70))
rect = node._rect_from_meta({"crop_x": 150, "crop_y": 150, "crop_w": 100, "crop_h": 100}, 200, 200)
check("_rect_from_meta 越界 clamp", rect == (150, 150, 200, 200))
check("_rect_from_meta 空 meta", node._rect_from_meta({}, 200, 200) is None)
check("_rect_from_meta crop_w=0", node._rect_from_meta({"crop_w": 0}, 200, 200) is None)
check("_rect_from_meta 退化为空", node._rect_from_meta({"crop_x": 10, "crop_y": 10, "crop_w": 0, "crop_h": 0}, 200, 200) is None)
check("_default_mask 尺寸", node._default_mask(64, 32) == "zeros")

m = node._default_mask if False else None
check("_identity_crop_info 非 tensor 回 None", node._identity_crop_info("x") is None)

# ── _crop_meta_from_widget 形状兼容（回归：Vue 下 DOM widget 值形状不定）──
fw = mod._crop_meta_from_widget
check("widget dict {crop_json}", fw({"crop_json": json.dumps({"crop_w": 100})})["crop_w"] == 100)
check("widget dict 直接 meta", fw({"crop_w": 100, "crop_x": 5})["crop_w"] == 100)
check("widget str JSON", fw('{"crop_w": 100}')["crop_w"] == 100)
check("widget str 套层 JSON", fw('{"crop_json": "{\\"crop_w\\": 100}"}')["crop_w"] == 100)
check("widget 坏 JSON", fw("{oops") == {})
check("widget None", fw(None) == {})
check("widget 非 dict/str", fw(123) == {})

# ── 磁盘源执行输出 sf_crop_source（回归：编辑器 Load Image 后节点预览
# 停留旧图——后端不输出源帧，前端 executed 事件无法刷新缓存）──
import json as _json
img_buf = io.BytesIO()
Image.new("RGB", (8, 8), (0, 255, 0)).save(img_buf, "PNG")
with open(os.path.join(_crop_dir, "crop_src_x.png"), "wb") as fh:
    fh.write(img_buf.getvalue())
node2 = mod.SFImageCrop()
disk_meta = {"src_path": "sfnodes_crop/crop_src_x.png",
             "crop_x": 0, "crop_y": 0, "crop_w": 4, "crop_h": 4,
             "doc_w": 8, "doc_h": 8}
res = node2.load_crop(SFCropJson=_json.dumps(disk_meta))
frame = res.get("ui", {}).get("sf_crop_source", [{}])[0] if isinstance(res, dict) else {}
check("磁盘源执行输出 sf_crop_source", isinstance(res, dict) and bool(res.get("ui", {}).get("sf_crop_source")))
check("源帧指向 sfnodes_crop/input", frame.get("filename") == "crop_src_x.png" and
      frame.get("subfolder") == "sfnodes_crop" and frame.get("type") == "input")
res_nosrc = node2.load_crop(SFCropJson=_json.dumps({"crop_w": 4}))
check("无 src_path 不输出源帧", not (isinstance(res_nosrc, dict) and res_nosrc.get("ui")))

# ── 注册键一致性（AST）──
import ast
src = open(os.path.join(root, "__init__.py")).read()
tree = ast.parse(src)
classmap = dispmap = None
for node_ in ast.walk(tree):
    if isinstance(node_, ast.Assign):
        for t in node_.targets:
            if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                classmap = {k.value for k in node_.value.keys}
            elif isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS":
                dispmap = {k.value for k in node_.value.keys}
check("注册键两字典一致", classmap == dispmap)
check("SFImageCrop 已注册", "SFImageCrop" in (classmap or set()))
check("SFImageUncrop 已注册", "SFImageUncrop" in (classmap or set()))

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
