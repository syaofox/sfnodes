# SFLoadImageResize 后端逻辑测试（Node/Python 直接运行：python tests/test_load_image_resize.py）
# 覆盖：
#   - sf_utils.resize_engine 纯引擎（无 ComfyUI 依赖，本机 PIL 直接跑）：
#     8 模式尺寸、mask 语义、snap floor、round-half-up、clamp、未知模式回退
#   - 节点模块（mock torch/folder_paths/node_helpers）：INPUT_TYPES 结构、
#     _parse_state/_parse_orig_name 兜底、注册键
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

# ── 1. resize 引擎（纯函数，无 mock）──
from sf_utils.resize_engine import (  # noqa: E402
    _resize_frame, _apply_snap, _round_half_up, _hex_to_rgb,
    parse_resize_state, RESIZE_DEFAULTS,
)
from PIL import Image  # noqa: E402

img = Image.new("RGB", (100, 50), (255, 0, 0))
mask = Image.new("L", (100, 50), 0)
img1024 = Image.new("RGB", (1024, 1024), (0, 255, 0))

def resize(state, w=100, h=50, img0=img):
    return _resize_frame(img0, mask, dict(RESIZE_DEFAULTS, **state), w, h)

rgb, m, w, h = resize({"mode": "off"})
check("off 直通", (w, h) == (100, 50) and rgb.size == (100, 50))

rgb, m, w, h = resize({"mode": "max_mp", "max_mp": 1.0}, 1024, 1024, img1024)
check("max_mp 1.0 下 1024² 不变", (w, h) == (1024, 1024))

rgb, m, w, h = resize({"mode": "max_mp", "max_mp": 0.5}, 1024, 1024, img1024)
check("max_mp 0.5 下 1024² -> 724²（round-half-up 对齐 JS）", (w, h) == (724, 724))

check("snap floor 724->704 (64)", _apply_snap(724, 724, 64) == (704, 704))
check("snap=0 不变", _apply_snap(723, 723, 0) == (723, 723))
check("round-half-up 62.5->63", _round_half_up(62.5) == 63)
check("round-half-up 62.4->62", _round_half_up(62.4) == 62)
check("hex 解析 #808080", _hex_to_rgb("#808080") == (128, 128, 128))
check("hex 短格式 #abc", _hex_to_rgb("#abc") == (170, 187, 204))
check("hex 非法回退黑", _hex_to_rgb("oops") == (0, 0, 0))

rgb, m, w, h = resize({"mode": "longest_side", "longest_side": 1024, "allow_upscale": False})
check("longest_side 禁放大直通", (w, h) == (100, 50))

rgb, m, w, h = resize({"mode": "scale_factor", "scale_factor": 2.0})
check("scale_factor 2x", (w, h) == (200, 100))

rgb, m, w, h = resize({"mode": "fit_inside", "fit_w": 1000, "fit_h": 40})
check("fit_inside 2000x40 内 fit", (w, h) == (80, 40))

rgb, m, w, h = resize({"mode": "cover", "cover_w": 50, "cover_h": 50, "crop_scale": False})
check("cover 直接裁切 1:1", (w, h) == (50, 50))

rgb, m, w, h = resize({"mode": "cover", "cover_w": 200, "cover_h": 100})
check("cover fill 缩放裁切", (w, h) == (200, 100))

rgb, m, w, h = resize({"mode": "match_ratio", "ratio_w": 4, "ratio_h": 3, "ratio_action": "crop"})
check("match_ratio 4:3 crop 100x50->67x50", (w, h) == (67, 50))

rgb, m, w, h = resize({"mode": "match_ratio", "ratio_w": 4, "ratio_h": 3, "ratio_action": "pad"})
check("match_ratio 4:3 pad 100x50->100x75", (w, h) == (100, 75))

rgb, m, w, h = resize({"mode": "pad", "pad_left": 8, "pad_right": 8, "pad_top": 4, "pad_bottom": 4})
check("pad 尺寸 100x50->116x58", (w, h) == (116, 58))
check("pad mask 边框 255（补丁区）", m.getpixel((0, 0)) == 255)
check("pad mask 原区域 0", m.getpixel((10, 10)) == 0)

rgb, m, w, h = resize({"mode": "scale_factor", "scale_factor": 1000})
check("clamp 上限 16384", w <= 16384 and h <= 16384)

rgb, m, w, h = resize({"mode": "bogus"})
check("未知模式回退 off", (w, h) == (100, 50))

st = parse_resize_state(None, RESIZE_DEFAULTS)
check("parse_resize_state None -> 默认", st["mode"] == "off")
st = parse_resize_state("not json", RESIZE_DEFAULTS)
check("parse_resize_state 坏 JSON -> 默认", st["mode"] == "off")
st = parse_resize_state(json.dumps({"mode": "pad", "evil": 1}), RESIZE_DEFAULTS)
check("parse_resize_state 过滤未知键", st["mode"] == "pad" and "evil" not in st)

# ── 2. 节点模块（mock torch / folder_paths / node_helpers）──
torch = types.ModuleType("torch")
torch.float32 = "float32"
torch.zeros = lambda *a, **k: "zeros"
sys.modules["torch"] = torch

tmp_in = tempfile.mkdtemp(prefix="sf_li_input_")
open(os.path.join(tmp_in, "a.png"), "w").close()
os.makedirs(os.path.join(tmp_in, "Studio1"), exist_ok=True)
open(os.path.join(tmp_in, "Studio1", "b.jpg"), "w").close()
open(os.path.join(tmp_in, "note.txt"), "w").close()

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_input_directory = lambda: tmp_in
folder_paths.filter_files_content_types = lambda files, cts: [f for f in files if not f.endswith(".txt")]
folder_paths.get_annotated_filepath = lambda name: os.path.join(tmp_in, name)
folder_paths.exists_annotated_filepath = lambda name: os.path.isfile(os.path.join(tmp_in, name))
sys.modules["folder_paths"] = folder_paths

node_helpers = types.ModuleType("node_helpers")
node_helpers.pillow = lambda fn, *a, **k: fn(*a, **k)
sys.modules["node_helpers"] = node_helpers

# 注册 sfnodes 包结构，使节点的相对导入（from ...sf_utils.resize_engine import）可解析
_sf_pkg = types.ModuleType("sfnodes")
_sf_pkg.__path__ = [root]
sys.modules["sfnodes"] = _sf_pkg
_sf_nodes_pkg = types.ModuleType("sfnodes.nodes")
_sf_nodes_pkg.__path__ = [os.path.join(root, "nodes")]
sys.modules["sfnodes.nodes"] = _sf_nodes_pkg
_sf_image_pkg = types.ModuleType("sfnodes.nodes.image")
_sf_image_pkg.__path__ = [os.path.join(root, "nodes", "image")]
sys.modules["sfnodes.nodes.image"] = _sf_image_pkg

spec = importlib.util.spec_from_file_location(
    "sfnodes.nodes.image.load_image_resize",
    os.path.join(root, "nodes", "image", "load_image_resize.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

check("模块已加载", hasattr(mod, "SFLoadImageResize"))
check("CATEGORY", mod.SFLoadImageResize.CATEGORY == "sfnodes/image")
check("DESCRIPTION 非空", isinstance(mod.SFLoadImageResize.DESCRIPTION, str) and len(mod.SFLoadImageResize.DESCRIPTION) > 0)
check("RETURN_TYPES", mod.SFLoadImageResize.RETURN_TYPES == ("IMAGE", "MASK", "INT", "INT", "STRING", "INT", "INT"))
check("FUNCTION", mod.SFLoadImageResize.FUNCTION == "load_image")
check("无 OUTPUT_NODE", getattr(mod.SFLoadImageResize, "OUTPUT_NODE", False) is False)

it = mod.SFLoadImageResize.INPUT_TYPES()
check("required 含 image + image_upload", "image" in it["required"] and it["required"]["image"][1].get("image_upload") is True)
vals = it["required"]["image"][0]
check("递归列表含子文件夹", "Studio1/b.jpg" in vals and "a.png" in vals)
check("过滤非图片", "note.txt" not in vals)
check("hidden 含 SFLoadImageResizeState", "SFLoadImageResizeState" in it["hidden"])

cls = mod.SFLoadImageResize
check("_parse_state None -> 默认", mod._parse_state(None)["mode"] == "off")
check("_parse_state 坏 JSON -> 默认", mod._parse_state("{")["mode"] == "off")
check("_parse_state 过滤未知键", "evil" not in mod._parse_state(json.dumps({"evil": 1})))
check("_parse_orig_name 正常", mod._parse_orig_name(json.dumps({"orig_name": "cat.png"})) == "cat.png")
check("_parse_orig_name 缺失", mod._parse_orig_name(json.dumps({"mode": "off"})) == "")
check("_parse_orig_name 坏 JSON", mod._parse_orig_name("{") == "")

check("VALIDATE_INPUTS 存在文件", cls.VALIDATE_INPUTS("a.png") is True)
check("VALIDATE_INPUTS 缺失文件返回错误串", isinstance(cls.VALIDATE_INPUTS("nope.png"), str))

# ── 3. 注册键一致性（AST）──
import ast
src = open(os.path.join(root, "__init__.py")).read()
tree = ast.parse(src)
classmap = dispmap = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                classmap = {k.value for k in node.value.keys}
            elif isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS":
                dispmap = {k.value for k in node.value.keys}
check("注册键两字典一致", classmap == dispmap)
check("SFLoadImageResize 已注册", "SFLoadImageResize" in (classmap or set()))
check("显示名映射", dispmap is not None and "SFLoadImageResize" in dispmap)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
