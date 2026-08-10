# SFLoraPlot / SFLoraPlotImageSaver 测试（Python 直接运行：python tests/test_lora_plot.py）
# 覆盖：
#   - lora_plot 纯逻辑：sanitize_filename（路径/非法字符/空）、build/parse
#     metadata 双向（下划线文件名）、color_to_rgba（命名/hex6/hex8/rgb/未知）、
#     pick_font（拉丁/CJK 判定 + 缓存）、add_text_overlay（尺寸不变/输入不改/
#     文字盒像素）
#   - lora_cache 修剪：note_applied 峰值逐出、trim last/all/none 语义
#   - 节点结构：SFLoraPlot 类、INPUT_TYPES（hidden LoraLoaderState）、
#     OUTPUT_IS_LIST、apply 全链路（mock comfy/folder_paths：开关过滤、
#     缺失跳过、全关 raise、列表并行、cacheMode 修剪、强度驱动 metadata）
#   - SFLoraPlotImageSaver：单帧/多帧标注、metadata 广播、长度不匹配 raise
import importlib.util
import os
import sys
import tempfile
import types

import numpy as np
from PIL import Image

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = []

def check(name, cond):
    if cond:
        print(f"PASS: {name}")
    else:
        failures.append(name)
        print(f"FAIL: {name}")

# ── mock comfy / torch / folder_paths（本机无 ComfyUI 运行时）────────────────
comfy = types.ModuleType("comfy"); comfy.__path__ = []; sys.modules["comfy"] = comfy
comfy_utils = types.ModuleType("comfy.utils"); sys.modules["comfy.utils"] = comfy_utils
comfy_sd = types.ModuleType("comfy.sd"); sys.modules["comfy.sd"] = comfy_sd
comfy.utils = comfy_utils
comfy.sd = comfy_sd

torch = types.ModuleType("torch"); sys.modules["torch"] = torch
torchvision = types.ModuleType("torchvision"); sys.modules["torchvision"] = torchvision
tv_transforms = types.ModuleType("torchvision.transforms"); sys.modules["torchvision.transforms"] = tv_transforms
tv_v2 = types.ModuleType("torchvision.transforms.v2"); sys.modules["torchvision.transforms.v2"] = tv_v2

load_calls = []
apply_calls = []

def fake_load_torch_file(path, safe_load=True, return_metadata=False):
    load_calls.append(path)
    if return_metadata:
        return {"tensor": 1}, None
    return {"tensor": 1}

def fake_load_lora_for_models(model, clip, lora, sm, sc, lora_metadata=None):
    apply_calls.append((sm, sc))
    return ("m" + str(model), "c" + str(clip))

comfy_utils.load_torch_file = fake_load_torch_file
comfy_utils.common_upscale = lambda *a, **k: None
comfy_sd.load_lora_for_models = fake_load_lora_for_models

folder_paths = types.ModuleType("folder_paths"); sys.modules["folder_paths"] = folder_paths

LORAS_DIR = tempfile.mkdtemp(prefix="sf_lora_plot_test_")

def fake_get_full_path(folder, name):
    if folder != "loras":
        return None
    return os.path.join(LORAS_DIR, name.replace("/", os.sep))

folder_paths.get_full_path = fake_get_full_path

# ── 注册 sfnodes 包结构（相对导入需要）──────────────────────────────────────
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.model"); pkg3.__path__ = [os.path.join(root, "nodes", "model")]; sys.modules["sfnodes.nodes.model"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg4

def load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

load_as("sfnodes.sf_utils.logger", os.path.join(root, "sf_utils", "logger.py"))
plot_utils = load_as("sf_utils_lora_plot", os.path.join(root, "sf_utils", "lora_plot.py"))
cache_utils = load_as("sf_utils_lora_cache", os.path.join(root, "sf_utils", "lora_cache.py"))
load_as("sfnodes.sf_utils.lora_plot", os.path.join(root, "sf_utils", "lora_plot.py"))
load_as("sfnodes.sf_utils.lora_cache", os.path.join(root, "sf_utils", "lora_cache.py"))
load_as("sfnodes.sf_utils.lora_reader", os.path.join(root, "sf_utils", "lora_reader.py"))
load_as("sfnodes.sf_utils.image_convert", os.path.join(root, "sf_utils", "image_convert.py"))

node_mod = load_as("sfnodes.nodes.model.lora_plot", os.path.join(root, "nodes", "model", "lora_plot.py"))

# ── sanitize_filename / metadata 双向 ──
sf_ = plot_utils.sanitize_filename
check("sanitize 剥路径", sf_("models/loras/foo.safetensors") == "foo")
check("sanitize 非法字符替换", sf_("lo<ra>|x?*.safetensors") == "lo_ra__x__")
check("sanitize 首尾点空格", sf_("  .hidden.safetensors") == "hidden")
check("sanitize 版本号保留点", sf_("MoXin_v1.0.safetensors") == "MoXin_v1.0")
check("sanitize 空回退 lora", sf_("...") == "lora" and sf_(None) == "lora")

bm_ = plot_utils.build_metadata
check("build_metadata 基本", bm_("a/b.safetensors", 0.8) == "b_0.8")
check("build_metadata 强度字符串化", bm_("c.ckpt", 1.0) == "c_1.0")
pm_ = plot_utils.parse_metadata
check("parse_metadata 基本", pm_("b_0.8") == ("b", "0.8"))
check("parse_metadata 文件名含下划线", pm_("my_lora_1.0") == ("my_lora", "1.0"))
check("parse_metadata 无下划线", pm_("plain") == ("plain", ""))
check("parse_metadata 空", pm_(None) == ("", "") and pm_("") == ("", ""))
check("parse/build 双向", pm_(bm_("dir/中文 名.safetensors", 0.5)) == ("中文 名", "0.5"))

# ── color_to_rgba ──
cr_ = plot_utils.color_to_rgba
check("color 命名色", cr_("white", 0.8) == (255, 255, 255, 204))
check("color hex6", cr_("#ff8000", 1.0) == (255, 128, 0, 255))
check("color hex8 覆盖 alpha", cr_("#ff000080", 1.0) == (255, 0, 0, 128))
check("color rgb()", cr_("rgb(1,2,3)", 0.5) == (1, 2, 3, 127))
check("color 未知回退白", cr_("nope", 1.0) == (255, 255, 255, 255))
check("color alpha 钳制", cr_("red", 1.7)[3] == 255 and cr_("red", -0.2)[3] == 0)
check("color 大小写无关", cr_("WHITE", 1.0) == (255, 255, 255, 255))

# ── pick_font（缓存 + CJK 判定; 弱断言——本机字体集不可控）──
pf_ = plot_utils.pick_font
f1 = pf_(20)
check("pick_font 拉丁非空", f1 is not None)
check("pick_font 同尺寸缓存", pf_(20) is f1)
check("pick_font CJK 命中不同缓存", pf_(24, "中文测试") is not None and pf_(24) is not f1)
check("pick_font 字号生效", int(pf_(18).size) == 18)

# ── add_text_overlay ──
img = Image.new("RGB", (400, 150), (0, 0, 0))
out = plot_utils.add_text_overlay(img, "my_lora\nStrength: 0.8", "white", "black", 24, 10, 0.8)
check("overlay 尺寸不变", out.size == img.size)
check("overlay 不改输入", list(img.getdata())[0] == (0, 0, 0))
rgb_out = np.array(out.convert("RGB"))
check("overlay 左下角未动", rgb_out[110:150, 0:40].max() == 0)
diff = (rgb_out != np.array(img))[..., :3].any(axis=2)
ys, xs = np.where(diff)
check("overlay 有差异像素", len(xs) > 0)
check("overlay 差异集中在右上角", len(xs) == 0 or xs.min() >= 180 and ys.max() <= 90)

# ── LoraCache 修剪 ──
c = cache_utils.LoraCache()
c.store("p1", 1); c.store("p2", 2)
r = c.note_applied("p1", "last", None)
check("cache note_applied 返回路径", r == "p1")
r = c.note_applied("p2", "last", r)
check("cache last 逐出上一行", c.get("p1") is None and c.get("p2") is not None)
c.trim("last", {"p2"}, r)
check("cache trim last 保留最后", c.get("p2") is not None and len(c._data) == 1)

c = cache_utils.LoraCache()
c.store("p1", 1); c.store("p2", 2)
c.trim("all", {"p1"}, None)
check("cache trim all 只留使用", c.get("p1") is not None and c.get("p2") is None)

c = cache_utils.LoraCache()
c.store("p1", 1); c.store("p2", 2)
c.trim("none", {"p1"}, None)
check("cache trim none 清空", c._data == {} and c._last_path is None)

c = cache_utils.LoraCache()
c.store("p1", 1)
c.trim("last", set(), None)  # 本 run 没用任何东西, 保留条目也不在 used 里
check("cache trim last 空 run 清空", c._data == {})

# ── 节点结构 ──
cls_ = node_mod.SFLoraPlot
types_ = cls_.INPUT_TYPES()
check("plot INPUT_TYPES required model/clip", "model" in types_["required"] and "clip" in types_["required"])
check("plot INPUT_TYPES hidden LoraLoaderState", types_["hidden"].get("LoraLoaderState") is not None)
check("plot RETURN_TYPES", cls_.RETURN_TYPES == ("MODEL", "CLIP", "STRING"))
check("plot OUTPUT_IS_LIST", cls_.OUTPUT_IS_LIST == (True, True, True))
check("plot RETURN_NAMES metadata", cls_.RETURN_NAMES[2] == "metadata")
check("plot CATEGORY", cls_.CATEGORY == "sfnodes/model")
check("plot DESCRIPTION", isinstance(cls_.DESCRIPTION, str) and len(cls_.DESCRIPTION) > 20)

saver_cls = node_mod.SFLoraPlotImageSaver
s_types = saver_cls.INPUT_TYPES()["required"]
check("saver INPUT_TYPES 字段", {"images", "metadata", "text_color", "background_color", "font_size", "padding", "opacity"} <= set(s_types))
check("saver OUTPUT_IS_LIST", saver_cls.OUTPUT_IS_LIST == (True,))
check("saver CATEGORY", saver_cls.CATEGORY == "sfnodes/model")
check("saver COLOR_OPTIONS 与纯逻辑一致", list(saver_cls.COLOR_OPTIONS) == plot_utils.COLOR_OPTIONS)

# ── SFLoraPlot.apply 全链路（mock 文件 + 计数）──
for fn in ("a.safetensors", "b.safetensors", "c.safetensors"):
    with open(os.path.join(LORAS_DIR, fn), "wb") as f:
        f.write(b"x")

def state(loras):
    return {"loras": loras, "sep": ", ", "cacheMode": "last"}

node = node_mod.SFLoraPlot()
load_calls.clear(); apply_calls.clear()
rows = [
    {"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8},
    {"name": "b.safetensors", "on": False, "sm": 0.5, "sc": 0.5},  # 关 -> 跳过
    {"name": "c.safetensors", "on": True, "sm": 1.0, "sc": 1.0},
]
m, c, meta = node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(state(rows)))
check("apply 只处理开启行", len(m) == 2 and len(c) == 2 and len(meta) == 2)
check("apply 从同一基础克隆", m == ["m" + "model", "m" + "model"] and c == ["c" + "clip", "c" + "clip"])
check("apply 强度各用各的", apply_calls == [(0.8, 0.8), (1.0, 1.0)])
check("apply metadata 并行", meta == ["a_0.8", "c_1.0"])
check("apply 缓存命中不重复加载", len(load_calls) == 2)
check("apply clip 强度跟随 sm", all(sc == sm for sm, sc in apply_calls))

# 再次运行 -> "last" 模式跨 run 只保留最近一个文件：首行 a 重新加载，c 命中
load_calls.clear()
node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(state(rows)))
check("apply last 二次运行只重读首行", len(load_calls) == 1)

# 缺失文件行跳过
load_calls.clear(); apply_calls.clear()
rows_missing = [
    {"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8},
    {"name": "nope.safetensors", "on": True, "sm": 0.5, "sc": 0.5},
]
m2, c2, meta2 = node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(state(rows_missing)))
check("apply 缺失行跳过且并行", len(m2) == 1 and meta2 == ["a_0.8"])

# 全关 -> raise
try:
    node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(state(
        [{"name": "a.safetensors", "on": False, "sm": 0.8, "sc": 0.8}])))
    check("apply 全关 raise", False)
except ValueError:
    check("apply 全关 raise", True)

# 坏状态字符串不炸（空行列表 -> 与全关同语义 raise）
try:
    node.apply("model", "clip", LoraLoaderState="not json")
    check("apply 坏状态 raise", False)
except ValueError:
    check("apply 坏状态 raise", True)

# cacheMode all -> 整栈跨 run 保留，连续运行零加载
load_calls.clear()
node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(
    {"loras": [{"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8}], "cacheMode": "all"}))
load_calls.clear()
node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(
    {"loras": [{"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8}], "cacheMode": "all"}))
check("apply cacheMode all 连续运行零加载", len(load_calls) == 0)

# cacheMode none -> 缓存清空
load_calls.clear()
node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(
    {"loras": [{"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8}], "cacheMode": "none"}))
check("apply cacheMode none 清空缓存", node._cache._data == {})
load_calls.clear()
node.apply("model", "clip", LoraLoaderState=__import__("json").dumps(
    {"loras": [{"name": "a.safetensors", "on": True, "sm": 0.8, "sc": 0.8}], "cacheMode": "all"}))
check("apply cacheMode all 保留", node._cache.get(os.path.join(LORAS_DIR, "a.safetensors")) is not None)

# ── SFLoraPlotImageSaver ──
# image_convert 依赖 torch（本机无）——替换为纯 numpy 实现再测节点逻辑。
def fake_tensor2pil(image):
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def fake_pil2tensor(image):
    return (np.array(image).astype(np.float32)[np.newaxis, ...] / 255.0)

node_mod.tensor2pil = fake_tensor2pil
node_mod.pil2tensor = fake_pil2tensor

saver = node_mod.SFLoraPlotImageSaver()
img_arr = np.zeros((1, 64, 64, 3), dtype=np.float32)
(out,) = saver.save_with_overlay(img_arr, "foo_0.8")
check("saver 单帧单 meta", len(out) == 1 and out[0].shape == (1, 64, 64, 3))
check("saver 标注生效", out[0][0, 40:64, 40:64].max() > 0)

img3 = np.zeros((3, 64, 64, 3), dtype=np.float32)
(out3,) = saver.save_with_overlay(img3, "foo_0.8")
check("saver 多帧广播单 meta", len(out3) == 3)

(outl,) = saver.save_with_overlay([img_arr, img_arr], ["foo_0.8", "bar_1.0"])
check("saver 列表双 meta", len(outl) == 2)

try:
    saver.save_with_overlay([img_arr, img_arr], ["one", "two", "three"])
    check("saver 数量不匹配 raise", False)
except ValueError:
    check("saver 数量不匹配 raise", True)

(outz,) = saver.save_with_overlay(img_arr, "中文名_0.5")
check("saver 中文名解析", outz is not None)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASSED")
