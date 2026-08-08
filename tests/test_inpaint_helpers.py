# SFInpaintCrop / SFInpaintStitch 后端逻辑测试（Node/Python 直接运行：python tests/test_inpaint_helpers.py）
# 覆盖：
#   - 结构：两个类、CATEGORY、RETURN_TYPES/NAMES、注册键、OUTPUT_NODE、DESCRIPTION
#   - INPUT_TYPES：required 参数、hidden SFInpaintJson、optional image/mask + any_type 回退
#   - 纯函数：merge_params、compute_region（keep/force/free/边界）、mask_bbox、
#     mask_to_np、preprocess_mask、fill_holes（本机无 scipy -> 走 PIL fallback）、
#     resolve_inpaint_mask、_sanitize_id/_safe_join/_decode_image、_inpaint_meta_from_widget
#   - 数值：apply_inpaint_crop（裁剪+遮罩输出）、stitch_back（mask/whole_crop 贴回）
# mock：torch（FakeTensor numpy 代理）/ aiohttp / folder_paths（numpy/PIL 本机真实可用）
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

# ── FakeTensor：numpy 代理，够跑 helper 的裁剪/缝合路径 ──
class FakeTensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def numpy(self):
        return self.data

    def __array__(self, dtype=None):
        return self.data.astype(dtype) if dtype else self.data

    def clone(self):
        return FakeTensor(self.data.copy())

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def contiguous(self):
        return self

    def dim(self):
        return self.data.ndim

    def numel(self):
        return self.data.size

    def view(self, *shape):
        return FakeTensor(self.data.reshape(shape[0] if len(shape) == 1 else shape))

    def new_zeros(self, shape):
        return FakeTensor(np.zeros(tuple(shape), dtype=np.float32))

    def repeat(self, *shape):
        return FakeTensor(np.tile(self.data, tuple(shape)))

    def clamp(self, lo=None, hi=None):
        return FakeTensor(np.clip(self.data, lo, hi))

    def to(self, device=None, dtype=None):
        return self

    def __getattr__(self, name):
        attr = getattr(self.data, name)
        if callable(attr):
            def wrapped(*a, **k):
                r = attr(*a, **k)
                return FakeTensor(r) if isinstance(r, np.ndarray) else r
            return wrapped
        return attr

    def __getitem__(self, k):
        r = self.data[k]
        return FakeTensor(r) if isinstance(r, np.ndarray) else r

    def __setitem__(self, k, v):
        self.data[k] = v.data if isinstance(v, FakeTensor) else v

    def __add__(self, o):
        return FakeTensor(self.data + (o.data if isinstance(o, FakeTensor) else o))

    def __sub__(self, o):
        return FakeTensor(self.data - (o.data if isinstance(o, FakeTensor) else o))

    def __mul__(self, o):
        return FakeTensor(self.data * (o.data if isinstance(o, FakeTensor) else o))

    def __truediv__(self, o):
        return FakeTensor(self.data / (o.data if isinstance(o, FakeTensor) else o))

    def __rsub__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) - self.data)

    def __radd__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) + self.data)

    def __rmul__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) * self.data)

    def __rtruediv__(self, o):
        return FakeTensor((o.data if isinstance(o, FakeTensor) else o) / self.data)

    def __neg__(self):
        return FakeTensor(-self.data)

    def __gt__(self, o):
        return self.data > (o.data if isinstance(o, FakeTensor) else o)

    def __ge__(self, o):
        return self.data >= (o.data if isinstance(o, FakeTensor) else o)

    def __lt__(self, o):
        return self.data < (o.data if isinstance(o, FakeTensor) else o)

    def __le__(self, o):
        return self.data <= (o.data if isinstance(o, FakeTensor) else o)

    def __float__(self):
        return float(self.data)

    def __bool__(self):
        return bool(self.data)

    def __eq__(self, o):
        return self.data == (o.data if isinstance(o, FakeTensor) else o)

# ── mock torch ──
import numpy as np
torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.from_numpy = lambda a: FakeTensor(a)

def _tshape(s):
    return s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s


torch.zeros = lambda *s, **k: FakeTensor(np.zeros(tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.ones = lambda *s, **k: FakeTensor(np.ones(tuple(int(x) for x in _tshape(s)), dtype=np.float32))
torch.arange = lambda n, **k: FakeTensor(np.arange(n))
torch.float32 = "float32"
torch.minimum = lambda a, b: FakeTensor(np.minimum(
    a.data if isinstance(a, FakeTensor) else a, b.data if isinstance(b, FakeTensor) else b))
torch.stack = lambda seq, dim=0: FakeTensor(np.stack(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
torch.cat = lambda seq, dim=0: FakeTensor(np.concatenate(
    [s.data if isinstance(s, FakeTensor) else s for s in seq], axis=dim))
sys.modules["torch"] = torch

# ── mock aiohttp ──
aiohttp = types.ModuleType("aiohttp")
aiohttp.web = types.ModuleType("aiohttp.web")
aiohttp.web.json_response = lambda *a, **k: types.SimpleNamespace(status=200, body=a)
aiohttp.web.Response = types.SimpleNamespace
sys.modules["aiohttp"] = aiohttp
sys.modules["aiohttp.web"] = aiohttp.web

# ── mock folder_paths ──
tmp_in = tempfile.mkdtemp(prefix="sf_inpaint_input_")
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_input_directory = lambda: tmp_in
folder_paths.get_temp_directory = lambda: os.path.join(tmp_in, "temp")
sys.modules["folder_paths"] = folder_paths

# ── 注册 sfnodes 包结构 ──
pkg = types.ModuleType("sfnodes"); pkg.__path__ = [root]; sys.modules["sfnodes"] = pkg
pkg2 = types.ModuleType("sfnodes.nodes"); pkg2.__path__ = [os.path.join(root, "nodes")]; sys.modules["sfnodes.nodes"] = pkg2
pkg3 = types.ModuleType("sfnodes.nodes.inpaint"); pkg3.__path__ = [os.path.join(root, "nodes", "inpaint")]; sys.modules["sfnodes.nodes.inpaint"] = pkg3
pkg4 = types.ModuleType("sfnodes.sf_utils"); pkg4.__path__ = [os.path.join(root, "sf_utils")]; sys.modules["sfnodes.sf_utils"] = pkg4


def load(modpath, modname):
    spec = importlib.util.spec_from_file_location(modname, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


helper = load(os.path.join(root, "sf_utils", "inpaint_helpers.py"), "sfnodes.sf_utils.inpaint_helpers")
mod = load(os.path.join(root, "nodes", "inpaint", "inpaint_editor.py"), "sfnodes.nodes.inpaint.inpaint_editor")

# ── 结构断言 ──
check("SFInpaintCrop 已加载", hasattr(mod, "SFInpaintCrop"))
check("SFInpaintStitch 已加载", hasattr(mod, "SFInpaintStitch"))
check("CATEGORY", mod.SFInpaintCrop.CATEGORY == "sfnodes/inpaint" and mod.SFInpaintStitch.CATEGORY == "sfnodes/inpaint")
check("Crop DESCRIPTION", isinstance(mod.SFInpaintCrop.DESCRIPTION, str) and len(mod.SFInpaintCrop.DESCRIPTION) > 0)
check("Stitch DESCRIPTION", isinstance(mod.SFInpaintStitch.DESCRIPTION, str) and len(mod.SFInpaintStitch.DESCRIPTION) > 0)
check("Crop OUTPUT_NODE", mod.SFInpaintCrop.OUTPUT_NODE is True)
check("Stitch 非 OUTPUT_NODE", getattr(mod.SFInpaintStitch, "OUTPUT_NODE", False) is False)
check("Crop RETURN_TYPES", mod.SFInpaintCrop.RETURN_TYPES == ("IMAGE", "MASK", "SF_CROP_INFO", "INT", "INT"))
check("Stitch RETURN_TYPES", mod.SFInpaintStitch.RETURN_TYPES == ("IMAGE", "IMAGE"))
check("SF_CROP_INFO 一致", helper.SF_CROP_INFO == "SF_CROP_INFO" and mod.SF_CROP_INFO == "SF_CROP_INFO")
check("IS_CHANGED 存在", callable(mod.SFInpaintCrop.IS_CHANGED))

it = mod.SFInpaintCrop.INPUT_TYPES()
check("Crop required 9 键", list(it["required"].keys()) == [
    "size_mode", "target", "multiple", "context_px", "mask_grow",
    "mask_blur", "softness", "blend_mode", "invert_mask"])
check("Crop hidden 声明 SFInpaintJson", it["hidden"]["SFInpaintJson"][0] == "STRING")
check("Crop optional 含 image/mask", it["optional"]["image"][0] == "IMAGE" and it["optional"]["mask"][0] == "MASK")
check("Crop image tooltip 中文", "接入上游 IMAGE" in it["optional"]["image"][1]["tooltip"])
check("Crop 任意键回退 any_type", it["optional"]["AnythingElse"][0] == "*")
check("Crop size_mode 默认", it["required"]["size_mode"][1]["default"] == "keep shape (long side)")
check("Crop multiple 默认 8", it["required"]["multiple"][1]["default"] == 8)

its = mod.SFInpaintStitch.INPUT_TYPES()
check("Stitch required 含 image", its["required"]["image"][0] == "IMAGE")
check("Stitch optional 含 crop_info", its["optional"]["crop_info"][0] == "SF_CROP_INFO")
check("Stitch softness 默认 -1", its["optional"]["softness"][1]["default"] == -1)
check("Stitch blend_mode 默认 from crop", its["optional"]["blend_mode"][1]["default"] == "from crop")
check("Stitch color_match 默认 off", its["optional"]["color_match"][1]["default"] == "off")

# ── merge_params ──
mp = helper.merge_params({})
check("merge_params 默认", mp["size_mode"] == "keep" and mp["blend"] == 16 and mp["multiple"] == 8)
mp = helper.merge_params({"size_mode": "FORCE", "target": 512.0, "multiple": "16", "context_px": "12"})
check("merge_params 覆盖+类型校正", mp["size_mode"] == "force" and mp["target"] == 512 and mp["multiple"] == 16 and mp["context_px"] == 12)
mp = helper.merge_params({"size_mode": "bogus"})
check("merge_params 非法 size_mode 回退", mp["size_mode"] == "keep")
mp = helper.merge_params({"resample": "BOGUS"})
check("merge_params 非法 resample 回退", mp["resample"] == "lanczos")
check("merge_params min/max 校正", helper.merge_params({"min_size": 4})["min_size"] == 8)
check("merge_params max>=min", helper.merge_params({"min_size": 512, "max_size": 100})["max_size"] == 512)

# ── compute_region ──
cr = helper.compute_region
# keep：bbox 20x10，target 1024 -> 长边 1024，短边 512
r = cr((10, 10, 30, 20), 100, 100, {"size_mode": "keep", "target": 1024, "multiple": 8,
                                     "context_px": 0, "blend": 0, "context_pct": 0, "min_size": 256})
check("keep 长边到 target", r["out_w"] == 1024 and r["out_h"] == 512)
check("keep 源区域包含 bbox", r["rx"] <= 10 and r["ry"] <= 10 and r["rx"] + r["rw"] >= 30 and r["ry"] + r["rh"] >= 20)
# keep：target 太小 -> min_size 抬升（20x10 bbox: s=3.2 -> 64x32 -> 抬升 x8 -> 512x256）
r = cr((10, 10, 30, 20), 200, 200, {"size_mode": "keep", "target": 64, "multiple": 8,
                                    "context_px": 0, "blend": 0, "context_pct": 0, "min_size": 256, "max_size": 2048})
check("keep min/max 夹紧", r["out_w"] == 512 and r["out_h"] == 256)
# force：恒 target x target 方形
r = cr((10, 10, 110, 60), 200, 200, {"size_mode": "force", "target": 512, "target_w": 512, "target_h": 512,
                                     "multiple": 8, "context_px": 0, "blend": 0, "context_pct": 0})
check("force 输出方形", r["out_w"] == 512 and r["out_h"] == 512)
check("force 源宽高比 == 1", r["rw"] == r["rh"])
# free：源尺寸仅对齐倍数（bank rounding：20->16, 10->8）
r = cr((10, 10, 30, 20), 100, 100, {"size_mode": "free", "multiple": 8,
                                    "context_px": 0, "blend": 0, "context_pct": 0, "max_size": 2048})
check("free 对齐倍数(bank)", r["out_w"] == 16 and r["out_h"] == 8)
# bbox None -> 整图
r = cr(None, 100, 50, {"size_mode": "free", "multiple": 8, "context_px": 0, "blend": 0,
                       "context_pct": 0, "max_size": 2048})
check("bbox None 整图", r["rx"] == 0 and r["ry"] == 0 and r["rw"] == 100 and r["rh"] == 50)
# 边缘遮罩 -> 源区域 clamp 到图像内
r = cr((0, 0, 20, 20), 100, 100, {"size_mode": "keep", "target": 2048, "multiple": 8,
                                  "context_px": 0, "blend": 0, "context_pct": 0, "max_size": 2048})
check("边缘遮罩 clamp", r["rx"] >= 0 and r["ry"] >= 0 and r["rx"] + r["rw"] <= 100 and r["ry"] + r["rh"] <= 100)
# context_pct：bbox 的 10% 总扩展
r = cr((10, 10, 30, 20), 200, 200, {"size_mode": "free", "multiple": 8, "context_px": 0,
                                    "blend": 0, "context_pct": 10, "max_size": 2048})
check("context_pct 生效", 20 <= r["rw"] < 25 and 10 <= r["rh"] < 15)

# ── mask 工具 ──
check("mask_bbox 空", helper.mask_bbox(np.zeros((10, 10), bool)) is None)
b = np.zeros((10, 10), bool); b[2:6, 3:8] = True
check("mask_bbox 非空", helper.mask_bbox(b) == (3, 2, 8, 6))
mn = helper.mask_to_np(None, 8, 8)
check("mask_to_np None 全零", mn.shape == (8, 8) and not mn.any())
mn = helper.mask_to_np(np.full((4, 4), 0.5), 8, 8)
check("mask_to_np NEAREST 缩放", mn.shape == (8, 8))
mn = helper.mask_to_np(FakeTensor(np.ones((2, 3))), 4, 4)
check("mask_to_np 3D 取首帧", mn.shape == (4, 4))
mn = helper.mask_to_np(np.ones((3, 3, 3)), 4, 4)
check("mask_to_np 非 2D 回退全零", not mn.any())
# fill_holes（本机无 scipy -> PIL MaxFilter/MinFilter fallback）：小孔被填
fh = np.zeros((64, 64), bool); fh[8:56, 8:56] = True; fh[24:32, 24:32] = False
out = helper.fill_holes(fh)
check("fill_holes 小孔被填", out[28, 28])
# 大的主体形孔洞保持不填（fallback 闭运算内核 9 太小够不着）
fh2 = np.zeros((128, 128), bool); fh2[8:120, 8:120] = True; fh2[30:98, 30:98] = False
out2 = helper.fill_holes(fh2)
check("fill_holes 大孔保留", not out2[60, 60])
pm = helper.preprocess_mask(np.zeros((16, 16), np.float32), helper.merge_params({"mask_grow": 2}))
check("preprocess_mask 空遮罩", not pm.any())
pm = helper.preprocess_mask(
    np.pad(np.ones((4, 4), np.float32), 6), helper.merge_params({"mask_grow": 2}))
check("preprocess_mask 膨胀生效", pm.shape == (16, 16) and pm.max() == 1.0)
# resolve_inpaint_mask
disk_empty = FakeTensor(np.zeros((1, 8, 8)))
disk_full = FakeTensor(np.ones((1, 8, 8)))
up = FakeTensor(np.full((1, 8, 8), 0.5))
check("resolve 空磁盘回退上游", helper.resolve_inpaint_mask(disk_empty, up) is up)
check("resolve 非空磁盘胜出", helper.resolve_inpaint_mask(disk_full, up) is disk_full)
check("resolve 均无内容返回磁盘", helper.resolve_inpaint_mask(disk_empty, None) is disk_empty)

# ── 安全函数 ──
check("_sanitize_id 清洗路径穿越", mod._sanitize_id("../../etc/passwd", "fb") == "etcpasswd")
check("_sanitize_id 保留字与连字符", mod._sanitize_id("a-b_c.1", "fb") == "a-b_c1")
check("_sanitize_id 空回退", mod._sanitize_id(None, "fb") == "fb")
inp_dir = os.path.join(tmp_in, "sfnodes_inpaint")
os.makedirs(inp_dir, exist_ok=True)
open(os.path.join(inp_dir, "inpaint_src_x.png"), "w").close()
check("_safe_join 合法相对路径", mod._safe_join("sfnodes_inpaint/inpaint_src_x.png") is not None)
check("_safe_join 不存在的文件", mod._safe_join("sfnodes_inpaint/nope.png") is None)
check("_safe_join 绝对路径拒绝", mod._safe_join("/etc/passwd") is None)
check("_safe_join .. 穿越拒绝", mod._safe_join("../../x.png") is None)
check("_safe_join UNC 拒绝", mod._safe_join("\\\\host\\share\\x.png") is None)
check("_safe_join 空拒绝", mod._safe_join("") is None)

import base64
import io
from PIL import Image
buf = io.BytesIO()
Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, "PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
check("_decode_image dataURL", mod._decode_image(f"data:image/png;base64,{b64}") is not None)
check("_decode_image 非法", mod._decode_image("not-base64!!") is None)

# ── _inpaint_meta_from_widget 形状兼容 ──
fw = mod._inpaint_meta_from_widget
check("widget dict {state_json}", fw({"state_json": json.dumps({"project_id": "a"})})["project_id"] == "a")
check("widget dict 直接 meta", fw({"project_id": "a", "mask_path": "x"})["project_id"] == "a")
check("widget str JSON", fw('{"project_id": "a"}')["project_id"] == "a")
check("widget 坏 JSON", fw("{oops") == {})
check("widget None", fw(None) == {})
check("widget 非 dict/str", fw(123) == {})

# ── IS_CHANGED：状态/参数变化 + 磁盘 mtime ──
open(os.path.join(inp_dir, "inpaint_mask_m.png"), "w").close()
meta = {"project_id": "p1", "mask_path": "sfnodes_inpaint/inpaint_mask_m.png"}
k1 = mod.SFInpaintCrop.IS_CHANGED(SFInpaintJson=json.dumps(meta), size_mode="keep shape (long side)")
k2 = mod.SFInpaintCrop.IS_CHANGED(SFInpaintJson=json.dumps(meta), size_mode="force size (square)")
check("IS_CHANGED 参数变化", k1 != k2)
k3 = mod.SFInpaintCrop.IS_CHANGED(SFInpaintJson=json.dumps(meta), size_mode="keep shape (long side)")
check("IS_CHANGED 同状态同参数", k1 == k3)
meta2 = dict(meta); meta2["mask_path"] = "sfnodes_inpaint/inpaint_mask_n.png"
k5 = mod.SFInpaintCrop.IS_CHANGED(SFInpaintJson=json.dumps(meta2), size_mode="keep shape (long side)")
check("IS_CHANGED 状态变化", k1 != k5)
os.utime(os.path.join(inp_dir, "inpaint_mask_m.png"), (os.path.getmtime(os.path.join(inp_dir, "inpaint_mask_m.png")) + 10,) * 2)
k4 = mod.SFInpaintCrop.IS_CHANGED(SFInpaintJson=json.dumps(meta))
check("IS_CHANGED mtime 变化", k4 != k1)
check("IS_CHANGED 空状态安全", isinstance(mod.SFInpaintCrop.IS_CHANGED(), str))

# ── apply_inpaint_crop 数值 ──
node = mod.SFInpaintCrop()
img = FakeTensor(np.zeros((1, 100, 120, 3)))
img[0, 30:70, 40:80, :] = 1.0
m = FakeTensor(np.zeros((1, 100, 120)))
m[0, 40:60, 50:70] = 1.0
cropped, cmask, info, ow, oh = helper.apply_inpaint_crop(
    img, m, {"size_mode": "free", "multiple": 8, "context_px": 0, "blend": 0,
             "context_pct": 0, "mask_grow": 0, "mask_blur": 0, "max_size": 2048})
check("apply 输出尺寸", (ow, oh) == (cropped.shape[2], cropped.shape[1]))
check("apply 输出形状", cropped.shape == (1, oh, ow, 3) and cmask.shape == (1, oh, ow))
check("apply crop_info 键齐全", all(k in info for k in ("image", "mask", "x", "y", "w", "h", "orig_w", "orig_h")))
check("apply crop_info 区域", info["x"] >= 40 and info["y"] >= 30 and info["w"] >= 20 and info["h"] >= 20)
check("apply crop_info 携带全帧", info["image"].shape == (1, 100, 120, 3) and info["mask"].shape == (1, 100, 120))
# 裁剪图在遮罩区域为白
cimg = cropped.data
cmsk = cmask.data[0]
check("apply 遮罩区域为白", bool(cimg[0][cmsk > 0.5].mean() > 0.9))
check("apply 输出对齐 8", ow % 8 == 0 and oh % 8 == 0)

# ── stitch_back 数值：mask 模式 ──
base = FakeTensor(np.zeros((1, 64, 64, 3)))
base[:, 16:48, 16:48, :] = 0.5
full_mask = FakeTensor(np.zeros((1, 64, 64)))
full_mask[0, 24:40, 24:40] = 1.0
info = {"image": base, "mask": full_mask, "x": 16, "y": 16, "w": 32, "h": 32, "orig_w": 64, "orig_h": 64}
patch = FakeTensor(np.ones((1, 32, 32, 3)))
res, orig = helper.stitch_back(info, patch, None, 0, "mask", "off")
check("stitch original 原图", orig.shape == (1, 64, 64, 3) and float(orig[0, 0, 0, 0]) == 0.0)
d = res.data[0]
check("stitch 遮罩内替换", float(d[32, 32, 0]) == 1.0)
check("stitch 遮罩外保留", float(d[8, 8, 0]) == 0.0)
check("stitch 区域外不动", float(d[60, 60, 0]) == 0.0)
# blend>0：仅向外羽化——遮罩内部保持 1，外圈 blend 像素过渡（取遮罩外 2px 处）
res2, _ = helper.stitch_back(info, patch, None, 4, "mask", "off")
e = res2.data[0, 24, 22, 0]  # x=22 在遮罩左缘 (x=24) 外 2px，blend=4 过渡带内
check("stitch blend 软化过渡", 0.0 < float(e) < 1.0)
check("stitch 遮罩内不透出", float(res2.data[0, 24, 32, 0]) == 1.0)

# ── stitch_back：whole_crop 模式 + 矩形羽化 ──
res3, _ = helper.stitch_back(info, patch, None, 0, "whole_crop", "off")
d3 = res3.data[0]
check("whole_crop 整区域替换", float(d3[20, 20, 0]) == 1.0)
check("whole_crop 区域外保留", float(d3[8, 8, 0]) == 0.0)
res4, _ = helper.stitch_back(info, patch, None, 4, "whole_crop", "off")
e4 = res4.data[0, 17, 32, 0]  # 矩形边界内侧（羽化带内）
check("whole_crop feather 过渡", 0.0 < float(e4) < 1.0)

# ── stitch_back：接入遮罩优先于 crop_info 遮罩 ──
alt = FakeTensor(np.zeros((1, 32, 32)))
alt[0, :16, :, ] = 1.0
res5, _ = helper.stitch_back(info, patch, alt, 0, "mask", "off")
d5 = res5.data[0]
check("接入遮罩优先(上半)", float(d5[20, 20, 0]) == 1.0 and float(d5[40, 20, 0]) == 0.5)

# ── stitch_back：无 mask 字段的 crop_info（SF Image Crop 互操作）→ 矩形羽化 ──
info_no_mask = {"image": base, "x": 16, "y": 16, "w": 32, "h": 32, "orig_w": 64, "orig_h": 64}
res6, _ = helper.stitch_back(info_no_mask, patch, None, 2, "mask", "off")
check("无 mask 互操作贴回", float(res6.data[0, 32, 32, 0]) == 1.0)

# ── stitch_back：batch 对齐（1 原图 + N 修复图）──
base2 = FakeTensor(np.zeros((1, 64, 64, 3)))
info2 = {"image": base2, "x": 0, "y": 0, "w": 64, "h": 64, "orig_w": 64, "orig_h": 64}
patch2 = FakeTensor(np.ones((2, 64, 64, 3)))
res7, orig7 = helper.stitch_back(info2, patch2, None, 0, "mask", "off")
check("stitch batch 展开", res7.shape == (2, 64, 64, 3) and orig7.shape == (2, 64, 64, 3))

# ── resolve_seam ──
check("resolve_seam 继承", helper.resolve_seam({"blend": 30, "blend_mode": "mask"}, -1, "from crop") == (30, "mask"))
check("resolve_seam 覆盖", helper.resolve_seam({"blend": 30, "blend_mode": "mask"}, 12, "whole crop") == (12, "whole_crop"))
check("resolve_seam 夹紧", helper.resolve_seam({}, 500, "from crop")[0] == 150)
check("resolve_seam 非法模式回退", helper.resolve_seam({}, -1, "bogus")[1] == "mask")

# ── 节点 run：无有效输入 -> empty 兜底 ──
res_empty = node.run()
check("run 空输入兜底", res_empty[1].shape == (1, 1024, 1024) and res_empty[3] == 1024)

# ── Stitch run：无 crop_info -> 双输出透传 ──
stitch = mod.SFInpaintStitch()
p = FakeTensor(np.ones((1, 4, 4, 3)))
r1, r2 = stitch.run(p)
check("Stitch 无 crop_info 透传", r1 is p and r2 is p)
crop_info = dict(info)
r1b, r2b = stitch.run(patch, crop_info=crop_info, softness=0)
check("Stitch 正常缝合", r1b.shape == (1, 64, 64, 3) and r2b.shape == (1, 64, 64, 3))

# ── 注册键一致性（AST）──
import ast
src = open(os.path.join(root, "__init__.py")).read()
tree = ast.parse(src)
classmap = dispmap = None
for n_ in ast.walk(tree):
    if isinstance(n_, ast.Assign):
        for t in n_.targets:
            if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                classmap = {k.value for k in n_.value.keys}
            elif isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS":
                dispmap = {k.value for k in n_.value.keys}
check("注册键两字典一致", classmap == dispmap)
check("SFInpaintCrop 已注册", "SFInpaintCrop" in (classmap or set()))
check("SFInpaintStitch 已注册", "SFInpaintStitch" in (classmap or set()))
check("SFInpaintExtendOutpaint 已注册", "SFInpaintExtendOutpaint" in (classmap or set()))
imports_src = open(os.path.join(root, "__init__.py")).read()
check("旧 InpaintCrop 不再导入", "from .nodes.inpaint.cropstitch import InpaintCrop" not in imports_src)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
