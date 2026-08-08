"""Inpaint Crop / Inpaint Stitch 共享的几何 + 遮罩 + 缝合工具（移植自
comfyui-pixaroma nodes/_inpaint_helpers.py）。

本模块是 inpaint 区域计算的**唯一 Python 事实来源**；`web/sf_inpaint_geometry.js`
与 `compute_region` 1:1 镜像，保证编辑器里实时预览的裁剪框与节点实际输出一致。

零额外依赖：torch + PIL + numpy 均由 ComfyUI 环境提供；scipy 仅在可用时用于
真正的洞填充/距离变换，不可用时回退到纯 PIL/numpy 实现（try/except 包裹，
遵循离线优先原则）。
"""

import math

import numpy as np
import torch
from PIL import Image, ImageFilter

from .logger import get_logger

try:
    from scipy import ndimage as _ndimage  # 可选，用于真正的 binary_fill_holes
    _HAS_SCIPY = True
except Exception:
    _ndimage = None
    _HAS_SCIPY = False

logger = get_logger(__name__)

# import 成功并不等于 scipy 真的能用：针对 numpy 1.x 构建/编写的 scipy 在用户
# 升级到 numpy 2 后能正常导入、却在调用深处抛错（来自社区的真实报告：
#   [PixaromaInpaintCrop] crop error: Alias 'bool8' was removed in NumPy 2.0.
# 该名字出自我们仅调用的库，本插件自己的代码从未包含）。import 时无法发现，
# 所以每个 scipy 调用处也单独 try 并回退到下方已写好的纯 PIL/numpy 路径。
# 失败会**会话锁存**，坏安装只带来一次警告而不是每个遮罩抛一次异常。
_SCIPY_DEAD = False


def _scipy_ok():
    return _HAS_SCIPY and _ndimage is not None and not _SCIPY_DEAD


def _scipy_call(name, *args, **kwargs):
    """执行一次 scipy 调用。返回 (True, 结果)，失败则锁存回退并返回 (False, None)。

    只有 scipy 调用本身在 try 内；外围的 numpy/PIL 运算绝不能触发锁存——
    否则我们自己的 np.bincount 若抛 MemoryError，会误把 scipy 禁掉整个会话，
    并在三个调用点各打一次甩锅警告。
    """
    if not _scipy_ok():
        return False, None
    try:
        return True, getattr(_ndimage, name)(*args, **kwargs)
    except Exception as e:
        _scipy_failed("ndimage." + name, e)
        return False, None


def _scipy_failed(where, err):
    global _SCIPY_DEAD
    if not _SCIPY_DEAD:
        _SCIPY_DEAD = True
        logger.warning(
            f"scipy 已安装但当前环境不可用 - {type(err).__name__}: {err} "
            f"(调用位置 {where})。本会话改用内置遮罩代码，结果近似一致。"
            f"若你近期升级过 NumPy，为对应 NumPy 版本重装 scipy 可修复。"
        )


# 类型名与节点文件保持一致——并与 Image Crop / Image Uncrop 对用的 SF_CROP_INFO
# 相同，使两条链路可互换。与 crop.py 一样以普通字符串重复声明（无跨文件导入链）。
SF_CROP_INFO = "SF_CROP_INFO"

_RESAMPLE = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}

# 默认参数值。web/sf_inpaint.js 的 DEFAULT_STATE 必须与这里保持同步
# （与其它 Pixaroma 节点同等风险类别）。
DEFAULTS = {
    "size_mode": "keep",        # keep | force | free
    "target": 1024,             # keep 模式的长边目标
    "target_w": 1024,           # force 模式
    "target_h": 1024,           # force 模式
    "multiple": 8,              # 8 | 16 | 32 | 64
    "context_px": 24,           # 每侧绝对上下文填充像素
    "context_pct": 10.0,        # bbox 尺寸比例的额外上下文（占总长百分比）
    "mask_grow": 4,             # 测量 bbox 前先膨胀遮罩
    "mask_blur": 4,             # 软化输出遮罩边缘（conditioning），像素
    "blend": 16,                # 接缝羽化像素（stitch）；同时扩张裁剪上下文
    "invert_mask": False,       # 裁剪前翻转遮罩（1 - mask）
    "fill_holes": True,
    "min_size": 256,
    "max_size": 2048,
    "resample": "lanczos",
    "allow_upscale": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# 小工具

def _round_mult(v, m):
    m = max(1, int(m))
    return int(max(m, round(float(v) / m) * m))


def _clampi(v, lo, hi):
    return int(max(lo, min(hi, int(round(v)))))


def merge_params(p):
    """用 DEFAULTS 补全缺失键并校正类型。"""
    out = dict(DEFAULTS)
    if isinstance(p, dict):
        out.update({k: p[k] for k in p if k in DEFAULTS})
    out["size_mode"] = str(out["size_mode"]).lower()
    if out["size_mode"] not in ("keep", "force", "free"):
        out["size_mode"] = "keep"
    out["resample"] = str(out["resample"]).lower()
    if out["resample"] not in _RESAMPLE:
        out["resample"] = "lanczos"
    for k in ("target", "target_w", "target_h", "multiple", "context_px",
              "mask_grow", "mask_blur", "blend", "min_size", "max_size"):
        out[k] = int(round(float(out[k])))
    out["context_pct"] = float(out["context_pct"])
    out["fill_holes"] = bool(out["fill_holes"])
    out["allow_upscale"] = bool(out["allow_upscale"])
    out["invert_mask"] = bool(out["invert_mask"])
    out["multiple"] = max(1, out["multiple"])
    out["min_size"] = max(8, out["min_size"])
    out["max_size"] = max(out["min_size"], out["max_size"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 遮罩工具（numpy float HxW，1 = 待修复区域）

def mask_to_np(mask, h, w):
    """把 ComfyUI MASK（[1,H,W] / [H,W]）转为 float HxW numpy（0..1），
    尺寸不符时缩放到 (h, w)。None -> 全零。"""
    if mask is None:
        return np.zeros((h, w), dtype=np.float32)
    m = mask
    if isinstance(m, torch.Tensor):
        if m.dim() == 4:
            m = m[:, 0] if m.shape[1] == 1 else m[..., 0]
        if m.dim() == 3:
            m = m[0]
        m = m.detach().cpu().float().clamp(0, 1).numpy()
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        return np.zeros((h, w), dtype=np.float32)
    if m.shape != (h, w):
        pim = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), "L")
        pim = pim.resize((w, h), Image.NEAREST)
        m = np.asarray(pim, dtype=np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)


def _max1d(a, k):
    """沿轴 1 的一维 max filter，奇数窗口 k，边缘填充。快速盒式膨胀的可分离
    构件：O(W*H*k)，而非 PIL MaxFilter 的 O(W*H*k^2)（后者在大的 mask_grow
    上曾挂起约 50 秒）。"""
    r = k // 2
    ap = np.pad(a, ((0, 0), (r, r)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(ap, k, axis=1)
    return win.max(axis=2)


def _dilate(m_bool, px):
    if px <= 0:
        return m_bool
    k = 2 * int(px) + 1
    # 可分离 max filter -> O(W*H)，即便内核巨大也快
    ok, filtered = _scipy_call("maximum_filter", m_bool, size=k)
    if ok:
        return filtered > 0
    # 无 scipy：两趟可分离 numpy max（先行后列）
    a = m_bool.astype(np.uint8)
    a = _max1d(a, k)
    a = _max1d(np.ascontiguousarray(a.T), k).T
    return a > 0


def fill_holes(m_bool):
    """只填充**小**的封闭孔洞（画笔斑点/缺口），绝不填充大的主体形孔洞。
    剪影/背景遮罩（白色包围主体）若被 scipy 的 binary_fill_holes 填充，
    整个遮罩会塌成实心——即"裁剪后遮罩消失"的 bug。scipy 可用时做
    （大小受限的真填充）；否则用 PIL 形态学闭运算（小内核，天然只限小孔）。"""
    ok, filled = _scipy_call("binary_fill_holes", m_bool)
    if ok:
        try:
            added = filled & ~m_bool          # binary_fill_holes 会填掉的像素
            if not added.any():
                return filled
            # 只保留不超过图像面积小比例的孔洞（斑点/缺口）；主体孔洞远大于
            # 此值，保持不填充，遮罩得以存活。
            H, W = m_bool.shape
            limit = max(256, int(0.005 * H * W))   # 约图像 0.5%
            ok2, labelled = _scipy_call("label", added)
            if ok2:
                lbl = labelled[0]
                sizes = np.bincount(lbl.ravel())
                small = np.where(sizes <= limit)[0]
                small = small[small != 0]          # 去掉标签 0（非孔洞区域）
                return m_bool | np.isin(lbl, small)
        except Exception:
            # 是我们的 numpy 运算失败，不是 scipy 的。仅本次遮罩回退，scipy
            # 保持存活——此处锁存会甩锅 scipy 并在本会话内悄悄降级膨胀+羽化。
            pass
    pim = Image.fromarray((m_bool * 255).astype(np.uint8), "L")
    k = 9
    pim = pim.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    return np.asarray(pim, dtype=np.uint8) > 127


def gaussian_blur_np(m, px):
    if px <= 0:
        return m
    pim = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), "L")
    pim = pim.filter(ImageFilter.GaussianBlur(radius=float(px)))
    return np.asarray(pim, dtype=np.float32) / 255.0


def preprocess_mask(m, p):
    """在 0..1 的 float 遮罩上做 填洞 + 膨胀 -> 用于 bbox 的 float 遮罩，
    同时以全帧形式带进 crop_info（缝合羽化时使用）。"""
    mb = m > 0.5
    if p["fill_holes"]:
        mb = fill_holes(mb)
    if p["mask_grow"] > 0:
        mb = _dilate(mb, p["mask_grow"])
    return mb.astype(np.float32)


def mask_bbox(m_bool):
    ys, xs = np.where(m_bool)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resolve_inpaint_mask(disk_mask, upstream_mask):
    """决定裁剪使用哪个遮罩：编辑器画在磁盘上的，还是接入的 MASK 输入。

    编辑器遮罩**只有在实际画了像素时才胜出**。清空编辑器（或没画就保存）
    会写出全黑遮罩文件但 mask_path 仍在——若"有磁盘遮罩就优先"，
    这个空遮罩会悄悄覆盖接入的遮罩，导致裁剪看不到任何绘制区域而退回
    整图（已知 bug）。所以空/缺失的编辑器遮罩回退到接入遮罩。返回选中的
    遮罩张量；两者都没有内容时返回 disk_mask。"""
    if (isinstance(disk_mask, torch.Tensor) and disk_mask.numel()
            and float(disk_mask.detach().max()) > 1e-6):
        return disk_mask
    if isinstance(upstream_mask, torch.Tensor):
        return upstream_mask
    return disk_mask


# ─────────────────────────────────────────────────────────────────────────────
# 几何（js/inpaint_crop/geometry.mjs 的镜像）

def compute_region(bbox, W, H, p):
    """从遮罩 bbox (x0,y0,x1,y1) 和图像尺寸，算出源裁剪区域 + 模型友好输出尺寸。

    返回 {rx, ry, rw, rh, out_w, out_h}。`r*` 为整数**源**像素（要裁剪的矩形）；
    `out_*` 是裁剪图缩放到的尺寸（即节点的 width/height 输出）。bbox 为 None
    时使用整图。"""
    p = merge_params(p)
    W = int(W); H = int(H)
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = bbox
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    # 上下文扩张：每侧 max(context_px, blend) + bbox 的 context_pct 比例。
    # 与 blend 取 max（Option B）让接缝羽化有空间在裁剪图**内部**衰减到 0，
    # 大 softness 因此扩张裁剪图而不是在缝合时被截断。
    ctx = max(p["context_px"], p["blend"])
    rw = bw + 2.0 * ctx + bw * p["context_pct"] / 100.0
    rh = bh + 2.0 * ctx + bh * p["context_pct"] / 100.0

    mode = p["size_mode"]
    mult = p["multiple"]

    # force 模式：先把期望区域扩到目标宽高比（输出为下方设定的固定目标尺寸）。
    tw = th = 0
    if mode == "force":
        tw = max(mult, _round_mult(p["target_w"], mult))
        th = max(mult, _round_mult(p["target_h"], mult))
        target_aspect = tw / float(th)
        if rw / rh < target_aspect:
            rw = rh * target_aspect
        else:
            rh = rw / target_aspect

    # 在图像内放置并夹紧源区域
    rw_i = max(1, min(int(round(rw)), W))
    rh_i = max(1, min(int(round(rh)), H))
    if mode == "force":
        # 图像边界夹紧（可能只裁掉一个轴）后仍保持源宽高比 == 目标宽高比，
        # 缩放才不会拉伸。
        aspect = tw / float(th)
        if rw_i > rh_i * aspect:
            rw_i = max(1, int(round(rh_i * aspect)))
        else:
            rh_i = max(1, int(round(rw_i / aspect)))
    rx = _clampi(cx - rw_i / 2.0, 0, W - rw_i)
    ry = _clampi(cy - rh_i / 2.0, 0, H - rh_i)

    # 输出尺寸由**夹紧后的**源矩形（rw_i, rh_i）推导，而不是未夹紧的期望
    # 区域——裁剪图按真实宽高比缩放，图像边缘裁掉区域时（如大 softness 配
    # 边缘遮罩：源被图像边界夹紧，输出必须跟着夹紧，否则裁剪图/遮罩被压扁）
    # 也绝不会被拉伸。
    if mode == "force":
        out_w, out_h = tw, th
    elif mode == "free":
        # free = "只对齐倍数"：保持裁剪矩形自身尺寸，仅对齐到 multiple。
        # 长边超过 max_size 时**两轴乘同一系数**缩放（而不是各轴独立封顶）
        # 以保持源宽高比——独立逐轴封顶会拉伸宽/高裁剪图。不做 min_size
        # 抬升（free 保持源尺寸，只有 max 上限生效）。
        ow, oh = float(rw_i), float(rh_i)
        big = max(ow, oh)
        if big > p["max_size"]:
            k = p["max_size"] / big
            ow *= k; oh *= k
        out_w = _round_mult(ow, mult)
        out_h = _round_mult(oh, mult)
    else:  # keep shape：裁剪矩形长边缩放到 target，保持宽高比
        long_side = max(rw_i, rh_i)
        s = p["target"] / long_side if long_side > 0 else 1.0
        if not p["allow_upscale"]:
            s = min(s, 1.0)
        ow = rw_i * s
        oh = rh_i * s
        # 先做 min_size 抬升（两轴同倍放大使短边到 min_size），最后 max_size
        # 作为硬上限收尾——极端宽高比的裁剪图长边不会突破 max_size 造出
        # OOM 张量（短边可能因此 < min_size，可接受）。
        small = min(ow, oh)
        if small < p["min_size"]:
            k = p["min_size"] / small
            ow *= k; oh *= k
        big = max(ow, oh)
        if big > p["max_size"]:
            k = p["max_size"] / big
            ow *= k; oh *= k
        out_w = _round_mult(ow, mult)
        out_h = _round_mult(oh, mult)

    out_w = max(mult, int(out_w))
    out_h = max(mult, int(out_h))
    return {"rx": rx, "ry": ry, "rw": rw_i, "rh": rh_i,
            "out_w": out_w, "out_h": out_h}


# ─────────────────────────────────────────────────────────────────────────────
# 图像 / 遮罩缩放（PIL 以获得高质量 + Lanczos 支持）

def resize_image_tensor(t, w, h, resample="lanczos"):
    """[B,H,W,3] float 0..1 -> [B,h,w,3]。逐帧走 PIL（支持 Lanczos）。
    尺寸已匹配时短路返回，恒等裁剪保持像素精确（PIL 同尺寸 resize 仍会重采样）。"""
    if int(w) == int(t.shape[2]) and int(h) == int(t.shape[1]):
        return t
    filt = _RESAMPLE.get(resample, Image.LANCZOS)
    frames = []
    for i in range(int(t.shape[0])):
        arr = (t[i].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        pim = Image.fromarray(arr, "RGB").resize((int(w), int(h)), filt)
        frames.append(np.asarray(pim, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames, 0))


def resize_mask_np(m, w, h, resample="bilinear"):
    if int(w) == int(m.shape[1]) and int(h) == int(m.shape[0]):
        return np.clip(m, 0, 1).astype(np.float32)
    filt = _RESAMPLE.get(resample, Image.BILINEAR)
    pim = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), "L")
    pim = pim.resize((int(w), int(h)), filt)
    return np.asarray(pim, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 裁剪（node_inpaint_crop.py 调用）

def apply_inpaint_crop(image, mask, p):
    """image [B,H,W,3], mask [1,H,W]|None, p = 参数 dict。

    返回 (cropped_image[B,out_h,out_w,3], out_mask[1,out_h,out_w], crop_info,
    out_w, out_h)。out_mask 是 conditioning 遮罩（膨胀+模糊后）的输出分辨率版；
    crop_info 携带完整原图 + 全帧处理后遮罩，供缝合贴回并产出全帧结果。
    """
    p = merge_params(p)
    # 输入转 RGB。Remove Background Pixaroma（及任何抠图）给的是 4 通道 RGBA；
    # 原样保留会进入 crop_info["image"] 并在 Inpaint Stitch 的 3-vs-4 通道粘贴
    # 时抛错（其 except 随后把结果静默透传成"original"）。RGBA 在黑色上预乘，
    # 让抠图读起来是黑底上的主体（与编辑器/预览一致），而不是残留背景还压在
    # alpha 底下。
    if image.shape[-1] == 4:
        image = (image[..., :3] * image[..., 3:4]).contiguous()
    elif image.shape[-1] > 4:
        image = image[..., :3].contiguous()
    B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])

    raw = mask_to_np(mask, H, W)
    if p["invert_mask"] and mask is not None:
        raw = 1.0 - raw                       # 翻转修复区域（Invert 开关）
    proc = preprocess_mask(raw, p)            # bbox 用的二值核心（填洞+膨胀）
    bbox = mask_bbox(proc > 0.5)
    # 保留画笔的软边缘：填洞+膨胀后的核心不透明，核心外软画的边缘得以保留
    # （软边画笔 + mask_blur 才能真正进入 conditioning 遮罩而不被阈值化掉）。
    softm = np.maximum(raw, proc)
    region = compute_region(bbox, W, H, p)
    rx, ry, rw, rh = region["rx"], region["ry"], region["rw"], region["rh"]
    out_w, out_h = region["out_w"], region["out_h"]

    # 裁剪 + 缩放图像（所有 batch 帧同矩形）。结果保持输入的 device/dtype
    # （缩放经 PIL 走 CPU）。
    crop = image[:, ry:ry + rh, rx:rx + rw, :].contiguous()
    cropped_image = resize_image_tensor(crop, out_w, out_h, p["resample"]).to(image.device, image.dtype)

    # 遮罩用 NEAREST 缩放（bilinear 会自带渐变光晕）；mask_blur 的高斯才是
    # conditioning 唯一预期的软化。
    mreg = softm[ry:ry + rh, rx:rx + rw]
    mout = resize_mask_np(mreg, out_w, out_h, "nearest")
    mout = gaussian_blur_np(mout, p["mask_blur"])
    out_mask = torch.from_numpy(np.clip(mout, 0, 1)[None, ...].astype(np.float32)).to(image.device)

    # 携带的遮罩保持 float32（而非 image.dtype），与 out_mask 一致，严格的下游
    # 遮罩运算在 fp16 图像流水线上也不会遇到 fp16 遮罩。
    full_mask = torch.from_numpy(softm[None, ...].astype(np.float32)).to(image.device)
    crop_info = {
        "image": image, "mask": full_mask,
        "x": rx, "y": ry, "w": rw, "h": rh,
        "orig_w": W, "orig_h": H,
    }
    return cropped_image, out_mask, crop_info, out_w, out_h


# ─────────────────────────────────────────────────────────────────────────────
# 缝合（node_inpaint_stitch.py 调用）

def _feather_alpha(alpha, feather):
    """把 alpha 在矩形边缘外 `feather` 像素内斜坡衰减到 0（到边距离），
    使贴回的裁剪图融入原图。与 Image Uncrop 的 _feather_alpha 思路相同，
    但作用于任意 [ch,cw] 的 alpha。"""
    k = int(feather)
    if k <= 0:
        return alpha
    ch, cw = int(alpha.shape[-2]), int(alpha.shape[-1])
    # 羽化上限约为短边的一半，保证内部保持完全不透明。比这更宽的羽化任何
    # 地方都到不了 1，整个矩形贴图都会半透明并透出原图（只在小裁剪图 + 大
    # feather 时出现，例如紧凑裁剪图上用大的 Stitch softness 覆盖）。
    k = min(k, max(1, (min(ch, cw) - 1) // 2))
    ys = torch.arange(ch, dtype=torch.float32).view(ch, 1)
    xs = torch.arange(cw, dtype=torch.float32).view(1, cw)
    dist = torch.minimum(torch.minimum(ys, (ch - 1) - ys),
                         torch.minimum(xs, (cw - 1) - xs))
    ramp = (dist / float(k)).clamp(0.0, 1.0)
    return (alpha * ramp).clamp(0.0, 1.0)


def _blur_alpha(alpha, blend):
    """按 `blend` 像素软化遮罩**自身边缘**，让遮罩式粘贴平滑融入周围
    （区别于 _feather_alpha 淡化矩形边界——whole-crop 模式用）。

    **仅向外羽化**：遮罩内部及边缘 alpha 全为 1.0，向外 `blend` 像素斜坡
    到 0。新内容完整覆盖遮罩区域（旧内容在新物体自身边缘处永远透不出来），
    只有过渡到周围的部分被软化。居中式羽化会让遮罩边缘本身半透明，读起来
    像旧内容透过来的软重影/光晕。覆盖率/膨胀是裁剪节点的 mask_grow 的职责；
    这里只软化外圈接缝。

    裁剪节点现在把裁剪上下文扩到 max(context_px, blend)（compute_region，
    Option B），遮罩始终离裁剪边 >= blend 像素，向外羽化自然在边界前到达 0。
    这取代了旧的矩形边守卫 + 核心不透明夹紧（会碾碎过宽羽化、让靠近裁剪边
    的遮罩透出重影）；smoothstep 构造上内部不透明，无需夹紧。
    """
    k = int(blend)
    if k <= 0:
        return alpha
    a_np = np.clip(alpha.detach().cpu().numpy(), 0.0, 1.0)
    mb = a_np > 0.5
    if not mb.any() or mb.all():
        return torch.from_numpy(a_np.astype(np.float32))
    soft = None
    # 到遮罩边缘的有符号距离（内部 +，外部 -，单位像素）。映射为
    # signed >= 0 -> 1.0，signed 在 [-k,0] 斜坡 1 -> 0（smoothstep）。
    # 两个 scipy 调用各自锁存；下方的算术是我们的，绝不能锁存。
    ok_in, d_in = _scipy_call("distance_transform_edt", mb)
    ok_out, d_out = _scipy_call("distance_transform_edt", ~mb) if ok_in else (False, None)
    if ok_in and ok_out:
        try:
            signed = d_in - d_out
            t = np.clip(signed / float(k) + 1.0, 0.0, 1.0)
            soft = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        except Exception:
            # 这些是我们的代码，绝不能锁存（这正是收窄守卫的意义）——但仍需
            # 兜底，与 fill_holes 保持的一样。两种落点：本链在两次距离变换之上
            # 还分配若干全尺寸 float64 临时量，大遮罩内存紧张时会在这失败；
            # 以及 scipy 返回了被污染数组而不是抛错——减法处失败，正是这个
            # 回退存在的 numpy-2 破坏场景。soft 保持 None 交给下方高斯路径。
            soft = None
    if soft is None:
        # 回退（无 scipy）：高斯模糊二值遮罩做向外衰减，内部强制回到 1.0
        # （仅向外）。
        mbf = mb.astype(np.float32)
        blurred = gaussian_blur_np(mbf, max(1, int(k / 1.7)))
        soft = np.where(mbf > 0.5, 1.0, blurred).astype(np.float32)
    return torch.from_numpy(np.clip(soft, 0.0, 1.0).astype(np.float32))


def _color_match(patch, ref, region_mask, strength):
    """把 patch 的颜色统计向 ref 对齐（仅遮罩区域内）。strength:
    'subtle' = 匹配均值，'strong' = 匹配均值 + 标准差。patch/ref [ch,cw,3]。"""
    if strength == "off":
        return patch
    w = region_mask.reshape(-1, 1)
    wsum = float(w.sum()) + 1e-6
    pf = patch.reshape(-1, 3)
    rf = ref.reshape(-1, 3)
    pm = (pf * w).sum(0) / wsum
    rm = (rf * w).sum(0) / wsum
    if strength == "strong":
        pv = ((pf - pm) ** 2 * w).sum(0) / wsum
        rv = ((rf - rm) ** 2 * w).sum(0) / wsum
        scale = (rv.clamp_min(1e-6).sqrt()) / (pv.clamp_min(1e-6).sqrt())
        scale = scale.clamp(0.5, 2.0)
        out = (pf - pm) * scale + rm
    else:
        out = pf - pm + rm
    return out.reshape(patch.shape).clamp(0.0, 1.0)


def resolve_seam(crop_info, softness, blend_mode):
    """计算缝合的实际接缝羽化（像素）+ blend 模式，让 Inpaint Stitch 节点能
    **覆盖** crop_info 上从 Crop 节点带来的值——混合可在缝合时调优而
    **无需重跑采样器**。

    softness: int；< 0（-1 的"继承"默认）保留 crop_info['blend']。
    blend_mode: 'from crop' / '' = 保留 crop_info['blend_mode']；否则
    'mask' 或 'whole crop'（空格 -> 下划线）。返回 (blend:int 0..150, mode:str)。
    """
    if not isinstance(crop_info, dict):
        crop_info = {}
    try:
        s = int(softness)
    except (TypeError, ValueError):
        s = -1
    if s < 0:
        try:
            s = int(crop_info.get("blend", 16))
        except (TypeError, ValueError):
            s = 16
    blend = max(0, min(150, s))

    bm = str(blend_mode if blend_mode is not None else "from crop").strip().lower()
    if bm in ("", "from crop", "inherit"):
        bm = str(crop_info.get("blend_mode", "mask"))
    bm = bm.replace(" ", "_")
    if bm not in ("mask", "whole_crop"):
        bm = "mask"
    return blend, bm


def stitch_back(crop_info, image, mask, blend, blend_mode, color_match):
    """把修复后的 `image` 按记录的区域贴回 crop_info['image']，无缝混合。
    返回 (result[B,H,W,3], original[B,H,W,3])。"""
    base = crop_info["image"]
    # 防御性 RGB 转换：来自 RGBA 源的 crop_info（手工构造的 dict，或喂了 RGBA
    # 图的 Image Crop）会在下方粘贴时与 RGB patch 形状不匹配。Inpaint Crop
    # 已转换自己的输入，这里只兜其他来源。
    if isinstance(base, torch.Tensor) and base.dim() == 4 and base.shape[-1] == 4:
        base = (base[..., :3] * base[..., 3:4]).contiguous()
    elif isinstance(base, torch.Tensor) and base.dim() == 4 and base.shape[-1] > 4:
        base = base[..., :3].contiguous()
    H, W = int(base.shape[1]), int(base.shape[2])
    x = _clampi(crop_info.get("x", 0), 0, W - 1)
    y = _clampi(crop_info.get("y", 0), 0, H - 1)
    cw = int(max(1, min(int(crop_info.get("w", W)), W - x)))
    ch = int(max(1, min(int(crop_info.get("h", H)), H - y)))

    patch = image
    if not isinstance(patch, torch.Tensor) or patch.dim() != 4:
        patch = base.new_zeros((1, ch, cw, base.shape[3]))
    if int(patch.shape[1]) != ch or int(patch.shape[2]) != cw:
        patch = resize_image_tensor(patch, cw, ch, "lanczos").to(base.device, base.dtype)

    # 区域 alpha（1 = 取 patch）
    if blend_mode == "whole_crop":
        a = torch.ones((ch, cw), dtype=torch.float32)
    else:  # mask-aware：优先接入的遮罩，否则 crop_info 里画的遮罩
        if isinstance(mask, torch.Tensor):
            # mask_to_np 已经用 NEAREST 缩放到 (ch,cw) ——保持锐利；接缝软化
            # 是 _blur_alpha 的职责（这里 bilinear 缩放会在羽化之上再糊一层
            # 渐变 = 可见光晕）。
            a = torch.from_numpy(np.ascontiguousarray(mask_to_np(mask, ch, cw), dtype=np.float32))
        elif isinstance(crop_info.get("mask"), torch.Tensor):
            fm = mask_to_np(crop_info["mask"], H, W)[y:y + ch, x:x + cw]
            a = torch.from_numpy(np.ascontiguousarray(fm, dtype=np.float32))
        else:
            a = torch.ones((ch, cw), dtype=torch.float32)
    # whole_crop：淡化矩形边界。mask：软化遮罩自身边缘。
    if blend_mode == "whole_crop":
        a = _feather_alpha(a.clamp(0, 1), blend)
    else:
        ac = a.clamp(0, 1)
        ab = ac > 0.5
        if not bool(ab.any()) or bool(ab.all()):
            # 没有遮罩边缘可软化——Image Crop 的 crop_info 没有逐像素遮罩
            # （全零）、全一遮罩、或什么都没画的整图修复。_blur_alpha 会返回
            # 全零（什么都不贴——修复结果丢失）或全一（硬矩形接缝）。回退到
            # 整裁剪区域的矩形羽化，编辑过的裁剪图以软接缝贴回，Image Crop
            # 互操作时 Softness 滑块仍然有效。
            a = _feather_alpha(torch.ones((ch, cw), dtype=torch.float32), blend)
        else:
            a = _blur_alpha(ac, blend)

    out = base.clone()
    B = int(out.shape[0])
    if patch.shape[0] != B:
        if patch.shape[0] == 1:
            patch = patch.repeat(B, 1, 1, 1)
        elif B == 1:
            out = out.repeat(patch.shape[0], 1, 1, 1)
            B = patch.shape[0]
        else:
            n = min(B, patch.shape[0])
            logger.warning(
                f"batch 不匹配：原图 {B} vs 修复图 {patch.shape[0]} - 取前 {n} 帧"
            )
            out, patch = out[:n], patch[:n]
            B = n
    patch = patch.to(out.device, out.dtype)

    if color_match and color_match != "off":
        patch = patch.clone()   # 不要原地改调用方的张量（上游可能被缓存）
        # 参考 = 未遮罩的上下文（遮罩**外**的周围），不是遮罩区域或整个裁剪图。
        # 匹配任何含遮罩区域的内容都会把修复**故意改的颜色**拉回原色（红裙变
        # 白后又被拉回粉）。匹配上下文只校正模型在未变周围引入的光照/色调漂移，
        # 这才是接缝消失的关键。
        am = np.clip(a.detach().cpu().numpy(), 0.0, 1.0)
        ctx = (am < 0.5).astype(np.float32)
        if ctx.sum() < 0.02 * ctx.size:   # 遮罩几乎填满裁剪图 -> 没有上下文可匹配
            ctx = np.ones_like(am, dtype=np.float32)
        ac = torch.from_numpy(np.ascontiguousarray(ctx))
        for b in range(B):  # 匹配每一帧，不只是第 0 帧（视频 / batch）
            region_b = out[b, y:y + ch, x:x + cw, :3].detach().cpu()
            p_b = patch[b, :, :, :3].detach().cpu()
            matched = _color_match(p_b, region_b, ac, color_match)
            patch[b, :, :, :3] = matched.to(patch.device, patch.dtype)

    av = a[None, ..., None].to(out.device, out.dtype)
    region = out[:, y:y + ch, x:x + cw, :]
    out[:, y:y + ch, x:x + cw, :] = patch[..., :region.shape[-1]] * av + region * (1.0 - av)

    original = base
    if original.shape[0] != out.shape[0]:
        if original.shape[0] == 1:
            original = original.repeat(out.shape[0], 1, 1, 1)
        else:
            # 上面结果被裁剪到 min(B,C)，original 同步裁剪，否则两个输出
            # batch 数不一致。
            original = original[:out.shape[0]]
    return out.clamp(0, 1), original.clamp(0, 1)
