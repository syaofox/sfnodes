"""SF Image Outpaint / SF Image Outpaint Stitch — 移植自 comfyui-pixaroma
node_outpaint.py + node_outpaint_stitch.py (PixaromaOutpaint / PixaromaOutpaintStitch)。

Outpaint: 用纯色填充给图片四周扩展（默认中灰），供外绘模型填充新区域；可选按
百万像素上限缩放结果并报告最终尺寸。可替代 Load Image 的 Pad 模式 +
Scale Image to Total Pixels + Get Image Size 链。输出额外携带 "SF_OUTPAINT_INFO"
自定义线型（原始图片张量 + 各边 pad 量），Stitch 据此把原始图贴回。

Stitch: 把原始图贴回外绘结果，只保留模型新生成的区域。外绘对大图会先缩小，
经 VAE 编解码与缩小后原图部分被软化，此节点把结果放大回完整填充尺寸、
把原始图逐像素贴回其原位，仅对与新区域相邻的边缘做羽化融合，并可做连续
色域匹配消除接缝色调差。纯 torch，无磁盘、无 JS。

文件内两节点共享 SF_OUTPAINT_INFO 常量（同文件天然解耦，无需跨文件复制字符串）。
"""

import json
import math
import os
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...sf_utils.resize_engine import _apply_max_mp, _apply_pad, _round_half_up

try:
    import folder_paths
except ImportError:  # 保持模块在裸测试环境可导入
    folder_paths = None

_CATEGORY = "sfnodes/image"

# 自定义线型：携带 Stitch 需要的一切——完整原始图片 + 各边 pad 量（这样 Stitch
# 知道原始图在填充画布中的位置）。纯字符串类型，与节点类保持解耦。
SF_OUTPAINT_INFO = "SF_OUTPAINT_INFO"

DEFAULT_STATE = {
    "version": 1,
    "mode": "ratio",
    "ratio": "3:2",
    "anchor": "centre",
    "top": 0, "bottom": 0, "left": 0, "right": 0,
    "limit": 0,
    # 中灰而非绿色：训练在绿色填充上的 LoRA 会把颜色连同形状一起学会，
    # 生成的整张图都会带上绿色色偏（2026-07-17 实际使用反馈）。
    # 中性灰没有色相可渗。必须与 web/sf_outpaint_core.js 的 DEFAULT_STATE 一致。
    "color": "#808080",
    "snap": 0,
    "collapsed": False,
}

_MAX_MP = 64  # _apply_max_mp 的上限；自定义 limit 可以是到它的任意值
_SNAPS = (0, 8, 16, 32, 64)
_ANCHORS = ("left", "centre", "right", "top", "middle", "bottom")
_MAX_PAD = 8192
_MAX_DIM = 16384  # resize_engine._clamp_dims 的上限；填充不得超过它


def _fit_pad(pad_a, pad_b, extent):
    """把相对两边的 pad 收缩到 extent + pad_a + pad_b <= _MAX_DIM（与 _apply_pad
    结果的 clamp 上限一致）。按比例拆分，图片保持 anchor 放置的位置。
    返回 (pad_a, pad_b)，已满足时不改变。在 _apply_pad 分配画布之前运行，
    过大的 pad 不再可能 OOM（见 outpaint）。"""
    room = max(0, _MAX_DIM - int(extent))
    total = int(pad_a) + int(pad_b)
    if total <= room:
        return int(pad_a), int(pad_b)
    if total <= 0:
        return 0, 0
    fa = int(pad_a) * room // total
    return fa, room - fa


def _tensor_to_pils(image_t):
    """[B,H,W,C] float32 0..1 -> RGB PIL 图片列表。与 crop.py 的副本镜像；
    各节点保留自己的副本而非膨胀共享引擎。"""
    out = []
    arr = image_t.detach().cpu().numpy()
    for i in range(arr.shape[0]):
        frame = np.clip(arr[i] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # ComfyUI 的 IMAGE 是 3 或 4 通道，但不守规矩的上游节点可能给出 1、2
        # 或 5+ 通道；Image.fromarray(..., "RGB") 会直接崩。像 Save Image 的
        # 防御式转换一样把所有情况归一为 RGB，而不是让坏线害死整个外绘。
        ch = frame.shape[2]
        if ch >= 3:
            frame = frame[:, :, :3]           # RGB，丢弃 alpha / 多余通道
        elif ch == 2:
            frame = np.repeat(frame[:, :, :1], 3, axis=2)  # 灰 + 多余通道 -> 灰
        else:  # ch == 1
            frame = np.repeat(frame, 3, axis=2)
        out.append(Image.fromarray(frame, "RGB"))
    return out


def _parse_state(state_json):
    """把隐藏状态 JSON 合并到默认值上，逐个字段做强制转换。
    必须容忍一切：手改的 API 文件可以放任意类型进来。"""
    st = dict(DEFAULT_STATE)
    if not state_json or not isinstance(state_json, str):
        return st
    try:
        raw = json.loads(state_json)
    except Exception:
        return st
    if not isinstance(raw, dict):
        return st

    if raw.get("mode") in ("ratio", "sides"):
        st["mode"] = raw["mode"]
    if isinstance(raw.get("ratio"), str):
        st["ratio"] = raw["ratio"]
    if raw.get("anchor") in _ANCHORS:
        st["anchor"] = raw["anchor"]
    for k in ("top", "bottom", "left", "right"):
        try:
            st[k] = max(0, min(int(raw.get(k, 0)), _MAX_PAD))
        # OverflowError 要单独处理：json.loads 按文档扩展接受字面量 Infinity，
        # int(inf) 抛 OverflowError 而非 ValueError——不捕获它的话，手改 API
        # 文件带 Infinity 会让整个节点倒下，"容忍一切"的承诺就破了。
        except (TypeError, ValueError, OverflowError):
            st[k] = 0
    try:
        lim = float(raw.get("limit", 0))
        # 用户可自定义任意百万像素目标，不接受固定白名单。[0, _MAX_MP] 内任何
        # 有限值都收（0 = 不缩放）；_apply_max_mp 会夹到同样的上限。
        st["limit"] = lim if (math.isfinite(lim) and 0 <= lim <= _MAX_MP) else 0
    except (TypeError, ValueError, OverflowError):
        st["limit"] = 0
    c = raw.get("color")
    if isinstance(c, str) and len(c) == 7 and c[0] == "#":
        try:
            int(c[1:], 16)
            st["color"] = c
        except ValueError:
            pass
    try:
        sn = int(raw.get("snap", 0))
        st["snap"] = sn if sn in _SNAPS else 0
    except (TypeError, ValueError, OverflowError):
        st["snap"] = 0
    return st


def _parse_ratio(text):
    """'3:2' -> (3.0, 2.0)。不可用时返回 None。"""
    if not isinstance(text, str) or ":" not in text:
        return None
    a, _, b = text.partition(":")
    try:
        rw, rh = float(a), float(b)
    except (TypeError, ValueError):
        return None
    # float() 接受 "inf" 和 "nan"；前端 core.mjs 的 FINITE_NUMBER 正则不接受，
    # 不拒绝的话两侧会不一致、预览会说谎。二者也都能溜过下面的守卫：inf > 0，
    # nan 与任何比较都为 False。直接拒绝，而不是按无穷或莫名其妙的值填充。
    if not math.isfinite(rw) or not math.isfinite(rh):
        return None
    if rw <= 0 or rh <= 0:
        return None
    return rw, rh


def _pads_for_ratio(src_w, src_h, ratio_text, anchor):
    """把 src 增长到目标宽高比。永远只有一个轴增长，所以 UI 只需三个 anchor
    芯片而不是九宫格。返回 (t, b, l, r)。

    anchor 命名"新空间去哪边"：anchor "right" 在右边填充。这与
    resize_engine._anchor_offsets 的约定刻意相反（那边 anchor 命名图片贴在哪）。
    两个原因：anchor 行存在的意义是选单侧绿色 LoRA 重绘哪边，而 "sides" 模式
    已经是每边绿色（right: 512 = 右边 512px 绿色），两种模式下同一个词必须
    同义。不要把它"纠正"回 _anchor_offsets 的约定。"""
    r = _parse_ratio(ratio_text)
    if not r:
        return 0, 0, 0, 0
    rw, rh = r
    target = rw / rh
    cur = src_w / src_h if src_h else 1.0

    if abs(target - cur) < 1e-6:
        return 0, 0, 0, 0

    if target > cur:  # 更宽：横向增长
        add = _round_half_up(src_h * target) - src_w
        if add <= 0:
            return 0, 0, 0, 0
        if anchor in ("left", "top"):
            return 0, 0, add, 0
        if anchor in ("right", "bottom"):
            return 0, 0, 0, add
        half = add // 2
        return 0, 0, half, add - half

    # 用 _round_half_up 而非内建 round()：Python 的 round() 是银行家舍入
    # （round(1498.5) = 1498）而 JS Math.round 总是进位，用内建 round() 会让
    # 实时预览与真实输出在恰好 .5 边界上不一致——999 高的源在 3:2 下正好踩中。
    add = _round_half_up(src_w / target) - src_h  # 更高：纵向增长
    if add <= 0:
        return 0, 0, 0, 0
    if anchor in ("top", "left"):
        return add, 0, 0, 0
    if anchor in ("bottom", "right"):
        return 0, add, 0, 0
    half = add // 2
    return half, add - half, 0, 0


class SFImageOutpaint:
    DESCRIPTION = (
        "用纯色给图片四周扩展（外绘 Outpainting），可选按百万像素上限缩放并输出"
        "最终尺寸。填充区域供外绘模型生成新内容。\n\n"
        "默认填充色为中灰。任意颜色都可用，但强色填充可能让整张生成图带上色偏——"
        "训练在替换填充上的模型会把颜色连同形状一起学会。灰色是中性的，没有色相可渗。\n\n"
        "To ratio 模式把图片增长到目标宽高比，anchor（加空间位置）决定新区域出现在"
        "哪一边；By side 模式按边设置精确像素数。百万像素上限可选：关闭时保持填充后尺寸。\n\n"
        "width / height 输出报告最终尺寸，可直接接入空 Latent。outpaint_info 携带原始"
        "图片与位置信息，接给 SF Image Outpaint Stitch 可把原始图以全分辨率贴回。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "要填充扩展的图片。任意图片源都可接入。",
                }),
            },
            "hidden": {"SFOutpaintState": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", SF_OUTPAINT_INFO)
    RETURN_NAMES = ("image", "width", "height", "outpaint_info")
    OUTPUT_TOOLTIPS = (
        "填充（及设置百万像素上限时缩放）后的图片。",
        "最终宽度（像素），填充与缩放之后。",
        "最终高度（像素），填充与缩放之后。",
        "供 SF Image Outpaint Stitch 使用的信息——携带原始图片与其在填充画布中的"
        "位置，模型填充新区域后可把原始图以全质量贴回。可选，不接也可。",
    )
    FUNCTION = "outpaint"
    CATEGORY = _CATEGORY

    def outpaint(self, image, SFOutpaintState=""):
        st = _parse_state(SFOutpaintState)
        pils = _tensor_to_pils(image)
        if not pils:
            return (image, 0, 0, None)

        src_w, src_h = pils[0].size
        if st["mode"] == "ratio":
            t, b, l, r = _pads_for_ratio(src_w, src_h, st["ratio"], st["anchor"])
        else:
            t, b, l, r = st["top"], st["bottom"], st["left"], st["right"]

        # 在构建画布之前先把填充尺寸压到引擎上限之内。_apply_pad 以未夹紧的
        # 尺寸分配 Image.new，_clamp_dims 只把结果缩小——无界的 pad 会在 clamp
        # 运行前就分配数 GB 然后 MemoryError。两条可达路径：手改极端比例
        # （1:1000 -> 数百万像素，_pads_for_ratio 没有 _MAX_PAD 夹紧），以及
        # sides 模式四边全开 8192 配大源图（最大 32768² = 3 GB）。无论如何
        # 最终尺寸都会被夹到 16384，先收缩 pad 不损失任何东西，只避开巨分配。
        t, b = _fit_pad(t, b, src_h)
        l, r = _fit_pad(l, r, src_w)

        limit = st["limit"]
        # Snap 只触发一次。开 limit 时 pad 过程不吸附、max_mp 过程负责吸附；
        # 否则两边都吸附、后者与前者的结果互相打架。
        pad_state = {
            "pad_top": t, "pad_bottom": b, "pad_left": l, "pad_right": r,
            "pad_color": st["color"],
            "snap": 0 if limit else st["snap"],
            "resample": "auto",
        }
        mp_state = {
            "max_mp": limit,
            "allow_upscale": True,  # 精确缩放到上限，放大或缩小
            "snap": st["snap"],
            "resample": "auto",
        }

        out_frames = []
        out_w = out_h = 0
        for pil in pils:
            # _apply_pad 需要一个 mask；本节点没有 mask 输出，用一次性白板。
            blank = Image.new("L", pil.size, 0)
            rgb, msk, w, h = _apply_pad(pil, blank, pad_state, *pil.size)
            if limit:
                rgb, msk, w, h = _apply_max_mp(rgb, msk, mp_state, w, h)
            out_w, out_h = w, h
            out_frames.append(
                torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]
            )

        out = torch.cat(out_frames, dim=0).to(image.device)

        # 预览第二层。节点吃张量，浏览器看不见；上游是 Load Image 时前端有自己的
        # imgs[0] 可画，但 VAE Decode 之类链路中段不会填充任何东西，预览会永远
        # 空白。把输入帧经 temp/ 递过去（Text Overlay 就是这么做的，且它也不是
        # OUTPUT_NODE——证明普通节点的 ui payload 能到达 JS）。
        #
        # 刻意用全分辨率：预览按图片 naturalWidth 推算 pad 与尺寸徽章，
        # 缩小的存档会让两者一起说谎。
        ui = {}
        try:
            if folder_paths is not None:
                temp_dir = folder_paths.get_temp_directory()
                os.makedirs(temp_dir, exist_ok=True)
                # 每次运行全新 uuid 兼作缓存失效：复用一个名字会让浏览器显示
                # 上一轮的帧。
                fname = "sf_outpaint_base_%s.png" % uuid.uuid4().hex[:12]
                pils[0].save(os.path.join(temp_dir, fname), "PNG", optimize=False)
                ui["sf_outpaint_base"] = [
                    {"filename": fname, "subfolder": "", "type": "temp"}
                ]
        except Exception as e:
            # 预览永远不值得让一次真实运行失败。
            print("[SF Image Outpaint] base preview stash failed:", e)

        # ui 里的一切必须严格 JSON 安全。一个 NaN 就会让前端对整个 websocket
        # 消息的 JSON.parse 抛错，连带丢掉其他所有节点的 payload。这里只有
        # 纯字符串到达，无需消毒。
        #
        # outpaint_info 携带原始图（未改动的输入张量）与 _fit_pad 之后的各边
        # pad，Stitch 据此精确知道原始图在填充画布中的位置 (l, t) 与完整尺寸
        # (src_w, src_h)。canvas_w/h 是 _apply_pad 构建的尺寸（max_mp 之前）——
        # Stitch 把结果放大回这个尺寸。info 从不离开 Python（是类型化线，不在
        # ui 里），所以这里放张量没问题。
        info = {
            "original": image,
            "left": int(l), "top": int(t), "right": int(r), "bottom": int(b),
            "orig_w": int(src_w), "orig_h": int(src_h),
            "canvas_w": int(src_w + l + r), "canvas_h": int(src_h + t + b),
        }
        return {"ui": ui, "result": (out, int(out_w), int(out_h), info)}


# ── 色域匹配参数 ─────────────────────────────────────────────────────────────

# 色域匹配模糊半径，按原始图较小边的比例。~0.21 是基准测试甜点
# （D:\Claude Tests\_outpaint_colormatch_bench.py）：大到足以平均掉生成区自身的
# 纹理，小到仍能跟随墙面/地板色调边界。尺度不变，一个系数适用任意分辨率。
# 夹紧后极小/极大的图片保持合理。
_MATCH_BLUR_FRAC = 0.21
_MATCH_BLUR_MIN = 12
_MATCH_BLUR_MAX = 800

# 色域匹配接缝采样带，按模糊半径的比例。生成侧的色调只在紧贴接缝的细带内读取
# （而非整条区域），模型画在生成区内部一点的主体不会被采样、不能驱动修正
# （光晕 bug）。保持薄（~0.2R）：沿缝低通后足够稳定，又足够窄以避开近缝主体。
_MATCH_SEAM_BAND_FRAC = 0.2


class SFImageOutpaintStitch:
    DESCRIPTION = (
        "把原始图片贴回外绘结果，只保留模型新生成的部分。用于大图必须缩小后"
        "过模型的情况：原始半边以全质量回归，而不是经过模型软化的缩小版。\n\n"
        "把 SF Image Outpaint 的 outpaint_info 输出接入 outpaint_info，把成品"
        "（VAE Decode 之后）接入 image。节点会把结果放大回完整填充尺寸、把原始图"
        "精确贴回原位并融合接缝。\n\n"
        "feather 软化原始图与新区域之间的接缝。接缝不会完全隐形——新区域是与"
        "重编码的原始副本融合的，而非与原始图本身——所以少量羽化通常观感最佳。"
        "只有贴着新区域的边被软化，真实图片边缘保持锐利。\n\n"
        "color match 修正新区域与原始图交界处可能出现的颜色/色调阶差：把原始图"
        "沿接缝的颜色延续进新区域。它按区域跟随背景，因此能均化亮墙/暗地板场景，"
        "而不只是单一纯色。0 关闭，100 完全匹配（通常甜点），100 以上过度匹配，"
        "用于罕见的顽固阶差。它只均化色调，绝不碰纹理或细节，不会引入伪影。\n\n"
        "输出重组后的全分辨率图片，外加标记新生成区域的遮罩（白 = 生成，黑 = "
        "未动的原始图）——之后想只对新部分做轻量精修时很方便。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # 槽位顺序 image, outpaint_info, feather，与 SF Image Outpaint 的
        # image / outpaint_info 输出对齐（连线横着直通）。outpaint_info 可选，
        # 接错时降级为干净的透传而非崩溃（在 stitch() 中处理）。
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "外绘成品，VAE Decode 之后。会自动放大回完整填充尺寸，"
                        "所以可以是缩小后的生成尺寸。"
                    ),
                }),
            },
            "optional": {
                "outpaint_info": (SF_OUTPAINT_INFO, {
                    "tooltip": (
                        "接入 SF Image Outpaint 的 outpaint_info 输出。它携带原始"
                        "图片与其在填充画布中的位置，据此把原始图精确贴回。不接时"
                        "图片原样透传。"
                    ),
                }),
                "feather": ("INT", {
                    "default": 64, "min": 0, "max": 1024, "step": 1,
                    "tooltip": (
                        "按该像素数把原始图边缘向新区域羽化，软化接缝。"
                        "0 = 硬边（原始图完整保留到接缝）。更大值融合更柔，"
                        "但会在接缝处吃掉一点原始图。"
                    ),
                }),
                "color_match": ("INT", {
                    "default": 100, "min": 0, "max": 200, "step": 1,
                    "display": "slider",
                    "tooltip": (
                        "消除新生成区域与原始图交界处的颜色/色调阶差：把原始图"
                        "沿接缝的颜色延续进新区域。按区域跟随背景，能修正亮墙/"
                        "暗地板场景而非单一纯色。0 = 关，100 = 完全匹配（通常"
                        "甜点）；100 以上过度匹配，用于仍有阶差的罕见情况。"
                        "只均化色调，不碰纹理或细节，不会引入伪影。"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = (
        "全分辨率图片：模型新生成的区域 + 贴回原位、以全质量呈现的原始图。",
        "新生成区域的遮罩（白 = 生成，黑 = 未动的原始图），接缝羽化对齐。"
        "想只对新部分精修/重绘时可接入。",
    )
    FUNCTION = "stitch"
    CATEGORY = _CATEGORY

    # ─────────────────────────────────────────────────────────────────────

    def _resize_bhwc(self, t, target_w, target_h):
        """把图像张量 [B,H,W,C] 缩放到 [B,target_h,target_w,C]。双线性，
        与 SFImageUncrop 一致，两节点缩放行为相同。"""
        x = t.permute(0, 3, 1, 2)  # [B,C,H,W]
        x = F.interpolate(x, size=(int(target_h), int(target_w)),
                          mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()

    def _feather_sides(self, h, w, feather, left, top, right, bottom):
        """0..1 的 alpha 图 [h,w]：内部 1.0，只在标记为 True 的边（贴着新生成
        区域的填充边）向内按 feather 像素斜坡降到 0。标记为 False 的边保持硬边
        ——它们是真实图片边界，原始内容一直延伸到边缘，绝不能淡进生成区。

        与 SFImageUncrop._feather_alpha 相同的到边向内斜坡（box blur 会做错——
        在边缘卡在 ~0.5，出现可见的 50% 阶差），只作用于选定边，真实图片边缘
        保持锐利。feather<=0（或无填充边）= 全 1（处处硬边）。"""
        a = torch.ones((int(h), int(w)), dtype=torch.float32)
        k = int(feather)
        if k <= 0 or not (left or top or right or bottom):
            return a
        hh, ww = int(h), int(w)
        ys = torch.arange(hh, dtype=torch.float32).view(hh, 1)
        xs = torch.arange(ww, dtype=torch.float32).view(1, ww)
        # 斜坡永远够不到的"远"，未标记的边永不淡出。
        big = float(max(hh, ww) + k + 1)
        d_left = xs if left else torch.full((1, ww), big)
        d_right = ((ww - 1) - xs) if right else torch.full((1, ww), big)
        d_top = ys if top else torch.full((hh, 1), big)
        d_bottom = ((hh - 1) - ys) if bottom else torch.full((hh, 1), big)
        # [1,w] min [1,w] -> [1,w]；[h,1] min [h,1] -> [h,1]；再广播 [h,w]。
        dist = torch.minimum(torch.minimum(d_left, d_right),
                             torch.minimum(d_top, d_bottom))
        ramp = (dist / float(k)).clamp(0.0, 1.0)  # 填充边 0 -> k 像素内 1
        return (a * ramp).clamp(0.0, 1.0)

    def _passthrough_mask(self, image):
        """全零（黑 = "处处是原始图"，没有生成内容）遮罩，尺寸跟随图片、设备一致，
        用于无 info 透传的情况。"""
        dev = image.device if isinstance(image, torch.Tensor) else "cpu"
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            b, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
            return torch.zeros((b, h, w), dtype=torch.float32, device=dev)
        return torch.zeros((1, 1, 1), dtype=torch.float32, device=dev)

    def _box_blur_axis(self, x, R, dim):
        """x 沿 dim 的单轴 box blur，用累加和（积分图）做到 O(N)，大半径与小
        半径同价。边缘 = 收缩窗口（只平均实际存在的像素），无需填充。
        R 以像素计。"""
        n = int(x.shape[dim])
        if R < 1 or n < 2:
            return x
        R = min(int(R), n - 1)
        cs = torch.cumsum(x, dim=dim)
        zero = torch.zeros_like(x.narrow(dim, 0, 1))
        cs = torch.cat([zero, cs], dim=dim)  # 长 n+1，cs[k] = sum(x[0:k])
        idx = torch.arange(n, device=x.device)
        hi = torch.clamp(idx + R + 1, max=n)
        lo = torch.clamp(idx - R, min=0)
        cnt = (hi - lo).to(x.dtype).clamp(min=1.0)
        hi_v = torch.index_select(cs, dim, hi)
        lo_v = torch.index_select(cs, dim, lo)
        shape = [1] * x.dim()
        shape[dim] = n
        return (hi_v - lo_v) / cnt.view(shape)

    def _box_blur(self, x, R, passes=2):
        """[B,H,W,C] 的可分离 box blur。2 次近似高斯。1:1 移植色域匹配
        基准中的 numpy 模糊。"""
        out = x
        for _ in range(int(passes)):
            out = self._box_blur_axis(out, R, 1)  # H
            out = self._box_blur_axis(out, R, 2)  # W
        return out

    def _color_match(self, canvas, orig_use, left, top, right, bottom, strength01):
        """连续、内容盲的低频色域匹配：把整图的平滑色调朝原始图边缘向外延续的
        方向偏移，让生成区按区域吸收原始色调，无接缝——且不响应模型画了什么。

        ref = 原始图把边框向外涂抹铺满填充画布（最近边外推）：每个像素的目标
        色调（逐行与角点都免费覆盖）。

        src = 画布中各生成条带被替换成自己的接缝边细带、再向外涂抹的版本。
        关键就在这：只在接缝边界读生成色调（干净外绘里那是背景），模型画在
        生成区内部的主体永远不进低通，无法驱动修正。

        delta = lowpass(ref) - lowpass(src) 加到整个画布上。ref 与 src 都是
        接缝锚定的涂抹，所以 delta 按区域是接缝处的色调阶差，在生成条带内
        近似恒定——一个柔和、均匀、跟随区域的微调，绝不是内容形状的光晕。

        绝不能减 lowpass(canvas)：对原始画布做模糊会把模型的新内容拖进低通，
        生成区里的亮/暗主体会让 delta 在其上及其周围巨大化，色域匹配画出一片
        随强度增长、把主体洗掉的光晕（2026-07-21 用户反馈：暗舞者随 cm 64/127/200
        爬升到接近墙面亮度）。接缝带 src 修正它——主体不受影响，背景阶差仍被修正。

        必须跨接缝连续——不能重新加一个只位移生成区的硬性 * G mask。那会让羽化
        带的软像素不动、旁边的条带却位移，接缝出现随强度增长的硬边（2026-07-20
        早期 bug）。施加平滑的全画布 delta 让条带与羽化带同步移动。原始图深处
        delta 自消（那里 lowpass(ref) == lowpass(src)），且之后原始图会被贴回
        覆盖，内部保持像素级精确；只有羽化带残留平滑残差、斜坡过渡到原始副本。

        无填充边 -> 无处可延续 -> 原样返回画布。"""
        if not (left > 0 or top > 0 or right > 0 or bottom > 0):
            return canvas
        H, W = int(canvas.shape[1]), int(canvas.shape[2])
        oh, ow = int(orig_use.shape[1]), int(orig_use.shape[2])
        dev = canvas.device

        ys = (torch.arange(H, device=dev) - int(top)).clamp(0, oh - 1)
        xs = (torch.arange(W, device=dev) - int(left)).clamp(0, ow - 1)
        ref = orig_use.index_select(1, ys).index_select(2, xs)  # [b,H,W,C]

        R = int(min(_MATCH_BLUR_MAX, max(_MATCH_BLUR_MIN,
                                         round(_MATCH_BLUR_FRAC * min(oh, ow)))))

        # 构建 src：从画布出发，在每个填充（生成）条带里把色调替换为该条带
        # 接缝生成侧细带的均值并涂抹整条。细带上限为该边 pad 宽，绝不溢出
        # 条带。只在接缝边界读值，让修正对生成区内的主体盲视。
        #
        # 角点：横竖两个 pad 重叠处（如 top + right），两条带都覆盖角。累加
        # 各边的贡献 + 计数再相除，角 = 两条带均值——与顺序无关，而非后者覆盖
        # 前者。cnt 在普通条带为 1（src = 该带，与单边填充相同），角为 2，
        # 原始区为 0（src 保持 = 画布，delta 在那里自消）。
        ox0, ox1 = int(left), int(left) + ow          # 原始矩形，x 跨度（画布）
        oy0, oy1 = int(top), int(top) + oh            # 原始矩形，y 跨度（画布）
        bw = max(2, int(round(R * _MATCH_SEAM_BAND_FRAC)))
        # anchor 映射把每条带的涂抹夹到原始图跨度内，让推入角点的条带延续最近
        # 直边接缝的色调（锚定在真实原始-相邻背景上）而非采样角点自己的内容。
        # 没有它，模型画在角里的主体会污染该角的条带、在角落重新引入光晕。
        # 单轴填充时垂直跨度是整张画布，夹紧是恒等（无变化）——只有真角点
        # （相邻两边填充）受影响。row_anchor 修正左右（逐行）条带，col_anchor
        # 修正上下（逐列）条带。
        row_anchor = torch.arange(H, device=dev).clamp(oy0, oy1 - 1)
        col_anchor = torch.arange(W, device=dev).clamp(ox0, ox1 - 1)
        acc = torch.zeros_like(canvas)
        cnt = torch.zeros((canvas.shape[0], H, W, 1), device=dev, dtype=canvas.dtype)
        if right > 0:
            k = max(1, min(bw, int(right)))
            band = canvas[:, :, ox1:ox1 + k, :].mean(dim=2, keepdim=True).index_select(1, row_anchor)
            acc[:, :, ox1:W, :] += band
            cnt[:, :, ox1:W, :] += 1.0
        if left > 0:
            k = max(1, min(bw, int(left)))
            band = canvas[:, :, ox0 - k:ox0, :].mean(dim=2, keepdim=True).index_select(1, row_anchor)
            acc[:, :, 0:ox0, :] += band
            cnt[:, :, 0:ox0, :] += 1.0
        if top > 0:
            k = max(1, min(bw, int(top)))
            band = canvas[:, oy0 - k:oy0, :, :].mean(dim=1, keepdim=True).index_select(2, col_anchor)
            acc[:, 0:oy0, :, :] += band
            cnt[:, 0:oy0, :, :] += 1.0
        if bottom > 0:
            k = max(1, min(bw, int(bottom)))
            band = canvas[:, oy1:oy1 + k, :, :].mean(dim=1, keepdim=True).index_select(2, col_anchor)
            acc[:, oy1:H, :, :] += band
            cnt[:, oy1:H, :, :] += 1.0
        # 有边覆盖的像素用平均带，其余保留原始画布。
        src = torch.where(cnt > 0, acc / cnt.clamp(min=1.0), canvas)

        delta = self._box_blur(ref, R) - self._box_blur(src, R)   # 连续 + 内容盲
        return canvas + delta * float(strength01)

    def stitch(self, image, outpaint_info=None, feather=64, color_match=100):
        # 无/非法 outpaint_info -> 没有可贴的，图片透传 + 黑 mask（绝不因接错线
        # 崩溃，与 SFImageUncrop 一致）。
        if (not isinstance(outpaint_info, dict)
                or not isinstance(outpaint_info.get("original"), torch.Tensor)):
            print("[SF Image Outpaint Stitch] no outpaint_info wired - passing image through")
            return (image, self._passthrough_mask(image))

        original = outpaint_info["original"]
        if original.dim() != 4 or not isinstance(image, torch.Tensor) or image.dim() != 4:
            return (image, self._passthrough_mask(image))

        oh, ow = int(original.shape[1]), int(original.shape[2])  # 原始 H, W
        left = max(0, int(outpaint_info.get("left", 0)))
        top = max(0, int(outpaint_info.get("top", 0)))
        right = max(0, int(outpaint_info.get("right", 0)))
        bottom = max(0, int(outpaint_info.get("bottom", 0)))

        # 结果映射回的完整填充画布。由原始尺寸 + （已被 _fit_pad 夹紧的）pad
        # 计算，精确等于 _apply_pad 构建的尺寸，在 (left, top) 的粘贴必然落位。
        # 开百万像素上限（本节点存在的意义）时 pad 过程不吸附，这是像素级精确
        # 的；唯一的漂移是罕见的无上限 + snap 组合，由 feather 覆盖。
        canvas_w = ow + left + right
        canvas_h = oh + top + bottom

        # 把结果放大到完整填充尺寸——"恢复分辨率"步骤。结果已经是该尺寸时
        # （例如没用百万像素上限）是空操作。
        canvas = image
        if int(canvas.shape[1]) != canvas_h or int(canvas.shape[2]) != canvas_w:
            canvas = self._resize_bhwc(canvas, canvas_w, canvas_h)

        # 对齐通道（丢 alpha 等），保证粘贴逐像素对齐。
        orig_use = original
        if canvas.shape[3] != orig_use.shape[3]:
            c = min(int(canvas.shape[3]), int(orig_use.shape[3]))
            canvas = canvas[..., :c]
            orig_use = orig_use[..., :c]

        canvas = canvas.clone()  # 往里面粘贴

        # 批次配对（SFImageUncrop 的做法）：把原始批次对齐到结果批次，
        # 多帧运行成对配对而非崩溃。
        b = int(canvas.shape[0])
        ob = int(orig_use.shape[0])
        if ob != b:
            if ob == 1:
                orig_use = orig_use.repeat(b, 1, 1, 1)
            elif b == 1:
                canvas = canvas.repeat(ob, 1, 1, 1)
                b = ob
            else:
                n = min(b, ob)
                canvas = canvas[:n]
                orig_use = orig_use[:n]
                b = n
        orig_use = orig_use.to(canvas.device, canvas.dtype)

        # 可选色域匹配：在粘贴之前，把原始图的边缘色调延续进生成区，让接缝
        # 阶差消失——亮墙/暗地板背景也能修，单一全局偏移做不到（见 _color_match
        # 与基准测试）。只位移平滑色调，从不碰纹理，不引入伪影，且只触及生成区
        # （随后原始图会被贴回覆盖）。100 = 完全匹配（甜点）；>100 过冲，仅当
        # 顽固阶差还在时有用。0 -> 跳过，输出与加功能前的节点逐字节一致。
        cm = max(0, min(200, int(color_match)))
        if cm > 0:
            canvas = self._color_match(
                canvas, orig_use, left, top, right, bottom, cm / 100.0)

        # 边选择性羽化：只把填充边淡入新区域。
        alpha = self._feather_sides(oh, ow, feather,
                                    left > 0, top > 0, right > 0, bottom > 0)  # [oh,ow] cpu
        a = alpha[None, ..., None].to(canvas.device, canvas.dtype)  # [1,oh,ow,1]

        region = canvas[:, top:top + oh, left:left + ow, :]
        canvas[:, top:top + oh, left:left + ow, :] = orig_use * a + region * (1.0 - a)

        # 遮罩：1 = 生成（可安全精修），0 = 原始图。原始矩形内为 1 - alpha，
        # 内部 0、到生成接缝处斜坡升到 1，与图像混合完全一致。
        mask = torch.ones((1, canvas_h, canvas_w), dtype=torch.float32)
        mask[:, top:top + oh, left:left + ow] = (1.0 - alpha)[None, ...]
        mask = mask.clamp(0.0, 1.0).to(canvas.device)
        # [1,H,W] -> [b,H,W]。repeat 的参数必须恰好等于 dim 数，多一个 4th 1
        # 会在前面加一个维，得到错误的 [b,1,H,W] 遮罩。
        if mask.shape[0] == 1 and b > 1:
            mask = mask.repeat(b, 1, 1)

        return (canvas.clamp(0.0, 1.0), mask)
