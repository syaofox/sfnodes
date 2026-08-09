"""SF Inpaint Crop / SF Inpaint Stitch — 移植自 comfyui-pixaroma
node_inpaint_crop.py + node_inpaint_stitch.py（PixaromaInpaintCrop /
PixaromaInpaintStitch）。

Crop: 全屏遮罩编辑器（web/sf_inpaint*.js）里涂抹待修复区域，节点自动找遮罩
外接框、加上下文边距、按模型友好的尺寸（8 的倍数、目标像素）裁剪。源图可来自
接入的 IMAGE、拖放、Ctrl+V 粘贴或编辑器内 Load Image。输出裁剪图、匹配遮罩、
crop_info 连线（与 SF Image Crop 同类型，可互换）与宽高。

Stitch: 把修复后的裁剪图按记录区域贴回原图，接缝无缝混合；softness / blend_mode
可在缝合节点覆盖（在采样器之后，调优不重跑采样器），color_match 校正模型引入的
色调漂移。输出完成图 + 原图（供前后对比）。

磁盘状态位于 input/sfnodes_inpaint/（路由守卫：dataURL 上传 + 每次读取 safe_join）。
编辑器画的遮罩与粘贴/拖放的源图都存在这里，state_json 随 workflow 保存。
"""

import json
import os
import uuid

import numpy as np
import torch
from PIL import Image
from aiohttp import web

import folder_paths

from ...sf_utils.common import AnyType
from ...sf_utils.disk_state import safe_join
from ...sf_utils.disk_state import sanitize_id as _sanitize_id
from ...sf_utils.disk_state import decode_image as _decode_image
from ...sf_utils.inpaint_helpers import (
    SF_CROP_INFO,
    DEFAULTS,
    apply_inpaint_crop,
    merge_params,
    resolve_inpaint_mask,
    resolve_seam,
    stitch_back,
)
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)

_CATEGORY = "sfnodes/inpaint"

any_type = AnyType("*")

# 磁盘状态根（ComfyUI input/ 内，与其它插件隔离）
_INPAINT_SUBDIR = "sfnodes_inpaint"


def _inpaint_dir() -> str:
    d = os.path.join(folder_paths.get_input_directory(), _INPAINT_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def _safe_join(rel: str) -> str:
    """把保存的相对路径解析为 input/ 目录下的绝对路径，越界或不存在返回 None。

    薄别名（sf_utils.disk_state 共享实现）：解析根是 input/ 本身，路径自带
    sfnodes_inpaint/ 前缀天然兼容，无需剥前缀。"""
    return safe_join(folder_paths.get_input_directory(), rel)


# _sanitize_id / _decode_image 为 sf_utils.disk_state 共享实现（import 别名）。


# ── 节点类 ────────────────────────────────────────────────────────────────


def _inpaint_meta_from_widget(state) -> dict:
    """把状态输入（无论前端以何种形状发送）解析为 meta dict。兼容：
      - dict {"state_json": "<json>"}         (DOM widget 值形状)
      - dict 直接含 project_id 的 meta
      - str  "<json>"                         (部分前端把 widget 值序列化为字符串)
    """
    if state is None:
        return {}
    raw = state
    if isinstance(raw, dict):
        sj = raw.get("state_json")
        if isinstance(sj, str):
            raw = sj
        else:
            return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class _InpaintOptionalInputs(dict):
    """Any-type 可选输入，同时声明具体的 IMAGE / MASK 输入（schema 里列出后，
    从 IMAGE 输出拖线搜索节点时能找到它），隐藏的编辑器状态值经 any_type
    回退进入 kwargs——与 Image Crop 的 _CropOptionalInputs 同一招。"""

    def __init__(self, type):
        super().__init__()
        self.type = type
        self["image"] = ("IMAGE", {
            "tooltip": "接入上游 IMAGE 进行修复（LoadImage、VAE Decode、任意来源）。也可以把图片文件拖到节点上或 Ctrl+V 粘贴——那些方式会直接加载图片并断开此连线。",
        })
        self["mask"] = ("MASK", {
            "tooltip": "可选。待修复区域的遮罩（如透明 PNG 的 alpha，或任何 MASK 输出）。编辑器里没画遮罩时按原样使用它——所以清空编辑器会回退到这个接入的遮罩。编辑器里画的遮罩优先。",
        })

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        return (self.type,)


class SFInpaintCrop:
    DESCRIPTION = (
        "可视化修复裁剪：打开全屏编辑器，在要修复的区域上涂抹遮罩（画笔/橡皮/清空/"
        "反选，可调笔刷大小）。节点自动找到遮罩外接框，加上上下文边距，裁剪出一块"
        "模型友好的图（8 的倍数、按目标尺寸缩放，小遮罩区域也能拿到足够分辨率）。\n\n"
        "打开 invert_mask 可翻转遮罩修复**相反**区域（如抠图背景而非主体），无需"
        "再接 Invert Mask 节点。\n\n"
        "把裁剪图和遮罩接入修复模型（KSampler、Flux、编辑模型），再把 crop_info 连线"
        "接入 SF Inpaint Stitch 把结果贴回原图精确位置。\n\n"
        "输出裁剪图、匹配的裁剪遮罩、crop_info 连线与裁剪宽高（可用于空 latent）。"
        "crop_info 与 SF Image Crop 同类型，两条链路可互换。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (
                    ["keep shape (long side)", "force size (square)", "free (multiple only)"],
                    {
                        "default": "keep shape (long side)",
                        "tooltip": (
                            "Keep shape: 遮罩区域长边缩放到目标尺寸，不拉伸（质量最佳）。"
                            "Force size: 恒输出 target x target 方块。Free: 自然尺寸，"
                            "仅对齐到 multiple。"
                        ),
                    },
                ),
                "target": (
                    "INT",
                    {
                        "default": 1024, "min": 64, "max": 8192, "step": 8,
                        "tooltip": "目标长边（keep）或方块边长（force），单位像素。",
                    },
                ),
                "multiple": (
                    [8, 16, 32, 64],
                    {
                        "default": 8,
                        "tooltip": "裁剪尺寸对齐到该倍数，兼容模型。",
                    },
                ),
                "context_px": (
                    "INT",
                    {
                        "default": 24, "min": 0, "max": 1024, "step": 1,
                        "tooltip": "每侧额外包含多少像素的周围上下文。",
                    },
                ),
                "mask_grow": (
                    "INT",
                    {
                        "default": 4, "min": 0, "max": 256, "step": 1,
                        "tooltip": "裁剪前把画的遮罩向外扩展这么多像素。",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 4, "min": 0, "max": 256, "step": 1,
                        "tooltip": "软化输出遮罩边缘这么多像素，修复过渡更平滑。",
                    },
                ),
                "softness": (
                    "INT",
                    {
                        "default": 16, "min": 0, "max": 150, "step": 1,
                        "tooltip": (
                            "SF Inpaint Stitch 贴回时接缝羽化的距离。编辑器里实时预览。"
                        ),
                    },
                ),
                "blend_mode": (
                    ["mask", "whole crop"],
                    {
                        "default": "mask",
                        "tooltip": (
                            "SF Inpaint Stitch 贴回方式。'mask': 只替换画过的区域，"
                            "裁剪图其余部分保留原图（常规修复，安全默认）。'whole crop': "
                            "整个裁剪区域都被模型版本替换（模型也改了周围光照/环境时，"
                            "或对整块裁剪做 img2img 式处理时使用）。编辑器里实时预览。"
                        ),
                    },
                ),
                "invert_mask": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "翻转遮罩，修复**相反**区域（主体与背景互换）。对接入的遮罩"
                            "或画的遮罩均生效——内置的 Invert Mask 节点替代品。未接遮罩"
                            "时无效果。"
                        ),
                    },
                ),
            },
            "optional": _InpaintOptionalInputs(any_type),
            # 隐藏状态输入：必须在 Python 侧声明，否则前端 validatePrompt 会把不在
            # 节点 schema 中的输入从 prompt 剥离。前端同名隐藏 STRING widget 的值
            # 走标准 widget 收集通道（AGENTS.md §widget 值传后端经验）。
            "hidden": {
                "SFInpaintJson": ("STRING", {"default": "{}"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", SF_CROP_INFO, "INT", "INT")
    RETURN_NAMES = ("image", "mask", "crop_info", "width", "height")
    OUTPUT_TOOLTIPS = (
        "裁剪区域，已缩放到模型友好的输出尺寸。",
        "同尺寸的裁剪遮罩（按设置膨胀和模糊过）。接入 SetLatentNoiseMask / 修复 conditioning",
        "供 SF Inpaint Stitch 使用的裁剪信息——携带原图与裁剪位置，修复结果可精确贴回。与 SF Image Crop 同类型，可互换",
        "裁剪输出宽度（像素）",
        "裁剪输出高度（像素）",
    )
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """任一参数变化或编辑器遮罩/源图文件变化时重跑。上游 IMAGE 变化已由
        ComfyUI 的输入哈希机制捕获；磁盘源另以 mtime 参与键值。"""
        parts = [str(kwargs.get(k)) for k in (
            "size_mode", "target", "multiple", "context_px", "mask_grow",
            "mask_blur", "softness", "blend_mode", "invert_mask",
        )]
        # SFInpaintJson（隐藏 STRING widget，标准 widget 值通道）优先——Vue 前端
        # 对 DOM widget（InpaintCropWidget）的值收集在某些版本/渲染器下不可靠；
        # 回退 InpaintCropWidget 兼容旧 prompt。
        state = kwargs.get("SFInpaintJson")
        if state is None:
            state = kwargs.get("InpaintCropWidget")
        try:
            meta = _inpaint_meta_from_widget(state)
            parts.append(json.dumps(meta, sort_keys=True))  # 键序无关，结果确定
            mp = meta.get("mask_path", "")
            if mp:
                full = _safe_join(mp)
                if full:
                    parts.append(str(os.path.getmtime(full)))
            sp = meta.get("src_path", "")
            if sp:
                fs = _safe_join(sp)
                if fs:
                    parts.append(str(os.path.getmtime(fs)))
        except Exception:
            parts.append(str(state))
        return "|".join(parts)

    def _save_source_temp(self, tensor):
        """把输入张量（batch 槽 0）存到 ComfyUI temp/ 下的 UUID 命名 PNG，
        前端编辑器 + 迷你预览经 /view?type=temp 拉取。尽力而为。"""
        try:
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 4 or tensor.shape[0] == 0:
                return None
            arr = tensor[0].clamp(0.0, 1.0).cpu().numpy()
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
            img = Image.fromarray(arr)
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            fname = f"sf_inpaint_src_{uuid.uuid4().hex}.png"
            img.save(os.path.join(temp_dir, fname), "PNG")
            return fname
        except Exception as e:
            logger.warning(f"临时源图保存失败: {e}")
            return None

    def _load_disk_image(self, rel_path):
        full = _safe_join(rel_path)
        if not full:
            return None
        try:
            arr = np.array(Image.open(full).convert("RGB")).astype(np.float32) / 255.0
            return torch.from_numpy(arr)[None,]
        except Exception as e:
            logger.warning(f"源图加载失败: {e}")
            return None

    def _load_disk_mask(self, rel_path):
        full = _safe_join(rel_path)
        if not full:
            return None
        try:
            arr = np.array(Image.open(full).convert("L")).astype(np.float32) / 255.0
            return torch.from_numpy(arr)[None,]
        except Exception as e:
            logger.warning(f"遮罩加载失败: {e}")
            return None

    def _empty(self):
        img = torch.ones((1, 1024, 1024, 3), dtype=torch.float32)
        mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        info = {"image": img, "mask": mask, "x": 0, "y": 0, "w": 1024, "h": 1024,
                "orig_w": 1024, "orig_h": 1024}
        return (img, mask, info, 1024, 1024)

    def _params(self, size_mode, target, multiple, context_px, mask_grow, mask_blur):
        mode = {
            "keep shape (long side)": "keep",
            "force size (square)": "force",
            "free (multiple only)": "free",
        }.get(size_mode, "keep")
        p = dict(DEFAULTS)
        p.update({
            "size_mode": mode, "target": int(target),
            "target_w": int(target), "target_h": int(target),
            "multiple": int(multiple), "context_px": int(context_px),
            "mask_grow": int(mask_grow), "mask_blur": int(mask_blur),
        })
        return merge_params(p)

    def run(self, size_mode="keep shape (long side)", target=1024, multiple=8,
            context_px=24, mask_grow=4, mask_blur=4, softness=16, blend_mode="mask",
            invert_mask=False, **kwargs):
        upstream = kwargs.get("image")
        upstream_mask = kwargs.get("mask")
        state = kwargs.get("SFInpaintJson")
        if state is None:
            state = kwargs.get("InpaintCropWidget")

        meta = {}
        if state is not None:
            try:
                parsed = _inpaint_meta_from_widget(state)
                if parsed:
                    meta = parsed
            except Exception as e:
                logger.warning(f"状态解析失败: {e}")

        # ── 源图：接入的 IMAGE 优先，否则编辑器存到磁盘的 src ──
        ui_payload = None
        image = upstream if isinstance(upstream, torch.Tensor) else None
        if image is not None:
            src_fname = self._save_source_temp(image)
            if src_fname:
                ui_payload = {"sf_inpaint_source": [
                    {"filename": src_fname, "subfolder": "", "type": "temp"}]}
        else:
            image = self._load_disk_image(meta.get("src_path", ""))
            # 磁盘源（粘贴 / 拖放 / 编辑器 Load Image）也向执行期事件暴露源帧，
            # 否则前端 executed 事件收不到 sf_inpaint_source，节点预览缓存停在
            # 旧图（运行结果正确但预览不刷新）。帧指向 input/sfnodes_inpaint/。
            if image is not None:
                src_path = meta.get("src_path", "")
                if src_path:
                    ui_payload = {"sf_inpaint_source": [
                        {"filename": os.path.basename(src_path.replace("\\", "/")),
                         "subfolder": _INPAINT_SUBDIR, "type": "input"}
                    ]}

        if not isinstance(image, torch.Tensor):
            return self._empty()

        # ── 遮罩：编辑器画的遮罩胜出；清空/全黑的编辑器遮罩（mask_path 仍在）
        # 回退到接入的遮罩，所以清空编辑器后接入遮罩原样生效。resolve_inpaint_mask
        # 拥有这套规则。
        disk_mask = self._load_disk_mask(meta.get("mask_path", ""))
        mask = resolve_inpaint_mask(disk_mask, upstream_mask)

        params = self._params(size_mode, target, multiple, context_px, mask_grow, mask_blur)
        # 接缝软度（节点 softness 旋钮，编辑器同步镜像）：喂给几何，使裁剪
        # 上下文按 max(context_px, blend) 扩张（Option B——compute_region 内部），
        # 再由 crop_info 带到 SF Inpaint Stitch 作为接缝羽化宽度。
        sb = max(0, min(150, int(softness)))
        params["blend"] = sb
        params["invert_mask"] = bool(invert_mask)   # 裁剪前翻转遮罩
        try:
            img_t, mask_t, crop_info, ow, oh = apply_inpaint_crop(image, mask, params)
        except Exception as e:
            logger.warning(f"裁剪失败: {e}")
            return self._empty()

        # blend_mode 现在是节点 widget（编辑器 Blend mode 胶囊镜像它），节点
        # widget 是事实来源，不再读 state_json。
        crop_info["blend"] = sb
        crop_info["blend_mode"] = "whole_crop" if blend_mode == "whole crop" else "mask"

        result = (img_t, mask_t, crop_info, ow, oh)
        if ui_payload:
            return {"ui": ui_payload, "result": result}
        return result


class SFInpaintStitch:
    DESCRIPTION = (
        "把修复后的裁剪图贴回原图精确位置，接缝无缝混合消失。\n\n"
        "把 SF Inpaint Crop 的 crop_info 输出接入 crop_info，修复后的裁剪图（模型"
        "之后）接入 image。节点把裁剪图缩放回原区域，默认只混合画过的区域，遮罩外"
        "的一切保持像素级原样。\n\n"
        "接缝软度与 blend 模式随 crop_info 来自 SF Inpaint Crop 节点，但可以在这里"
        "**覆盖**（softness -1 = 用裁剪节点的值）。因为本节点在采样器之后，改 softness、"
        "blend 模式或 color match 只重跑本节点（固定 seed 的采样器保持缓存），可即时"
        "微调混合而无需重新生成。color match 校正模型引入的色彩/色调偏移。\n\n"
        "输出完成的全图，外加原未裁剪图——两条线接入 SF Image Compare 即可即时"
        "前后对比。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        # 槽位顺序 image, mask, crop_info 与 Inpaint Crop 的 image / mask /
        # crop_info 输出对齐，连线横平竖直。
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "修复后的裁剪图（模型之后）。尺寸与原裁剪区域不同时自动缩放。",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "可选。把混合限制在这个区域内（缩放到裁剪区域尺寸）。不接时用 crop_info 里携带的画过的遮罩做遮罩感知混合。",
                }),
                "crop_info": (SF_CROP_INFO, {
                    "tooltip": "接入 SF Inpaint Crop 的 crop_info 输出。携带原图与裁剪位置，修复结果可精确贴回。不接时图片直接透传。",
                }),
                "softness": ("INT", {
                    "default": -1, "min": -1, "max": 150, "step": 1,
                    "tooltip": (
                        "接缝羽化，覆盖 SF Inpaint Crop 节点的 softness 值。"
                        "-1 = 用裁剪节点的值。设 0-150 可在这里调混合——因为 Stitch 在"
                        "采样器之后，只有本节点重跑（固定 seed 的采样器保持缓存），即时生效。"
                        "比裁剪上下文留出的空间更大时边缘可能略硬——调大裁剪节点的"
                        "softness 留出更多空间。"
                    ),
                }),
                "blend_mode": (["from crop", "mask", "whole crop"], {
                    "default": "from crop",
                    "tooltip": (
                        "覆盖裁剪节点的 blend 模式。'from crop' = 用裁剪节点设的。"
                        "'mask' = 只替换画过的区域。'whole crop' = 替换整个裁剪框。"
                        "和 softness 一样，在这里改只重跑本节点（不重新采样）。"
                    ),
                }),
                "color_match": (["off", "subtle", "strong"], {
                    "default": "off",
                    "tooltip": (
                        "校正模型引入的色彩/色调偏移，匹配遮罩周围未变化的区域。"
                        "故意改色时保持 Off（会把颜色拉回去）。无实时预览——设置后重跑。"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "original")
    OUTPUT_TOOLTIPS = (
        "修复裁剪图混合回原位后的原图。",
        "完整原未裁剪图（来自 crop_info）——与结果一起接入 SF Image Compare 做前后对比。",
    )
    FUNCTION = "run"
    CATEGORY = _CATEGORY

    def run(self, image, crop_info=None, mask=None, softness=-1,
            blend_mode="from crop", color_match="off"):
        # 无有效 crop_info -> 没有可贴回的东西；把 image 作为两个输出透传，
        # 下游连线仍可工作。要求几何键齐全，残缺 dict 不会静默贴在 (0,0) 整图。
        if (not isinstance(crop_info, dict)
                or not isinstance(crop_info.get("image"), torch.Tensor)
                or crop_info["image"].dim() != 4
                or any(k not in crop_info for k in ("x", "y", "w", "h"))):
            # 残缺/缺失 crop_info（含非 [B,H,W,C] 的 image）-> 双输出透传，图仍可跑。
            # 这里查秩是防坏图进入 stitch_back（否则会抛错，下面 except 会把
            # `original` 静默变成结果的副本）。
            logger.info("未接入有效 crop_info - 图片透传")
            return (image, image)

        # 接缝混合 + 模式随 crop_info 来自裁剪节点，但本节点的 softness /
        # blend_mode widget 设置时覆盖它们（混合可在这里调优而无需重跑采样器）。
        # color_match 是本节点自己的旋钮（结果后微调，无实时预览）。
        blend, bm = resolve_seam(crop_info, softness, blend_mode)
        cm = str(color_match)
        color_match = cm if cm in ("off", "subtle", "strong") else "off"

        try:
            result, original = stitch_back(crop_info, image, mask, blend, bm, color_match)
        except Exception as e:
            # stitch_back 内部真实故障（如 CUDA OOM）。修复图透传为结果，但
            # `original` 保留**真正的**未裁剪原图（crop_info["image"] 已通过上面
            # dim==4 守卫），下游前后对比看到的是真原图而非补丁副本。
            logger.warning(f"缝合失败: {e}")
            return (image, crop_info["image"])
        return (result, original)


# ── 路由 ────────────────────────────────────────────────────────────────


def _register_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/inpaint/upload_src")
        async def _upload_inpaint_source(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            project_id = _sanitize_id(data.get("project_id", ""), uuid.uuid4().hex)
            img = _decode_image(data.get("image", ""))
            if img is None:
                return web.json_response({"error": "Invalid image data"}, status=400)
            filename = f"inpaint_src_{project_id}.png"
            try:
                img.convert("RGB").save(os.path.join(_inpaint_dir(), filename), "PNG")
            except Exception:
                return web.json_response({"error": "Save failed"}, status=500)
            return web.json_response({
                "status": "success",
                "path": os.path.join(_INPAINT_SUBDIR, filename).replace("\\", "/"),
            })

        @routes.post("/api/sfnodes/inpaint/save_mask")
        async def _save_inpaint_mask(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            project_id = _sanitize_id(data.get("project_id", ""), uuid.uuid4().hex)
            img = _decode_image(data.get("mask", ""))
            if img is None:
                return web.json_response({"error": "Invalid mask data"}, status=400)
            filename = f"inpaint_mask_{project_id}.png"
            try:
                # 画的遮罩：白色 = 这里要修复。存为 8 位灰度。
                img.convert("L").save(os.path.join(_inpaint_dir(), filename), "PNG")
            except Exception:
                return web.json_response({"error": "Save failed"}, status=500)
            return web.json_response({
                "status": "success",
                "path": os.path.join(_INPAINT_SUBDIR, filename).replace("\\", "/"),
            })
    except Exception:
        pass


_register_routes()
