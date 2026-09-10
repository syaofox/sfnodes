"""SF Image Brush Mask — SAM 右键图层（核心 SAM3_Detect 委托）。

在刷子节点上经右键菜单用文本 prompt 跑 SAM 分割，结果转为 fill 矢量笔触
直接并入笔触列表——与手绘统一管理（添加/擦除/撤销全通用，不再区分
SAM 层与手绘层，见 §45.9）。位图→矢量的轮廓追踪在
``sf_utils/brush_mask.py::mask_to_fill_strokes``（cv2）。

委托链（全部 ComfyUI 核心 API，零新依赖、零第三方模型代码）：
  1. ``folder_paths.get_full_path_or_raise("checkpoints", CKPT)``
     —— ``sam3.1_multiplex_fp16.safetensors``，core 判型 SAM31；
  2. ``comfy.sd.load_checkpoint_guess_config`` 同文件拆出 MODEL + CLIP；
  3. ``nodes.CLIPTextEncode().encode`` 编码 prompt（空按 RMBG 惯例回退 "object"）；
  4. ``comfy_extras.nodes_sam3.SAM3_Detect.execute``
     （threshold/refine_iterations=2/individual_masks=False 取并集）。

模型/CLIP 按 checkpoint 常驻缓存（复用 core 的 cached_patcher_init 机制之余，
模块级再包一层避免重复拆文件）；右键“卸载 SAM 模型”清缓存 + empty_cache。
全部 import 均在函数内 lazy——RMBG/旧版 core 缺席时节点照常加载，
点击菜单才报清晰中文错误。

路由（副作用注册，改动需重启容器；与 crop.py 同款 try/except 包裹）：
  POST /api/sfnodes/brush_mask/sam         {src_path, prompt, threshold}
  POST /api/sfnodes/brush_mask/sam_unload  {}
  GET  /api/sfnodes/brush_mask/sam_status  （门禁诊断：core/模型/已加载三态）
"""

import os

import numpy as np
import torch
from PIL import Image
from aiohttp import web

import folder_paths

from .crop import _safe_join

_CKPT = "sam3.1_multiplex_fp16.safetensors"

_cache = {}  # ckpt_path -> (model_patcher, clip)


def sam_status():
    """可用性三态（ pure readout，供菜单门禁与诊断）。"""
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect  # noqa: F401
    except Exception as e:
        return {"available": False, "reason": f"当前 ComfyUI 核心无 SAM3_Detect（{e}），请升级 ComfyUI",
                "model_found": False, "loaded": False}
    try:
        ckpt = folder_paths.get_full_path("checkpoints", _CKPT)
    except Exception:
        ckpt = None
    if not ckpt or not os.path.isfile(ckpt):
        return {"available": False, "reason": f"模型缺失：请把 {_CKPT} 放入 models/checkpoints/",
                "model_found": False, "loaded": False}
    return {"available": True, "reason": "", "model_found": True,
            "loaded": ckpt in _cache, "ckpt": _CKPT}


def _load_stack():
    """加载（MODEL, CLIP）并常驻缓存。缺 core/模型时抛中文 RuntimeError。"""
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"当前 ComfyUI 核心无 SAM3_Detect（{e}），请升级 ComfyUI")
    import comfy.sd
    try:
        ckpt = folder_paths.get_full_path_or_raise("checkpoints", _CKPT)
    except Exception:
        raise RuntimeError(f"模型缺失：请把 {_CKPT} 放入 models/checkpoints/")
    if ckpt not in _cache:
        model, clip, *_ = comfy.sd.load_checkpoint_guess_config(
            ckpt, output_vae=False, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            output_model=True)
        if model is None or clip is None:
            raise RuntimeError(f"模型解析失败：{_CKPT} 未拆出 MODEL/CLIP")
        _cache[ckpt] = (model, clip)
    return _cache[ckpt]


def unload_sam():
    """清缓存并释放显存，返回是否释放过。"""
    had = bool(_cache)
    _cache.clear()
    try:
        import comfy.model_management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _ = comfy.model_management  # 显式引用，守卫缺席 core
    except Exception:
        pass
    return had


def run_sam_mask(image_tensor, prompt, threshold=0.5):
    """对 [1,H,W,3] 张量跑 SAM，返回 (H, W) float32 并集遮罩。"""
    from nodes import CLIPTextEncode
    from comfy_extras.nodes_sam3 import SAM3_Detect

    model, clip = _load_stack()
    text = (prompt or "").strip() or "object"
    try:
        thr = float(threshold)
    except Exception:
        thr = 0.5
    thr = max(0.0, min(1.0, thr))

    cond = CLIPTextEncode().encode(clip, text)[0]
    # 路由跑在队列执行上下文之外：新版 core 的 PROGRESS_BAR_HOOK 会读
    # PromptServer.last_prompt_id 做进度广播，该属性此时不存在 → 推理期
    # 临时置空 hook（ProgressBar 为 None 时跳过广播），finally 还原，
    # 不污染并发中的正常执行（已创建的 bar 持有旧 hook 引用不受影响）。
    import comfy.utils
    prev_hook = comfy.utils.PROGRESS_BAR_HOOK
    comfy.utils.PROGRESS_BAR_HOOK = None
    try:
        out = SAM3_Detect.execute(model, image_tensor, conditioning=cond,
                                  threshold=thr, refine_iterations=2,
                                  individual_masks=False)
    finally:
        comfy.utils.PROGRESS_BAR_HOOK = prev_hook
    masks = out[0]  # [B, H, W]（union）
    m = masks[0].detach().to("cpu", torch.float32).numpy()
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _handle_sam(data):
    """纯逻辑入口（可裸测）：data -> (http_status, payload)。"""
    if not isinstance(data, dict):
        data = {}
    src_path = data.get("src_path", "") or ""
    full = _safe_join(src_path) if src_path else None
    if not full:
        return 400, {"error": "节点尚未加载源图，请先 Load / Browse / 拖放 / 粘贴图片"}
    try:
        pil = Image.open(full).convert("RGB")
    except Exception as e:
        return 400, {"error": f"源图读取失败：{e}"}
    arr = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(arr)[None,]
    try:
        mask_arr = run_sam_mask(image, data.get("prompt", ""), data.get("threshold", 0.5))
    except Exception as e:
        print(f"[SFImageBrushMask:sam] inference failed: {e}")
        return 500, {"error": f"SAM 推理失败：{e}"}
    h, w = int(mask_arr.shape[0]), int(mask_arr.shape[1])
    if (w, h) != (pil.size[0], pil.size[1]):
        # 防御：尺寸漂移时按源图重采样后再追踪（NEAREST 保二值）
        mask_pil = Image.fromarray((mask_arr * 255.0 + 0.5).astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize(pil.size, Image.NEAREST)
        mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
    from ...sf_utils.brush_mask import mask_to_fill_strokes
    strokes = mask_to_fill_strokes(mask_arr)
    coverage = round(float((mask_arr > 0.5).mean()), 4)
    return 200, {"status": "success", "strokes": strokes,
                 "count": len(strokes), "coverage": coverage,
                 "width": pil.size[0], "height": pil.size[1]}


def _register_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/brush_mask/sam")
        async def _sam_run(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            code, payload = _handle_sam(data)
            return web.json_response(payload, status=code)

        @routes.post("/api/sfnodes/brush_mask/sam_unload")
        async def _sam_unload(request: web.Request) -> web.Response:
            return web.json_response({"status": "success", "unloaded": unload_sam()})

        @routes.get("/api/sfnodes/brush_mask/sam_status")
        async def _sam_status(request: web.Request) -> web.Response:
            return web.json_response(sam_status())
    except Exception:
        pass


_register_routes()
