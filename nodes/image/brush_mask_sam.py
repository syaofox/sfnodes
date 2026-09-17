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
  POST /api/sfnodes/brush_mask/sam         {src_path, prompt, threshold, refine_iterations,
                                             positive_coords/negative_coords/bbox 可选（点/框提示，
                                             list 或 JSON 串；仅点框时 conditioning=None）}
  POST /api/sfnodes/brush_mask/sam_unload  {}
  GET  /api/sfnodes/brush_mask/sam_status  （门禁诊断：core/模型/已加载 + busy）

忙时熔断（2026-09，真机报错修复）：
  工作流执行期间路由推理会与执行线程**并发做模型加载**——comfy-aimdo 的进程级
  全局 file reader 非线程安全（hostbuf_file_reader_read failed: active slot
  already has a completion event），且核心模型管理全局态本就只支持执行线程单线
  程使用。故 `queue_busy()` 非空（队列/正在运行）时直接 409 拒绝；路由同步处理
  （阻塞事件循环），检查通过后 /prompt 无法入队，竞态窗口实际被关闭。
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from aiohttp import web

import folder_paths

from .crop import _safe_join

_CKPT = "sam3.1_multiplex_fp16.safetensors"

_cache = {}  # ckpt_path -> (model_patcher, clip)

_BUSY_ERROR = ("ComfyUI 正在执行工作流（可能是 SAM/模型加载任务）。为避免与运行时"
               "模型加载冲突，请等当前任务结束后再试")


def queue_busy():
    """当前是否有工作流任务在跑/排队（熔断判据，见模块 docstring）。

    取不到 PromptServer（测试/无服务器）视为不忙——不阻塞功能，服务端仍有
    独立兜底（409 只在真实队列非空时触发）。
    """
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        q = getattr(ins, "prompt_queue", None)
        if q is None:
            return False
        return bool(q.get_tasks_remaining() > 0)
    except Exception:
        return False


def sam_status():
    """可用性三态 + busy（pure readout，供菜单预检与诊断）。"""
    busy = queue_busy()
    try:
        from comfy_extras.nodes_sam3 import SAM3_Detect  # noqa: F401
    except Exception as e:
        return {"available": False, "reason": f"当前 ComfyUI 核心无 SAM3_Detect（{e}），请升级 ComfyUI",
                "model_found": False, "loaded": False, "busy": busy}
    try:
        ckpt = folder_paths.get_full_path("checkpoints", _CKPT)
    except Exception:
        ckpt = None
    if not ckpt or not os.path.isfile(ckpt):
        return {"available": False, "reason": f"模型缺失：请把 {_CKPT} 放入 models/checkpoints/",
                "model_found": False, "loaded": False, "busy": busy}
    return {"available": True, "reason": "", "model_found": True,
            "loaded": ckpt in _cache, "busy": busy, "ckpt": _CKPT}


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


def run_sam_mask(image_tensor, prompt="", threshold=0.5, refine_iterations=2,
                 positive_coords=None, negative_coords=None, bbox=None):
    """对 [1,H,W,3] 张量跑 SAM，返回 (H, W) float32 并集遮罩。

    提示源（可文本+点/框组合）：
      - 文本 prompt（空且无点/框时按 RMBG 惯例回退 "object"）；
      - positive_coords/negative_coords：core 约定 JSON 串
        ``[{"x": int, "y": int}, ...]``（正/负点，像素坐标）；
      - bbox：``{"x", "y", "width", "height"}`` 像素矩形。
    仅点/框时 ``conditioning=None``，core 走 SAM decoder 路径（已核实支持）。
    """
    from nodes import CLIPTextEncode
    from comfy_extras.nodes_sam3 import SAM3_Detect

    model, clip = _load_stack()
    text = (prompt or "").strip()
    if not text and not positive_coords and not negative_coords and not bbox:
        text = "object"
    try:
        thr = float(threshold)
    except Exception:
        thr = 0.5
    thr = max(0.0, min(1.0, thr))
    try:
        refine = int(float(refine_iterations))
    except Exception:
        refine = 2
    refine = max(0, min(5, refine))

    cond = CLIPTextEncode().encode(clip, text)[0] if text else None
    # 路由跑在队列执行上下文之外：新版 core 的 PROGRESS_BAR_HOOK 会读
    # PromptServer.last_prompt_id 做进度广播，该属性此时不存在 → 推理期
    # 临时置空 hook（ProgressBar 为 None 时跳过广播），finally 还原，
    # 不污染并发中的正常执行（已创建的 bar 持有旧 hook 引用不受影响）。
    import comfy.utils
    prev_hook = comfy.utils.PROGRESS_BAR_HOOK
    comfy.utils.PROGRESS_BAR_HOOK = None
    try:
        out = SAM3_Detect.execute(model, image_tensor, conditioning=cond,
                                  bboxes=bbox,
                                  positive_coords=positive_coords,
                                  negative_coords=negative_coords,
                                  threshold=thr, refine_iterations=refine,
                                  individual_masks=False)
    finally:
        comfy.utils.PROGRESS_BAR_HOOK = prev_hook
    masks = out[0]  # [B, H, W]（union）
    m = masks[0].detach().to("cpu", torch.float32).numpy()
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _normalize_points(raw, width, height):
    """点提示归一（纯函数）：接受 list[{"x","y"}] / JSON 串 / None。

    越界/非法点丢弃；无有效点返回 None。返回 core 约定的 JSON 串。
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, list):
        return None
    pts = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x = int(round(float(item.get("x"))))
            y = int(round(float(item.get("y"))))
        except Exception:
            continue
        if 0 <= x < int(width) and 0 <= y < int(height):
            pts.append({"x": x, "y": y})
    return json.dumps(pts) if pts else None


def _normalize_bbox(raw, width, height):
    """框提示归一（纯函数）：接受 [x1,y1,x2,y2] 或 {"x","y","width","height"}。

    钳制到图像内；面积小于 2×2 返回 None。返回 core 约定的 dict。
    """
    if raw is None:
        return None
    x = y = w = h = None
    if isinstance(raw, dict):
        try:
            x = float(raw.get("x"))
            y = float(raw.get("y"))
            w = float(raw.get("width"))
            h = float(raw.get("height"))
        except Exception:
            return None
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in raw]
        except Exception:
            return None
        x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
    if x is None or w is None or h is None:
        return None
    W, H = float(width), float(height)
    bx1 = max(0.0, min(W, x))
    by1 = max(0.0, min(H, y))
    bx2 = max(0.0, min(W, x + w))
    by2 = max(0.0, min(H, y + h))
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        return None
    return {"x": int(round(bx1)), "y": int(round(by1)),
            "width": int(round(bx2 - bx1)), "height": int(round(by2 - by1))}


def _handle_sam(data, busy=None):
    """纯逻辑入口（可裸测）：data -> (http_status, payload)。

    busy: None = 实时查询 queue_busy()；True/False 由调用方（测试）注入。
    忙时 409 直接拒绝，绝不触碰模型加载/推理（见模块 docstring 忙时熔断）。
    """
    if busy is None:
        busy = queue_busy()
    if busy:
        return 409, {"busy": True, "error": _BUSY_ERROR}
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
    # 点/框提示归一（显式提供但全部无效 → 400，避免静默退化成 "object" 文本检测）
    pos = _normalize_points(data.get("positive_coords"), pil.size[0], pil.size[1])
    neg = _normalize_points(data.get("negative_coords"), pil.size[0], pil.size[1])
    box = _normalize_bbox(data.get("bbox"), pil.size[0], pil.size[1])
    if (data.get("positive_coords") or data.get("negative_coords")) and not (pos or neg):
        return 400, {"error": "点提示无效：需要图像内的 x/y 像素坐标"}
    if data.get("bbox") and not box:
        return 400, {"error": "框提示无效：需要图像内至少 2×2 的矩形"}
    arr = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(arr)[None,]
    try:
        mask_arr = run_sam_mask(image, data.get("prompt", ""), data.get("threshold", 0.5),
                                data.get("refine_iterations", 2),
                                positive_coords=pos, negative_coords=neg, bbox=box)
    except Exception as e:
        print(f"[SFImageBrushMask:sam] inference failed: {e}")
        # 极小残留窗口（任务刚结束、aimdo 预取线程未退）仍可能撞原生 file reader：
        # 给出可操作提示而不是裸英文原生错误
        hint = "（模型加载读取冲突：请在工作流空闲时重试）" if "hostbuf" in str(e).lower() else ""
        return 500, {"error": f"SAM 推理失败：{e}{hint}"}
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
            # 同步执行（不经线程池）：事件循环被本推理阻塞期间 /prompt 无法入队，
            # 配合 _handle_sam 的队列空闲检查封死"与执行线程并发读模型"的竞态
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
