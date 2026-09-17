"""SF 画笔节点右键菜单扩展后端：人物部位 / YOLO 检测分割 / 导入遮罩 / 统一卸载。

路由（副作用注册，改动需重启容器；与 brush_mask_sam.py 同款 try/except 包裹）：
  POST /api/sfnodes/brush_mask/person_parts  {src_path, parts[], confidence, refine}
  GET  /api/sfnodes/brush_mask/yolo_models   （bbox/segm 权重清单，扫描 ultralytics/{bbox,segm} + yolo）
  GET  /api/sfnodes/brush_mask/yolo_classes  ?kind&model（类别清单 + 任务类型；不熔断）
  POST /api/sfnodes/brush_mask/yolo          {src_path, kind: bbox|segm, model, conf, box_shape,
                                              imgsz: 640|960|1280, classes: [id 或类名]}
  POST /api/sfnodes/brush_mask/import_mask   {src_path, src_w, src_h}
  POST /api/sfnodes/brush_mask/unload_all

模型路由与 SAM 同规矩：`queue_busy()` 忙时熔断（409）+ 同步处理（见
brush_mask_sam 模块 docstring / experience §91），避免与执行线程并发模型加载
撞 comfy-aimdo 全局 file reader。

依赖边界：
  - 人物部位：MediaPipe selfie multiclass（requirements 已声明），模型经
    ModelManager(sub_dir="person_mask") 自动下载/缓存，分割核心复用
    ``sf_utils/person_mask.py``（与 SFPersonMask 节点单源）。
  - YOLO：``ultralytics`` 为**运行时可选依赖**（AGPL-3.0，仅本地调用、不拷贝其
    代码；缺失时给中文报错，不进 requirements.txt），权重扫
    ``models/ultralytics/{bbox,segm}``（Impact 惯例目录，本包不注册 folder 键）。
  - 导入遮罩：纯本地文件→笔触（无模型、无熔断），灰图 NEAREST 缩放到源图尺寸
    后走 ``mask_to_fill_strokes``。
"""

import os

import numpy as np
from PIL import Image
from aiohttp import web

import folder_paths

from ...sf_utils import person_mask as _person_mask
from ...sf_utils.brush_mask import mask_to_fill_strokes
from .brush_mask_sam import _BUSY_ERROR, queue_busy, unload_sam
from .crop import open_src_image

_person_cache = {"buffer": None}   # MediaPipe tflite 字节
_yolo_cache = {}                   # 模型绝对路径 -> ultralytics.YOLO

_YOLO_KINDS = ("bbox", "segm")

# kind → 候选目录（相对 models_dir，按序查找首命中）：
#   bbox = ultralytics/bbox（主）+ yolo（ComfyUI-RMBG 的 legacy 目录，只读扫描，
#          让 person/COCO 等通用检测器也进菜单；不移动文件、不影响 RMBG）
#   segm = ultralytics/segm
_YOLO_DIRS = {
    "bbox": ("ultralytics/bbox", "yolo"),
    "segm": ("ultralytics/segm",),
}

# imgsz 白名单（小目标 nipples/eyes 用 960/1280；非法回退 640）
_IMGSZ_CHOICES = (640, 960, 1280)


# ── 人物部位（MediaPipe）────────────────────────────────────────────────────

def load_person_buffer():
    """读取/缓存 MediaPipe tflite 字节（首次触发 ModelManager 下载）。"""
    if _person_cache["buffer"] is None:
        _person_cache["buffer"] = _person_mask.load_model_buffer()
    return _person_cache["buffer"]


def _handle_person_parts(data, busy=None):
    """纯逻辑入口（可裸测）：人物部位分割 → fill 笔触。"""
    if busy is None:
        busy = queue_busy()
    if busy:
        return 409, {"busy": True, "error": _BUSY_ERROR}
    if not isinstance(data, dict):
        data = {}
    pil = open_src_image(data.get("src_path", "") or "", "[SFBrushMask:person]")
    if pil is None:
        return 400, {"error": "节点尚未加载源图，请先 Load / Browse / 拖放 / 粘贴图片"}
    parts = _person_mask.normalize_parts(data.get("parts"))
    try:
        confidence = float(data.get("confidence", 0.4))
    except Exception:
        confidence = 0.4
    confidence = max(0.01, min(1.0, confidence))
    try:
        mask_arr = _person_mask.segment_mask(
            pil, parts, confidence, bool(data.get("refine")), load_person_buffer())
    except Exception as e:
        print(f"[SFBrushMask:person] inference failed: {e}")
        hint = "（模型加载读取冲突：请在工作流空闲时重试）" if "hostbuf" in str(e).lower() else ""
        return 500, {"error": f"人物部位分割失败：{e}{hint}"}
    strokes = mask_to_fill_strokes(mask_arr)
    return 200, {"status": "success", "strokes": strokes, "count": len(strokes),
                 "coverage": round(float((mask_arr > 0.5).mean()), 4), "parts": parts,
                 "width": pil.size[0], "height": pil.size[1]}


# ── YOLO 检测/分割（运行时可选依赖）────────────────────────────────────────

def _yolo_dirs(kind):
    """kind 的候选目录绝对路径（缺失目录也返回，扫描时忽略）。"""
    return [os.path.join(folder_paths.models_dir, *d.split("/")) for d in _YOLO_DIRS.get(kind, ())]


def list_yolo_models():
    """扫描各候选目录的 .pt 权重清单（多目录合并去重，前序目录优先）。"""
    out = {}
    for kind in _YOLO_KINDS:
        names = []
        seen = set()
        for d in _yolo_dirs(kind):
            try:
                cur = sorted(f for f in os.listdir(d) if f.lower().endswith(".pt"))
            except OSError:
                cur = []
            for n in cur:
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        out[kind] = names
    return out


def resolve_yolo_model(kind, name):
    """白名单解析权重绝对路径：kind 合法 + 单段文件名 + 在扫描清单内
    （多目录按序首命中）。"""
    if kind not in _YOLO_KINDS or not isinstance(name, str):
        return None
    if os.path.basename(name) != name or not name.lower().endswith(".pt"):
        return None
    if name not in list_yolo_models().get(kind, []):
        return None
    for d in _yolo_dirs(kind):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _check_yolo_task(model, kind):
    """任务自检（纯函数，可注入 fake）：返回 (ok, warning)。

    - bbox 选到分割权重 → 可用（只出框），warning 提示可切 segm 拿掩码
    - segm 选到非分割权重 → 不可用（masks 恒空会"静默未检出"），ok=False
    - task 取不到（旧/异常权重）→ 不拦截
    """
    task = getattr(model, "task", None)
    if task == "segment" and kind == "bbox":
        return True, "该权重是分割模型，当前按检测框输出（切换类型为 segm 可得到掩码）"
    if kind == "segm" and task not in (None, "segment"):
        return False, f"该权重是 {task} 模型，不能用于分割：请切换类型为 bbox 或换用 -seg 权重"
    return True, None


def _normalize_classes(raw, names_map):
    """类别过滤归一（纯函数）：接受 id 列表或类名列表（混合亦可）。

    未知项丢弃；重复去重；无有效项返回 None（= 不过滤）。
    """
    if not raw:
        return None
    if isinstance(raw, str):
        raw = [c.strip() for c in raw.split(",")]
    if not isinstance(raw, (list, tuple)):
        return None
    valid = set()
    by_name = {}
    for cid, label in (names_map or {}).items():
        try:
            cid = int(cid)
        except Exception:
            continue
        valid.add(cid)
        by_name.setdefault(str(label), []).append(cid)
    out = []
    for item in raw:
        try:
            cid = int(item)
            if cid in valid and cid not in out:
                out.append(cid)
            continue
        except (TypeError, ValueError):
            pass
        for cid in by_name.get(str(item).strip(), []):
            if cid not in out:
                out.append(cid)
    return sorted(out) or None


def _load_yolo(path):
    try:
        from ultralytics import YOLO  # 运行时可选依赖（AGPL-3.0，见模块 docstring）
    except Exception as e:
        raise RuntimeError(f"未安装 ultralytics（可选依赖）：{e}")
    model = _yolo_cache.get(path)
    if model is None:
        model = YOLO(path)
        _yolo_cache[path] = model
    return model


def _boxes_to_mask(boxes, width, height, shape="rect"):
    """bbox 列表 → (H, W) float32 并集掩码（rect 实心框 / ellipse 内切椭圆）。

    纯 numpy（不依赖 cv2），便于裸测与无 cv2 环境降级。
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes or []:
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except Exception:
            continue
        ix1, iy1 = max(0, int(round(min(x1, x2)))), max(0, int(round(min(y1, y2))))
        ix2, iy2 = min(width, int(round(max(x1, x2)))), min(height, int(round(max(y1, y2))))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if shape == "ellipse":
            cx = (ix1 + ix2) / 2.0
            cy = (iy1 + iy2) / 2.0
            rx = max(0.5, (ix2 - ix1) / 2.0)
            ry = max(0.5, (iy2 - iy1) / 2.0)
            ys, xs = np.ogrid[iy1:iy2, ix1:ix2]
            inside = (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0
            sub = mask[iy1:iy2, ix1:ix2]
            mask[iy1:iy2, ix1:ix2] = np.maximum(sub, inside.astype(np.float32))
        else:
            mask[iy1:iy2, ix1:ix2] = 1.0
    return mask


def _polygons_to_mask(polys, width, height):
    """ultralytics masks.xy 多边形（原图像素坐标）→ (H, W) float32 并集掩码。"""
    from PIL import ImageDraw

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys or []:
        try:
            pts = [(float(p[0]), float(p[1])) for p in poly]
        except Exception:
            continue
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(img).astype(np.float32) / 255.0


def run_yolo(np_rgb, kind, model_path, conf=0.25, box_shape="rect",
             classes=None, imgsz=640):
    """跑 YOLO 并返回 (mask (H,W) float32, detected 数, labels 类名列表)。

    bbox 权重 → 实心框/椭圆并集；segm 权重 → masks.xy 多边形并集。
    classes: 类别过滤（id 或类名列表，未知名丢弃；空 = 不过滤）；
    imgsz: 640/960/1280（白名单外回退 640）。
    """
    model = _load_yolo(model_path)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.25
    conf = max(0.01, min(1.0, conf))
    try:
        imgsz = int(float(imgsz))
    except Exception:
        imgsz = 640
    if imgsz not in _IMGSZ_CHOICES:
        imgsz = 640
    class_ids = _normalize_classes(classes, getattr(model, "names", None))
    # ⚠ ultralytics 对 list-of-numpy 输入在 preprocess 里无条件 div_(255)（按 uint8
    # 0..255 处理）；传 float 0..1 会被再除一次 → 近全黑 → 漏检（真机踩坑，见 §94.6）。
    # 故这里统一转 uint8 BGR：float 输入按 0..1 语义映射到 0..255。
    img = np.asarray(np_rgb)
    if img.dtype != np.uint8:
        img = (np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    kwargs = {"source": np.ascontiguousarray(img[..., ::-1]),  # RGB → BGR
              "conf": conf, "verbose": False, "imgsz": imgsz}
    if class_ids:
        kwargs["classes"] = class_ids
    results = model.predict(**kwargs)
    r = results[0]
    h, w = int(np_rgb.shape[0]), int(np_rgb.shape[1])
    names = getattr(r, "names", None) or {}
    labels = []
    if kind == "segm":
        polys = []
        masks = getattr(r, "masks", None)
        if masks is not None:
            polys = list(getattr(masks, "xy", None) or [])
        mask = _polygons_to_mask(polys, w, h)
        detected = len(polys)
    else:
        boxes = []
        boxes_obj = getattr(r, "boxes", None)
        if boxes_obj is not None:
            try:
                xyxy = boxes_obj.xyxy
                cls = getattr(boxes_obj, "cls", None)
                for i in range(len(xyxy)):
                    boxes.append([float(v) for v in xyxy[i].tolist()])
                    if cls is not None:
                        try:
                            labels.append(str(names.get(int(cls[i].item()), int(cls[i].item()))))
                        except Exception:
                            pass
            except Exception:
                boxes = []
        mask = _boxes_to_mask(boxes, w, h, box_shape)
        detected = len(boxes)
    return mask, detected, sorted(set(labels))


def _handle_yolo(data, busy=None):
    """纯逻辑入口（可裸测）：YOLO bbox/segm → fill 笔触。"""
    if busy is None:
        busy = queue_busy()
    if busy:
        return 409, {"busy": True, "error": _BUSY_ERROR}
    if not isinstance(data, dict):
        data = {}
    kind = data.get("kind", "bbox")
    path = resolve_yolo_model(kind, data.get("model"))
    if not path:
        return 400, {"error": "YOLO 模型无效：请从 models/ultralytics/{bbox,segm} 的清单中选择 .pt"}
    pil = open_src_image(data.get("src_path", "") or "", "[SFBrushMask:yolo]")
    if pil is None:
        return 400, {"error": "节点尚未加载源图，请先 Load / Browse / 拖放 / 粘贴图片"}
    box_shape = data.get("box_shape", "rect")
    if box_shape not in ("rect", "ellipse"):
        box_shape = "rect"
    arr = np.array(pil)  # uint8 RGB（run_yolo 内转 uint8 BGR，勿预除 255）
    try:
        model = _load_yolo(path)
    except Exception as e:
        print(f"[SFBrushMask:yolo] load failed: {e}")
        return 500, {"error": f"YOLO 模型加载失败：{e}"}
    task = getattr(model, "task", None)
    ok, warning = _check_yolo_task(model, kind)
    if not ok:
        return 400, {"error": warning, "task": task, "model": data.get("model")}
    try:
        mask_arr, detected, labels = run_yolo(arr, kind, path, data.get("conf", 0.25), box_shape,
                                              data.get("classes"), data.get("imgsz", 640))
    except Exception as e:
        print(f"[SFBrushMask:yolo] inference failed: {e}")
        hint = "（模型加载读取冲突：请在工作流空闲时重试）" if "hostbuf" in str(e).lower() else ""
        return 500, {"error": f"YOLO 推理失败：{e}{hint}"}
    strokes = mask_to_fill_strokes(mask_arr)
    out = {"status": "success", "strokes": strokes, "count": len(strokes),
           "coverage": round(float((mask_arr > 0.5).mean()), 4),
           "detected": detected, "labels": labels, "task": task,
           "width": pil.size[0], "height": pil.size[1]}
    if warning:
        out["warning"] = warning
    return 200, out


def _handle_yolo_classes(kind, model_name):
    """纯逻辑入口（可裸测）：类别清单 + 任务类型（不推理，不加忙时熔断）。"""
    path = resolve_yolo_model(kind, model_name)
    if not path:
        return 400, {"error": "YOLO 模型无效：请从 models/ultralytics/{bbox,segm} 的清单中选择 .pt"}
    try:
        model = _load_yolo(path)
    except Exception as e:
        print(f"[SFBrushMask:yolo] load failed: {e}")
        return 500, {"error": f"YOLO 模型加载失败：{e}"}
    names = getattr(model, "names", None) or {}
    out = {}
    for cid, label in names.items():
        try:
            out[str(int(cid))] = str(label)
        except Exception:
            continue
    return 200, {"names": out, "task": getattr(model, "task", None), "model": model_name}


# ── 导入遮罩文件（纯本地，无模型/无熔断）──────────────────────────────────

def _handle_import_mask(data):
    """纯逻辑入口（可裸测）：遮罩文件 → fill 笔触（缩放到当前源图尺寸）。"""
    if not isinstance(data, dict):
        data = {}
    pil = open_src_image(data.get("src_path", "") or "", "[SFBrushMask:import]")
    if pil is None:
        return 400, {"error": "遮罩文件读取失败（上传未成功或路径越界）"}
    gray = np.array(pil.convert("L")).astype(np.float32) / 255.0
    try:
        tw = int(data.get("src_w") or 0)
        th = int(data.get("src_h") or 0)
    except Exception:
        tw = th = 0
    if tw > 0 and th > 0 and (tw, th) != (gray.shape[1], gray.shape[0]):
        resized = Image.fromarray((gray * 255.0 + 0.5).astype(np.uint8), mode="L").resize(
            (tw, th), Image.NEAREST)
        gray = np.array(resized).astype(np.float32) / 255.0
    strokes = mask_to_fill_strokes(gray)
    return 200, {"status": "success", "strokes": strokes, "count": len(strokes),
                 "coverage": round(float((gray > 0.5).mean()), 4),
                 "width": gray.shape[1], "height": gray.shape[0]}


# ── 统一卸载 ───────────────────────────────────────────────────────────────

def unload_all():
    """清空 SAM/人物/YOLO 全部菜单模型缓存并释放显存，返回各域是否驻留过。"""
    had_sam = unload_sam()
    had_person = _person_cache["buffer"] is not None
    had_yolo = len(_yolo_cache)
    _person_cache["buffer"] = None
    _yolo_cache.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"sam": had_sam, "person": had_person, "yolo": had_yolo}


# ── 路由注册 ───────────────────────────────────────────────────────────────

def _register_routes():
    try:
        from server import PromptServer
        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.post("/api/sfnodes/brush_mask/person_parts")
        async def _person_parts(request: web.Request) -> web.Response:
            # 同步执行（同 SAM 路由）：事件循环被推理阻塞期间 /prompt 无法入队，
            # 配合 queue_busy() 熔断封死与执行线程并发读模型的竞态
            try:
                data = await request.json()
            except Exception:
                data = {}
            code, payload = _handle_person_parts(data)
            return web.json_response(payload, status=code)

        @routes.get("/api/sfnodes/brush_mask/yolo_models")
        async def _yolo_models(request: web.Request) -> web.Response:
            return web.json_response(list_yolo_models())

        @routes.get("/api/sfnodes/brush_mask/yolo_classes")
        async def _yolo_classes(request: web.Request) -> web.Response:
            # 只构造模型读元数据（懒加载 + 缓存），不推理 → 不加忙时熔断，
            # 运行工作流时也能浏览类别
            code, payload = _handle_yolo_classes(request.query.get("kind", "bbox"),
                                                 request.query.get("model", ""))
            return web.json_response(payload, status=code)

        @routes.post("/api/sfnodes/brush_mask/yolo")
        async def _yolo_run(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            code, payload = _handle_yolo(data)
            return web.json_response(payload, status=code)

        @routes.post("/api/sfnodes/brush_mask/import_mask")
        async def _import_mask(request: web.Request) -> web.Response:
            try:
                data = await request.json()
            except Exception:
                data = {}
            code, payload = _handle_import_mask(data)
            return web.json_response(payload, status=code)

        @routes.post("/api/sfnodes/brush_mask/unload_all")
        async def _unload_all(request: web.Request) -> web.Response:
            return web.json_response({"status": "success", **unload_all()})
    except Exception:
        pass


_register_routes()
