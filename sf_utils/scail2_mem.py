"""SCAIL-2 预处理内存优化：整段张量按帧分块（运行时补丁，不改核心文件）。

comfy_extras/nodes_scail.py 的两个纯函数会一次性构造整段 T 帧的中间张量：

- ``_render_colored_masks``：``unpack_masks`` → ``[T,N,H,W]`` f32，再
  ``interpolate`` + ``color_overlay`` ``[T,H,W,3]`` f32。2500 帧 @704×1280、N=1
  时约 30–55 GB。
- ``_extract_mask_to_28ch``：7 通道阈值图 ``[T,7,H,W]`` f32（同尺寸约 16 GB）。

默认 ``intermediate_device`` 为 CPU 时这是系统内存峰值，``--gpu-only`` 时直接
爆显存。本模块保持输出逐元素一致，只把中间量按 ``chunk_frames`` 分块计算，
把峰值从 O(T) 降到 O(chunk)：

- ``any / argmax / interpolate(mode="area")`` 对时间维独立；
- ``where`` 与背景色广播逐帧独立；
- ``extract_mask_to_28ch`` 的时间打包顺序由显式索引写入复现（帧 0 → padded[0:4]，
  帧 j≥1 → padded[3+j]）。

另外包装 ``WanSCAILToVideo.execute``：原生对整段 ``reference_image_mask`` 做
``common_upscale``（``[T,3,H,W]`` f32，可数十 GB），但实际只用前 ``n_ref`` 帧
（``min(i,n_masks-1)`` 与 ``[:1]``）。仅当 mask 帧数大于 reference_image 批数时
裁剪，语义逐元素等价。

安装方式对齐 ``sf_utils/scail2_context.py``（一次性幂等 patch）与
``sf_utils/vhs_loadvideo_filename.py``（install / 守卫 / logger）。设置读
``sfnodes.SCAIL2Mem.{Enabled,ChunkFrames,HalfPrecision}``，读盘复用
``sf_utils/llm_client.read_comfy_settings``（comfy.settings.json 唯一读取实现）。

``Enabled`` 为**主开关**：关闭时完全按原生执行（不分块、不裁剪参考蒙版、
``HalfPrecision`` 也不生效），只有开启时其余两项才有效。
"""

import functools

from .logger import get_logger

logger = get_logger("sfnodes.scail2_mem")

DEFAULT_CHUNK_FRAMES = 32

SETTING_ENABLED = "sfnodes.SCAIL2Mem.Enabled"
SETTING_CHUNK = "sfnodes.SCAIL2Mem.ChunkFrames"
SETTING_HALF = "sfnodes.SCAIL2Mem.HalfPrecision"

_PATCHED = False
_MARK = "_sf_scail2_mem_patched"


# ── 设置读取 ────────────────────────────────────────────────────────────
def read_options(settings=None):
    """返回 (enabled, chunk_frames, half_precision)。settings 可注入（测试/复用）。"""
    if settings is None:
        try:
            from .llm_client import read_comfy_settings

            settings = read_comfy_settings()
        except Exception:
            settings = {}
    if not isinstance(settings, dict):
        settings = {}
    enabled = bool(settings.get(SETTING_ENABLED, True))
    half = bool(settings.get(SETTING_HALF, True))
    raw_chunk = settings.get(SETTING_CHUNK, DEFAULT_CHUNK_FRAMES)
    try:
        chunk = int(raw_chunk)
    except Exception:
        chunk = DEFAULT_CHUNK_FRAMES
    if chunk < 1:
        chunk = DEFAULT_CHUNK_FRAMES
    return enabled, chunk, half


# ── 分块纯函数（依赖注入，逐元素对齐核心实现）────────────────────────────
def render_colored_masks_chunked(track_data, background, *, torch, unpack_masks, interpolate,
                                 palette, device, dtype, chunk_frames, half=False):
    """核心 ``_render_colored_masks`` 的分块版：同参同结果，中间量按帧分块。

    ``half=True`` 时输出 float16（值仅 0/1，与 f32 逐元素等价，仅省一半常驻）。
    """
    packed = track_data.get("packed_masks")
    H, W = track_data["orig_size"]
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    out_dtype = torch.float16 if half else dtype

    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch.empty(T, H, W, 3, device=device, dtype=dtype)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out.to(out_dtype)

    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [palette[i % len(palette)] for i in range(N_obj)], device=device, dtype=dtype
    )
    bg_tensor = torch.tensor(bg_rgb, device=device, dtype=dtype).view(1, 1, 1, 3)
    out = torch.empty(T, H, W, 3, device=device, dtype=out_dtype)

    for start in range(0, T, chunk_frames):
        end = min(start + chunk_frames, T)
        c = end - start
        masks_full = unpack_masks(packed[start:end].to(device)).float()
        Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
        masks_full = (
            interpolate(masks_full.view(c * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest")
            .view(c, N_obj, H, W)
            > 0.5
        )
        any_mask = masks_full.any(dim=1)
        color_overlay = colors[masks_full.to(torch.uint8).argmax(dim=1)]
        out[start:end] = torch.where(
            any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay)
        ).to(out_dtype)
    return out


def extract_mask_to_28ch_chunked(rgb_video, *, torch, interpolate, chunk_frames):
    """核心 ``_extract_mask_to_28ch`` 的分块版：同参同结果。

    输入 ``(T, H, W, 3)`` 彩色蒙版；要求 ``T ≡ 1 (mod 4)``（核心 view 的前置条件，
    调用方先截断到 4n+1）。仅空间下采样对时间分块独立，时间打包按索引写入复现。
    """
    T, H, W, _ = rgb_video.shape
    on_thresh = 225.0 / 255.0

    H_lat, W_lat = H, W
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    T_latent = (T - 1) // 4 + 1

    padded = torch.empty(
        T_latent * 4, 7, H_lat, W_lat, device=rgb_video.device, dtype=torch.float32
    )
    for start in range(0, T, chunk_frames):
        end = min(start + chunk_frames, T)
        seg = rgb_video[start:end].movedim(-1, 1).float()
        R = (seg[:, 0:1] > on_thresh).float()
        G = (seg[:, 1:2] > on_thresh).float()
        B = (seg[:, 2:3] > on_thresh).float()
        nR, nG, nB = 1 - R, 1 - G, 1 - B
        binary_7ch = torch.cat(
            [
                R * G * B,    # white
                R * nG * nB,  # red
                nR * G * nB,  # green
                nR * nG * B,  # blue
                R * G * nB,   # yellow
                R * nG * B,   # magenta
                nR * G * B,   # cyan
            ],
            dim=1,
        )
        binary_7ch = interpolate(binary_7ch, size=(H_lat, W_lat), mode="area")
        for local in range(end - start):
            j = start + local
            if j == 0:
                padded[0:4] = binary_7ch[local]
            else:
                padded[3 + j] = binary_7ch[local]

    return padded.view(T_latent, 28, H_lat, W_lat).unsqueeze(0)


def _to_half(value, enabled):
    if not enabled:
        return value
    try:
        import torch

        return value.to(torch.float16)
    except Exception:
        return value


# ── 包装器 ──────────────────────────────────────────────────────────────
def _wrap_render(palette, orig):
    def _render_colored_masks(track_data, background="black"):
        enabled, chunk, half = read_options()
        if not enabled:
            # Enabled 是主开关：关闭时完全按原生执行（含不转 f16）。
            return orig(track_data, background)
        packed = track_data.get("packed_masks") if isinstance(track_data, dict) else None
        n = int(packed.shape[0]) if packed is not None else 0
        if packed is None or packed.shape[1] == 0 or n <= chunk:
            return _to_half(orig(track_data, background), half)
        try:
            import torch
            import torch.nn.functional as F
            from comfy import model_management
            from comfy.ldm.sam3.tracker import unpack_masks

            return render_colored_masks_chunked(
                track_data,
                background,
                torch=torch,
                unpack_masks=unpack_masks,
                interpolate=F.interpolate,
                palette=palette,
                device=model_management.intermediate_device(),
                dtype=model_management.intermediate_dtype(),
                chunk_frames=chunk,
                half=half,
            )
        except Exception as exc:
            logger.warning("彩色蒙版分块渲染失败，回退原生：%s", exc)
            return orig(track_data, background)

    return _render_colored_masks


def _wrap_extract(orig):
    def _extract_mask_to_28ch(rgb_video):
        enabled, chunk, _half = read_options()
        try:
            t = int(rgb_video.shape[0])
        except Exception:
            return orig(rgb_video)
        if not enabled or t <= chunk or (t - 1) % 4 != 0:
            return orig(rgb_video)
        try:
            import torch
            import torch.nn.functional as F

            return extract_mask_to_28ch_chunked(
                rgb_video, torch=torch, interpolate=F.interpolate, chunk_frames=chunk
            )
        except Exception as exc:
            logger.warning("28ch 分块提取失败，回退原生：%s", exc)
            return orig(rgb_video)

    return _extract_mask_to_28ch


def _wrap_execute(orig_func):
    """包装 WanSCAILToVideo.execute：裁剪过长的 reference_image_mask。

    调用方有两条路径：执行引擎 ``method(cls, **inputs)``（全 kwargs）与
    SF ``nodes/video/scail2.py::_run_native_scail_chunk`` 的**位置参数**调用。
    故先按原函数签名 bind 出参名，再按需改参，兼容两种调用形态。
    """
    try:
        import inspect

        sig = inspect.signature(orig_func)
        has_params = all(
            name in sig.parameters
            for name in ("reference_image_mask", "reference_image")
        )
    except Exception:
        sig = None
        has_params = False

    @functools.wraps(orig_func)
    def execute(cls, *args, **kwargs):
        enabled, _chunk, _half = read_options()
        if enabled and has_params:
            try:
                bound = sig.bind_partial(cls, *args, **kwargs)
                ref_mask = bound.arguments.get("reference_image_mask")
                ref_img = bound.arguments.get("reference_image")
                if ref_mask is not None and ref_img is not None:
                    n_ref = int(ref_img.shape[0])
                    n_mask = int(ref_mask.shape[0])
                    if n_ref >= 1 and n_mask > n_ref:
                        bound.arguments["reference_image_mask"] = ref_mask[:n_ref]
                        return orig_func(*bound.args, **bound.kwargs)
            except Exception:
                pass
        return orig_func(cls, *args, **kwargs)

    return execute


# ── 安装 ────────────────────────────────────────────────────────────────
def _execute_globals(cls):
    """取类 execute 方法所在命名空间（模块 globals dict）。"""
    fn = getattr(cls, "execute", None)
    fn = getattr(fn, "__func__", fn)
    ns = getattr(fn, "__globals__", None)
    return ns if isinstance(ns, dict) else None


def _is_scail_class(cls):
    return "nodes_scail" in (getattr(cls, "__module__", "") or "")


def _target_namespaces():
    """收集 SCAIL-2 相关的命名空间。

    注意 ComfyUI 用 ``load_custom_node`` 以文件路径为模块名加载
    ``comfy_extras/nodes_scail.py``，而 ``from comfy_extras import nodes_scail``
    会得到**另一个**模块对象。故以注册类的 ``execute.__globals__`` 为准（那才是
    原生节点实际查名字的命名空间），再补上 ``comfy_extras.nodes_scail`` 模块，
    覆盖 SF ``nodes/video/scail2.py`` 走 import 的路径。
    """
    namespaces = []
    try:
        import nodes as comfy_nodes

        mapping = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        classes = list(mapping.values())
    except Exception:
        classes = []
    for cls in classes:
        if not _is_scail_class(cls):
            continue
        ns = _execute_globals(cls)
        if ns is not None:
            namespaces.append(ns)
    try:
        from comfy_extras import nodes_scail

        namespaces.append(vars(nodes_scail))
    except Exception:
        pass

    uniq, seen = [], set()
    for ns in namespaces:
        if id(ns) not in seen:
            seen.add(id(ns))
            uniq.append(ns)
    return uniq


def _patch_namespace(ns):
    count = 0
    orig_render = ns.get("_render_colored_masks")
    if callable(orig_render) and not getattr(orig_render, _MARK, False):
        wrapper = _wrap_render(ns.get("DEFAULT_PALETTE", []), orig_render)
        wrapper._sf_scail2_mem_patched = True
        ns["_render_colored_masks"] = wrapper
        count += 1
    orig_extract = ns.get("_extract_mask_to_28ch")
    if callable(orig_extract) and not getattr(orig_extract, _MARK, False):
        wrapper = _wrap_extract(orig_extract)
        wrapper._sf_scail2_mem_patched = True
        ns["_extract_mask_to_28ch"] = wrapper
        count += 1
    return count


def _patch_execute_class(cls):
    try:
        raw = cls.__dict__.get("execute")
        func = getattr(raw, "__func__", raw)
        if not callable(func) or getattr(func, "_sf_scail2_mem_exec_patched", False):
            return 0
        wrapper = _wrap_execute(func)
        wrapper._sf_scail2_mem_exec_patched = True
        setattr(cls, "execute", classmethod(wrapper))
        return 1
    except Exception as exc:
        logger.warning("WanSCAILToVideo.execute 参考蒙版裁剪补丁失败：%s", exc)
        return 0


def _candidate_classes(module=None):
    classes = []
    if module is not None:
        cls = getattr(module, "WanSCAILToVideo", None)
        if cls is not None:
            classes.append(cls)
    else:
        try:
            import nodes as comfy_nodes

            mapping = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        except Exception:
            mapping = {}
        for cls in mapping.values():
            if getattr(cls, "__name__", "") == "WanSCAILToVideo":
                classes.append(cls)
    try:
        from comfy_extras import nodes_scail

        cls = getattr(nodes_scail, "WanSCAILToVideo", None)
        if cls is not None:
            classes.append(cls)
    except Exception:
        pass
    uniq, seen = [], set()
    for cls in classes:
        if id(cls) not in seen:
            seen.add(id(cls))
            uniq.append(cls)
    return uniq


def install(module=None):
    """给 core nodes_scail 打一次性幂等内存补丁；返回补丁项数（已装/不可用为 0）。

    module 仅用于测试注入（一个假模块对象）；正常路径按注册类命名空间定位。
    """
    global _PATCHED
    if _PATCHED:
        return 0
    if module is not None:
        namespaces = [vars(module) if not isinstance(module, dict) else module]
    else:
        namespaces = _target_namespaces()
    if not namespaces:
        logger.warning("未找到 SCAIL-2 目标命名空间，内存补丁未安装")
        return 0

    count = 0
    for ns in namespaces:
        count += _patch_namespace(ns)
    for cls in _candidate_classes(module):
        count += _patch_execute_class(cls)

    if count:
        _PATCHED = True
        logger.info("SCAIL-2 预处理内存补丁：%d 项", count)
    return count
