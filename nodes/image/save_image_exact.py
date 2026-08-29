"""SF Save Image Exact — 精确文件名保存（无计数后缀）。

原生 SaveImage 强制 `_{counter:05}_.png` 后缀，
`filename_prefix="seedvr/xiaoguo-v3gai/a1"` 必然落盘 `a1_00001_.png`，
无法得到 `a1.png`。本节点提供精确名/可选覆盖语义：

- `filename="seedvr/xiaoguo-v3gai/a1"` 或 `"seedvr/xiaoguo-v3gai/a1.png"`
  直接保存为 `output/seedvr/xiaoguo-v3gai/a1.png`（按 `format` 决定扩展名，
  输入中的扩展名被剥离后重加，避免 `a1.png_00001_.png` 双扩展名）。
- `overwrite=True`：精确覆盖同名文件；batch>1 时首帧为精确名，后续帧
  `_{1,2...}`（`_1` 风格，非 ` _00001_`），重复执行覆盖同组文件。
- `overwrite=False`：永不覆盖，自动找空闲 `base.ext / base_1.ext ...`。
- `format` 支持 png/jpeg/webp，`quality` 控制 jpeg/webp 质量，png 固定
  compress_level=4（与原生 SaveImage 一致）。
- 沿用原生 SaveImage 的 metadata 嵌入（png 仅，尊重 --disable-metadata）、
  `is_within_directory` 越界检查与 `_safe_prefix` 风格的非法字符清洗。

无前端 JS，纯后端节点。
"""

import json
import os
import re

import folder_paths
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

_CATEGORY = "sfnodes/image"

# ── 清洗（复刻 preview_routes._safe_prefix / _sanitize_segment，禁内联副本漂移） ──

_DISALLOWED_CHAR_RE = re.compile(r'[<>:"|?*\x00-\x1f\x7f]')
_MULTI_UNDERSCORE_RE = re.compile(r"_+")
_PREFIX_MAX_LEN = 256
_PREFIX_OUTPUT_MAX = 100
_WIN_RESERVED_NAMES = frozenset((
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
))

# 输入 filename 中被剥离的扩展名（大小写不敏感），剥离后由 format 决定最终扩展名
_STRIP_EXTS = frozenset((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"))


def _sanitize_segment(seg):
    cleaned = _DISALLOWED_CHAR_RE.sub("_", seg)
    cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned)
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = cleaned.strip().strip("_").rstrip(". ")
    if cleaned and cleaned.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES:
        cleaned += "_"
    return cleaned


def _safe_filename(raw):
    """清洗 filename（可含子目录），剥离扩展名，返回清洗后的相对路径（无扩展名）或 ""。

    管道与 preview_routes._safe_prefix 1:1：逐段替换 Windows 非法字符、
    折叠重复 '_'、剥离边沿、守卫保留设备名；先检查 leading '/' 与 '..'。
    额外：末段剥离已知图片扩展名（.png/.jpg 等），由 format 参数重加。
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().replace("\\", "/")
    if not s or len(s) > _PREFIX_MAX_LEN:
        return ""
    if s.startswith("/"):
        return ""
    parts = s.split("/")
    if any(p == ".." for p in parts):
        return ""
    cleaned_parts = [_sanitize_segment(p) for p in parts if p]
    cleaned_parts = [p for p in cleaned_parts if p]
    if not cleaned_parts:
        return ""
    result = "/".join(cleaned_parts)
    if len(result) > _PREFIX_OUTPUT_MAX:
        result = result[:_PREFIX_OUTPUT_MAX].rstrip("/_-")
        if not result:
            return ""
    # 剥离扩展名（仅末段）
    # result 已含子目录，最后一段是文件名
    dirname = os.path.dirname(result)
    basename = os.path.basename(result)
    # 剥离扩展名（大小写不敏感）
    lower = basename.lower()
    for ext in _STRIP_EXTS:
        if lower.endswith(ext):
            basename = basename[: -len(ext)]
            # 剥离后可能残留尾点/尾下划线
            basename = basename.rstrip(". ")
            break
    basename = basename.strip()
    if not basename:
        return ""
    # basename 可能因剥离后变成保留设备名裸露，需二次守卫
    if basename.split(".", 1)[0].upper() in _WIN_RESERVED_NAMES:
        basename += "_"
    if dirname:
        result = dirname + "/" + basename
    else:
        result = basename
    # 二次长度截断（剥离后可能变短，不会超限）
    if len(result) > _PREFIX_OUTPUT_MAX:
        result = result[:_PREFIX_OUTPUT_MAX].rstrip("/_-")
    return result


def _metadata_disabled():
    try:
        from comfy.cli_args import args as _comfy_cli_args
        return bool(getattr(_comfy_cli_args, "disable_metadata", False))
    except Exception:
        return False


class SFSaveImageExact:
    DESCRIPTION = (
        "精确文件名保存图片（无计数后缀）：filename=\"seedvr/xiaoguo-v3gai/a1\" "
        "直接保存为 output/seedvr/xiaoguo-v3gai/a1.png，重复执行按 overwrite 决定覆盖或递增 "
        "（_1 风格，非 _00001_）。支持 png/jpeg/webp，batch>1 时首帧为精确名，后续帧 _1/_2..."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "要保存的图片批次 [B,H,W,C]"}),
                "filename": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "相对 output 目录的保存路径，可含子目录，可带/不带扩展名（扩展名由 format 决定，输入中的 .png/.jpg 等会被剥离重加），例 seedvr/xiaoguo-v3gai/a1 或 a1.png",
                }),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "label_on": "overwrite",
                    "label_off": "increment",
                    "tooltip": "True=精确覆盖同名文件（重复执行覆盖）；False=永不覆盖，自动找空闲文件名 _1/_2...",
                }),
                "format": (["png", "jpeg", "webp"], {
                    "default": "png",
                    "tooltip": "保存格式，决定最终扩展名（png/.png, jpeg/.jpg, webp/.webp）",
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "jpeg/webp 质量 1-100（png 忽略，png 固定 compress_level=4）",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    def save(self, images, filename="ComfyUI", overwrite=True, format="png", quality=95,
             prompt=None, extra_pnginfo=None):
        # 清洗 filename，空则回退 ComfyUI
        cleaned = _safe_filename(filename) or "ComfyUI"
        output_dir = folder_paths.get_output_directory()
        subfolder = os.path.dirname(os.path.normpath(cleaned))
        base = os.path.basename(os.path.normpath(cleaned))

        # format 决定扩展名
        fmt = str(format).lower() if isinstance(format, str) else "png"
        if fmt not in ("png", "jpeg", "webp", "jpg"):
            fmt = "png"
        if fmt == "jpg":
            fmt = "jpeg"
        ext_map = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
        ext = ext_map[fmt]

        full_output_folder = os.path.join(output_dir, subfolder) if subfolder else output_dir
        # 越界检查（与 folder_paths.get_save_image_path 一致）
        if not folder_paths.is_within_directory(output_dir, full_output_folder):
            err = "**** ERROR: Saving image outside the output folder is not allowed.\n full_output_folder: " + os.path.abspath(full_output_folder) + "\n         output_dir: " + output_dir
            raise Exception(err)
        os.makedirs(full_output_folder, exist_ok=True)

        # quality 归一
        try:
            q = int(quality)
        except Exception:
            q = 95
        q = max(1, min(100, q))

        results = []
        # 用于 overwrite=False 时避免同批次内碰撞的已占用集合
        used_in_run = set()

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # ── 文件名决策 ──
            if overwrite:
                # 精确覆盖：首帧精确名，后续 _1/_2...
                if batch_number == 0:
                    fname = f"{base}{ext}"
                else:
                    fname = f"{base}_{batch_number}{ext}"
            else:
                # 递增不覆盖：从 0 开始找首个不存在且未被本批次占用的文件名
                found = None
                idx = 0
                for _ in range(100000):
                    cand = f"{base}{ext}" if idx == 0 else f"{base}_{idx}{ext}"
                    full = os.path.join(full_output_folder, cand)
                    if cand not in used_in_run and not os.path.exists(full):
                        found = cand
                        break
                    idx += 1
                if found is None:
                    raise RuntimeError("[SFSaveImageExact] 无法找到空闲文件名（超出尝试上限）")
                fname = found

            used_in_run.add(fname)
            full_path = os.path.join(full_output_folder, fname)

            # ── 保存 ──
            if fmt == "png":
                metadata = None
                if not _metadata_disabled():
                    metadata = PngInfo()
                    if prompt is not None:
                        try:
                            metadata.add_text("prompt", json.dumps(prompt))
                        except Exception:
                            pass
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            try:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                            except Exception:
                                pass
                img.save(full_path, "PNG", pnginfo=metadata, compress_level=4)
            elif fmt == "jpeg":
                # jpeg 不支持 alpha，强制 RGB
                if img.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", img.size, (0, 0, 0))
                    bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(full_path, "JPEG", quality=q, subsampling=0, optimize=True)
            else:  # webp
                img.save(full_path, "WEBP", quality=q, method=4)

            results.append({
                "filename": fname,
                "subfolder": subfolder,
                "type": "output",
            })

        return {"ui": {"images": results}, "result": (images,)}
