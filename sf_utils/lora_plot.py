"""LoRA Plot 纯逻辑(SFLoraPlot / SFLoraPlotImageSaver 共用)。

仅标准库 + PIL——无 comfy、无 folder_paths、无 torch,可在 ComfyUI 之外单测
(tests/test_lora_plot.py 直跑)。

双端契约:build_metadata 生成 "{文件名}_{强度}" 字符串,parse_metadata 解析回
(名称, 强度)。metadata 格式与原 ComfyUI-LoRAPlotNode 插件一致——旧工作流
保存的 metadata 仍可被 SFLoraPlotImageSaver 直接解析。
"""
import os
import re

from PIL import Image, ImageDraw, ImageFont

# 文字覆盖的颜色名 -> RGB。也是两个节点组合下拉的选项来源。
_NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (169, 169, 169),
}
COLOR_OPTIONS = list(_NAMED_COLORS)

# 拉丁字体候选(轻、加载快),覆盖 Linux / macOS / Windows。
_LATIN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "arialbd.ttf"),  # Windows
    os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "arial.ttf"),  # Windows
]

# CJK 字体候选(中文 LoRA 文件名画文字盒,否则豆腐块)。仅当文本含中文等
# 字符时使用——避免全拉丁场景为每个大小加载几十 MB 的 ttc。
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (Arch/Fedora)
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",  # Linux (Debian/Ubuntu)
    "/usr/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf",  # Linux (Debian 变体)
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux (文泉驿)
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "msyh.ttc"),  # Windows 雅黑
    os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "msyhbd.ttc"),  # Windows
]

# CJK 判断:中日韩统一表意文字 + 兼容区 + 假名 + 谚文 + 谚文字母。
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\u1100-\u11ff]")

# 字体按 (size, 是否 CJK) 缓存,避免每个 size 重复扫描磁盘。
_font_cache = {}


def sanitize_filename(filename):
    """LoRA 文件名 -> 用于 metadata 的安全名。去掉目录与扩展名、替换非法
    字符、剥掉首尾点/空格;结果为空时回退 "lora"。"""
    basename = os.path.basename(filename or "")
    name_without_ext = os.path.splitext(basename)[0]
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name_without_ext)
    sanitized = sanitized.strip('. ')
    return sanitized if sanitized else "lora"


def build_metadata(name, strength):
    """"{文件名}_{强度}"。strength 是 float,字符串化与原插件一致
    (1.0 -> "1.0")。"""
    return "{}_{}".format(sanitize_filename(name), strength)


def parse_metadata(meta):
    """把 metadata 拆回 (名称, 强度)。按最后一个下划线切分——文件名自身
    可能含下划线,强度是无下划线的数字串,从右切最稳。无下划线时返回
    (meta, "")。"""
    meta = meta or ""
    parts = meta.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return meta, ""


def color_to_rgba(color_str, alpha):
    """颜色字符串 -> RGBA tuple。支持:命名色(12 色)、"#RRGGBB"、
    "#RRGGBBAA"、rgb(r,g,b)。未知 -> 白色。永不抛错。"""
    color_str = (color_str or "").strip().lower()
    if color_str in _NAMED_COLORS:
        r, g, b = _NAMED_COLORS[color_str]
    elif color_str.startswith("#"):
        hex_color = color_str[1:]
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            alpha = int(hex_color[6:8], 16) / 255.0
        else:
            r, g, b = 255, 255, 255
    elif color_str.startswith("rgb"):
        match = re.search(r"rgb\((\d+),(\d+),(\d+)\)", color_str)
        if match:
            r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            r, g, b = 255, 255, 255
    else:
        r, g, b = 255, 255, 255
    return (r, g, b, int(max(0.0, min(1.0, alpha)) * 255))


def pick_font(font_size, text=None):
    """按字号取字体,缓存。text 含 CJK 字符时优先 CJK 候选(否则中文画成
    豆腐块);全拉丁走轻量拉丁候选。全部失败回退 Pillow 默认字体。"""
    cjk = bool(text) and _CJK_RE.search(text)
    key = (int(font_size), cjk)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    candidates = _CJK_FONT_CANDIDATES if cjk else _LATIN_FONT_CANDIDATES
    font = None
    for font_path in candidates:
        try:
            font = ImageFont.truetype(font_path, int(font_size))
            break
        except Exception:
            continue
    if font is None:
        try:
            # Pillow >= 10.1 默认字体支持 size 参数
            font = ImageFont.load_default(size=int(font_size))
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def add_text_overlay(image, text, text_color, background_color, font_size=24,
                     padding=10, opacity=0.8):
    """在图片右上角画半透明背景文字盒(名称 + 强度)。

    image: PIL Image(RGB/RGBA)。text: 多行文本(\n 分隔)。返回新 PIL Image,
    不修改原图。"""
    img = image.copy()
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    font = pick_font(font_size, text)

    lines = text.split("\n")
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    max_width = max(bbox[2] - bbox[0] for bbox in bboxes)
    line_heights = [bbox[3] - bbox[1] for bbox in bboxes]
    total_height = sum(line_heights) + (len(lines) - 1) * 5  # 5px 行距

    img_width, img_height = img.size
    box_width = max_width + (padding * 2)
    box_height = total_height + (padding * 2)
    x = img_width - box_width - padding
    y = padding

    bg_rgba = color_to_rgba(background_color, opacity)
    draw.rectangle([(x, y), (x + box_width, y + box_height)], fill=bg_rgba)

    text_rgba = color_to_rgba(text_color, 1.0)
    y_offset = y + padding
    for line, line_height in zip(lines, line_heights):
        draw.text((x + padding, y_offset), line, fill=text_rgba, font=font)
        y_offset += line_height + 5

    # 绘图层用 RGBA，输出回 RGB（与原插件一致——ComfyUI IMAGE 惯例 3 通道）。
    return img.convert("RGB")
