import json
import os
import re


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


def _parse_fill_color(fill_color):
    """Parse a fill color into an (r, g, b) tuple of 0-255 ints.

    Accepts a "#rrggbb" / "rrggbb" hex string or any 3-sequence of ints.
    (从 nodes/mask/masks.py 提升的公共实现——SFMaskFill 与 SFImageCropExpand 共用；
    前端取色器输出恒为 6 位 hex，故不支持 #RGB 缩写与 "r,g,b" 字符串。)
    """
    if isinstance(fill_color, str):
        hex_color = fill_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )
    return tuple(fill_color)


def json_safe(obj):
    """清洗 NaN/Inf 使对象保持合法 JSON（非法 float 转字符串）。

    快照 PNG 嵌入整个 prompt，其中任何节点的 IS_CHANGED 返回 NaN 都会贡献
    `is_changed: [NaN]`——不是合法 JSON，前端 JSON.parse 会抛错并丢弃整个
    payload。pause_image / pause_mask / preview_routes 曾各持一份逐字相同的
    内联副本，现收敛为单一实现。
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
    return obj


def parse_json_dict(raw):
    """把隐藏 STRING 真源解析为 dict，任何失败返回 {}。

    brush_mask / crop_expand 曾各持一份逐字相同的 _parse_state，现收敛。
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def lora_stem(lora_name):
    """LoRA 文件名去路径去扩展（只留 stem）。

    lora_loader / lora_loader_model_only / lora_selector 三处逐字相同。
    """
    return os.path.splitext(os.path.basename(lora_name))[0]


def collect_indexed(kwargs, prefix):
    """从 kwargs 收集形如 prefix+数字（如 image1、mask2）的输入，返回 {编号: 值}。

    动态槽位节点（Krea2 / Painter Flux Edit）后端统一入口：None 值忽略、非匹配名忽略。
    """
    pattern = re.compile(r"^{}(\d+)$".format(re.escape(prefix)))
    out = {}
    for key, value in kwargs.items():
        match = pattern.match(key)
        if match is not None and value is not None:
            out[int(match.group(1))] = value
    return out


def ordered_slot_items(kwargs, prefix):
    """按槽位编号（数字序）返回有序 [(名, 值)]；None 槽跳过（collect_indexed 单源）。

    `sorted(kwargs)` 是字典序，槽数 >9 时 image_10 会排到 image_2 之前——这里按
    编号整数排序修正。nodes/image/batch.py 两节点与 SFImagePromptRewriter 共用。
    """
    return [("{}{}".format(prefix, i), v) for i, v in sorted(collect_indexed(kwargs, prefix).items())]


def frame_to_pil(image, index):
    """IMAGE 张量 [B,H,W,C] 取指定帧转 PIL RGB；alpha 按黑底预乘。

    index 支持负值（-1 = 末帧），越界抛 ValueError。SFImageInterrogatorAPI 与
    SFImagePromptRewriter 共用（原 interrogator 节点内联实现提升至此）。
    """
    import numpy as np
    from PIL import Image

    total = int(image.shape[0])
    idx = index if index >= 0 else total + index
    if idx < 0 or idx >= total:
        raise ValueError(f"frame_index 越界：{index}（batch 帧数 {total}）")
    frame = image[idx]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    arr = np.clip(arr.astype("float32"), 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[-1] >= 4:
        arr = arr[..., :3] * arr[..., 3:4]
    elif arr.ndim == 3 and arr.shape[-1] != 3:
        arr = arr[..., :3]
    return Image.fromarray((arr * 255.0 + 0.5).astype("uint8"), "RGB")


def node_result(value):
    """把核心 V3 节点 execute 的返回值归一为 tuple。

    `io.NodeOutput.result`（可能为 None）→ tuple；已是 tuple → 原样；其余 → 单元素 tuple。
    SFSAM3PointTrack 与 SFSAM3ReanchorTrack 复用（原先内联在 sam3_point_track.py）。
    """
    if hasattr(value, "result"):
        result = value.result
        if result is None:
            return ()
        return tuple(result)
    if isinstance(value, tuple):
        return value
    return (value,)


def valid_name(name, max_len=None):
    """预设/库条目名合法性：非空字符串、无路径分隔符、无控制字符。

    krea2_presets 与 text_presets 的 _valid_name 仅长度上限不同（后者限
    200），max_len 参数化后收敛为单一实现。
    """
    if not isinstance(name, str):
        return False
    name = name.strip()
    if not name:
        return False
    if max_len is not None and len(name) > max_len:
        return False
    if "/" in name or "\\" in name:
        return False
    if any(ord(c) < 32 for c in name):
        return False
    return True
