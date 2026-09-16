"""SAM3_TRACK_DATA 纯逻辑运算（无 torch / ComfyUI 依赖，依赖注入）。

用于 `SFSAM3PointTrack` 的锚帧前补空帧，以及 `SFTrackDataSubtract` 的
track_data 层逐帧相减（排除男性/阴茎/精液等）、`SFTrackDataAdd` 的逐帧
并集合成单身份。位打包复用核心
`comfy.ldm.sam3.tracker.pack_masks/unpack_masks`（由调用方注入）。

track_data 形状约定：`packed_masks` 为 `[T, N, H, W//8]` 位打包张量
（N=对象数，T 可因 4 帧对齐等与 n_frames 一致），`unpack_masks` 返回
`[T, N, H, W*8]` bool；`pack_masks` 接受任意前置维 `[..., H, W]`。
"""


def is_track_data(value):
    return isinstance(value, dict) and "packed_masks" in value


def pad_width_to_8(masks, torch):
    """把最后一维宽度补到 8 的倍数（SAM3 位打包要求 W 可整除 8）。"""
    width = int(masks.shape[-1])
    padded = (width + 7) // 8 * 8
    if padded == width:
        return masks
    return torch.nn.functional.pad(masks, (0, padded - width))


def pad_track_data_front(track_data, total_frames, torch=None):
    """在 track_data 前补 `total_frames - n_frames` 帧空遮罩（锚帧非 0 的追踪回全长）。

    `packed_masks is None`（整条无对象）时仅补齐 `n_frames`；已足够长则原样返回。
    """
    packed = track_data.get("packed_masks")
    n_frames = int(packed.shape[0]) if packed is not None else int(track_data.get("n_frames") or 0)
    total = int(total_frames or 0)
    out = dict(track_data)
    if total <= n_frames:
        return out
    if packed is None:
        out["n_frames"] = total
        return out
    pad = total - n_frames
    zeros = torch.zeros(tuple([pad]) + tuple(packed.shape[1:]),
                        dtype=packed.dtype, device=packed.device)
    out["packed_masks"] = torch.cat([zeros, packed], dim=0)
    out["n_frames"] = total
    return out


def _exclusion_frames(exclusion, unpack_masks):
    """归一排除输入为逐帧 bool `[T, H, W]`；空追踪/None 返回 None。"""
    if exclusion is None:
        return None
    if is_track_data(exclusion):
        packed = exclusion.get("packed_masks")
        if packed is None or packed.shape[1] == 0:
            return None
        return unpack_masks(packed).any(dim=1)

    shape = exclusion.shape
    if len(shape) == 2:
        frames = exclusion.unsqueeze(0)
    elif len(shape) == 4:
        if shape[1] != 1:
            raise ValueError("SF Track Data Subtract: 排除 MASK 的通道维应为 1（[T,1,H,W]）")
        frames = exclusion[:, 0]
    elif len(shape) == 3:
        frames = exclusion
    else:
        raise ValueError(f"SF Track Data Subtract: 不支持的排除遮罩维度 {len(shape)}（期望 2D/3D/4D）")
    return frames > 0


def _resize_frames(frames, height, width, interpolate):
    fh, fw = frames.shape[-2], frames.shape[-1]
    if fh == height and fw == width:
        return frames
    up = frames.float().unsqueeze(1)
    up = interpolate(up, size=(height, width), mode="nearest")
    return up[:, 0] > 0.5


def subtract_from_track_data(track_data, exclusion_masks, pack_masks=None,
                             unpack_masks=None, torch=None, interpolate=None):
    """基础 track_data 逐帧减去若干排除遮罩，保留对象数。

    - 排除输入可为 `MASK`（`[T,H,W]` / `[H,W]` / `[T,1,H,W]`）或 `SAM3_TRACK_DATA`；
    - 各路先并集再逐对象相减：`base & ~(exclude_1 | exclude_2 | ...)`；
    - 排除遮罩帧数须与基础一致（不广播），否则报错；
    - 基础为空追踪 / 无有效排除时原样返回（浅拷贝）。
    """
    packed = track_data.get("packed_masks")
    if packed is None or packed.shape[1] == 0:
        return dict(track_data)

    frames = []
    for exclusion in (exclusion_masks or []):
        f = _exclusion_frames(exclusion, unpack_masks)
        if f is not None:
            frames.append(f)
    if not frames:
        return dict(track_data)

    masks = unpack_masks(packed)  # [T, N, H, W] bool
    total = masks.shape[0]
    height, width = masks.shape[-2], masks.shape[-1]

    union = None
    for f in frames:
        if f.shape[0] != total:
            raise ValueError(
                f"SF Track Data Subtract: 排除遮罩帧数 {f.shape[0]} 与基础 {total} 不一致"
            )
        f = _resize_frames(f, height, width, interpolate)
        union = f if union is None else (union | f)

    keep = masks & (~union).unsqueeze(1)
    out = dict(track_data)
    out["packed_masks"] = pack_masks(keep)
    return out


def add_to_track_data(track_data, add_masks, pack_masks=None, unpack_masks=None,
                      torch=None, interpolate=None):
    """基础 track_data 逐帧并上若干叠加遮罩，合成为单一身份。

    - 叠加输入可为 `MASK`（`[T,H,W]` / `[H,W]` / `[T,1,H,W]`）或 `SAM3_TRACK_DATA`；
    - 基础与各路叠加先各自并集（基础跨对象塌平），再逐帧 OR，输出对象维恒为 1；
    - 叠加遮罩帧数须与基础一致（不广播），否则报错；
    - 基础为空追踪且无有效叠加时原样返回（浅拷贝）；有叠加时以基础
      `orig_size`/`n_frames` 为准补零基础（缺省回退首个叠加的尺寸/帧数）；
    - `scores` 置 `[1.0]`（对齐 `invert_track_data` 塌单身份语义）。
    """
    packed = track_data.get("packed_masks")
    frames = []
    for add in (add_masks or []):
        f = _exclusion_frames(add, unpack_masks)
        if f is not None:
            frames.append(f)

    if packed is not None and packed.shape[1] > 0:
        base = unpack_masks(packed).any(dim=1)  # [T, H, W]
        total = base.shape[0]
        height, width = base.shape[-2], base.shape[-1]
    else:
        if not frames:
            return dict(track_data)
        orig = track_data.get("orig_size") or (0, 0)
        height, width = int(orig[0]), int(orig[1])
        if height <= 0 or width <= 0:
            height, width = frames[0].shape[-2], frames[0].shape[-1]
        total = int(track_data.get("n_frames") or 0) or frames[0].shape[0]
        base = torch.zeros((total, height, width), dtype=torch.bool)

    union = base
    for f in frames:
        if f.shape[0] != total:
            raise ValueError(
                f"SF Track Data Add: 叠加遮罩帧数 {f.shape[0]} 与基础 {total} 不一致"
            )
        union = union | _resize_frames(f, height, width, interpolate)

    out = dict(track_data)
    out["packed_masks"] = pack_masks(pad_width_to_8(union, torch)).unsqueeze(1)
    out["n_frames"] = int(total)
    out["orig_size"] = (int(height), int(width))
    out["scores"] = [1.0]
    return out
