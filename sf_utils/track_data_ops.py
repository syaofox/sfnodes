"""SAM3_TRACK_DATA 纯逻辑运算（无 torch / ComfyUI 依赖，依赖注入）。

用于 `SFSAM3PointTrack` 的锚帧前补空帧、`SFSAM3ReanchorTrack` 的分段
拼接回全长（每段位或塌单身份 + 空隙补零），以及 `SFTrackDataSubtract` 的
track_data 层逐帧相减（排除男性/阴茎/精液等）、`SFTrackDataAdd` 的逐帧
并集合成单身份、`SFTrackDataMerge` 的逐槽先减后加组合、
`SFTrackDataSlice` 的时间维区间切片、`SFTrackDataToMask` 的单帧遮罩输出、
`SFInvertTrackData` 与 `SFTrackDataToMask` 共用的对象索引解析。位打包复用核心
`comfy.ldm.sam3.tracker.pack_masks/unpack_masks`（由调用方注入）。

track_data 形状约定：`packed_masks` 为 `[T, N, H, W//8]` 位打包张量
（N=对象数，T 可因 4 帧对齐等与 n_frames 一致），`unpack_masks` 返回
`[T, N, H, W*8]` bool；`pack_masks` 接受任意前置维 `[..., H, W]`。
"""


def is_track_data(value):
    return isinstance(value, dict) and "packed_masks" in value


def parse_object_indices(spec, n_obj):
    """解析逗号分隔的对象索引：空/None=全部；非法与越界项忽略（可能返回空表）。

    语义对齐核心 `SAM3_TrackToMask`（`isdigit` 过滤 + 越界忽略），
    `SFInvertTrackData` 与 `frame_mask_from_track_data` 共用。
    """
    count = int(n_obj)
    if spec is None or not str(spec).strip():
        return list(range(count))
    out = []
    for part in str(spec).split(","):
        token = part.strip()
        if token.isdigit():
            index = int(token)
            if index < count:
                out.append(index)
    return out


def slice_track_data(track_data, start=0, length=0):
    """按帧区间切片 track_data（时间维），保留对象数 / `scores` / `orig_size`。

    - `start` 支持负值（相对末尾，-1=最后一帧），越界夹到 `[0, T]`；
    - `length <= 0` 表示切到结尾，超出尾部自动截断；
    - `packed_masks is None`（整条无对象）时仅调整 `n_frames`（时间语义一致）；
    - 输出为浅拷贝，不改原输入；空切片合法（首维 0、`n_frames=0`）。
    """
    out = dict(track_data)
    packed = track_data.get("packed_masks")
    total = int(packed.shape[0]) if packed is not None else int(track_data.get("n_frames") or 0)
    start = int(start or 0)
    if start < 0:
        start += total
    start = max(0, min(start, total))
    requested = int(length or 0)
    end = total if requested <= 0 else min(total, start + requested)
    span = max(0, end - start)
    if packed is None:
        out["n_frames"] = span
        return out
    out["packed_masks"] = packed[start:end].contiguous()
    out["n_frames"] = span
    return out


def frame_mask_from_track_data(track_data, frame_index=0, object_indices="",
                               unpack_masks=None, torch=None, interpolate=None):
    """取 track_data 指定帧的选中对象并集，还原真实分辨率输出 MASK `[1, H, W]`。

    用于 `SFTrackDataToMask`（从 `SFSAM3ReanchorTrack` 等追踪结果取单帧遮罩）：

    - `frame_index` 支持负值（相对末尾，-1=最后一帧，同 `slice_track_data`）；越界报错；
    - `object_indices` 逗号分隔（越界/非法项忽略），空=全部对象；无有效对象输出全零；
    - 空追踪（`packed_masks is None` / 0 对象）输出全零（需 `orig_size` 有效）；
    - 输出尺寸取 `orig_size`（追踪器方形工作网格 ≠ 真实宽高），双线性插值 +
      `align_corners=False` 与核心 `SAM3_TrackToMask` 一致；`orig_size` 缺失/非法报错；
    - 输出 `[1, H, W]` float（device 随输入）；不修改原 track_data。
    """
    packed = track_data.get("packed_masks")
    total = int(packed.shape[0]) if packed is not None else int(track_data.get("n_frames") or 0)
    index = int(frame_index or 0)
    if index < 0:
        index += total
    if total <= 0 or index < 0 or index >= total:
        raise ValueError(f"SF Track Data To Mask: 帧索引 {frame_index} 越界（总帧数 {total}）")

    orig = track_data.get("orig_size") or (0, 0)
    height, width = int(orig[0]), int(orig[1])
    if height <= 0 or width <= 0:
        raise ValueError("SF Track Data To Mask: orig_size 缺失或非法，无法确定输出尺寸")

    indices = parse_object_indices(object_indices, packed.shape[1]) if packed is not None else []
    if not indices:
        return torch.zeros((1, height, width), dtype=torch.float32,
                           device=getattr(packed, "device", None))

    union = packed[:, indices[0]]
    for other in indices[1:]:
        union = union | packed[:, other]
    frame = unpack_masks(union)[index].unsqueeze(0).unsqueeze(0).float()  # [1, 1, h, w]
    return interpolate(frame, size=(height, width), mode="bilinear", align_corners=False)[0]


def concat_track_data_segments(segments, total_frames, torch=None):
    """把若干 `(起始帧, track_data)` 段按时间偏移拼回全长的单身份 track_data。

    用于分段重锚追踪（`SFSAM3ReanchorTrack`）：每段独立跑一次追踪后，
    - 每段跨对象位或（packed uint8 按位或）塌成 1 个身份；
    - 段按起始帧升序、互不重叠；首/尾/段间空隙补零帧；
    - `packed_masks is None` / 0 对象 / 0 帧的段视为空段跳过（其帧区间为空格）；
    - 各段 packed 工作网格（H/W）必须一致，否则报错；起点重叠或超出总帧数报错；
    - 输出 `scores=[1.0]`（单身份语义，同 `add_to_track_data`），
      `orig_size` 取首个带 orig_size 的段；全空时 `packed_masks=None`。
    """
    total = int(total_frames or 0)
    ref = None  # 首个非空 packed，用于 dtype/device/网格基准
    ref_orig = (0, 0)
    parts = []
    cursor = 0
    for start, track_data in (segments or []):
        start = int(start)
        if track_data is None:
            continue
        orig = track_data.get("orig_size")
        if orig and ref_orig == (0, 0):
            ref_orig = (int(orig[0]), int(orig[1]))
        packed = track_data.get("packed_masks")
        if packed is None or packed.shape[0] == 0 or packed.shape[1] == 0:
            continue
        length = int(packed.shape[0])
        if start < cursor:
            raise ValueError(f"SF 分段拼接: 段起点 {start} 与前段重叠（当前游标 {cursor}）")
        if start + length > total:
            raise ValueError(f"SF 分段拼接: 段 [{start}, {start + length}) 超出总帧数 {total}")
        if ref is None:
            ref = packed
        elif tuple(packed.shape[2:]) != tuple(ref.shape[2:]):
            raise ValueError("SF 分段拼接: 各段 packed 工作网格不一致")
        if start > cursor:
            parts.append(torch.zeros((start - cursor, 1) + tuple(ref.shape[2:]),
                                     dtype=ref.dtype, device=ref.device))
        union = packed[:, 0]
        for i in range(1, int(packed.shape[1])):
            union = union | packed[:, i]
        parts.append(union.unsqueeze(1))
        cursor = start + length

    if ref is None:
        return {"packed_masks": None, "n_frames": total, "orig_size": ref_orig, "scores": []}
    if cursor < total:
        parts.append(torch.zeros((total - cursor, 1) + tuple(ref.shape[2:]),
                                 dtype=ref.dtype, device=ref.device))
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return {"packed_masks": out, "n_frames": total, "orig_size": ref_orig, "scores": [1.0]}


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
    - `scores` 置 `[1.0]`（对齐 `invert_track_data` 塌单身份语义）；
    - **`orig_size` 必须继承输入**：SAM3 追踪器在固定方形工作分辨率
      （`image_size`）上打包，`packed_masks` 的 H/W 与真实宽高无关，只有
      `orig_size` 记录真尺寸（`SAM3_VideoTrack` 事后写入）。若用 `unpack` 的
      H/W 覆盖 orig_size，下游 `SAM3_TrackToMask` 会按方形插值 → 宽高比变 1:1。
      工作分辨率（`unpack` 的 H/W）仅用于把各路叠加对齐到基础打包网格。
    """
    packed = track_data.get("packed_masks")
    frames = []
    for add in (add_masks or []):
        f = _exclusion_frames(add, unpack_masks)
        if f is not None:
            frames.append(f)

    orig = track_data.get("orig_size") or (0, 0)
    orig_h, orig_w = int(orig[0]), int(orig[1])
    has_orig = orig_h > 0 and orig_w > 0

    if packed is not None and packed.shape[1] > 0:
        base = unpack_masks(packed).any(dim=1)  # [T, work_h, work_w]（追踪器工作网格）
        total = base.shape[0]
        work_h, work_w = base.shape[-2], base.shape[-1]
        # 真实宽高以输入 orig_size 为准（缺失/非法才回退工作分辨率）
        out_orig = (orig_h, orig_w) if has_orig else (int(work_h), int(work_w))
    else:
        if not frames:
            return dict(track_data)
        work_h, work_w = (orig_h, orig_w) if has_orig else (frames[0].shape[-2], frames[0].shape[-1])
        total = int(track_data.get("n_frames") or 0) or frames[0].shape[0]
        base = torch.zeros((total, work_h, work_w), dtype=torch.bool)
        out_orig = (int(work_h), int(work_w))

    union = base
    for f in frames:
        if f.shape[0] != total:
            raise ValueError(
                f"SF Track Data Add: 叠加遮罩帧数 {f.shape[0]} 与基础 {total} 不一致"
            )
        union = union | _resize_frames(f, work_h, work_w, interpolate)

    out = dict(track_data)
    out["packed_masks"] = pack_masks(pad_width_to_8(union, torch)).unsqueeze(1)
    out["n_frames"] = int(total)
    out["orig_size"] = (int(out_orig[0]), int(out_orig[1]))
    out["scores"] = [1.0]
    return out


def merge_track_data(track_data, subtract_masks, add_masks, pack_masks=None,
                     unpack_masks=None, torch=None, interpolate=None):
    """基础 track_data 先逐对象相减、再有叠加时并集塌单身份。

    等价于把 `SFTrackDataSubtract` 与 `SFTrackDataAdd` 串联：
    `subtract_from_track_data` → （仅当存在至少一个有效叠加时）
    `add_to_track_data`。顺序无关（相减各路并集、叠加各路并集，相减先于叠加）。

    - 无有效叠加时保留基础对象数与 `scores`（不塌单身份）；
    - 空基础 + 有叠加走 `add_to_track_data` 的补零/回退尺寸逻辑。
    """
    out = subtract_from_track_data(
        track_data, subtract_masks,
        pack_masks=pack_masks, unpack_masks=unpack_masks,
        torch=torch, interpolate=interpolate,
    )
    valid_adds = [a for a in (add_masks or []) if a is not None]
    if valid_adds:
        out = add_to_track_data(
            out, valid_adds,
            pack_masks=pack_masks, unpack_masks=unpack_masks,
            torch=torch, interpolate=interpolate,
        )
    return out
