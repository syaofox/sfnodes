"""镜头切分纯逻辑（SFImageSceneSplit 用）。

无 torch / ComfyUI 依赖，可独立测试。
输入帧为 uint8 RGB [H,W,3] 或 float [0,1]，内部统一转 uint8 灰度小图后对比。

检测四类：
  - 硬切（跳切）：相邻帧直方图/像素差 > threshold
  - 黑场：连续黑帧（灰度均值 < black_threshold）段边界
  - 白闪：连续白帧（灰度均值 > white_threshold）段边界
  - 溶解/渐变：滑窗 W 内累积直方图距离 > threshold 且窗内单步均值 > dissolve_threshold 且 max 单步 < threshold

最短场景去抖：相邻切点距 < min_scene_len 则合并（删后者），尾段不足也合并。
"""

import math

import numpy as np


def _to_uint8_rgb(frame):
    """任意帧 -> uint8 RGB [H,W,3]。支持 float[0,1]/uint8，1ch/3ch/4ch。"""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"帧形状非法: {arr.shape}")
    # 通道归一
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"通道数非法: {arr.shape}")
    if arr.dtype != np.uint8:
        # float [0,1] -> uint8
        arr = np.clip(arr, 0, 1) if arr.dtype in (np.float32, np.float64) else np.clip(arr, 0, 255)
        if arr.max() <= 1.0 + 1e-6 and arr.dtype != np.uint8:
            # 启发式：若已在 0-1 范围则 *255
            # 若输入是 0-255 float，也会在上一步被 clip 到 255，但仍用 *1 判断
            is_float01 = arr.dtype in (np.float32, np.float64) and float(arr.max()) <= 1.0
            if is_float01:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _downscale_and_gray(frame_uint8, longest=160):
    """RGB uint8 -> 缩略灰度图 uint8 [h,w]，最长边 longest。"""
    h, w = frame_uint8.shape[:2]
    ls = max(h, w)
    if ls > longest:
        scale = longest / ls
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
    else:
        nh, nw = h, w
    # 优先 cv2，否则 PIL，否则最近邻采样
    small_rgb = None
    try:
        import cv2  # type: ignore

        small_rgb = cv2.resize(frame_uint8, (nw, nh), interpolation=cv2.INTER_AREA)
        small_gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        return small_gray
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        pil = Image.fromarray(frame_uint8)
        pil_small = pil.resize((nw, nh), Image.BILINEAR)
        small_rgb = np.array(pil_small)
        # luma
        small_gray = (0.2126 * small_rgb[:, :, 0] + 0.7152 * small_rgb[:, :, 1] + 0.0722 * small_rgb[:, :, 2]).astype(np.uint8)
        return small_gray
    except Exception:
        pass
    # 最近邻子采样
    step_h = max(1, h // nh)
    step_w = max(1, w // nw)
    sampled = frame_uint8[::step_h, ::step_w]
    sampled = sampled[:nh, :nw]
    if sampled.shape[0] < nh or sampled.shape[1] < nw:
        # 尺寸不足时 pad
        res = np.zeros((nh, nw, 3), dtype=np.uint8)
        rh = min(nh, sampled.shape[0])
        rw = min(nw, sampled.shape[1])
        res[:rh, :rw] = sampled[:rh, :rw]
        sampled = res
    small_gray = (0.2126 * sampled[:, :, 0] + 0.7152 * sampled[:, :, 1] + 0.0722 * sampled[:, :, 2]).astype(np.uint8)
    return small_gray


def _hist(gray_small, bins=32):
    hist, _ = np.histogram(gray_small, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _hist_distance(h1, h2):
    """Bhattacharyya 距离 1 - BC，0=相同，1=完全不同。"""
    bc = float(np.sum(np.sqrt(h1 * h2)))
    bc = max(0.0, min(1.0, bc))
    return 1.0 - bc


def _diff_distance(g1, g2):
    return float(np.mean(np.abs(g1.astype(np.int16) - g2.astype(np.int16))) / 255.0)


def _process_frame(frame, bins=32, longest=160):
    fu8 = _to_uint8_rgb(frame)
    gray_small = _downscale_and_gray(fu8, longest=longest)
    hist = _hist(gray_small, bins=bins)
    mean = float(np.mean(gray_small) / 255.0)
    return gray_small, hist, mean


def detect_scenes(frames, threshold=0.30, black_threshold=0.08, white_threshold=0.92,
                  min_scene_len=12, method="hist", dissolve_window=8, dissolve_threshold=0.18,
                  bins=32, longest=160):
    """帧序列 -> 切点列表 [0, cut1, ..., B]（含起止）。

    frames: np.ndarray [B,H,W,3] uint8/float 或 iterable[frame]
    method: "hist" | "diff"
    """
    grays = []
    hists = []
    means = []
    # 归一化迭代
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        b = frames.shape[0]
        iterator = (frames[i] for i in range(b))
    elif isinstance(frames, (list, tuple)):
        iterator = iter(frames)
    else:
        # generator / iterable
        try:
            iterator = iter(frames)
        except TypeError:
            raise ValueError("frames 必须是 [B,H,W,C] 数组或可迭代帧序列")
        # 无法预知长度，逐个消费
    count = 0
    for fr in iterator:
        gray_small, hist, mean = _process_frame(fr, bins=bins, longest=longest)
        grays.append(gray_small)
        hists.append(hist)
        means.append(mean)
        count += 1
    B = len(means)
    if B == 0:
        return [0, 0]
    if B == 1:
        return [0, 1]

    cuts = set()

    # 1) 硬切
    for i in range(B - 1):
        if method == "hist":
            d = _hist_distance(hists[i], hists[i + 1])
        else:
            d = _diff_distance(grays[i], grays[i + 1])
        if d > threshold:
            cuts.add(i + 1)

    # 2) 黑/白场连续段边界
    def add_runs(mask, is_black=True):
        # mask: bool list
        i = 0
        while i < B:
            if not mask[i]:
                i += 1
                continue
            s = i
            while i < B and mask[i]:
                i += 1
            e = i - 1  # inclusive
            run_len = e - s + 1
            # 过滤单帧噪点？黑场至少 1 帧即有意义，但若整段全黑则不切
            if run_len >= 1 and not (s == 0 and e == B - 1):
                # 段起点前一切，段终点后一切（若不在边界）
                if s > 0:
                    cuts.add(s)
                if e + 1 < B:
                    cuts.add(e + 1)
                # 对于长黑场，内部不再额外切
            # i 已在 e+1
    black_mask = [m < black_threshold for m in means]
    white_mask = [m > white_threshold for m in means]
    add_runs(black_mask, is_black=True)
    add_runs(white_mask, is_black=False)

    # 3) 溶解/渐变（滑窗累积）
    W = int(dissolve_window)
    if W >= 2 and B > W:
        # 预计算单步距离用于窗口内统计
        step_dists = []
        for i in range(B - 1):
            if method == "hist":
                d = _hist_distance(hists[i], hists[i + 1])
            else:
                d = _diff_distance(grays[i], grays[i + 1])
            step_dists.append(d)
        for i in range(B - W):
            # 累积距离
            if method == "hist":
                D = _hist_distance(hists[i], hists[i + W])
            else:
                D = _diff_distance(grays[i], grays[i + W])
            # 窗内单步统计
            window_steps = step_dists[i:i + W]
            avg_step = float(np.mean(window_steps)) if window_steps else 0.0
            max_step = float(np.max(window_steps)) if window_steps else 0.0
            # 溶解条件：累积显著，单步均值中等但无单步硬切
            if D > threshold and max_step < threshold and avg_step > dissolve_threshold:
                # 进一步：亮度单调性可选（对 fade 有效，不强制）
                # 若需要可放宽：只要直方图条件满足即判溶解
                # 额外用亮度斜率辅助过滤噪声：若窗口内亮度方差极小且 D 仅略超阈值，可能是噪声，跳过？
                # 这里保留宽松策略
                mid = i + W // 2 + 1  # 切在窗口中点后
                if 0 < mid < B:
                    cuts.add(mid)

    # 4) 去抖：相邻切点距 < min_scene_len 则合并（删后者）
    cand = sorted(cuts)
    # 去除 0/B 哨兵若被误加入
    cand = [c for c in cand if 0 < c < B]
    # 合并
    merged = []
    last = 0
    for c in cand:
        if c - last >= min_scene_len:
            merged.append(c)
            last = c
        # else 跳过（合并到上一段）
    # 尾段处理
    final = [0] + merged
    if final[-1] != B:
        if B - final[-1] >= min_scene_len or len(final) == 1:
            final.append(B)
        else:
            # 尾段过短，合并到上一段（用 B 替换最后一切点）
            # 若 merged 为空则直接 [0,B]
            if len(final) > 1:
                final[-1] = B
            else:
                final.append(B)
    # 去重排序
    final = sorted(set(final))
    # 确保首尾
    if final[0] != 0:
        final = [0] + final
    if final[-1] != B:
        final.append(B)
    final = sorted(set(final))
    return final


def split_scenes(cuts, B=None):
    """切点列表 -> [(s,e), ...] 段区间（左闭右开）。"""
    if not cuts:
        return []
    cuts = sorted(set(cuts))
    if B is not None and cuts[-1] != B:
        cuts.append(B)
    segs = []
    for i in range(len(cuts) - 1):
        s = int(cuts[i])
        e = int(cuts[i + 1])
        if e > s:
            segs.append((s, e))
    return segs
