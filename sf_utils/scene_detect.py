"""镜头切分纯逻辑（SFImageSceneSplit 用）。

无 torch / ComfyUI 依赖，可独立测试。
输入帧为 uint8 RGB [H,W,3] 或 float [0,1]，内部统一转 uint8 小图后对比。

检测四类（融合 max）：
  - 硬切（跳切）：RGB 直方图 + HSV 直方图 + 边缘差 + 2×2 分块直方图 max + RGB 像素差 的 max > 自适应阈值
  - 黑场：连续黑帧（灰度均值 < black_threshold）段边界
  - 白闪：连续白帧（灰度均值 > white_threshold）段边界
  - 溶解/渐变：滑窗 W 内累积 RGB 直方图距离 > threshold 且窗内单步 max < threshold 且 avg > dissolve_threshold

最短场景去抖：相邻切点距 < min_scene_len 则合并（删后者），尾段不足也合并。
自适应：threshold' = max(threshold, μ+3σ)，μ/σ 取前 32 帧融合距离滑动窗，抑制抖动误切。
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
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"通道数非法: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) if arr.dtype in (np.float32, np.float64) else np.clip(arr, 0, 255)
        if arr.max() <= 1.0 + 1e-6 and arr.dtype != np.uint8:
            is_float01 = arr.dtype in (np.float32, np.float64) and float(arr.max()) <= 1.0
            if is_float01:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _downscale_rgb(frame_uint8, longest=160):
    """RGB uint8 -> 缩略 RGB uint8 [h,w,3]，最长边 longest。"""
    h, w = frame_uint8.shape[:2]
    ls = max(h, w)
    if ls > longest:
        scale = longest / ls
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
    else:
        nh, nw = h, w
    try:
        import cv2  # type: ignore

        small_rgb = cv2.resize(frame_uint8, (nw, nh), interpolation=cv2.INTER_AREA)
        return small_rgb
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        pil = Image.fromarray(frame_uint8)
        pil_small = pil.resize((nw, nh), Image.BILINEAR)
        return np.array(pil_small)
    except Exception:
        pass
    step_h = max(1, h // nh)
    step_w = max(1, w // nw)
    sampled = frame_uint8[::step_h, ::step_w]
    sampled = sampled[:nh, :nw]
    if sampled.shape[0] < nh or sampled.shape[1] < nw:
        res = np.zeros((nh, nw, 3), dtype=np.uint8)
        rh = min(nh, sampled.shape[0])
        rw = min(nw, sampled.shape[1])
        res[:rh, :rw] = sampled[:rh, :rw]
        sampled = res
    return sampled


def _rgb_to_gray(small_rgb):
    return (0.2126 * small_rgb[:, :, 0] + 0.7152 * small_rgb[:, :, 1] + 0.0722 * small_rgb[:, :, 2]).astype(np.uint8)


def _hist(gray_small, bins=32):
    hist, _ = np.histogram(gray_small, bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _hist_rgb(small_rgb, bins=32):
    hists = []
    for c in range(3):
        hist, _ = np.histogram(small_rgb[:, :, c], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist /= s
        hists.append(hist)
    return hists


def _hist_hsv(small_rgb, bins=32):
    """H 通道直方图（HSV 对光照不敏感）。cv2 优先，PIL 回退，失败回退 R 通道。"""
    try:
        import cv2  # type: ignore

        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]  # 0-180
        hist, _ = np.histogram(h, bins=bins, range=(0, 180))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        pil = Image.fromarray(small_rgb)
        hsv = pil.convert("HSV")
        arr = np.array(hsv)
        h = arr[:, :, 0]
        hist, _ = np.histogram(h, bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist
    except Exception:
        # 回退 R 通道
        hist, _ = np.histogram(small_rgb[:, :, 0], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist


def _sobel_edge(small_gray):
    try:
        import cv2  # type: ignore

        gx = cv2.Sobel(small_gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(small_gray, cv2.CV_16S, 0, 1, ksize=3)
        mag = np.sqrt(gx.astype(float) ** 2 + gy.astype(float) ** 2)
        return np.clip(mag, 0, 255).astype(np.uint8)
    except Exception:
        # 简易梯度
        gx = np.diff(small_gray.astype(int), axis=1, prepend=small_gray[:, :1].astype(int))
        gy = np.diff(small_gray.astype(int), axis=0, prepend=small_gray[:1, :].astype(int))
        mag = np.sqrt(gx.astype(float) ** 2 + gy.astype(float) ** 2)
        return np.clip(mag, 0, 255).astype(np.uint8)


def _hist_distance(h1, h2):
    """Bhattacharyya 距离 1 - BC，0=相同，1=完全不同。"""
    bc = float(np.sum(np.sqrt(h1 * h2)))
    bc = max(0.0, min(1.0, bc))
    return 1.0 - bc


def _hist_distance_rgb(hs1, hs2):
    return float(np.mean([_hist_distance(h1, h2) for h1, h2 in zip(hs1, hs2)]))


def _diff_distance(g1, g2):
    return float(np.mean(np.abs(g1.astype(np.int16) - g2.astype(np.int16))) / 255.0)


def _diff_distance_rgb(rgb1, rgb2):
    return float(np.mean(np.abs(rgb1.astype(np.int16) - rgb2.astype(np.int16))) / 255.0)


def _block_hist_distance_rgb(small_rgb1, small_rgb2, bins=16, grid=4):
    h, w = small_rgb1.shape[:2]
    # 4×4 分块（同调色跳切多为局部替换，2×2 对 16×16 小块仅占块 25% 会稀释到 0.13）
    if h < 8 or w < 8:
        return _hist_distance_rgb(_hist_rgb(small_rgb1, bins=bins), _hist_rgb(small_rgb2, bins=bins))
    maxd = 0.0
    for i in range(grid):
        for j in range(grid):
            y = i * h // grid
            x = j * w // grid
            bh = h // grid if i < grid - 1 else h - y
            bw = w // grid if j < grid - 1 else w - x
            if bh <= 0 or bw <= 0:
                continue
            b1 = small_rgb1[y : y + bh, x : x + bw]
            b2 = small_rgb2[y : y + bh, x : x + bw]
            if b1.size == 0 or b2.size == 0:
                continue
            d = _hist_distance_rgb(_hist_rgb(b1, bins=bins), _hist_rgb(b2, bins=bins))
            if d > maxd:
                maxd = d
                if maxd >= 0.99:
                    return float(maxd)
    return float(maxd)


def _process_frame(frame, bins=32, longest=160):
    fu8 = _to_uint8_rgb(frame)
    small_rgb = _downscale_rgb(fu8, longest=longest)
    small_gray = _rgb_to_gray(small_rgb)
    hists_rgb = _hist_rgb(small_rgb, bins=bins)
    hist_hsv = _hist_hsv(small_rgb, bins=bins)
    edge = _sobel_edge(small_gray)
    mean = float(np.mean(small_gray) / 255.0)
    return small_rgb, small_gray, hists_rgb, hist_hsv, edge, mean


def detect_scenes(frames, threshold=0.22, black_threshold=0.08, white_threshold=0.92,
                  min_scene_len=12, method="auto", dissolve_window=8, dissolve_threshold=0.18,
                  bins=32, longest=160):
    """帧序列 -> 切点列表 [0, cut1, ..., B]（含起止）。

    frames: np.ndarray [B,H,W,3] uint8/float 或 iterable[frame]
    method: "auto"(融合 max) | "hist" | "diff"（后两者兼容，内部仍走融合以提升同调色召回）
    """
    rgbs = []
    grays = []
    hists_rgb_list = []
    hists_hsv_list = []
    edges = []
    means = []
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        b = frames.shape[0]
        iterator = (frames[i] for i in range(b))
    elif isinstance(frames, (list, tuple)):
        iterator = iter(frames)
    else:
        try:
            iterator = iter(frames)
        except TypeError:
            raise ValueError("frames 必须是 [B,H,W,C] 数组或可迭代帧序列")
    for fr in iterator:
        small_rgb, small_gray, hists_rgb, hist_hsv, edge, mean = _process_frame(fr, bins=bins, longest=longest)
        rgbs.append(small_rgb)
        grays.append(small_gray)
        hists_rgb_list.append(hists_rgb)
        hists_hsv_list.append(hist_hsv)
        edges.append(edge)
        means.append(mean)
    B = len(means)
    if B == 0:
        return [0, 0]
    if B == 1:
        return [0, 1]

    cuts = set()

    # 预计算融合距离（B-1）
    fused = []
    for i in range(B - 1):
        d_rgb = _hist_distance_rgb(hists_rgb_list[i], hists_rgb_list[i + 1])
        d_hsv = _hist_distance(hists_hsv_list[i], hists_hsv_list[i + 1])
        d_diff = _diff_distance_rgb(rgbs[i], rgbs[i + 1])
        d_edge = _diff_distance(edges[i], edges[i + 1])
        d_block = _block_hist_distance_rgb(rgbs[i], rgbs[i + 1], bins=16)
        fused.append(max(d_rgb, d_hsv, d_diff, d_edge, d_block))

    # 自适应阈值（32 帧滑动窗 μ+3σ），抑制抖动段误切
    adapt_thr = []
    window = 32
    for i in range(len(fused)):
        base = float(threshold)
        if i >= 5:
            win = fused[max(0, i - window) : i]
            if len(win) >= 4:
                mu = float(np.mean(win))
                sigma = float(np.std(win))
                # 仅当方差有意义时提升阈值
                if sigma > 1e-6:
                    cand = mu + 3.0 * sigma
                    if cand > base:
                        base = min(0.9, cand)
        adapt_thr.append(base)

    # 1) 硬切（融合 max）
    for i, d in enumerate(fused):
        # method 兼容：hist/diff 仍走融合，不再单指标漏检
        if d > adapt_thr[i]:
            cuts.add(i + 1)

    # 2) 黑/白场连续段边界
    def add_runs(mask, is_black=True):
        i = 0
        while i < B:
            if not mask[i]:
                i += 1
                continue
            s = i
            while i < B and mask[i]:
                i += 1
            e = i - 1
            run_len = e - s + 1
            if run_len >= 1 and not (s == 0 and e == B - 1):
                if s > 0:
                    cuts.add(s)
                if e + 1 < B:
                    cuts.add(e + 1)
    black_mask = [m < black_threshold for m in means]
    white_mask = [m > white_threshold for m in means]
    add_runs(black_mask, is_black=True)
    add_runs(white_mask, is_black=False)

    # 3) 溶解/渐变（滑窗累积，RGB 直方图）
    W = int(dissolve_window)
    if W >= 2 and B > W:
        step_dists = fused  # 复用融合距离作单步
        for i in range(B - W):
            # 累积距离用 RGB 直方图跨窗
            D = _hist_distance_rgb(hists_rgb_list[i], hists_rgb_list[i + W])
            # 也可用 HSV 跨窗取 max 更敏感
            D_hsv = _hist_distance(hists_hsv_list[i], hists_hsv_list[i + W])
            D = max(D, D_hsv)
            window_steps = step_dists[i : i + W]
            avg_step = float(np.mean(window_steps)) if window_steps else 0.0
            max_step = float(np.max(window_steps)) if window_steps else 0.0
            # 自适应阈值取窗起点对应阈值
            thr = adapt_thr[i] if i < len(adapt_thr) else float(threshold)
            if D > thr and max_step < thr and avg_step > dissolve_threshold:
                mid = i + W // 2 + 1
                if 0 < mid < B:
                    cuts.add(mid)

    # 4) 去抖：相邻切点距 < min_scene_len 则合并（删后者）
    cand = sorted(cuts)
    cand = [c for c in cand if 0 < c < B]
    merged = []
    last = 0
    for c in cand:
        if c - last >= min_scene_len:
            merged.append(c)
            last = c
    final = [0] + merged
    if final[-1] != B:
        if B - final[-1] >= min_scene_len or len(final) == 1:
            final.append(B)
        else:
            if len(final) > 1:
                final[-1] = B
            else:
                final.append(B)
    final = sorted(set(final))
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
