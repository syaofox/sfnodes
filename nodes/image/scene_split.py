import json

import numpy as np
import torch

_Category = "sfnodes/image"
_Category = "sfnodes/image"
_CATEGORY = "sfnodes/image"


class SFImageSceneSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "视频连续帧 [B, H, W, C] float [0,1]"}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.01, "tooltip": "硬切阈值（直方图 Bhattacharyya 距离 0-1），越大越迟钝"},
                ),
                "black_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "黑场亮度阈值（灰度均值 [0,1] 低于此视为黑帧）"},
                ),
                "white_threshold": (
                    "FLOAT",
                    {"default": 0.92, "min": 0.5, "max": 0.99, "step": 0.01, "tooltip": "白闪阈值（灰度均值高于此视为白场）"},
                ),
                "min_scene_len": (
                    "INT",
                    {"default": 12, "min": 1, "max": 1000, "step": 1, "tooltip": "最短场景帧数，小于此的切点被合并去抖"},
                ),
                "segment_index": (
                    "INT",
                    {"default": 0, "min": -100000, "max": 100000, "step": 1, "tooltip": "要输出第几段，0 起；负数倒数，越界抛错"},
                ),
                "max_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "限制输出帧数，0=不限制；>0 时取段内首 N 帧"},
                ),
                "method": (
                    ["hist", "diff"],
                    {"default": "hist", "tooltip": "hist=直方图距离（推荐，抗抖动），diff=缩略图像素差"},
                ),
                "dissolve_window": (
                    "INT",
                    {"default": 8, "min": 2, "max": 60, "step": 1, "tooltip": "溶解检测窗口帧数"},
                ),
                "dissolve_threshold": (
                    "FLOAT",
                    {"default": 0.18, "min": 0.05, "max": 0.8, "step": 0.01, "tooltip": "溶解单步阈值（窗口内单步均值需高于此才判溶解）"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "INT", "IMAGE")
    RETURN_NAMES = ("images", "count", "cuts", "scene_count", "all_segments")
    OUTPUT_IS_LIST = (False, False, False, False, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "视频帧镜头切分：检测硬切/黑场/白闪/溶解，按 segment_index 输出指定段（负数倒数，越界抛错），max_frames 取首 N 帧；all_segments 以 LIST 输出全部分段"

    def execute(self, images, threshold, black_threshold, white_threshold, min_scene_len,
                segment_index, max_frames, method, dissolve_window, dissolve_threshold):
        from ...sf_utils.scene_detect import detect_scenes

        if images is None or not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images 必须是 [B, H, W, C] 图像批次")
        B = int(images.shape[0])
        if B == 0:
            raise ValueError("images 批次为空")

        # 逐帧转 uint8 生成器，避免一次性 B*H*W 复制
        def frame_gen():
            for i in range(B):
                f = images[i]  # [H,W,C] tensor
                arr = f.detach().cpu().numpy()
                # arr in [0,1] float
                arr = np.clip(arr, 0.0, 1.0)
                arr_u8 = (arr * 255.0).astype(np.uint8)
                # 通道处理：若 C==1 复制，>3 截断
                if arr_u8.ndim == 3:
                    c = arr_u8.shape[2]
                    if c == 1:
                        arr_u8 = np.repeat(arr_u8, 3, axis=2)
                    elif c > 3:
                        arr_u8 = arr_u8[:, :, :3]
                elif arr_u8.ndim == 2:
                    arr_u8 = np.stack([arr_u8] * 3, axis=2)
                yield arr_u8

        cuts = detect_scenes(
            frame_gen(),
            threshold=float(threshold),
            black_threshold=float(black_threshold),
            white_threshold=float(white_threshold),
            min_scene_len=int(min_scene_len),
            method=str(method),
            dissolve_window=int(dissolve_window),
            dissolve_threshold=float(dissolve_threshold),
        )
        scene_count = len(cuts) - 1

        # 构建全部分段 LIST（保持原分辨率/设备/dtype）
        all_segments = []
        for i in range(scene_count):
            s = int(cuts[i])
            e = int(cuts[i + 1])
            seg = images[s:e]
            all_segments.append(seg)

        # 解析 segment_index（支持负数）
        orig_idx = int(segment_index)
        idx = orig_idx
        if idx < 0:
            idx = scene_count + idx
        if idx < 0 or idx >= scene_count:
            raise ValueError(
                f"segment_index {orig_idx} 越界，可用范围 [{-scene_count}, {scene_count - 1}]，当前共 {scene_count} 段，cuts={cuts}"
            )
        selected = all_segments[idx]
        if max_frames is not None and int(max_frames) > 0 and selected.shape[0] > int(max_frames):
            selected = selected[: int(max_frames)]

        count = int(selected.shape[0])
        cuts_json = json.dumps(cuts)

        return (selected, count, cuts_json, scene_count, all_segments)
