import json

_CATEGORY = "sfnodes/video"


def _has_points(spec):
    """点提示字符串是否含有效坐标（空串 / "[]" / "null" 视为无）。"""
    if not spec:
        return False
    try:
        return bool(json.loads(spec))
    except Exception:
        return True


def _node_result(value):
    if hasattr(value, "result"):
        result = value.result
        if result is None:
            return ()
        return tuple(result)
    if isinstance(value, tuple):
        return value
    return (value,)


class SFSAM3PointTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "驱动视频帧序列；锚帧检测后全程传播，输出与输入同长度"}),
                "model": ("MODEL", {"tooltip": "SAM3.1 模型（CheckpointLoaderSimple 的 MODEL 输出）"}),
                "anchor_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                         "tooltip": "锚定帧索引：在该帧做检测并从此向后传播；精液等中途出现的目标填其出现帧"}),
            },
            "optional": {
                "positive_coords": ("STRING", {"forceInput": True, "tooltip": "正向点 JSON（PointsEditor 的 positive_coords）"}),
                "negative_coords": ("STRING", {"forceInput": True, "tooltip": "负向点 JSON（PointsEditor 的 negative_coords）"}),
                "initial_mask": ("MASK", {"tooltip": "直接给定锚帧遮罩（如 SAM3 Detect 输出）；接了则跳过点检测，点提示可留空"}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1,
                                              "tooltip": "SAM 解码器精修轮数（0=用原始检测掩码）"}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "用点提示（或现成遮罩）在锚帧检测目标，再用 SAM3 全程传播为 SAM3_TRACK_DATA；输出与输入同长度，锚帧之前的帧为空（可接 SF Track Data Subtract 做排除）"

    def execute(self, images, model, anchor_frame=0, positive_coords=None,
                negative_coords=None, initial_mask=None, refine_iterations=2):
        import torch
        from ...sf_utils.track_data_ops import pad_track_data_front

        if images is None or images.ndim != 4:
            raise ValueError("SF SAM3 Point Track: images 必须是 [T,H,W,C] 的 IMAGE 张量")
        total = int(images.shape[0])
        anchor = int(anchor_frame)
        if anchor < 0 or anchor >= total:
            raise ValueError(f"SF SAM3 Point Track: anchor_frame={anchor} 越界（总帧数 {total}）")

        if initial_mask is not None:
            seed = initial_mask
            if seed.ndim == 2:
                seed = seed.unsqueeze(0)
        else:
            if not (_has_points(positive_coords) or _has_points(negative_coords)):
                raise ValueError("SF SAM3 Point Track: 请先在 PointsEditor 点选目标（刷新底图后 Shift+左键=正向点），或接入 initial_mask")
            from comfy_extras.nodes_sam3 import SAM3_Detect

            detect = _node_result(SAM3_Detect.execute(
                model=model,
                image=images[anchor:anchor + 1],
                positive_coords=positive_coords,
                negative_coords=negative_coords,
                refine_iterations=int(refine_iterations),
            ))
            seed = detect[0]

        from comfy_extras.nodes_sam3 import SAM3_VideoTrack

        result = _node_result(SAM3_VideoTrack.execute(
            images=images[anchor:],
            model=model,
            initial_mask=seed,
            conditioning=None,
            detection_threshold=0.5,
            max_objects=1,
            detect_interval=1,
        ))
        track_data = result[0]
        if anchor > 0:
            track_data = pad_track_data_front(track_data, total, torch=torch)
        return (track_data,)
