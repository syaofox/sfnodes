"""SFTrackDataToMask：从 SAM3_TRACK_DATA 取指定帧输出 MASK。

用于 `SFSAM3ReanchorTrack` 等追踪结果取单帧遮罩（首/尾帧或任一关键帧）：
选中对象并集后按 `orig_size` 双线性插值还原真实分辨率（与核心
`SAM3_TrackToMask` 一致，避免追踪器方形工作网格导致 1:1），输出 `[1, H, W]`。

纯逻辑在 `sf_utils/track_data_ops.py::frame_mask_from_track_data`（依赖注入可直测）。
"""

_CATEGORY = "sfnodes/image"


class SFTrackDataToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA", {"tooltip": "SAM3 追踪数据（如 SF SAM3 Reanchor Track 输出）"}),
                "frame_index": ("INT", {"default": 0, "min": -1000000, "max": 1000000, "step": 1,
                                        "tooltip": "取遮罩的帧序号（0 起；负值相对末尾，-1=最后一帧）；越界报错"}),
            },
            "optional": {
                "object_indices": ("STRING", {"default": "", "tooltip": "参与并集的对象索引，逗号分隔（如 '0,2'）。空=全部对象；越界/非法项忽略"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从 SAM3_TRACK_DATA 取指定帧的选中对象并集输出 MASK [1,H,W]（按 orig_size 还原真实分辨率，与 SAM3 Track to Mask 一致）；frame_index 支持负值取尾，空追踪/无有效对象输出全零"

    def execute(self, track_data, frame_index=0, object_indices=""):
        import torch
        from comfy.ldm.sam3.tracker import unpack_masks
        from ...sf_utils.track_data_ops import frame_mask_from_track_data, is_track_data

        if not is_track_data(track_data):
            raise ValueError("SF Track Data To Mask: 输入不是有效的 SAM3_TRACK_DATA")
        return (frame_mask_from_track_data(
            track_data, frame_index, object_indices,
            unpack_masks=unpack_masks, torch=torch,
            interpolate=torch.nn.functional.interpolate),)
