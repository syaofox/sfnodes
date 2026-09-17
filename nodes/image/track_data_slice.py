"""SFTrackDataSlice：SAM3_TRACK_DATA 时间维区间切片。

外部分段处理（长视频/长追踪逐段跑 SCAIL-2）时，整段追踪数据只需在循环外
跑一次（或读 `SFTrackDataCache` 缓存），循环内按段区间切片后送给
`SF SCAIL-2 Simple Video`，避免每段渲染整段彩色蒙版（O(T) 内存），并保证
段内 track 与段内 pose 帧一一对应。

纯逻辑在 `sf_utils/track_data_ops.py::slice_track_data`（无 torch 依赖，可直测）。
"""

_CATEGORY = "sfnodes/image"


class SFTrackDataSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA", {"tooltip": "待切片的追踪数据（通常为整段追踪结果或其磁盘缓存）"}),
                "start": ("INT", {"default": 0, "min": -1000000, "max": 1000000, "step": 1, "tooltip": "起始帧（0 起；负值相对末尾，-1=最后一帧）"}),
                "length": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "切片帧数（0=切到结尾；超出尾部自动截断）"}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "按帧区间切片 SAM3_TRACK_DATA（保留对象数/scores/真实宽高），供分段循环里逐段喂给 SCAIL-2 Simple Video；start 支持负值取尾、length=0 到结尾"

    def execute(self, track_data, start=0, length=0):
        from ...sf_utils.track_data_ops import is_track_data, slice_track_data

        if not is_track_data(track_data):
            raise ValueError("SF Track Data Slice: 输入不是有效的 SAM3_TRACK_DATA")
        return (slice_track_data(track_data, start, length),)
