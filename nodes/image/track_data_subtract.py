_CATEGORY = "sfnodes/image"

_MAX_EXCLUDES = 4


class SFTrackDataSubtract:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"exclude_{i}": (
                "MASK,SAM3_TRACK_DATA",
                {"tooltip": f"要从基础遮罩中逐帧排除的遮罩 / 追踪数据（第 {i} 路，未连接跳过）"},
            )
            for i in range(1, _MAX_EXCLUDES + 1)
        }
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA", {"tooltip": "基础追踪数据（如女性 SAM3 Video Track 输出）"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "在 SAM3_TRACK_DATA 层逐帧减去若干排除遮罩/追踪数据（男性、阴茎、精液等），保留对象数；输出可直接接 SCAIL-2 的 driving_track_data"

    def execute(self, track_data, **kwargs):
        import torch
        from comfy.ldm.sam3.tracker import pack_masks, unpack_masks
        from ...sf_utils.track_data_ops import is_track_data, subtract_from_track_data

        if not is_track_data(track_data):
            raise ValueError("SF Track Data Subtract: 输入不是有效的 SAM3_TRACK_DATA")

        excludes = [kwargs.get(f"exclude_{i}") for i in range(1, _MAX_EXCLUDES + 1)]
        return (subtract_from_track_data(
            track_data,
            excludes,
            pack_masks=pack_masks,
            unpack_masks=unpack_masks,
            torch=torch,
            interpolate=torch.nn.functional.interpolate,
        ),)
