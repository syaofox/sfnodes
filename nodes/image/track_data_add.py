_CATEGORY = "sfnodes/image"

_MAX_ADD_SLOTS = 20


class SFTrackDataAdd:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"add_{i}": (
                "MASK,SAM3_TRACK_DATA",
                {"tooltip": f"要逐帧并集叠加到基础追踪数据的遮罩 / 追踪数据（第 {i} 路，未连接跳过）"},
            )
            for i in range(1, _MAX_ADD_SLOTS + 1)
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
    DESCRIPTION = "把若干 MASK / SAM3_TRACK_DATA 逐帧并集叠加到基础追踪数据并合成为单一身份（SF Track Data Subtract 的逆操作）；输入槽随连接自动增删（初始 4，上限 20）；输出可直接接 SCAIL-2 的 driving/ref"

    def execute(self, track_data, **kwargs):
        import torch
        from comfy.ldm.sam3.tracker import pack_masks, unpack_masks
        from ...sf_utils.track_data_ops import add_to_track_data, is_track_data

        if not is_track_data(track_data):
            raise ValueError("SF Track Data Add: 输入不是有效的 SAM3_TRACK_DATA")

        adds = [v for k, v in kwargs.items() if k.startswith("add_") and v is not None]
        return (add_to_track_data(
            track_data,
            adds,
            pack_masks=pack_masks,
            unpack_masks=unpack_masks,
            torch=torch,
            interpolate=torch.nn.functional.interpolate,
        ),)
