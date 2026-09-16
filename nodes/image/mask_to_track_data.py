_CATEGORY = "sfnodes/image"


def _empty_track_data(n_frames=0, orig_size=(0, 0)):
    return {"packed_masks": None, "n_frames": int(n_frames), "orig_size": tuple(orig_size), "scores": []}


def mask_to_track_data(masks, pack_masks=None, torch=None):
    from ...sf_utils.track_data_ops import pad_width_to_8

    if masks is None:
        raise ValueError("SF Mask To Track Data: 输入 MASK 为空")
    dim = masks.dim()
    if dim == 2:
        masks = masks.unsqueeze(0).unsqueeze(0)
    elif dim == 3:
        masks = masks.unsqueeze(1)
    elif dim != 4:
        raise ValueError(f"SF Mask To Track Data: 不支持的 MASK 维度 {dim}（期望 2D/3D/4D）")

    t, n, h, w = masks.shape
    if n != 1:
        raise ValueError(f"SF Mask To Track Data: 仅支持单对象 MASK（对象维应为 1，实际 {n}）")
    if t == 0 or h <= 0 or w <= 0:
        return _empty_track_data(t, (h, w))

    masks = pad_width_to_8(masks, torch)

    return {
        "packed_masks": pack_masks(masks),
        "n_frames": int(t),
        "orig_size": (int(h), int(w)),
        "scores": [1.0],
    }


class SFMaskToTrackData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "逐帧追踪遮罩（batch [T,H,W]），如 SeC Video Segmentation 的 masks 输出"}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "把逐帧 MASK 打包成 SAM3_TRACK_DATA（复用核心 pack_masks），使 SeC 等 MASK 追踪结果可直接接入 SCAIL2ColoredMask 的 driving_track_data"

    def execute(self, masks):
        import torch
        from comfy.ldm.sam3.tracker import pack_masks

        return (mask_to_track_data(masks, pack_masks=pack_masks, torch=torch),)
