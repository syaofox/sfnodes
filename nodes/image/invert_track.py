_CATEGORY = "sfnodes/image"


def _empty_result(track_data):
    out = dict(track_data)
    out["packed_masks"] = None
    out["scores"] = []
    return out


def _full_frame_result(track_data, n_frames, orig_size, pack_masks, torch):
    if not n_frames or not orig_size:
        return _empty_result(track_data)
    h, w = int(orig_size[0]), int(orig_size[1])
    if h <= 0 or w <= 0:
        return _empty_result(track_data)
    wp = (w + 7) // 8 * 8
    full = torch.ones((n_frames, h, wp), dtype=torch.bool)
    out = dict(track_data)
    out["packed_masks"] = pack_masks(full).unsqueeze(1)
    out["scores"] = [1.0]
    return out


def invert_track_data(track_data, object_indices="", unpack_masks=None, pack_masks=None, torch=None):
    from ...sf_utils.track_data_ops import parse_object_indices

    packed = track_data.get("packed_masks")
    n_frames = track_data.get("n_frames")
    orig_size = track_data.get("orig_size")

    if packed is None or packed.shape[1] == 0:
        return _full_frame_result(track_data, n_frames, orig_size, pack_masks, torch)

    n_obj = packed.shape[1]
    indices = parse_object_indices(object_indices, n_obj)
    if not indices:
        return _empty_result(track_data)

    masks = unpack_masks(packed[:, indices])
    union = masks.any(dim=1)
    inv = ~union
    out = dict(track_data)
    out["packed_masks"] = pack_masks(inv).unsqueeze(1)
    out["scores"] = [1.0]
    return out


class SFInvertTrackData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA", {"tooltip": "SAM3 视频追踪数据（SAM3 Video Track 输出）"}),
            },
            "optional": {
                "object_indices": ("STRING", {"default": "", "tooltip": "参与反转的对象索引，逗号分隔（如 '0,2'）。空=全部对象"}),
            },
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "反转 SAM3 追踪数据：对选中对象并集取反后合成为单一身份（prompt 写 person 即得到 person 以外区域），可直接接 SCAIL2ColoredMask 的 driving/ref"

    def execute(self, track_data, object_indices=""):
        import torch
        from comfy.ldm.sam3.tracker import pack_masks, unpack_masks

        if not isinstance(track_data, dict) or "packed_masks" not in track_data:
            raise ValueError("SF Invert Track Data: 输入不是有效的 SAM3_TRACK_DATA")
        return (invert_track_data(track_data, object_indices=object_indices,
                                  unpack_masks=unpack_masks, pack_masks=pack_masks, torch=torch),)
