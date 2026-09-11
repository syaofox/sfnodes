_CATEGORY = "sfnodes/image"


def _resolve_range(count, start_index, num_frames):
    if start_index == -1:
        start = max(0, count - num_frames)
    else:
        start = start_index
    if start < 0 or start >= count:
        raise ValueError(f"start_index {start_index} 超出有效范围 [0, {count - 1}]（-1=取尾部）")
    return (start, min(start + num_frames, count))


class SFImageBatchRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_index": ("INT", {"default": 0, "min": -1, "max": 1000000, "step": 1, "tooltip": "起始索引（0 起，-1=从尾部取 num_frames 帧）"}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 1000000, "step": 1, "tooltip": "取出帧数，超出尾部自动截断"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "图像批次 [B, H, W, C]"}),
                "masks": ("MASK", {"tooltip": "遮罩批次 [B, H, W]"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "从图像/遮罩批次中按区间取出多帧（复刻 KJ GetImageRangeFromBatch）：start_index+num_frames 切片，-1 取尾部，尾部超出自动截断"

    def execute(self, start_index, num_frames, images=None, masks=None):
        chosen_images = None
        chosen_masks = None

        if images is not None:
            if images.ndim != 4:
                raise ValueError("images 必须是 [B, H, W, C] 图像批次")
            s, e = _resolve_range(int(images.shape[0]), start_index, num_frames)
            chosen_images = images[s:e]

        if masks is not None:
            if masks.ndim not in (2, 3):
                raise ValueError("masks 必须是 [B, H, W] 遮罩批次")
            s, e = _resolve_range(int(masks.shape[0]), start_index, num_frames)
            chosen_masks = masks[s:e]

        if chosen_images is None and chosen_masks is None:
            raise ValueError("SF Image Batch Range: 至少需要一路输入（images 或 masks）")

        return (chosen_images, chosen_masks)
