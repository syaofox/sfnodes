_CATEGORY = "sfnodes/image"


class SFImageBatchIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "图像批次 [B, H, W, C]"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "要取出的图片索引（0 起）"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "按索引从图像批次中取出一张（保留 batch 维），常用于循环内逐张处理"

    def execute(self, images, index):
        if images is None or images.ndim != 4:
            raise ValueError("images 必须是 [B, H, W, C] 图像批次")
        if index < 0 or index >= images.shape[0]:
            raise ValueError(f"index {index} 超出有效范围 [0, {images.shape[0] - 1}]")
        return (images[index:index + 1],)
