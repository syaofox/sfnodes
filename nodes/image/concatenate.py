import torch
from comfy.utils import common_upscale

_CATEGORY = "sfnodes/image"
_MAX_IMAGE_SLOTS = 16


class ImageConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, _MAX_IMAGE_SLOTS + 1):
            optional[f"image_{i}"] = ("IMAGE",)
        return {
            "required": {
                "direction": (
                    ["right", "down", "left", "up"],
                    {"default": "right"},
                ),
                "match_image_size": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = _CATEGORY
    DESCRIPTION = "拼接多张图片，输入端口随连接自动增减，支持右、下、左、上四个方向"

    def concatenate(self, direction, match_image_size, **kwargs):
        images = []
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if v is not None:
                images.append(v)

        if not images:
            return (torch.zeros((1, 64, 64, 3)),)

        result = images[0]
        for next_img in images[1:]:
            result = self._pair_concat(result, next_img, direction, match_image_size)
        return (result,)

    @staticmethod
    def _pair_concat(image_a, image_b, direction, match_image_size):
        batch_size1 = image_a.shape[0]
        batch_size2 = image_b.shape[0]

        if batch_size1 != batch_size2:
            max_batch = max(batch_size1, batch_size2)
            if max_batch - batch_size1 > 0:
                last = image_a[-1].unsqueeze(0).repeat(max_batch - batch_size1, 1, 1, 1)
                image_a = torch.cat([image_a.clone(), last], dim=0)
            if max_batch - batch_size2 > 0:
                last = image_b[-1].unsqueeze(0).repeat(max_batch - batch_size2, 1, 1, 1)
                image_b = torch.cat([image_b.clone(), last], dim=0)

        if match_image_size:
            orig_h, orig_w = image_b.shape[1], image_b.shape[2]
            aspect = orig_w / orig_h
            if direction in ("left", "right"):
                target_h = image_a.shape[1]
                target_w = int(target_h * aspect)
            else:
                target_w = image_a.shape[2]
                target_h = int(target_w / aspect)
            img = image_b.movedim(-1, 1)
            img = common_upscale(img, target_w, target_h, "lanczos", "disabled")
            image_b = img.movedim(1, -1)

        ch_a, ch_b = image_a.shape[-1], image_b.shape[-1]
        if ch_a != ch_b:
            if ch_a < ch_b:
                pad = torch.ones((*image_a.shape[:-1], ch_b - ch_a), device=image_a.device)
                image_a = torch.cat((image_a, pad), dim=-1)
            else:
                pad = torch.ones((*image_b.shape[:-1], ch_a - ch_b), device=image_b.device)
                image_b = torch.cat((image_b, pad), dim=-1)

        if direction == "right":
            return torch.cat((image_a, image_b), dim=2)
        elif direction == "down":
            return torch.cat((image_a, image_b), dim=1)
        elif direction == "left":
            return torch.cat((image_b, image_a), dim=2)
        elif direction == "up":
            return torch.cat((image_b, image_a), dim=1)


class ImageConcatFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_columns": ("INT", {"default": 3, "min": 1, "max": 255, "step": 1}),
                "match_image_size": ("BOOLEAN", {"default": False}),
                "max_resolution": ("INT", {"default": 4096}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat"
    CATEGORY = _CATEGORY
    DESCRIPTION = "将图片批次排列为网格拼接图，支持指定列数和尺寸匹配"

    def concat(self, images, num_columns, match_image_size, max_resolution):
        batch_size, height, width, channels = images.shape
        num_rows = (batch_size + num_columns - 1) // num_columns

        if match_image_size:
            target_shape = images[0].shape

            resized_images = []
            for image in images:
                original_height = image.shape[0]
                original_width = image.shape[1]
                original_aspect_ratio = original_width / original_height

                if original_aspect_ratio > 1:
                    target_height = target_shape[0]
                    target_width = int(target_height * original_aspect_ratio)
                else:
                    target_width = target_shape[1]
                    target_height = int(target_width / original_aspect_ratio)

                resized_image = common_upscale(
                    image.movedim(-1, 0),
                    target_width,
                    target_height,
                    "lanczos",
                    "disabled",
                )
                resized_image = resized_image.movedim(0, -1)
                resized_images.append(resized_image)

            images = torch.stack(resized_images)
            height, width = target_shape[:2]

        grid_height = num_rows * height
        grid_width = num_columns * width

        scale_factor = min(
            max_resolution / grid_height, max_resolution / grid_width, 1.0
        )

        scaled_height = height * scale_factor
        scaled_width = width * scale_factor

        height = max(1, int(round(scaled_height / 8) * 8))
        width = max(1, int(round(scaled_width / 8) * 8))

        if abs(scaled_height - height) > 4:
            height = max(1, int(round((scaled_height + 4) / 8) * 8))
        if abs(scaled_width - width) > 4:
            width = max(1, int(round((scaled_width + 4) / 8) * 8))

        grid_height = num_rows * height
        grid_width = num_columns * width

        grid = torch.zeros((grid_height, grid_width, channels), dtype=images.dtype)

        for idx, image in enumerate(images):
            resized_image = (
                torch.nn.functional.interpolate(
                    image.unsqueeze(0).permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                )
                .squeeze()
                .permute(1, 2, 0)
            )
            row = idx // num_columns
            col = idx % num_columns
            grid[
                row * height : (row + 1) * height, col * width : (col + 1) * width, :
            ] = resized_image

        return (grid.unsqueeze(0),)
