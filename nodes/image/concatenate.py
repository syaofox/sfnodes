import torch
from comfy.utils import common_upscale

_CATEGORY = "sfnodes/image"


class ImageConcanate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (
                    [
                        "right",
                        "down",
                        "left",
                        "up",
                    ],
                    {"default": "right"},
                ),
                "match_image_size": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = _CATEGORY

    def concatenate(
        self, image1, image2, direction, match_image_size, first_image_shape=None
    ):
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2

            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)

        if match_image_size:
            target_shape = (
                first_image_shape if first_image_shape is not None else image1.shape
            )

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ["left", "right"]:
                target_height = target_shape[1]
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ["up", "down"]:
                target_width = target_shape[2]
                target_height = int(target_width / original_aspect_ratio)

            image2_for_upscale = image2.movedim(-1, 1)
            image2_resized = common_upscale(
                image2_for_upscale, target_width, target_height, "lanczos", "disabled"
            )
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                alpha_channel = torch.ones(
                    (*image1.shape[:-1], channels_image2 - channels_image1),
                    device=image1.device,
                )
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                alpha_channel = torch.ones(
                    (*image2_resized.shape[:-1], channels_image1 - channels_image2),
                    device=image2_resized.device,
                )
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)

        if direction == "right":
            concatenated_image = torch.cat((image1, image2_resized), dim=2)
        elif direction == "down":
            concatenated_image = torch.cat((image1, image2_resized), dim=1)
        elif direction == "left":
            concatenated_image = torch.cat((image2_resized, image1), dim=2)
        elif direction == "up":
            concatenated_image = torch.cat((image2_resized, image1), dim=1)
        return (concatenated_image,)


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
