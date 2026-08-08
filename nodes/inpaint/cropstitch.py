"""SF Inpaint Extend Outpaint — 外绘图像扩展（移植自 comfyui-pixaroma 旧版
InpaintCrop 的外绘扩展逻辑）。

注：Pixaroma 官方的 InpaintCrop / InpaintStitch 已升级为编辑器版（本项目见
inpaint_editor.py 的 SFInpaintCrop / SFInpaintStitch），旧版无编辑器实现已删除，
此处仅保留独立的图像扩展节点。
"""

import nodes
import torch

_CATEGORY = "sfnodes/inpaint"

class InpaintExtendOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["factors", "pixels"], {"default": "factors"}),
                "expand_up_pixels": (
                    "INT",
                    {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
                "expand_up_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01},
                ),
                "expand_down_pixels": (
                    "INT",
                    {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
                "expand_down_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01},
                ),
                "expand_left_pixels": (
                    "INT",
                    {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
                "expand_left_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01},
                ),
                "expand_right_pixels": (
                    "INT",
                    {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1},
                ),
                "expand_right_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01},
                ),
            },
            "optional": {
                "optional_context_mask": ("MASK",),
            },
        }

    CATEGORY = _CATEGORY
    DESCRIPTION = "扩展图像边界用于外绘（Outpainting），支持按比例或像素扩展"

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "context_mask")

    FUNCTION = "inpaint_extend"

    def inpaint_extend(
        self,
        image,
        mask,
        mode,
        expand_up_pixels,
        expand_up_factor,
        expand_down_pixels,
        expand_down_factor,
        expand_left_pixels,
        expand_left_factor,
        expand_right_pixels,
        expand_right_factor,
        optional_context_mask=None,
    ):
        assert image.shape[0] == mask.shape[0], (
            "Batch size of images and masks must be the same"
        )
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], (
                "Batch size of optional_context_masks must be the same as images or None"
            )

        results_image = []
        results_mask = []
        results_context_mask = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)  # Adding batch dimension
            one_mask = mask[b].unsqueeze(0)  # Adding batch dimension
            one_context_mask = None
            if optional_context_mask is not None:
                one_context_mask = optional_context_mask[b].unsqueeze(0)

            # Validate or initialize mask
            if (
                one_mask.shape[1] != one_image.shape[1]
                or one_mask.shape[2] != one_image.shape[2]
            ):
                non_zero_indices = torch.nonzero(one_mask[0], as_tuple=True)
                if not non_zero_indices[0].size(0):
                    one_mask = torch.zeros_like(one_image[:, :, :, 0])
                else:
                    assert False, "mask size must match image size"

            # Validate or initialize context mask
            if one_context_mask is not None and (
                one_context_mask.shape[1] != one_image.shape[1]
                or one_context_mask.shape[2] != one_image.shape[2]
            ):
                non_zero_indices = torch.nonzero(one_context_mask[0], as_tuple=True)
                if not non_zero_indices[0].size(0):
                    one_context_mask = torch.zeros_like(one_image[:, :, :, 0])
                else:
                    assert False, "context_mask size must match image size"

            # Get original dimensions
            orig_height, orig_width = one_image.shape[1], one_image.shape[2]

            if mode == "factors":
                # Calculate new dimensions based on factors
                new_height = int(
                    orig_height * (expand_up_factor + expand_down_factor - 1)
                )
                new_width = int(
                    orig_width * (expand_left_factor + expand_right_factor - 1)
                )

                up_padding = int(orig_height * (expand_up_factor - 1))
                down_padding = new_height - orig_height - up_padding
                left_padding = int(orig_width * (expand_left_factor - 1))
                right_padding = new_width - orig_width - left_padding
            elif mode == "pixels":
                # Calculate new dimensions based on pixel expansion
                new_height = orig_height + expand_up_pixels + expand_down_pixels
                new_width = orig_width + expand_left_pixels + expand_right_pixels

                up_padding = expand_up_pixels
                down_padding = expand_down_pixels
                left_padding = expand_left_pixels
                right_padding = expand_right_pixels

            # Expand image
            new_image = torch.zeros(
                (one_image.shape[0], new_height, new_width, one_image.shape[3]),
                dtype=one_image.dtype,
            )
            new_image[
                :,
                up_padding : up_padding + orig_height,
                left_padding : left_padding + orig_width,
                :,
            ] = one_image.squeeze(0)

            start_y = up_padding
            start_x = left_padding
            initial_height = orig_height
            initial_width = orig_width

            # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
            available_top = min(start_y, initial_height)
            available_bottom = min(
                new_height - (start_y + initial_height), initial_height
            )
            available_left = min(start_x, initial_width)
            available_right = min(new_width - (start_x + initial_width), initial_width)
            # Top
            if available_top:
                new_image[
                    :,
                    start_y - available_top : start_y,
                    start_x : start_x + initial_width,
                    :,
                ] = torch.flip(image[:, :available_top, :, :], [1])
            # Bottom
            if available_bottom:
                new_image[
                    :,
                    start_y + initial_height : start_y
                    + initial_height
                    + available_bottom,
                    start_x : start_x + initial_width,
                    :,
                ] = torch.flip(image[:, -available_bottom:, :, :], [1])
            # Left
            if available_left:
                new_image[
                    :,
                    start_y : start_y + initial_height,
                    start_x - available_left : start_x,
                    :,
                ] = torch.flip(
                    new_image[
                        :,
                        start_y : start_y + initial_height,
                        start_x : start_x + available_left,
                        :,
                    ],
                    [2],
                )
            # Right
            if available_right:
                new_image[
                    :,
                    start_y : start_y + initial_height,
                    start_x + initial_width : start_x + initial_width + available_right,
                    :,
                ] = torch.flip(
                    new_image[
                        :,
                        start_y : start_y + initial_height,
                        start_x + initial_width - available_right : start_x
                        + initial_width,
                        :,
                    ],
                    [2],
                )
            # Top-left corner
            if available_top and available_left:
                new_image[
                    :,
                    start_y - available_top : start_y,
                    start_x - available_left : start_x,
                    :,
                ] = torch.flip(
                    new_image[
                        :,
                        start_y : start_y + available_top,
                        start_x : start_x + available_left,
                        :,
                    ],
                    [1, 2],
                )
            # Top-right corner
            if available_top and available_right:
                new_image[
                    :,
                    start_y - available_top : start_y,
                    start_x + initial_width : start_x + initial_width + available_right,
                    :,
                ] = torch.flip(
                    new_image[
                        :,
                        start_y : start_y + available_top,
                        start_x + initial_width - available_right : start_x
                        + initial_width,
                        :,
                    ],
                    [1, 2],
                )
            # Bottom-left corner
            if available_bottom and available_left:
                new_image[
                    :,
                    start_y + initial_height : start_y
                    + initial_height
                    + available_bottom,
                    start_x - available_left : start_x,
                    :,
                ] = torch.flip(
                    new_image[
                        :,
                        start_y + initial_height - available_bottom : start_y
                        + initial_height,
                        start_x : start_x + available_left,
                        :,
                    ],
                    [1, 2],
                )
            # Bottom-right corner
            if available_bottom and available_right:
                new_image[
                    :,
                    start_y + initial_height : start_y
                    + initial_height
                    + available_bottom,
                    start_x + initial_width : start_x + initial_width + available_right,
                    :,
                ] = torch.flip(
                    new_image[
                        :,
                        start_y + initial_height - available_bottom : start_y
                        + initial_height,
                        start_x + initial_width - available_right : start_x
                        + initial_width,
                        :,
                    ],
                    [1, 2],
                )

            # Expand mask
            new_mask = torch.ones(
                (one_mask.shape[0], new_height, new_width), dtype=one_mask.dtype
            )
            new_mask[
                :,
                up_padding : up_padding + orig_height,
                left_padding : left_padding + orig_width,
            ] = one_mask.squeeze(0)

            # Expand context mask if present
            if one_context_mask is not None:
                new_context_mask = torch.zeros(
                    (one_context_mask.shape[0], new_height, new_width),
                    dtype=one_context_mask.dtype,
                )
                new_context_mask[
                    :,
                    up_padding : up_padding + orig_height,
                    left_padding : left_padding + orig_width,
                ] = one_context_mask.squeeze(0)

            # Append results
            results_image.append(new_image.squeeze(0))
            results_mask.append(new_mask.squeeze(0))
            if one_context_mask is not None:
                results_context_mask.append(new_context_mask.squeeze(0))

        # Stack the results to form batches
        output_image = torch.stack(results_image, dim=0)
        output_mask = torch.stack(results_mask, dim=0)
        output_context_mask = None
        if optional_context_mask is not None:
            output_context_mask = torch.stack(results_context_mask, dim=0)

        return (output_image, output_mask, output_context_mask)
