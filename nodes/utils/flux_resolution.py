"""
Flux Resolution Calculator 节点：根据百万像素和宽高比计算分辨率，并生成预览图。
"""

import os
from PIL import Image, ImageDraw, ImageFont

from ...sf_utils.image_convert import pil2tensor


def _get_font(size):
    """跨平台字体加载：优先系统 TrueType，失败则使用默认字体。"""
    font_paths = [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in font_paths:
        try:
            if os.path.isabs(path):
                if os.path.isfile(path):
                    return ImageFont.truetype(path, size)
            else:
                return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


class FluxResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        megapixel_options = [f"{i / 10:.1f}" for i in range(1, 26)]  # 0.1 to 2.5

        return {
            "required": {
                "megapixel": (megapixel_options, {"default": "1.0"}),
                "aspect_ratio": (
                    [
                        "1:1 (Perfect Square)",
                        "2:3 (Classic Portrait)",
                        "3:4 (Golden Ratio)",
                        "3:5 (Elegant Vertical)",
                        "4:5 (Artistic Frame)",
                        "5:7 (Balanced Portrait)",
                        "5:8 (Tall Portrait)",
                        "7:9 (Modern Portrait)",
                        "9:16 (Slim Vertical)",
                        "9:19 (Tall Slim)",
                        "9:21 (Ultra Tall)",
                        "9:32 (Skyline)",
                        "3:2 (Golden Landscape)",
                        "4:3 (Classic Landscape)",
                        "5:3 (Wide Horizon)",
                        "5:4 (Balanced Frame)",
                        "7:5 (Elegant Landscape)",
                        "8:5 (Cinematic View)",
                        "9:7 (Artful Horizon)",
                        "16:9 (Panorama)",
                        "19:9 (Cinematic Ultrawide)",
                        "21:9 (Epic Ultrawide)",
                        "32:9 (Extreme Ultrawide)",
                    ],
                    {"default": "1:1 (Perfect Square)"},
                ),
                "divisible_by": (["8", "16", "32", "64"], {"default": "64"}),
                "custom_ratio": (
                    "BOOLEAN",
                    {"default": False, "label_on": "Enable", "label_off": "Disable"},
                ),
            },
            "optional": {
                "custom_aspect_ratio": ("STRING", {"default": "1:1"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("width", "height", "resolution", "preview")
    FUNCTION = "calculate_dimensions"
    CATEGORY = "sfnodes/Flux"
    OUTPUT_NODE = True

    def create_preview_image(self, width, height, resolution, ratio_display):
        preview_size = (1024, 1024)
        image = Image.new("RGB", preview_size, (0, 0, 0))
        draw = ImageDraw.Draw(image)

        grid_color = "#333333"
        grid_spacing = 50
        for x in range(0, preview_size[0], grid_spacing):
            draw.line([(x, 0), (x, preview_size[1])], fill=grid_color)
        for y in range(0, preview_size[1], grid_spacing):
            draw.line([(0, y), (preview_size[0], y)], fill=grid_color)

        preview_width = 800
        preview_height = int(preview_width * (height / width))

        if preview_height > 800:
            preview_height = 800
            preview_width = int(preview_height * (width / height))

        x_offset = (preview_size[0] - preview_width) // 2
        y_offset = (preview_size[1] - preview_height) // 2

        draw.rectangle(
            [
                (x_offset, y_offset),
                (x_offset + preview_width, y_offset + preview_height),
            ],
            outline="red",
            width=4,
        )

        font_large = _get_font(48)
        font_medium = _get_font(36)
        font_small = _get_font(32)

        text_y = y_offset + preview_height // 2

        draw.text(
            (preview_size[0] // 2, text_y),
            f"{width}x{height}",
            fill="red",
            anchor="mm",
            font=font_large,
        )

        draw.text(
            (preview_size[0] // 2, text_y + 60),
            f"({ratio_display})",
            fill="red",
            anchor="mm",
            font=font_medium,
        )

        draw.text(
            (preview_size[0] // 2, y_offset + preview_height + 60),
            f"Resolution: {resolution}",
            fill="white",
            anchor="mm",
            font=font_small,
        )

        return pil2tensor(image)

    def calculate_dimensions(
        self,
        megapixel,
        aspect_ratio,
        divisible_by,
        custom_ratio,
        custom_aspect_ratio=None,
    ):
        megapixel = float(megapixel)
        round_to = int(divisible_by)

        if custom_ratio and custom_aspect_ratio:
            numeric_ratio = custom_aspect_ratio
            ratio_display = custom_aspect_ratio
        else:
            numeric_ratio = aspect_ratio.split(" ")[0]
            ratio_display = numeric_ratio

        width_ratio, height_ratio = map(int, numeric_ratio.split(":"))

        total_pixels = megapixel * 1_000_000
        dimension = (total_pixels / (width_ratio * height_ratio)) ** 0.5
        width = int(dimension * width_ratio)
        height = int(dimension * height_ratio)

        width = round(width / round_to) * round_to
        height = round(height / round_to) * round_to

        resolution = f"{width} x {height}"

        preview = self.create_preview_image(width, height, resolution, ratio_display)

        return width, height, resolution, preview
