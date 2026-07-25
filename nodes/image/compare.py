# ==========================================================================
# SF Image Compare
# ==========================================================================


import torch
import numpy as np
from PIL import Image
from server import PromptServer  # type: ignore
from io import BytesIO
import base64

# Main class --------------


_CATEGORY = "sfnodes/image"


class ImageCompare:
    DESCRIPTION = "通过滑块和混合模式对比两张图片，输出差异遮罩"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
            },
            "optional": {
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "image_a",
        "diff_mask",
    )
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    def execute(
        self,
        image_a,
        image_b=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        if unique_id is not None:
            img_a_b64, img_b_b64 = None, None

            if image_a is not None:
                img_a_pil = Image.fromarray(
                    np.clip(255.0 * image_a[0].cpu().numpy(), 0, 255).astype(np.uint8)
                )
                buffered_a = BytesIO()
                img_a_pil.save(buffered_a, format="PNG")
                img_a_b64 = base64.b64encode(buffered_a.getvalue()).decode("utf-8")

            if image_b is not None:
                img_b_pil = Image.fromarray(
                    np.clip(255.0 * image_b[0].cpu().numpy(), 0, 255).astype(np.uint8)
                )
                buffered_b = BytesIO()
                img_b_pil.save(buffered_b, format="PNG")
                img_b_b64 = base64.b64encode(buffered_b.getvalue()).decode("utf-8")

            client_id = PromptServer.instance.client_id
            PromptServer.instance.send_sync(
                "sfnodes.image_compare_preview",
                {
                    "node_id": unique_id,
                    "image_a_data": img_a_b64,
                    "image_b_data": img_b_b64,
                },
                client_id,
            )
            print(
                f"[SFImageCompare] Sent preview event for node {unique_id}, image_a={img_a_b64 is not None}, image_b={img_b_b64 is not None}, client_id={client_id}"
            )

        diff_mask = torch.zeros_like(image_a[:, :, :, 0])

        if image_b is not None and image_a.shape == image_b.shape:
            grayscale_a = (
                0.2126 * image_a[..., 0]
                + 0.7152 * image_a[..., 1]
                + 0.0722 * image_a[..., 2]
            )
            grayscale_b = (
                0.2126 * image_b[..., 0]
                + 0.7152 * image_b[..., 1]
                + 0.0722 * image_b[..., 2]
            )
            diff_mask = torch.abs(grayscale_a - grayscale_b)

        return (
            image_a,
            diff_mask,
        )
