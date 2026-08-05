import torch
from PIL import Image

from ...sf_utils.blend import BLEND_MODES, chop_image
from ...sf_utils.image_convert import mask2pil, pil2tensor, tensor2pil
from ...sf_utils.logger import get_logger

_CATEGORY = "sfnodes/image"

logger = get_logger(__name__)


class SFImageBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (BLEND_MODES,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "layer_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "将图层图像按选定的混合模式与背景图像合成，支持透明度与遮罩控制；"
        "layer 为 RGBA 时自动取其 Alpha 通道作为遮罩，提供 layer_mask 时优先使用并支持反转"
    )

    def execute(
        self,
        background_image,
        layer_image,
        invert_mask,
        blend_mode,
        opacity,
        layer_mask=None,
    ):
        b_images = [torch.unsqueeze(b, 0) for b in background_image]
        l_images = [torch.unsqueeze(l, 0) for l in layer_image]

        l_masks = []
        for l in layer_image:
            m = tensor2pil(l)
            if m.mode == "RGBA":
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new("L", m.size, "white"))
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(mask2pil(m))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        ret_images = []
        for i in range(max_batch):
            background = b_images[i] if i < len(b_images) else b_images[-1]
            layer = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            _canvas = tensor2pil(background).convert("RGB")
            _layer = tensor2pil(layer).convert("RGB")

            if _mask.size != _layer.size:
                _mask = Image.new("L", _layer.size, "white")
                logger.warning("SFImageBlend mask mismatch, dropped!")

            _comp = chop_image(_canvas, _layer, blend_mode, opacity)
            _canvas.paste(_comp, mask=_mask)

            ret_images.append(pil2tensor(_canvas))

        return (torch.cat(ret_images, dim=0),)
