import math

import torch
from PIL import Image

from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import (
    WEIGHT_TYPES,
    IPAdapterAdvanced,
    ipadapter_execute,
)
from custom_nodes.ComfyUI_IPAdapter_plus.utils import contrast_adaptive_sharpening

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

_CATEGORY = "sfnodes/model"


def tile_image(image, attn_mask, sharpening, tile_size=256):
    _, oh, ow, _ = image.shape
    if attn_mask is None:
        attn_mask = torch.ones([1, oh, ow], dtype=image.dtype, device=image.device)
    img = image.permute([0, 3, 1, 2])
    mask = attn_mask.unsqueeze(1)
    mask = T.Resize(
        (oh, ow), interpolation=T.InterpolationMode.BICUBIC, antialias=True
    )(mask)
    if oh / ow > 0.75 and oh / ow < 1.33:
        img = T.CenterCrop(min(oh, ow))(img)
        resize = (tile_size * 2, tile_size * 2)
        mask = T.CenterCrop(min(oh, ow))(mask)
    else:
        resize = (
            (int(tile_size * ow / oh), tile_size)
            if oh < ow
            else (tile_size, int(tile_size * oh / ow))
        )
    imgs = []
    for im in img:
        im = T.ToPILImage()(im)
        im = im.resize(resize, resample=Image.Resampling["LANCZOS"])
        imgs.append(T.ToTensor()(im))
    img = torch.stack(imgs)
    del imgs, im
    mask = T.Resize(
        resize[::-1], interpolation=T.InterpolationMode.BICUBIC, antialias=True
    )(mask)
    if oh / ow > 4 or oh / ow < 0.25:
        crop = (tile_size, tile_size * 4) if oh < ow else (tile_size * 4, tile_size)
        img = T.CenterCrop(crop)(img)
        mask = T.CenterCrop(crop)(mask)
    mask = mask.squeeze(1)
    if sharpening > 0:
        img = contrast_adaptive_sharpening(img, sharpening)
    img = img.permute([0, 2, 3, 1])
    _, oh, ow, _ = img.shape
    tiles_x = math.ceil(ow / tile_size)
    tiles_y = math.ceil(oh / tile_size)
    overlap_x = max(0, (tiles_x * tile_size - ow) / (tiles_x - 1 if tiles_x > 1 else 1))
    overlap_y = max(0, (tiles_y * tile_size - oh) / (tiles_y - 1 if tiles_y > 1 else 1))
    base_mask = torch.zeros([mask.shape[0], oh, ow], dtype=img.dtype, device=img.device)
    tiles = []
    masks = []
    for y in range(tiles_y):
        for x in range(tiles_x):
            start_x = int(x * (tile_size - overlap_x))
            start_y = int(y * (tile_size - overlap_y))
            tiles.append(
                img[:, start_y : start_y + tile_size, start_x : start_x + tile_size, :]
            )
            m = base_mask.clone()
            m[:, start_y : start_y + tile_size, start_x : start_x + tile_size] = mask[
                :, start_y : start_y + tile_size, start_x : start_x + tile_size
            ]
            masks.append(m)
    del m
    return tiles, masks


class IPAdapterMSLayerWeights:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["SDXL", "SD15"],),
                "L0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L3_Composition": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01},
                ),
                "L4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L5": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L6_Style": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01},
                ),
                "L7": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L8": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L9": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L10": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L11": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L12": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L13": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L14": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
                "L15": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10, "step": 0.01}),
            }
        }

    INPUT_NAME = "layer_weights"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_weights",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "IPAdapter Mad Scientist Layer Weights"

    def execute(
        self,
        model_type,
        L0,
        L1,
        L2,
        L3_Composition,
        L4,
        L5,
        L6_Style,
        L7,
        L8,
        L9,
        L10,
        L11,
        L12,
        L13,
        L14,
        L15,
    ):
        if model_type == "SD15":
            return (
                f"0:{L0}, 1:{L1}, 2:{L2}, 3:{L3_Composition}, 4:{L4}, 5:{L5}, 6:{L6_Style}, 7:{L7}, 8:{L8}, 9:{L9}, 10:{L10}, 11:{L11},12:{L12},13:{L13},14:{L14},15:{L15}",
            )
        else:
            return (
                f"0:{L0}, 1:{L1}, 2:{L2}, 3:{L3_Composition}, 4:{L4}, 5:{L5}, 6:{L6_Style}, 7:{L7}, 8:{L8}, 9:{L9}, 10:{L10}, 11:{L11}",
            )


class IPAdapterMSTiled(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "image": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5, "step": 0.05},
                ),
                "weight_faceidv2": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05},
                ),
                "weight_type": (WEIGHT_TYPES,),
                "combine_embeds": (
                    ["concat", "add", "subtract", "average", "norm average"],
                ),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "embeds_scaling": (
                    ["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],
                ),
                "sharpening": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "layer_weights": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            },
        }

    CATEGORY = _CATEGORY
    DESCRIPTION = "IPAdapter Mad Scientist Tiled"

    RETURN_TYPES = (
        "MODEL",
        "IMAGE",
        "MASK",
        "IMAGE",
    )
    RETURN_NAMES = (
        "MODEL",
        "style_tiles",
        "masks",
        "composition_tiles",
    )

    def apply_ipadapter(
        self,
        model,
        ipadapter,
        image,
        weight,
        weight_faceidv2,
        weight_type,
        combine_embeds,
        start_at,
        end_at,
        embeds_scaling,
        layer_weights,
        sharpening,
        image_negative=None,
        attn_mask=None,
        clip_vision=None,
        insightface=None,
    ):
        # 1. Select the models
        if "ipadapter" in ipadapter:
            ipadapter_model = ipadapter["ipadapter"]["model"]
            clip_vision = (
                clip_vision
                if clip_vision is not None
                else ipadapter["clipvision"]["model"]
            )
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        # 2. Extract the tiles (调用tile_image)
        tile_size = 256
        tiles, masks = tile_image(image, attn_mask, sharpening, tile_size=tile_size)

        # 3. Apply the ipadapter to each group of tiles
        model = model.clone()
        for i in range(len(tiles)):
            ipa_args = {
                "image": tiles[i],
                "image_negative": image_negative,
                "weight": weight,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type,
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": masks[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface,
                "layer_weights": layer_weights,
            }
            model, _ = ipadapter_execute(
                model, ipadapter_model, clip_vision, **ipa_args
            )

        return (
            model,
            torch.cat(tiles),
            torch.cat(masks),
        )


class IPAdapterEmbedsMS:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "pos_embed": ("EMBEDS",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5, "step": 0.05},
                ),
                "weight_faceidv2": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05},
                ),
                "weight_type": (WEIGHT_TYPES,),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "embeds_scaling": (
                    ["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],
                ),
                "layer_weights": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = _CATEGORY

    def apply_ipadapter(
        self,
        model,
        ipadapter,
        pos_embed,
        weight,
        weight_faceidv2,
        weight_type,
        start_at,
        end_at,
        layer_weights,
        neg_embed=None,
        attn_mask=None,
        clip_vision=None,
        insightface=None,
        embeds_scaling="V only",
    ):
        ipa_args = {
            "pos_embed": pos_embed,
            "neg_embed": neg_embed,
            "weight": weight,
            "weight_faceidv2": weight_faceidv2,
            "weight_type": weight_type,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "embeds_scaling": embeds_scaling,
            "unfold_batch": self.unfold_batch,
            "layer_weights": layer_weights,
            "insightface": insightface,
        }

        if "ipadapter" in ipadapter:
            ipadapter_model = ipadapter["ipadapter"]["model"]
            clip_vision = (
                clip_vision
                if clip_vision is not None
                else ipadapter["clipvision"]["model"]
            )
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        # if clip_vision is None and neg_embed is None:
        #     raise Exception("Missing CLIPVision model.")

        del ipadapter

        return ipadapter_execute(
            model.clone(), ipadapter_model, clip_vision, **ipa_args
        )


class IPAdapterEmbedsMSBatch(IPAdapterEmbedsMS):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "pos_embed": ("EMBEDS",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5, "step": 0.05},
                ),
                "weight_faceidv2": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05},
                ),
                "weight_type": (WEIGHT_TYPES,),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "embeds_scaling": (
                    ["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],
                ),
                "layer_weights": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            },
        }


class IPAdapterStyleCompositionTiled(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("IPADAPTER",),
                "image_style": ("IMAGE",),
                "image_composition": ("IMAGE",),
                "weight_style": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5, "step": 0.05},
                ),
                "weight_composition": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 5, "step": 0.05},
                ),
                "expand_style": ("BOOLEAN", {"default": False}),
                "combine_embeds": (
                    ["concat", "add", "subtract", "average", "norm average"],
                    {"default": "average"},
                ),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "embeds_scaling": (
                    ["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],
                ),
                "sharpening": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            },
        }

    CATEGORY = _CATEGORY
    DESCRIPTION = "IPAdapter Style & Composition Tiled"

    RETURN_TYPES = (
        "MODEL",
        "IMAGE",  # style_tiles
        "IMAGE",  # composition_tiles
        "MASK",  # masks
    )
    RETURN_NAMES = (
        "MODEL",
        "style_tiles",
        "composition_tiles",
        "masks",
    )

    def apply_ipadapter(
        self,
        model,
        ipadapter,
        image_style,
        image_composition,
        weight_style,
        weight_composition,
        expand_style,
        combine_embeds,
        start_at,
        end_at,
        embeds_scaling,
        sharpening,
        image_negative=None,
        attn_mask=None,
        clip_vision=None,
        encode_batch_size=0,
    ):
        # 分别对style和composition做tile
        style_tiles, style_masks = tile_image(image_style, attn_mask, sharpening)
        comp_tiles, comp_masks = tile_image(image_composition, attn_mask, sharpening)
        all_masks = [
            torch.logical_or(sm.bool(), cm.bool()).float()
            for sm, cm in zip(style_masks, comp_masks)
        ]
        work_model = model.clone()
        # 先应用style tiles
        for i in range(len(style_tiles)):
            ipa_args = {
                "image": style_tiles[i],
                "image_negative": image_negative,
                "weight": weight_style,
                "weight_type": "style transfer"
                if not expand_style
                else "strong style transfer",
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": all_masks[i],
                "unfold_batch": False,
                "embeds_scaling": embeds_scaling,
            }
            if "ipadapter" in ipadapter:
                ipadapter_model = ipadapter["ipadapter"]["model"]
                clip_vision_model = (
                    clip_vision
                    if clip_vision is not None
                    else ipadapter["clipvision"]["model"]
                )
            else:
                ipadapter_model = ipadapter
                clip_vision_model = clip_vision
            work_model, _ = ipadapter_execute(
                work_model, ipadapter_model, clip_vision_model, **ipa_args
            )
        # 再应用composition tiles
        for i in range(len(comp_tiles)):
            ipa_args = {
                "image": comp_tiles[i],
                "image_negative": image_negative,
                "weight": weight_composition,
                "weight_type": "composition",
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": all_masks[i],
                "unfold_batch": False,
                "embeds_scaling": embeds_scaling,
                "encode_batch_size": encode_batch_size,
            }
            if "ipadapter" in ipadapter:
                ipadapter_model = ipadapter["ipadapter"]["model"]
                clip_vision_model = (
                    clip_vision
                    if clip_vision is not None
                    else ipadapter["clipvision"]["model"]
                )
            else:
                ipadapter_model = ipadapter
                clip_vision_model = clip_vision
            work_model, _ = ipadapter_execute(
                work_model, ipadapter_model, clip_vision_model, **ipa_args
            )
        return (
            work_model,
            torch.cat(style_tiles),
            torch.cat(comp_tiles),
            torch.cat(all_masks),
        )
