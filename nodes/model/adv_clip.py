from ...sf_utils.adv_encode import advanced_encode, advanced_encode_XL
from nodes import MAX_RESOLUTION

_CATEGORY = "sfnodes/model"


class AdvancedCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (
                    ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                ),
                # "affect_pooled": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = "高级 CLIP 文本编码，支持多种权重解析和归一化策略"

    def encode(
        self,
        clip,
        text,
        token_normalization,
        weight_interpretation,
        affect_pooled="disable",
    ):
        embeddings_final, pooled = advanced_encode(
            clip,
            text,
            token_normalization,
            weight_interpretation,
            w_max=1.0,
            apply_to_pooled=affect_pooled == "enable",
        )
        return ([[embeddings_final, {"pooled_output": pooled}]],)


class AddCLIPSDXLParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "target_width": (
                    "INT",
                    {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION},
                ),
                "target_height": (
                    "INT",
                    {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = "为 SDXL Conditioning 添加宽高、裁剪和目标尺寸参数"

    def encode(
        self, conditioning, width, height, crop_w, crop_h, target_width, target_height
    ):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["width"] = width
            n[1]["height"] = height
            n[1]["crop_w"] = crop_w
            n[1]["crop_h"] = crop_h
            n[1]["target_width"] = target_width
            n[1]["target_height"] = target_height
            c.append(n)
        return (c,)


class AddCLIPSDXLRParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                "ascore": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = "为 SDXL Refiner Conditioning 添加宽高和美学评分参数"

    def encode(self, conditioning, width, height, ascore):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["width"] = width
            n[1]["height"] = height
            n[1]["aesthetic_score"] = ascore
            c.append(n)
        return (c,)


class AdvancedCLIPTextEncodeSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_l": ("STRING", {"multiline": True}),
                "text_g": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (
                    ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                ),
                # "affect_pooled": (["disable", "enable"],),
                "balance": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = "高级 SDXL CLIP 文本编码，支持 text_l 和 text_g 双编码器"

    def encode(
        self,
        clip,
        text_l,
        text_g,
        token_normalization,
        weight_interpretation,
        balance,
        affect_pooled="disable",
    ):
        embeddings_final, pooled = advanced_encode_XL(
            clip,
            text_l,
            text_g,
            token_normalization,
            weight_interpretation,
            w_max=1.0,
            clip_balance=balance,
            apply_to_pooled=affect_pooled == "enable",
        )
        return ([[embeddings_final, {"pooled_output": pooled}]],)
