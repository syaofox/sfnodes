# SF Qwen Edit 编码节点（复刻自 ComfyUI-EditUtils 的 QwenEditTextEncode_EditUtils + QwenEditOutputExtractor_EditUtils）
# 与原版差异：
#   - 裁剪 rope offsets（reference_rope_offsets 无 ComfyUI 核心消费端）
#   - 每图独立 ref_longest_edge / ref_crop / mask（替代原版共享参数）
#   - ref_resize_mode 仅 longest_edge（原包装默认）
# 纯逻辑在 sf_utils/qwen_edit.py

from ...sf_utils.common import AnyType
from ...sf_utils import qwen_edit as qwe

_CATEGORY = "sfnodes/model"

any_type = AnyType("*")

_REF_CROPS = ["pad", "center", "disabled"]
_UPSCALE_METHODS = ["lanczos", "bicubic", "area"]
_VL_CROPS = ["center", "disabled"]


class SFQwenEditTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in (1, 2, 3):
            optional["image%d" % i] = ("IMAGE",)
            optional["mask%d" % i] = ("MASK",)
            optional["ref_longest_edge%d" % i] = (
                "INT", {"default": 1024, "min": 64, "max": 4096, "step": 8,
                        "tooltip": "第 %d 张参考图的最长边（像素）" % i},
            )
            optional["ref_crop%d" % i] = (
                _REF_CROPS, {"default": "pad",
                             "tooltip": "第 %d 张参考图裁剪方式；pad 仅对主图输出 pad_info" % i},
            )
        optional["ref_upscale"] = (_UPSCALE_METHODS, {"default": "lanczos"})
        optional["vl_target_size"] = (
            "INT", {"default": 384, "min": 128, "max": 2048, "step": 8,
                    "tooltip": "视觉塔输入的目标面积边长"},
        )
        optional["vl_crop"] = (_VL_CROPS, {"default": "center"})
        optional["vl_upscale"] = (_UPSCALE_METHODS, {"default": "lanczos"})
        return {"required": {
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
        }, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "LATENT", any_type, "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Qwen Edit 编码：参考图 VAE 编码（pad 画布/裁剪 + 每图独立尺寸）+ 视觉塔输入 + 文本，输出 conditioning、主图初始 latent、noise_mask 与全量中间产物"

    def execute(self, clip, vae, prompt, ref_upscale="lanczos",
                vl_target_size=384, vl_crop="center", vl_upscale="lanczos", **kwargs):
        entries = []
        for i in (1, 2, 3):
            image = kwargs.get("image%d" % i)
            if image is None:
                continue
            mask = kwargs.get("mask%d" % i)
            if mask is not None and not qwe.mask_matches(mask, image):
                print("SFQwenEditTextEncode: mask%d H/W 与 image%d 不符，忽略该 mask" % (i, i))
                mask = None
            entries.append({
                "image": image,
                "mask": mask,
                "ref_longest_edge": kwargs.get("ref_longest_edge%d" % i, 1024),
                "ref_crop": kwargs.get("ref_crop%d" % i, "pad"),
            })

        if not entries:
            print("SFQwenEditTextEncode: 未提供任何图片，执行纯文本编码（latent 输出为占位值）")

        conditioning, latent_out, custom_output, main_image, noise_mask = qwe.encode_qwen_edit(
            clip, vae, prompt, entries,
            ref_upscale=ref_upscale,
            vl_target_size=vl_target_size,
            vl_crop=vl_crop,
            vl_upscale=vl_upscale,
        )
        return (conditioning, latent_out, custom_output, main_image, noise_mask)


class SFQwenEditOutputExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "custom_output": (any_type,),
        }}

    RETURN_TYPES = (any_type, "CONDITIONING", "IMAGE", any_type, any_type, any_type, "STRING", "CONDITIONING", "MASK")
    RETURN_NAMES = ("pad_info", "full_refs_cond", "main_image", "vae_images",
                    "ref_latents", "vl_images", "full_prompt", "no_refs_cond", "mask")
    FUNCTION = "extract"
    CATEGORY = _CATEGORY
    DESCRIPTION = "拆解 SF Qwen Edit Text Encode 的 custom_output 中间产物"

    def extract(self, custom_output):
        get = custom_output.get if hasattr(custom_output, "get") else (lambda k: None)
        return (
            get("pad_info"),
            get("full_refs_cond"),
            get("main_image"),
            get("vae_images"),
            get("ref_latents"),
            get("vl_images"),
            get("full_prompt"),
            get("no_refs_cond"),
            get("mask"),
        )
