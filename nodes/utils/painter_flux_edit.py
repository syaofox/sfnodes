# SF Painter Flux Image Edit 节点（复刻 ComfyUI-PainterFluxImageEdit 的 PainterFluxImageEdit）
# Flux2(Klein) 文生图 & 图生图编辑一体化编码：无图时纯文生图，接入图片自动切换为图编。
# 与原版差异：
#   - 去掉 mode 枚举，image1..imageN 由前端动态槽位按连接自动增删（web/sf_painter_flux_edit.js）
#   - 不再复制 conditioning 到 batch_size 份（采样器自动广播）；width/height step 对齐 Flux2 16x
#   - 新增 negative_prompt / use_reference_latent_as_init / reference_latents_method / encode_vision
# 纯逻辑在 sf_utils/flux_edit.py

from ...sf_utils.common import collect_indexed
from ...sf_utils import flux_edit as fe

_CATEGORY = "sfnodes/model"

_REF_LATENT_METHODS = ["", "offset", "index", "index_timestep_zero", "uxo"]


class SFPainterFluxImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Flux2 的文本编码器。使用 CLIPLoader 加载，类型选择 flux2",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "正向文本提示词。无参考图时为纯文生图；接入图片后作为编辑指令",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "负向提示词（Flux2 蒸馏版 CFG=1 时无影响）",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "输出批大小。通过采样器的 latent batch 广播实现，条件不复制",
                }),
                "width": ("INT", {
                    "default": 1024, "min": 512, "max": 4096, "step": 16,
                    "tooltip": "输出图片宽度（像素）。Flux2 VAE 为 16x 下采样，须为 16 的倍数"
                               "（非 16 倍数会被 VAE 静默中心裁剪）",
                }),
                "height": ("INT", {
                    "default": 1024, "min": 512, "max": 4096, "step": 16,
                    "tooltip": "输出图片高度（像素）。Flux2 VAE 为 16x 下采样，须为 16 的倍数"
                               "（非 16 倍数会被 VAE 静默中心裁剪）",
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "VAE 加载器（必接，否则执行报错）。用于编码参考图与空 latent",
                }),
                "reference_latents_method": (_REF_LATENT_METHODS, {
                    "default": "",
                    "tooltip": "reference_latents 注入方式（仅接参考图时生效）。'' = 用模型默认"
                               "（Flux2 为 index）；多参考图可尝试 uxo",
                }),
                "use_reference_latent_as_init": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "开：有参考图时以首图 latent 为去噪起点（原版行为，denoise<1 时是"
                               " img2img）；关：始终从空 latent 开始",
                }),
                "encode_vision": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开：额外计算 VL 缩放并插入 vision token 前缀（原版行为）。当前 "
                               "ComfyUI 的 Flux2 Klein 文本编码器忽略视觉输入，默认关闭以省算力；"
                               "若运行环境支持视觉通路再开",
                }),
                "image1_mask": ("MASK", {
                    "tooltip": "可选遮罩，仅作用于第一张参考图，实现精确区域重绘。"
                               "内部会缩放到 latent 空间作为 noise_mask",
                }),
                "image1": ("IMAGE", {
                    "tooltip": "第一张参考图。连接后自动出现 image2/image3……至多 10 张。"
                               "无任何参考图时为纯文生图",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    OUTPUT_TOOLTIPS = (
        "正向条件：文本 + 参考图 reference_latents。",
        "负向条件（negative_prompt + 参考图 reference_latents）。",
        "初始 latent：默认有参考图时为对齐到 batch 的首图参考 latent，否则为按输出尺寸编码的空 latent。",
    )
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Flux2(Klein) 文生图 & 图片编辑一体化编码（复刻 PainterFluxImageEdit）："
                   "把 CLIP 文本编码、参考图 VAE 编码与 reference_latents 注入合并为"
                   " positive/negative/latent 三路，直连 Flux2 采样器。无参考图时为纯文生图；"
                   "接入图片自动切换为图编，最多 10 张（前端动态槽位），首图支持遮罩；"
                   "支持 negative_prompt、reference_latents_method、起点与 VL 开关")

    def encode(self, clip, prompt, negative_prompt="", batch_size=1, width=1024, height=1024,
               vae=None, reference_latents_method="", use_reference_latent_as_init=True,
               encode_vision=False, image1_mask=None, **kwargs):
        images = collect_indexed(kwargs, "image")
        ordered = [images[n] for n in sorted(images.keys())]

        positive, negative, latent = fe.encode_painter_flux(
            clip, vae, prompt, ordered, mask=image1_mask,
            negative_prompt=negative_prompt, width=width, height=height, batch_size=batch_size,
            reference_latents_method=reference_latents_method,
            use_reference_latent_as_init=use_reference_latent_as_init,
            encode_vision=encode_vision,
        )
        return (positive, negative, latent)
