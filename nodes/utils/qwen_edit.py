# SF Qwen Edit 编码节点（复刻自 ComfyUI-EditUtils 的 QwenEditTextEncode_EditUtils + QwenEditOutputExtractor_EditUtils）
# 以及 Krea2 管线用的 QwenConfigPreparer / EditTextEncode（SFKrea2ConfigPreparer / SFKrea2EditTextEncode）。
# 与原版差异：
#   - 每图独立 ref_longest_edge / ref_crop / mask（替代原版共享参数）
#   - rope offsets 已支持：非零时写入 conditioning 的 reference_rope_offsets（SFKrea2EditApply 消费）
# 纯逻辑在 sf_utils/qwen_edit.py

import copy

from ...sf_utils.common import AnyType, collect_indexed, parse_json_dict
from ...sf_utils import krea2_edit as ke
from ...sf_utils import qwen_edit as qwe

_CATEGORY = "sfnodes/model"

any_type = AnyType("*")

_REF_CROPS = ["pad", "center", "disabled"]
_UPSCALE_METHODS = ["lanczos", "bicubic", "area"]
_VL_CROPS = ["center", "disabled"]
_RESIZE_MODES = ["longest_edge", "area"]


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


class SFEasyKrea2Edit:
    """Krea2 / Qwen 编辑单节点编码（Easy_QwenEdit2509 等价，纯 sfnodes 实现）。

    动态 imageN 参考图槽位（前端自动增删）；latent_image 缺省时首个参考图兼作初始 latent
    （Easy 语义：独立 latent_image 不额外进参考）；auto_resize 把参考图统一到目标尺寸
    （crop/pad/stretch）；逐图 strength 经隐藏状态传入；直接输出 positive/zero_negative/latent。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "参考图（连接后前端自动追加 image2、image3…）"}),
                "latent_image": ("IMAGE", {"tooltip": "生成尺寸与初始 latent 的来源；缺省用 image1 兼作"}),
                "latent_mask": ("MASK", {"tooltip": "局部重绘遮罩（作用于 latent_image；无 latent_image 时作用于主图）"}),
                "auto_resize": (["crop", "pad", "stretch"], {
                    "default": "crop",
                    "tooltip": "参考图统一到目标尺寸：crop=覆盖缩放后居中裁剪 / pad=完整缩放后居中黑边 / stretch=强制拉伸",
                }),
                "vl_size": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64,
                           "tooltip": "视觉塔输入的目标面积边长（vl_size² 像素，与原 Easy 一致）"}),
                "system_prompt": ("STRING", {"multiline": False, "default": ke.DEFAULT_QWEN_EDIT_INSTRUCTION,
                                 "tooltip": "系统指令；留空回退 Qwen 编辑默认模板，包装规则与 Easy 相同"}),
                "reference_latents_method": (["", "index_timestep_zero", "index", "offset", "uxo/uno"], {
                    "default": "index_timestep_zero",
                    "tooltip": "写入 conditioning 的参考方法；空 = 不写，交给外部节点（如 Edit Model Reference Method）",
                }),
            },
            "hidden": {"SFEasyKrea2EditState": ("STRING", {"default": "{}"})},
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "zero_negative", "latent")
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Easy 式 Krea2/Qwen 编辑编码：多参考图（动态槽位，可选逐图 strength）+ 目标尺寸统一 "
        "（auto_resize）+ 初始 latent/noise_mask + 零负条件，直接替代 Easy_QwenEdit2509。"
    )

    def encode(self, clip, vae, prompt, image1=None, latent_image=None, latent_mask=None,
               auto_resize="crop", vl_size=384, system_prompt=ke.DEFAULT_QWEN_EDIT_INSTRUCTION,
               reference_latents_method="index_timestep_zero",
               SFEasyKrea2EditState="{}", **kwargs):
        images = collect_indexed(kwargs, "image")
        if image1 is not None:
            images[1] = image1

        raw_strengths = parse_json_dict(SFEasyKrea2EditState).get("strengths") or {}
        entries = []
        for n in sorted(images):
            try:
                strength = float(raw_strengths.get(str(n), raw_strengths.get(n, 1.0)))
            except (TypeError, ValueError):
                strength = 1.0
            entries.append({"image": images[n], "ref_strength": strength})

        if latent_image is None and latent_mask is not None and entries:
            entries[0]["mask"] = latent_mask  # 无独立 latent_image：遮罩作用于首个参考图（主图）
            latent_mask = None

        conditioning, latent_out, _custom, _main_image, _mask = qwe.encode_qwen_edit(
            clip, vae, prompt, entries,
            llama_template=ke.get_system_prompt(system_prompt),
            ref_upscale="bicubic",
            vl_target_size=vl_size, vl_crop="disabled", vl_upscale="area",
            init_image=latent_image, init_mask=latent_mask,
            ref_target_mode=auto_resize,
        )
        if reference_latents_method:
            conditioning = qwe.set_conditioning_values(
                conditioning, {"reference_latents_method": reference_latents_method})
        return (conditioning, qwe.zero_conditioning(conditioning), latent_out)


class SFKrea2ConfigPreparer:
    """Krea2 管线图片配置聚合（复刻 EditUtils QwenConfigPreparer_EditUtils）。

    把当前 image + 各项参数打包成一个 config 追加到 configs 列表；可串联多个
    Preparer 组合多图（顺序即 Picture 编号顺序）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "configs": ("LIST", {"default": None, "tooltip": "已有 configs 链（串联多个 Preparer）"}),
                "to_ref": ("BOOLEAN", {"default": True, "tooltip": "加入参考 latent"}),
                "ref_main_image": ("BOOLEAN", {"default": True,
                                  "tooltip": "设为主图（其 latent 作为输出初始 latent）"}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1,
                                    "tooltip": "输出 latent 的最长边；ref_resize_mode=area 时为总面积边长"}),
                "ref_crop": (_REF_CROPS, {"default": "pad", "tooltip": "参考图裁剪方式（pad 仅主图输出 pad_info）"}),
                "ref_upscale": (_UPSCALE_METHODS, {"default": "lanczos", "tooltip": "参考图缩放算法"}),
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "加入 Qwen-VL 视觉编码"}),
                "vl_resize": ("BOOLEAN", {"default": True, "tooltip": "视觉编码前是否缩放"}),
                "vl_target_size": ("INT", {"default": 384, "min": 384, "max": 2048,
                                  "tooltip": "视觉编码目标面积边长"}),
                "vl_crop": (_VL_CROPS, {"default": "center", "tooltip": "视觉编码裁剪方式"}),
                "vl_upscale": (_UPSCALE_METHODS, {"default": "lanczos", "tooltip": "视觉编码缩放算法"}),
                "mask": ("MASK",),
                "ref_resize_mode": (_RESIZE_MODES, {"default": "longest_edge",
                                    "tooltip": "longest_edge=最长边对齐 ref_longest_edge；"
                                               "area=总面积对齐 ref_longest_edge²"}),
                "rope_x_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8,
                                 "tooltip": "参考位置 ROPE 水平偏移（像素，VAE 对齐 step=8）"}),
                "rope_y_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8,
                                 "tooltip": "参考位置 ROPE 垂直偏移（像素，VAE 对齐 step=8）"}),
            },
        }

    RETURN_TYPES = ("LIST", any_type)
    RETURN_NAMES = ("configs", "config")
    FUNCTION = "prepare_config"
    CATEGORY = _CATEGORY
    DESCRIPTION = "把当前图片与各项参数打包成 config 追加到 configs 列表，供 Krea2 Edit Text Encode 使用"

    def prepare_config(self, image, configs=None, to_ref=True, ref_main_image=True,
                       ref_longest_edge=1024, ref_crop="pad", ref_upscale="lanczos",
                       to_vl=True, vl_resize=True, vl_target_size=384, vl_crop="center",
                       vl_upscale="lanczos", mask=None, ref_resize_mode="longest_edge",
                       rope_x_offset=0, rope_y_offset=0):
        if configs is None:
            configs = []
        config = {
            "image": image,
            "to_ref": to_ref,
            "ref_main_image": ref_main_image,
            "ref_longest_edge": ref_longest_edge,
            "ref_crop": ref_crop,
            "ref_upscale": ref_upscale,
            "ref_resize_mode": ref_resize_mode,
            "to_vl": to_vl,
            "vl_resize": vl_resize,
            "vl_target_size": vl_target_size,
            "vl_crop": vl_crop,
            "vl_upscale": vl_upscale,
            "rope_x_offset": rope_x_offset,
            "rope_y_offset": rope_y_offset,
        }
        config_output = copy.deepcopy(configs)
        if mask is not None:
            if not qwe.mask_matches(mask, image):
                print("SFKrea2ConfigPreparer: mask H/W 与 image 不符，跳过该 mask")
            else:
                config["mask"] = mask
        config_output.append(config)
        return (config_output, config)


class SFKrea2EditTextEncode:
    """Krea2 管线文本+图像编码（复刻 EditUtils EditTextEncode_EditUtils 的 qwen 路径）。

    Krea2 文本编码器基于 Qwen、VAE 是 Qwen-Image 的 VAE，因此走 qwen 编码分支。
    输出 conditioning（含 reference_latents，供 SF Krea2 Edit Apply 消费）、主图初始
    latent、noise_mask、主图与 pad_info。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "model_config": ("DICT", {"default": None}),
            },
            "optional": {
                "configs": ("LIST", {"default": None,
                            "tooltip": "图片配置列表；未提供时执行纯文本编码"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", any_type, "IMAGE", "MASK", "ANY")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask", "pad_info")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Krea2 文本+图像编码：configs 参考图经 VAE/视觉塔处理，输出 conditioning、"
                   "主图初始 latent、mask 与 pad_info")

    def execute(self, clip, vae, prompt, model_config=None, configs=None):
        if not isinstance(model_config, dict):
            raise ValueError(
                "SFKrea2EditTextEncode: model_config 未连接或非法，请接 SF Krea2 Model Config"
            )
        model_name = model_config.get("model_name")
        if model_name not in (None, "", "qwen"):
            raise ValueError(
                f"SFKrea2EditTextEncode: 仅支持 Krea2/qwen 编码路径，收到 model_name={model_name!r}"
            )
        vae_unit = model_config.get("vae_unit", qwe.VQE_UNIT)
        llama_template = model_config.get("llama_template", "")

        entries = []
        for cfg in (configs or []):
            if not isinstance(cfg, dict) or "image" not in cfg:
                continue
            entries.append(cfg)
        if not entries:
            print("SFKrea2EditTextEncode: 未提供图片配置，执行纯文本编码")

        conditioning, latent_out, custom_output, main_image, noise_mask = qwe.encode_qwen_edit(
            clip, vae, prompt, entries,
            llama_template=llama_template or qwe.DEFAULT_LLAMA_TEMPLATE,
            vae_unit=vae_unit,
        )
        return (conditioning, latent_out, custom_output, main_image, noise_mask,
                custom_output.get("pad_info"))
