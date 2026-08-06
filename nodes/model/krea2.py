"""Krea2（K2）模型视觉感知文本编码节点。

Krea2 的条件编码器是 12 层 Qwen3-VL-4B 的 tap（见 comfy/text_encoders/krea2.py）。
由于该文本编码器是视觉语言模型，可将参考图送入其视觉通路，使条件编码获得图像感知
能力，无需 VAE / reference-latent。Krea2 的 DiT（comfy/ldm/krea2/model.py）是纯
文生图模型，token 序列为 [text_tokens, noisy_image_patches]，没有 reference latent
的插槽，因此本节点刻意不提供 VAE 输入（接 VAE 也只会被静默丢弃）。

每张参考图可配一张可选遮罩：连接遮罩后，图片会在送入视觉编码器前裁剪到遮罩的
包围盒，VLM 只"看"被遮罩标记的区域。这是参考图遮罩，不是局部重绘 —— Krea2 没有
concat/inpaint 通路。

与 TextEncodeQwenImageEdit 的区别：
  * 即使连接了图片也强制使用 Krea2 的 descriptor 条件模板（核心 Qwen-Edit 节点
    在带图时会回退到 Qwen3-VL 的普通图片模板）；
  * 无 VAE 输入，且支持无界、自动增长的 imageN/maskN 插槽（由 web 端扩展实现）。
"""

import math
import re

import torch

import comfy.utils
from aiohttp import web

# 与模型自身模板保持一致；非 Krea2 版本的 ComfyUI 上回退为字面量副本。
try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
except Exception:  # 可移植性兜底
    KREA2_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )

# system_prompt 输入只保存系统消息文本，由节点包装进聊天模板。默认值（Krea2 训练时的
# 描述指令）从模板中提取，保证与 ComfyUI 内置版本同步。
_sys = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", KREA2_TEMPLATE, re.S)
KREA2_SYSTEM_DEFAULT = _sys.group(1) if _sys else (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)

# instruct/编辑风格指令（类似 TextEncodeQwenImageEditPlus）：填入 system_prompt 可让
# VLM 将用户文本与参考图融合，而不是只描述图片。对 Krea2 训练时的描述指令来说属于
# 分布外，实验性。
KREA2_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image, especially the characters' facial "
    "expressions, gender, hairstyle, hair color and ethnicity, as well as the color, shape, "
    "size, texture, objects and background. Then explain how the user's instruction should "
    "combine with or alter it, and generate a new image meeting the instruction while keeping "
    "the characters' expressions, gender, hairstyle, hair color, ethnicity and the reference "
    "consistent where appropriate:"
)

# Krea2SystemPrompt 的预设：键为 combo 显示名，值为系统指令文本。'none' = 自定义。
# 唯一数据源；前端 web/krea2_system_prompt.js 通过 GET /api/sfnodes/krea2_presets
# 获取，切换预设时自动填充 text widget。
KREA2_PRESETS = {
    "none": "",
    "default": KREA2_INSTRUCT_SYSTEM,
    "不描述人物相貌与身材": (
        "Describe the key features of the reference image, including each character's facial "
        "expressions, gender, hairstyle and hair color, as well as the color, shape, "
        "size, texture, objects and background. But DO NOT describe any character's facial "
        "features, appearance, facial contours or body shape and height. Generate a new image following the "
        "user's instruction, keeping the characters' expressions, gender, hairstyle "
        "and hair color consistent with the reference where appropriate, but without "
        "referencing their facial looks or physique."
    ),
    "不描述人物相貌与身材（保持姿势动作）": (
        "Describe the key features of the reference image, including each character's facial "
        "expressions, gender, hairstyle and hair color, poses and actions, as well as the color, "
        "shape, size, texture, objects and background. But DO NOT describe any character's facial "
        "features, appearance, facial contours or body shape and height. Generate a new image "
        "following the user's instruction, keeping the characters' expressions, gender, hairstyle, "
        "hair color, poses and actions consistent with the reference where appropriate, but "
        "without referencing their facial looks or physique."
    ),
    "不描述相貌身材和发型发色": (
        "Describe the key features of the reference image, including each character's facial "
        "expressions and gender, as well as the color, shape, size, texture, objects and "
        "background. But DO NOT describe any character's facial features, appearance, facial "
        "contours, body shape and height, hairstyle or hair color. Generate a new image "
        "following the user's instruction, keeping the characters' expressions and gender "
        "consistent with the reference where appropriate, but without referencing their "
        "facial looks, physique, hairstyle or hair color."
    ),
    "不描述相貌身材和发型发色（保持姿势动作）": (
        "Describe the key features of the reference image, including each character's facial "
        "expressions, gender, poses and actions, as well as the color, shape, size, texture, "
        "objects and background. But DO NOT describe any character's facial features, appearance, "
        "facial contours, body shape and height, hairstyle or hair color. Generate a new image "
        "following the user's instruction, keeping the characters' expressions, gender, poses and "
        "actions consistent with the reference where appropriate, but without referencing their "
        "facial looks, physique, hairstyle or hair color."
    ),
    "黑白漫画转真人": (
        "Describe the key features of the reference image, including the characters' facial "
        "expressions, gender, hairstyle and hair color, poses, clothing, objects, composition "
        "and background, but DO NOT describe any character's facial features, appearance, body "
        "shape or ethnicity, and DO NOT describe or preserve the "
        "black-and-white manga art style. The output MUST be a full-color realistic photograph "
        "with natural skin texture, realistic lighting, soft natural shadows and vibrant "
        "colors, rendered photographically with realistic detail; it must NOT be "
        "black-and-white or manga style in any way. Keep the characters' expressions, gender, "
        "hairstyle, hair color and the composition, poses and scene consistent with "
        "the reference where appropriate."
    ),
    "真人转黑白漫画": (
        "Describe the key features of the reference image, including the characters' facial "
        "expressions, gender, hairstyle and hair color, poses, clothing, objects, composition "
        "and background, but DO NOT describe any character's facial features, appearance, body "
        "shape or ethnicity, and DO NOT describe or preserve realistic "
        "photographic details such as skin texture, lighting and color. The output MUST be a "
        "black-and-white manga illustration with clean line art, screentone shading and high "
        "contrast, and must NOT look like a realistic photograph. Keep the characters' "
        "expressions, gender, hairstyle, hair color and the composition, poses and "
        "scene consistent with the reference where appropriate."
    ),
    "动画截图转真人": (
        "Describe the key features of the reference image, including the characters' facial "
        "expressions, gender, hairstyle and hair color, poses, clothing, objects, composition "
        "and background, but DO NOT describe any character's facial features, appearance, body "
        "shape or ethnicity, and DO NOT describe or preserve the anime art "
        "style. The output MUST be a full-color realistic photograph with natural skin texture, "
        "realistic lighting, shadows and vibrant colors. The output must NOT be anime, "
        "illustration, or cartoon style in any way. Keep the characters' expressions, gender, "
        "hairstyle, hair color and the composition, poses and scene consistent with "
        "the reference where appropriate."
    ),
    "真人转动漫截图": (
        "Describe the key features of the reference image, including the characters' facial "
        "expressions, gender, hairstyle and hair color, poses, clothing, objects, composition "
        "and background, but DO NOT describe any character's facial features, appearance, body "
        "shape or ethnicity, and DO NOT describe or preserve realistic "
        "photographic details such as skin texture, lighting and color. The output MUST be an "
        "anime-style illustration with clean line art and vibrant colors, and must NOT look "
        "like a realistic photograph. Keep the characters' expressions, gender, hairstyle, "
        "hair color and the composition, poses and scene consistent with the "
        "reference where appropriate."
    ),
    "任意图片转真人": (
        "Describe the key features of the reference image, including the characters' facial "
        "expressions, gender, hairstyle and hair color, poses, clothing, objects, composition "
        "and background, but DO NOT describe any character's facial features, appearance, body "
        "shape or ethnicity, and DO NOT describe or preserve any "
        "illustration, anime, manga, cartoon, painting, sketch or other non-photographic art "
        "style. The output MUST be a full-color realistic photograph with natural skin "
        "texture, realistic lighting, shadows and vibrant colors, rendered photographically, "
        "and must NOT retain any non-photographic art style. Keep the characters' expressions, "
        "gender, hairstyle, hair color and the composition, poses and scene "
        "consistent with the reference where appropriate."
    ),
    "Krea2 提示词扩展（官方规则）": (
        "You are an expert prompt engineer for text-to-image models. Expand the user's prompt "
        "into a highly effective image-generation prompt for Krea 2. Think step by step about "
        "the subject and mood, suitable visual styles and lighting, and composition and framing "
        "details, then output a single expanded prompt paragraph. Rules: 1) Faithfulness "
        "first: preserve all original subjects, actions, colors and spatial relationships; do "
        "not add new objects, props, characters or animals unless the user clearly implies "
        "them. 2) Use practical T2I structure: "
        "group subjects with their own attributes and actions, use grounded phrasing for poses, "
        "interactions and spatial layout. 3) Keep style planning internal; do not emit "
        "planning tags or wrappers. 4) If visible text is requested, specify the exact words "
        "wrapped in quotes. 5) Avoid over-specification: do not invent clothing, colors, "
        "materials or scene details unless supported. 6) Output one cohesive paragraph, no "
        "bullets, JSON or markdown. 7) If the prompt is already detailed, lightly polish "
        "rather than heavily expand. 8) Describe camera angle, shot size and perspective "
        "naturally (e.g. low-angle perspective, extreme close-up, high-angle wide perspective, "
        "over-the-shoulder framing) when they serve the image. 9) Preserve the user's stated "
        "medium: when the user explicitly requests a medium (e.g. photo, photograph, "
        "illustration, painting, sketch, 3D render), honor it and do not pivot to a different "
        "medium to avoid difficulty. 10) Respect the human form: "
        "assume clothing covers intimate anatomy."
    ),
}

_CATEGORY = "sfnodes/model"


class TextEncodeKrea2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Krea2 的 CLIP 模型。使用 CLIPLoader 加载，类型选择 krea2",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "文本提示词。支持多行和动态提示词（dynamicPrompts）语法",
                }),
            },
            "optional": {
                # system_prompt 放在图片插槽上方。
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "可选系统指令输入。连接文本节点可覆盖 VLM 组织参考图与提示词的方式；"
                               "不连接则使用 Krea2 训练时的描述指令（分布内，推荐）。如需让提示词"
                               "与图片互动，可改用 instruct 编辑风格指令（见 SFKrea2SystemPrompt "
                               "节点）。只需提供指令文本，节点会自动包装聊天模板",
                }),
                # image1/mask1 为起始插槽对；web 端扩展会在连接后自动追加 image2/mask2、……
                "image1": ("IMAGE", {
                    "tooltip": "参考图片，送入 Qwen3-VL 视觉通路，使条件编码感知图片内容。"
                               "连接后会自动出现新的 image2/mask2 插槽",
                }),
                "mask1": ("MASK", {
                    "tooltip": "可选遮罩。连接后图片会先裁剪到遮罩的包围盒，VLM 只看到遮罩区域"
                               "（参考图遮罩，不是局部重绘）",
                }),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "每张参考图送入 Qwen3-VL 视觉编码器前的最大尺寸（百万像素）。"
                               "超过此上限的参考图会被缩小；较小的（如紧密遮罩裁剪）保持原始"
                               "大小，不会被放大",
                }),
                "mask_padding": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02,
                    "tooltip": "裁剪前在遮罩四周保留的上下文比例（按图像尺寸每侧各加的比例）。"
                               "0 = 紧贴遮罩裁剪；0.1 ≈ 每侧 10% 边距。仅在连接了遮罩时生效",
                }),
                "vision_position": (["before prompt", "after prompt"], {
                    "default": "before prompt",
                    "tooltip": "用户回合中图片（视觉）token 相对文本的位置。'before prompt' = "
                               "图片在前（默认）；'after prompt' = 文本在前。无图片时无效果，实验性",
                }),
                "print_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "将最终组装好的 Qwen3-VL 提示词（系统指令 + 视觉占位符 + 文本）"
                               "打印到 ComfyUI 控制台，便于调试",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Krea2（K2）文本条件编码，支持视觉提示。参考图通过 Qwen3-VL 视觉通路送入；"
                   "每张参考图可选遮罩裁剪到遮罩区域。不使用 VAE（Krea2 无 reference-latent 通路）")

    @staticmethod
    def _collect_indexed(kwargs, prefix):
        """从 kwargs 中收集形如 prefix + 数字（如 image1、mask2）的输入，返回 {编号: 值}。"""
        pattern = re.compile(r"^{}(\d+)$".format(prefix))
        out = {}
        for key, value in kwargs.items():
            match = pattern.match(key)
            if match is not None and value is not None:
                out[int(match.group(1))] = value
        return out

    @staticmethod
    def _crop_to_mask(image, mask, padding=0.0):
        """将图片 (B,H,W,C) 裁剪到遮罩的包围盒，并按 `padding`（图像尺寸的比例）向四周扩展。
        遮罩为空/未连接时为无操作。"""
        if mask is None:
            return image

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:  # (B,1,H,W) 等 -> (B,H,W)
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])

        h, w = image.shape[1], image.shape[2]
        if mask.shape[-2:] != (h, w):
            resized = comfy.utils.common_upscale(mask.unsqueeze(1), w, h, "bilinear", "disabled")
            mask = resized[:, 0]

        presence = (mask > 0.5).any(dim=0)  # 合并 batch -> (H,W)
        if not bool(presence.any()):
            return image  # 无选中区域：保留整张图

        rows = torch.where(torch.any(presence, dim=1))[0]
        cols = torch.where(torch.any(presence, dim=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])

        if padding > 0.0:  # 向外扩展保留上下文，并限制在图像范围内
            pad_x = round(padding * w)
            pad_y = round(padding * h)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)

        return image[:, y0:y1 + 1, x0:x1 + 1, :]

    @classmethod
    def _prepare_vision(cls, kwargs, vision_megapixels, mask_padding):
        """裁剪+缩放每张已连接的参考图，并组装视觉 token 字符串。"""
        images = cls._collect_indexed(kwargs, "image")
        masks = cls._collect_indexed(kwargs, "mask")
        ordered = sorted(images.keys())

        images_vl = []
        image_prompt = ""
        total = int(vision_megapixels * 1024 * 1024)

        for slot, n in enumerate(ordered):
            image = cls._crop_to_mask(images[n], masks.get(n), padding=mask_padding)
            samples = image.movedim(-1, 1)
            # vision_megapixels 是上限而不是固定目标：只缩小过大的参考图，绝不放大
            # （否则紧密遮罩裁剪的小图会被放大）。
            scale_by = min(1.0, math.sqrt(total / (samples.shape[3] * samples.shape[2])))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])
            if len(ordered) > 1:
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(slot + 1)
            else:
                image_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
        return images_vl, image_prompt

    @staticmethod
    def _build_text(system_prompt, prompt, image_prompt, vision_position):
        """组装用户文本（含视觉 token）与聊天模板。"""
        system = system_prompt.strip() or KREA2_SYSTEM_DEFAULT
        template = ("<|im_start|>system\n" + system + "<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        text = (prompt + image_prompt) if vision_position == "after prompt" else (image_prompt + prompt)
        return text, template

    @staticmethod
    def _fp8_hint(exc, images_vl):
        """把晦涩的 FP8 视觉崩溃映射为可操作的错误信息；不匹配则返回 None。

        ComfyUI 的 Qwen3-VL 视觉塔（qwen35.py fast_pos_embed_interpolate）在叠加
        位置编码权重时未做类型转换，FP8 加载的文本编码器在图像路径上会崩溃。"""
        if images_vl and isinstance(exc, NotImplementedError) and "Float8" in str(exc):
            return RuntimeError(
                "Krea2: Qwen3-VL 文本编码器以 FP8 加载，ComfyUI 的视觉塔无法在图像路径上运行"
                "（'add_stub not implemented for Float8_e4m3fn'）。使用图片参考时，请通过 "
                "CLIPLoader（类型 krea2）加载 bf16/fp16 的 Qwen3-VL-4B 文本编码器"
                "（如 qwen3vl_4b 的 *bf16* 文件）。FP8 编码器仅支持纯文本模式。"
            )
        return None

    def encode(self, clip, prompt, vision_megapixels=1.0, mask_padding=0.0,
               system_prompt=KREA2_SYSTEM_DEFAULT, vision_position="before prompt",
               print_prompt=False, **kwargs):
        images_vl, image_prompt = self._prepare_vision(kwargs, vision_megapixels, mask_padding)
        text, template = self._build_text(system_prompt, prompt, image_prompt, vision_position)

        if print_prompt:
            print("\n========== Text Encode (Krea2) -> Qwen3-VL prompt ==========")
            print(template.replace("{}", text, 1))  # 字面替换：对大括号安全
            print("---- references: {} ----".format(len(images_vl)))
            print("===========================================================\n")

        tokens = clip.tokenize(text, images=images_vl, llama_template=template)
        try:
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        except NotImplementedError as exc:
            hint = self._fp8_hint(exc, images_vl)
            if hint is not None:
                raise hint from exc
            raise
        return (conditioning,)


class Krea2SystemPrompt:
    """预置系统提示词的文本节点。输出接入 TextEncodeKrea2 的 system_prompt 输入。
    支持预设下拉（风格转换、特征控制），选择后自动填充并可继续手动编辑。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(KREA2_PRESETS.keys()), {
                    "default": "default",
                    "tooltip": "预设系统指令：'none' = 自定义；'default' = 默认 instruct 编辑"
                               "风格指令；其他预设为风格转换或特征控制指令。选择预设会自动"
                               "填充下方文本，之后仍可手动编辑",
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": KREA2_INSTRUCT_SYSTEM,
                    "tooltip": "Krea2 VLM 的系统指令。选择预设会自动填充；'none' 时自由编辑。"
                               "默认（instruct 编辑风格指令）可使提示词与参考图融合，实验性、"
                               "分布外；粘贴普通的描述指令可回退到 Krea2 默认行为",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("预置 instruct 风格系统提示词的文本节点，输出接入 Text Encode (Krea2) "
                   "的 system_prompt 输入。支持预设下拉（风格转换、特征控制）与自定义")

    def run(self, preset, text):
        text = (text or "").strip()
        if not text:
            text = KREA2_PRESETS.get(preset, "")
        return (text,)


def _register_krea2_routes():
    """提供预设数据 API，供前端 web/krea2_system_prompt.js 获取（唯一数据源）。"""
    try:
        from server import PromptServer

        ins = getattr(PromptServer, "instance", None)
        if ins is None or not hasattr(ins, "routes"):
            return
        routes = ins.routes

        @routes.get("/api/sfnodes/krea2_presets")
        async def _krea2_presets(request: web.Request) -> web.Response:
            return web.json_response(KREA2_PRESETS)

        @routes.get("/api/sfnodes/interrogator_presets")
        async def _interrogator_presets(request: web.Request) -> web.Response:
            return web.json_response(INTERROGATOR_PRESETS)

    except Exception:
        pass


_register_krea2_routes()


# 图像反推的默认指令（用户提示词会替换模板占位符）。
INTERROGATOR_DEFAULT_PROMPT = (
    "Describe this image in detail, including the subjects, facial expressions, gender, "
    "hairstyle, hair color, ethnicity, clothing, objects, scene, art style, colors and "
    "composition. Provide the description as a usable image generation prompt."
)

# SFImageInterrogator 的反推指令预设：键为 combo 显示名，值为指令文本。唯一数据源；
# 前端 web/krea2_interrogator.js 通过 GET /api/sfnodes/interrogator_presets 获取。
# 相貌相关预设与 KREA2_PRESETS 语义对称：保留性别/表情/发型发色，仅排除相貌特征。
INTERROGATOR_PRESETS = {
    "default": INTERROGATOR_DEFAULT_PROMPT,
    "简单描述": (
        "Generate a detailed paragraph that combines the subject, actions, environment, "
        "lighting, and mood into 2-3 cohesive sentences. Focus on accurate visual details "
        "rather than speculation."
    ),
    "不描述人物相貌": (
        "Describe this image in detail, including each character's facial expressions, gender, "
        "hairstyle and hair color, poses, actions, clothing, objects, scene, art style, colors, "
        "lighting and composition. But DO NOT describe any character's facial features, "
        "appearance, facial contours or looks. Provide the description as a usable image "
        "generation prompt."
    ),
    "不描述人物相貌与身材": (
        "Describe this image in detail, including each character's facial expressions, gender, "
        "hairstyle and hair color, poses, actions, clothing, objects, scene, art style, colors, "
        "lighting and composition. But DO NOT describe any character's facial features, "
        "appearance, facial contours, body shape or height. Provide the description as a usable "
        "image generation prompt."
    ),
    "不描述相貌身材和发型发色": (
        "Describe this image in detail, including each character's facial expressions and gender, "
        "poses, actions, clothing, objects, scene, art style, colors, lighting and composition. "
        "But DO NOT describe any character's facial features, appearance, facial contours, body "
        "shape, height, hairstyle or hair color. Provide the description as a usable image "
        "generation prompt."
    ),
}


class SFImageInterrogator:
    """图像反推节点：用 Krea2 的 CLIP（Qwen3-VL-4B）将输入图片生成为描述文本。

    ComfyUI 官方即支持 Krea2 的 CLIP 做多模态生成（见 comfy/sd.py 中 Krea2 的注释
    "12-layer tap for conditioning + multimodal generate"）：先 tokenize 文本与图片，
    再 clip.generate() 采样生成 token，最后 clip.decode() 还原为字符串。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Krea2 的 CLIP 模型。使用 CLIPLoader 加载，类型选择 krea2",
                }),
                "image": ("IMAGE", {
                    "tooltip": "待反推的图片，由 Krea2 的 Qwen3-VL 视觉通路理解并生成描述文本",
                }),
                "preset": (list(INTERROGATOR_PRESETS.keys()), {
                    "default": "default",
                    "tooltip": "反推指令预设（含不描述人物相貌等特征控制指令）。选择后自动填充"
                               "下方文本，之后仍可手动编辑；留空文本时回退到所选预设",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": INTERROGATOR_DEFAULT_PROMPT,
                    "tooltip": "给 VLM 的指令文本，默认要求详细描述图片内容以便作为生成提示词使用，"
                               "可按需修改；留空时使用所选预设的指令",
                }),
                "max_length": ("INT", {
                    "default": 256, "min": 8, "max": 4096,
                    "tooltip": "生成文本的最大 token 数上限",
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否采样生成。关闭时使用贪心解码（确定性输出），适合追求稳定",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "采样温度。越高越随机，越低越保守",
                }),
                "top_k": ("INT", {
                    "default": 64, "min": 0, "max": 1000,
                    "tooltip": "仅从概率最高的 top_k 个 token 中采样。0 = 禁用",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "核采样：从累计概率不超过 top_p 的最小 token 集合中采样",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "重复惩罚。>1 抑制重复，<1 鼓励重复",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "随机种子。相同参数与种子可复现相同结果",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "可选系统指令输入。不连接则使用 Krea2 训练时的描述指令（默认模板）",
                }),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "图片送入视觉编码器前的最大尺寸（百万像素）。超过上限会缩小，"
                               "较小的保持原始大小，不会被放大",
                }),
                # 注意：user_prompt 必须保持在 optional 的最后一个 widget 位置——ComfyUI
                # 前端按 widget 数组索引恢复旧工作流的值（widgets_values 位置敏感），新增
                # widget 若插在中间会导致旧工作流值错位（如 vision_megapixels 的 1 落入
                # 本字段）。新增 widget 一律追加到末尾。
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "可选用户提示词（兼容 Impact Pack Interrogator 的 user_prompt）："
                               "以独立段落附加到指令文本末尾，可结合自己的诉求引导反推（如强调"
                               "保留特定内容）。留空则只使用指令文本",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "interrogate"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("图像反推：用 Krea2 的 CLIP（Qwen3-VL-4B）将输入图片生成为描述文本，"
                   "可接 CLIP Text Encode / Text Encode (Krea2) 作为提示词使用。"
                   "支持预设（含不描述人物相貌/身材等特征控制指令）")

    @staticmethod
    def _scale_image(image, megapixels):
        """将单张图片 (B,H,W,C) 缩放到 megapixels 上限（只缩小不放大），返回 [B,H,W,C] 列表。"""
        samples = image.movedim(-1, 1)
        total = int(megapixels * 1024 * 1024)
        scale_by = min(1.0, math.sqrt(total / (samples.shape[3] * samples.shape[2])))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        return [s.movedim(1, -1)[:, :, :, :3]]

    def interrogate(self, clip, image, preset, prompt, max_length, do_sample, temperature, top_k,
                    top_p, repetition_penalty, seed, user_prompt=None, system_prompt=None,
                    vision_megapixels=1.0):
        images_vl = self._scale_image(image, vision_megapixels)
        prompt = (prompt or "").strip() or INTERROGATOR_PRESETS.get(preset, INTERROGATOR_DEFAULT_PROMPT)
        user_prompt = (user_prompt or "").strip()
        if user_prompt:
            prompt = prompt + "\n" + user_prompt
        system = (system_prompt or "").strip() or KREA2_SYSTEM_DEFAULT
        template = ("<|im_start|>system\n" + system + "<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        # 关键：必须在文本流中插入视觉占位符，tokenize 才会把图片嵌入 token 序列
        # （qwen3vl 仅在遇到 <|image_pad|> token 时替换为图片 embedding），否则
        # 模型生成时看不到图片，只会产生与图无关的幻觉描述。
        text = "<|vision_start|><|image_pad|><|vision_end|>" + prompt

        tokens = clip.tokenize(text, images=images_vl, llama_template=template)
        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        return (clip.decode(generated_ids),)
