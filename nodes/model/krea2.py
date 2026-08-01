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
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
)

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
    """预置 instruct/编辑风格系统提示词的文本节点。输出接入 TextEncodeKrea2 的
    system_prompt 输入，让提示词与参考图融合（实验性 / 分布外）。文本可自由编辑。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": KREA2_INSTRUCT_SYSTEM,
                    "tooltip": "Krea2 VLM 的系统指令。默认为 instruct 编辑风格指令，可使提示词"
                               "与参考图融合（实验性，分布外）。可按需编辑；粘贴普通的描述指令"
                               "可回退到默认行为",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("预置 instruct 风格系统提示词的文本节点，输出接入 Text Encode (Krea2) "
                   "的 system_prompt 输入")

    def run(self, text):
        return (text,)
