"""SFImageInterrogatorAPI - 通过大模型 API 对图片反推描述/提示词。

与本地版 SFImageInterrogator（Krea2 CLIP / Qwen3-VL）不同，本节点把单张图片编码为
base64 data URL，调用 OpenAI 兼容的视觉 LLM（默认 DeepSeek `deepseek-flash`，
原生多模态），把返回文本作为提示词输出。适合不想加载本地 VLM 的场景。

API key / base_url / model 来自 ComfyUI Settings（`sfnodes.LLM.*`，见 web/sf_llm_settings.js），
后端经 sf_utils/llm_client.py 读服务器上的 comfy.settings.json，节点内不接触密钥。

反推预设（内置 + 用户覆盖 + 墓碑删除）与本地版共用同一预设库
（sf_utils/krea2_presets.py，kind="interrogator"），节点上「⚙ 管理预设」由前端挂载。

只处理单帧（frame_index，-1 取末帧）：一次请求一张图，成本/延迟可控。请求失败或
结果为空抛异常（工作流红色报错，UI 显示原因）。
"""

import numpy as np

# 顶层包导入时 `...` 正常；测试以 `nodes.model.image_interrogator_api` 顶层导入时
# `...` 越界，回退绝对导入（krea2.py 同款可移植性兜底）。
try:
    from ...sf_utils.llm_client import (
        build_image_content,
        chat_completion_sync,
        get_llm_config,
        image_to_data_url,
    )
except Exception:  # pragma: no cover - 测试/移植性兜底
    from sf_utils.llm_client import (  # type: ignore
        build_image_content,
        chat_completion_sync,
        get_llm_config,
        image_to_data_url,
    )

# 反推预设库与本地版共用（内置 + 用户覆盖，见 sf_utils/krea2_presets.py）
from .krea2 import INTERROGATOR_DEFAULT_PROMPT, INTERROGATOR_PRESETS, _merged_presets

_CATEGORY = "sfnodes/model"

_DEFAULT_SYSTEM_PROMPT = (
    "You are a professional image description assistant. Follow the user's "
    "instruction precisely and output only the requested description."
)


class SFImageInterrogatorAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "待反推的图片（IMAGE batch）。只取 frame_index 指定的一帧编码后发送给视觉 LLM",
                }),
                "preset": (list(INTERROGATOR_PRESETS.keys()), {
                    "default": "default",
                    "tooltip": "反推指令预设（内置 + 用户自定义，与本地版 SFImage Interrogator 共用）。"
                               "选择后自动覆盖 prompt 文本；prompt 留空时执行时回退到该预设。"
                               "可通过节点内「⚙ 管理预设」新增/修改/删除/复位",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": INTERROGATOR_DEFAULT_PROMPT,
                    "tooltip": "主指令文本（发给视觉 LLM 的任务描述）。留空时回退到所选预设的默认指令；"
                               "会与 user_prompt 拼接后发送",
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "可选附加约束，以独立段落追加到主指令末尾（如“保留服装细节/不描述背景”）；留空则仅用主指令",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "采样温度。越低越稳定确定（反推建议 0.2~0.5）；越高越发散",
                }),
                "max_tokens": ("INT", {
                    "default": 512, "min": 8, "max": 8192,
                    "tooltip": "返回文本的最大 token 数（DeepSeek thinking 已关闭，不会占用推理预算）",
                }),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "发送前图片缩放上限（百万像素，按面积等比，只缩小不放大）。控制请求体积/成本",
                }),
                "detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "视觉细节级别（OpenAI 兼容字段）：low 降采样更省，high 保留原分辨率细节，auto 由接口决定",
                }),
                "frame_index": ("INT", {
                    "default": 0, "min": -1, "max": 0xffffffff,
                    "tooltip": "IMAGE batch 中要反推的帧序号（0 起；-1 = 末帧）。越界报错",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "可选系统指令，覆盖内置默认系统提示词；不连接则用内置描述助手指令",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_TOOLTIPS = ("视觉 LLM 返回的描述文本，可直接作为提示词使用。",)
    FUNCTION = "interrogate"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "图像反推（LLM API 版）：把输入图片编码为 base64 发给 OpenAI 兼容的视觉大模型"
        "（默认 DeepSeek deepseek-flash），返回描述文本作为提示词。与本地版 SF Image "
        "Interrogator 共用反推预设库。API key/模型在 设置 → SF LLM 配置；只处理单帧"
        "（frame_index），失败报错。"
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # combo 选项由前端按用户预设动态重建，值可能超出 INPUT_TYPES 静态列表。
        return True

    @staticmethod
    def _frame_to_pil(image, index):
        """取 [B,H,W,C] 中指定帧为 PIL RGB（alpha 按黑底预乘，与本地版一致）。"""
        from PIL import Image

        total = int(image.shape[0])
        idx = index if index >= 0 else total + index
        if idx < 0 or idx >= total:
            raise ValueError(f"frame_index 越界：{index}（batch 帧数 {total}）")
        frame = image[idx]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame)
        arr = np.clip(arr.astype("float32"), 0.0, 1.0)
        if arr.ndim == 3 and arr.shape[-1] >= 4:
            # 透明区按黑底预乘，避免残留 RGB 被视觉模型当作真实颜色
            arr = arr[..., :3] * arr[..., 3:4]
        elif arr.ndim == 3 and arr.shape[-1] != 3:
            arr = arr[..., :3]
        return Image.fromarray((arr * 255.0 + 0.5).astype("uint8"), "RGB")

    @classmethod
    def _encode_frame(cls, image, index, megapixels):
        """单帧 -> base64 data URL（默认 JPEG）。测试可打桩。"""
        return image_to_data_url(cls._frame_to_pil(image, index), max_megapixels=megapixels)

    def interrogate(self, image, preset, prompt, user_prompt, temperature, max_tokens,
                    vision_megapixels, detail, frame_index, system_prompt=None):
        instruction = (prompt or "").strip() or _merged_presets(
            "interrogator", INTERROGATOR_PRESETS).get(preset, INTERROGATOR_DEFAULT_PROMPT)
        extra = (user_prompt or "").strip()
        if extra:
            instruction = instruction + "\n" + extra
        system = (system_prompt or "").strip() or _DEFAULT_SYSTEM_PROMPT

        data_url = self._encode_frame(image, frame_index, vision_megapixels)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": build_image_content(instruction, data_url, detail)},
        ]
        text = chat_completion_sync(
            get_llm_config(), messages, temperature=temperature, max_tokens=max_tokens,
        )
        return (text,)
