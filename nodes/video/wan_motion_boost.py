"""SFWanMotionBoost —— Wan2.2 图生视频慢动作增强（原生 WanImageToVideo 超集）。

针对 4-step 蒸馏 LoRA（如 lightx2v）常见的动作幅度偏小/慢动作问题：
在原生 `WanImageToVideo` 输出上，把 mask 标记的灰填充占位帧相对**最后
一个条件帧**的 latent 差异放大 `motion_amplitude` 倍（保均值，不整体
偏色），间接驱动更大的动作与运镜幅度。

实现要点（与第三方 PainterI2V / PainterI2Vadvanced 的差异见
doc/experience/nodes-video.md §105）：

- **零复制原生逻辑**：concat latent / mask / 多帧 `start_image` / 类别
  全部委托原生 `WanImageToVideo.execute`，本节点只在返回 conditioning 上
  做后处理——原生后续改进（多帧输入、inpaint 掩码等）自动继承。
- **多帧支持**：占位帧范围由原生 `concat_mask` 决定，基准取最后一个条件帧
  （单帧时与 PainterI2V 行为一致）。
- **色彩保护**：缩放保持每帧每通道均值；`latent_clamp` 截断后再精确恢复
  占位帧均值，消除截断造成的偏灰/偏绿（`color_protect=True` 默认开）。
- **类型校验**：可选接 `model`，非 Wan 系列（WAN21/WAN22 家族，含 GGUF
  量化经 ComfyUI-GGUF 加载）时告警并原样直通，避免误用。
- **不做 placebo**：不再注入 `reference_latents`——标准 Wan2.2 I2V
  checkpoint 无 `ref_conv`，该条件会被核心静默忽略（仅 SCAIL/Animate 等
  变体消费），第三方节点宣称的主体一致性增强对 I2V 无实际作用。
"""

import logging

from nodes import MAX_RESOLUTION

from ...sf_utils.common import node_result
from ...sf_utils.wan_motion_boost import boost_conditioning

_CATEGORY = "sfnodes/video"


def _is_wan_model(model):
    """MODEL 是否为 Wan 系列（WAN21 及其全部子类；无法判定时返回 True 不拦截）。"""
    try:
        import comfy.model_base

        inner = getattr(model, "model", None)
        return isinstance(inner, comfy.model_base.WAN21)
    except Exception:
        return True


class SFWanMotionBoost:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "正向条件（CLIP Text Encode）"}),
                "negative": ("CONDITIONING", {"tooltip": "负向条件（CLIP Text Encode）"}),
                "vae": ("VAE", {"tooltip": "Wan 视频 VAE（VAELoader）"}),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16,
                                  "tooltip": "输出宽度；与原版 WanImageToVideo 保持一致"}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16,
                                   "tooltip": "输出高度；与原版 WanImageToVideo 保持一致"}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4,
                                   "tooltip": "视频帧数（4n+1 效果最佳）"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096,
                                       "tooltip": "latent 批次，与原版一致"}),
                "motion_amplitude": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05,
                                               "tooltip": "运动幅度倍率：1.0=原生不加成；建议 1.1~1.2 起步，"
                                                          "快速动作 1.25~1.35；过高会出现不受控运镜/偏色"}),
                "color_protect": ("BOOLEAN", {"default": True,
                                              "tooltip": "色彩保护：latent_clamp 截断后精确恢复占位帧每通道均值，"
                                                         "抑制高幅度下的偏灰偏绿"}),
                "latent_clamp": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5,
                                           "tooltip": "缩放后 latent 绝对值上限（0=不限制）；默认 6.0 与原版 Wan latent 量级一致"}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "可选：仅用于类型校验。接非 Wan 系列模型时告警并跳过运动增强（原样直通）；GGUF 量化模型（ComfyUI-GGUF 的 GGUFModelPatcher）同样识别"}),
                "start_image": ("IMAGE", {"tooltip": "起始图（多帧输入时按原生语义取 start_image[:length]，前段为条件帧）；"
                                                     "不接则无 concat 可增强，等同原生透传"}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP Vision 编码（I2V 通常需要）"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = ("Wan2.2 图生视频慢动作增强：原生 WanImageToVideo 全功能 + motion_amplitude "
                   "占位帧运动放大（保均值、可 clamp 后色彩保护），缓解 4 步蒸馏 LoRA 动作幅度小的问题；"
                   "只改 conditioning 不改权重，GGUF 量化模型同样有效")

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.15, color_protect=True, latent_clamp=6.0,
                model=None, start_image=None, clip_vision_output=None):
        from comfy_extras.nodes_wan import WanImageToVideo

        if positive is None or negative is None:
            raise ValueError("SF Wan Motion Boost: positive / negative 条件不能为空")

        result = node_result(WanImageToVideo.execute(
            positive=positive,
            negative=negative,
            vae=vae,
            width=int(width),
            height=int(height),
            length=int(length),
            batch_size=int(batch_size),
            start_image=start_image,
            clip_vision_output=clip_vision_output,
        ))
        if len(result) != 3:
            raise RuntimeError(
                f"SF Wan Motion Boost: 原生 WanImageToVideo 返回 {len(result)} 个输出，与预期 3 个不符"
            )
        positive_out, negative_out, latent_out = result

        if start_image is None or float(motion_amplitude) <= 1.0:
            return (positive_out, negative_out, latent_out)

        if model is not None and not _is_wan_model(model):
            logging.warning(
                "[SFWanMotionBoost] 接入的 model 不是 Wan 系列（WAN21/WAN22 家族），"
                "已跳过运动增强并原样透传原生结果"
            )
            return (positive_out, negative_out, latent_out)

        boosted_positive = boost_conditioning(
            positive_out, float(motion_amplitude), float(latent_clamp), bool(color_protect))
        boosted_negative = boost_conditioning(
            negative_out, float(motion_amplitude), float(latent_clamp), bool(color_protect))
        return (boosted_positive, boosted_negative, latent_out)


__all__ = ["SFWanMotionBoost"]
