"""Krea2 参考图编辑节点（复刻 ComfyUI-EditUtils 的 Krea2ModelConfig / Krea2EditApply）。

- SFKrea2ModelConfig：输出 Krea2 管线用的 model_config（Krea2 文本编码器基于 Qwen、
  VAE 是 Qwen-Image 的 VAE，因此路由到 qwen 编码分支，vae_unit=8）。
- SFKrea2EditApply：给 Krea2（SingleStreamDiT）模型打参考图编辑补丁。用户只需连
  model 线——参考图 latent 通过 EditUtils 条件链（conditioning 上的
  ``reference_latents``）经 ``extra_conds`` 补丁自动流入，无需额外连线。

补丁逻辑在 sf_utils/krea2_edit.py（逐行移植），本文件只放节点壳。
"""

from ...sf_utils import krea2_edit as ke

_CATEGORY = "sfnodes/model"


class SFKrea2ModelConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Describe the image by detailing the color, shape, size, texture, "
                        "quantity, text, spatial relationships of the objects and background:"
                    ),
                    "tooltip": "Krea2 系统指令（留空回退内置推荐值）",
                }),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Krea2 模型配置：输出 {model_name:'qwen', vae_unit:8, config_for:'krea2', "
        "llama_template} 供 Krea2 Edit Text Encode 使用（Krea2 文本编码器基于 Qwen）"
    )

    def configure_model(self, instruction=None):
        if not instruction:
            instruction = ke.DEFAULT_INSTRUCTION
        config = {
            "model_name": "qwen",
            "vae_unit": 8,
            "config_for": "krea2",
        }
        config["llama_template"] = ke.get_system_prompt(instruction)
        return (config,)


class SFKrea2EditApply:
    """给 Krea2 模型打参考图条件补丁。

    序列 ``[text | target | ref₁ | …]``；target 位置 id=(0,h,w)、refₙ=(+n,h,w)；
    ``editutils`` 模式 ref token t=0 调制，``krea2edit`` 模式 ref 用真实 timestep。
    补丁经 ``model.clone()`` + ``add_object_patch`` 隔离，不影响原模型实例；
    非 Krea2 模型原样返回。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "mode": (["editutils", "krea2edit"], {
                    "default": "editutils",
                    "tooltip": "参考 token 的 timestep 约定。editutils=ref 按干净 latent "
                               "调制（t=0，UnifiedTrainer 配方）；krea2edit=ref 用真实 "
                               "timestep（ai-toolkit predict_velocity_edit），用于跑 "
                               "comfyui-krea2edit 训练的 LoRA 的原生推理几何",
                }),
                "ref_pos_match_target": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "把 ref 位置 id 中心对齐覆盖到 target token 网格。ref 与 "
                               "target 分辨率不一致时开启，否则小 ref 只对齐 target 左上角",
                }),
                "ref_kv_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "缓存参考图注意力 K/V：首步捕获每层 ref K/V，后续步与重复"
                               "生成复用（约 2x 加速）。冻结参考近似，非逐位精确",
                }),
                "ref_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "参考图约束采样多长时间。1.0=ref 全程参与；0.5=ref 只参与"
                               "进度前 50%，之后纯文生图；0.0=ref 完全不参与。需开启 "
                               "ref_kv_cache 才生效",
                }),
                "reset_cache": ("BOOLEAN", {"default": True,
                               "tooltip": "每次节点执行时清空 KV 缓存（重新捕获）"}),
                "debug_log": ("BOOLEAN", {"default": False,
                             "tooltip": "打印 KV 缓存捕获/复用日志"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "给 Krea2 模型打参考图编辑补丁：只需连 model 线，参考图 latent 经 conditioning "
        "的 reference_latents 自动流入。支持多参考图、ref 位置对齐、参考 K/V 缓存与 "
        "ref_strength 调度。补丁经 clone + add_object_patch 隔离"
    )

    def apply_patch(self, model, mode="editutils", ref_pos_match_target=True,
                    ref_kv_cache=True, ref_strength=1.0, reset_cache=True,
                    debug_log=False):
        if not ke.is_krea2_model(model):
            return (model,)

        m = model.clone()
        dit = m.get_model_object("diffusion_model")
        setattr(dit, "_editutils_ref_pos_match_target", bool(ref_pos_match_target))
        setattr(dit, "_editutils_ref_timestep_mode", str(mode))

        ke.apply_krea2_edit_patch(m)

        if ref_kv_cache:
            ke.install_ref_kv_cache_patch(m, dit, reset_cache=reset_cache,
                                          ref_strength=ref_strength,
                                          debug_log=debug_log)

        return (m,)
