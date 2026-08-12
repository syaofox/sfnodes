import os
import folder_paths
import comfy.utils
from typing import Union

from nodes import LoraLoader
from ...sf_utils.logger import get_logger
from ...sf_utils.common import AnyType

logger = get_logger(__name__)

_CATEGORY = "sfnodes/model"

any_type = AnyType("*")


class FlexibleOptionalInputType(dict):
    def __init__(self, type, data=None):
        self.type = type
        self.data = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


def get_lora_by_filename(file_path, lora_paths=None):
    lora_paths = (
        lora_paths
        if lora_paths is not None
        else folder_paths.get_filename_list("loras")
    )

    if file_path in lora_paths:
        return file_path

    lora_paths_no_ext = [os.path.splitext(x)[0] for x in lora_paths]

    file_path_force_no_ext = os.path.splitext(file_path)[0]
    if file_path_force_no_ext in lora_paths_no_ext:
        found = lora_paths[lora_paths_no_ext.index(file_path_force_no_ext)]
        return found

    # Fuzzy match: basename
    basename = os.path.basename(file_path_force_no_ext)
    for i, p in enumerate(lora_paths_no_ext):
        if os.path.basename(p) == basename:
            return lora_paths[i]

    logger.warning(f"[PowerLoraLoader] Could not find lora: {file_path}")
    return None


def _load_sd_direct(path):
    """直接读盘返回 (sd, meta)（Power 无缓存；SFLoraStack 用自己的 _get_lora）。"""
    try:
        lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    except TypeError:
        # 旧 ComfyUI：没有 return_metadata 参数。
        lora, meta = comfy.utils.load_torch_file(path, safe_load=True), None
    return lora, meta


class PowerLoraLoader:
    """A powerful, flexible node to add multiple loras to a model/clip with custom UI."""
    DESCRIPTION = (
        "功能强大的多 LoRA 加载器，支持动态槽位、权重归一化与预设输入（连接后预设优先）。"
        "节点上方的 Merge method 下拉框可切换叠加方式：Linear（线性，默认，逐行相加）"
        "或 Ortho GS（Gram-Schmidt 输入空间正交化，减少相似 LoRA 之间的干扰；"
        "行顺序即优先级——第一个 LoRA 保持原样、后续让位、可能损失幅度；"
        "仅 UNet 层正交化，CLIP 仍线性叠加）。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "开启归一化权重",
                    },
                ),
                "normalize_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "归一化目标总权重",
                    },
                ),
                "merge_method": (
                    ["linear", "ortho_gs"],
                    {
                        "default": "linear",
                        "tooltip": "叠加方式：Linear（线性，默认，逐个相加）；"
                            "Ortho GS（Gram-Schmidt 输入空间正交化，减少相似 LoRA "
                            "干扰；行顺序即优先级，第一个 LoRA 保持原样、后续让位、"
                            "可能损失幅度；仅 UNet 层正交化，CLIP 仍线性叠加）",
                    },
                ),
            },
            "optional": FlexibleOptionalInputType(
                type=any_type,
                data={
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "preset": ("SF_LORA_PRESET",),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = _CATEGORY

    def load_loras(
        self, normalize, normalize_weight, merge_method="linear",
        model=None, clip=None, preset=None, **kwargs
    ):
        # Collect enabled loras
        enabled_loras = []
        if isinstance(preset, dict) and isinstance(preset.get("loras"), list):
            # 传入预设优先：忽略 widget 配置，使用预设的顺序/强度/normalize
            normalize = bool(preset.get("normalize", normalize))
            normalize_weight = float(preset.get("normalize_weight", normalize_weight))
            for item in preset["loras"]:
                if not isinstance(item, dict) or not item.get("on"):
                    continue
                strength_model = item.get("strength", 0)
                strength_clip = item.get("strengthTwo", None)
                if strength_model != 0 or (
                    strength_clip is not None and strength_clip != 0
                ):
                    enabled_loras.append((None, item, strength_model, strength_clip))
        else:
            for key, value in kwargs.items():
                key_upper = key.upper()
                if (
                    key_upper.startswith("LORA_")
                    and isinstance(value, dict)
                    and "on" in value
                    and "lora" in value
                    and "strength" in value
                    and value["on"]
                ):
                    strength_model = value["strength"]
                    strength_clip = value.get("strengthTwo", None)
                    if strength_model != 0 or (
                        strength_clip is not None and strength_clip != 0
                    ):
                        enabled_loras.append(
                            (key, value, strength_model, strength_clip)
                        )

        if not enabled_loras:
            return (model, clip)

        # Calculate normalization
        total_weight = sum(abs(s) for _, _, s, _ in enabled_loras)

        if normalize:
            logger.info(
                f"[PowerLoraLoader] normalize=ON, weight={normalize_weight}, "
                f"total_abs_weight={total_weight:.4f}, loras={len(enabled_loras)}"
            )

        # 应用计划：[(规范文件名, path, norm_s_model, norm_s_clip)]（栈顺序）。
        # 第一项必须是 get_lora_by_filename 的规范化结果（短名/无扩展名输入已
        # 解析为列表完整条目）——官方 LoraLoader 内部 get_full_path_or_raise
        # 只做精确解析，原始短名会失败。归一化/路径解析在此统一完成，顺序与
        # ortho 两条路径共用。
        plan = []
        for key, value, strength_model, strength_clip in enabled_loras:
            lora_name = value["lora"]
            lora = get_lora_by_filename(lora_name)
            if lora is None or model is None:
                continue

            if clip is None:
                if strength_clip is not None and strength_clip != 0:
                    logger.warning(
                        "[PowerLoraLoader] Received clip strength even though no clip supplied!"
                    )
                strength_clip = 0
            else:
                strength_clip = (
                    strength_clip if strength_clip is not None else strength_model
                )

            # Apply normalization
            if normalize and total_weight > 0:
                norm_s_model = (abs(strength_model) / total_weight) * normalize_weight
                if strength_model < 0:
                    norm_s_model = -norm_s_model
                if strength_clip != 0:
                    ratio = (
                        strength_clip / strength_model if strength_model != 0 else 1.0
                    )
                    norm_s_clip = norm_s_model * ratio
                else:
                    norm_s_clip = 0
                logger.info(
                    f"[PowerLoraLoader] {lora_name}: "
                    f"raw_model={strength_model:.2f} raw_clip={strength_clip:.2f} -> "
                    f"norm_model={norm_s_model:.4f} norm_clip={norm_s_clip:.4f}"
                )
            else:
                norm_s_model = strength_model
                norm_s_clip = strength_clip

            if norm_s_model != 0 or norm_s_clip != 0:
                plan.append(
                    (lora, folder_paths.get_full_path("loras", lora),
                     norm_s_model, norm_s_clip)
                )

        if not plan:
            return (model, clip)

        # Ortho GS：≥2 行且 key map 构建成功才正交化，否则静默回落线性
        #（DuoNodes 同款兜底——ComfyUI 内部结构变化时绝不报错）。
        if merge_method == "ortho_gs" and len(plan) >= 2:
            from ...sf_utils import lora_ortho_load as OL

            result = OL.ortho_apply(model, clip, plan, _load_sd_direct)
            if result is not None:
                new_model, new_clip, _ok_paths, (ortho_keys, pass_keys) = result
                logger.info(
                    f"[PowerLoraLoader] ortho: {ortho_keys} key(s) orthogonalized, "
                    f"{pass_keys} pass-through"
                )
                return (new_model, new_clip)
            logger.info("[PowerLoraLoader] ortho key map failed; falling back to linear")

        # 线性路径：plan 里存的是 get_lora_by_filename 规范化后的文件名
        #（短名/无扩展名输入已解析为列表中的完整条目；官方 LoraLoader 内部
        # get_full_path_or_raise 只做精确解析，原始短名会失败）。
        for lora_name, _path, norm_s_model, norm_s_clip in plan:
            model, clip = LoraLoader().load_lora(
                model, clip, lora_name, norm_s_model, norm_s_clip
            )

        return (model, clip)

    @classmethod
    def get_enabled_loras_from_prompt_node(
        cls, prompt_node: dict
    ) -> list[dict[str, Union[str, float]]]:
        result = []
        for name, lora in prompt_node["inputs"].items():
            if name.startswith("lora_") and isinstance(lora, dict) and lora.get("on"):
                lora_file = get_lora_by_filename(lora["lora"])
                if lora_file is not None:
                    lora_dict = {
                        "name": lora["lora"],
                        "strength": lora["strength"],
                        "path": folder_paths.get_full_path("loras", lora_file),
                    }
                    if "strengthTwo" in lora:
                        lora_dict["strength_clip"] = lora["strengthTwo"]
                    result.append(lora_dict)
        return result


# 导入以触发 LoRA 笔记/预设 HTTP 路由注册
from ...sf_utils import lora_notes  # noqa: F401, E402
from ...sf_utils import lora_presets  # noqa: F401, E402
