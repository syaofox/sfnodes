import os
import folder_paths
from typing import Union

from nodes import LoraLoader
from ...sf_utils.logger import get_logger

logger = get_logger(__name__)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


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

    logger.warning(f"[SFPowerLoraLoader] Could not find lora: {file_path}")
    return None


class SFPowerLoraLoader:
    """A powerful, flexible node to add multiple loras to a model/clip with custom UI."""

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
            },
            "optional": FlexibleOptionalInputType(
                type=any_type,
                data={
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    def load_loras(self, normalize, normalize_weight, model=None, clip=None, **kwargs):
        # Collect enabled loras
        enabled_loras = []
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
                    enabled_loras.append((key, value, strength_model, strength_clip))

        if not enabled_loras:
            return (model, clip)

        # Calculate normalization
        total_weight = sum(abs(s) for _, _, s, _ in enabled_loras)

        if normalize:
            logger.info(
                f"[SFPowerLoraLoader] normalize=ON, weight={normalize_weight}, "
                f"total_abs_weight={total_weight:.4f}, loras={len(enabled_loras)}"
            )

        for key, value, strength_model, strength_clip in enabled_loras:
            lora_name = value["lora"]
            lora = get_lora_by_filename(lora_name)
            if lora is None or model is None:
                continue

            if clip is None:
                if strength_clip is not None and strength_clip != 0:
                    logger.warning(
                        "[SFPowerLoraLoader] Received clip strength even though no clip supplied!"
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
                    f"[SFPowerLoraLoader] {lora_name}: "
                    f"raw_model={strength_model:.2f} raw_clip={strength_clip:.2f} -> "
                    f"norm_model={norm_s_model:.4f} norm_clip={norm_s_clip:.4f}"
                )
            else:
                norm_s_model = strength_model
                norm_s_clip = strength_clip

            if norm_s_model != 0 or norm_s_clip != 0:
                model, clip = LoraLoader().load_lora(
                    model, clip, lora, norm_s_model, norm_s_clip
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
