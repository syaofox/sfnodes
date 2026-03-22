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
            "required": {},
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

    def load_loras(self, model=None, clip=None, **kwargs):
        for key, value in kwargs.items():
            key_upper = key.upper()
            if (
                key_upper.startswith("LORA_")
                and isinstance(value, dict)
                and "on" in value
                and "lora" in value
                and "strength" in value
            ):
                strength_model = value["strength"]
                strength_clip = value.get("strengthTwo", None)
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
                if value["on"] and (strength_model != 0 or strength_clip != 0):
                    lora = get_lora_by_filename(value["lora"])
                    if model is not None and lora is not None:
                        model, clip = LoraLoader().load_lora(
                            model, clip, lora, strength_model, strength_clip
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
