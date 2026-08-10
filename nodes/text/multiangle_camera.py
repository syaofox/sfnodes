import random

from comfy.comfy_types.node_typing import IO

_CATEGORY = "sfnodes/text"
_RANDOM = "随机"

H_DIRECTIONS = [
    "front view",
    "front-right quarter view",
    "right side view",
    "back-right quarter view",
    "back view",
    "back-left quarter view",
    "left side view",
    "front-left quarter view",
]

V_DIRECTIONS = [
    "low-angle shot",
    "eye-level shot",
    "elevated shot",
    "high-angle shot",
]

DISTANCES = [
    "wide shot",
    "medium shot",
    "close-up",
]

_SKS = "<sks>"


def _resolve_choice(selection, options):
    if selection == _RANDOM:
        return random.choice(options)
    return selection


class SFMultiangleCamera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "horizontal_direction": ([_RANDOM] + H_DIRECTIONS, {
                    "default": "front view",
                    "tooltip": "镜头水平方向：随机 / 指定方向",
                }),
                "vertical_direction": ([_RANDOM] + V_DIRECTIONS, {
                    "default": "eye-level shot",
                    "tooltip": "镜头垂直角度：随机 / 指定角度",
                }),
                "distance": ([_RANDOM] + DISTANCES, {
                    "default": "medium shot",
                    "tooltip": "镜头景别（距离）：随机 / 指定景别",
                }),
                "add_sks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否在提示词前加 <sks> 触发词（Qwen Image Edit 专用）",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "相机角度组合：选择水平方向/垂直角度/景别（可选随机），输出组合后的镜头提示词"

    @classmethod
    def IS_CHANGED(cls, horizontal_direction, vertical_direction, distance, add_sks=True):
        if _RANDOM in (horizontal_direction, vertical_direction, distance):
            return random.random()
        return (horizontal_direction, vertical_direction, distance)

    def execute(self, horizontal_direction, vertical_direction, distance, add_sks=True):
        h_direction = _resolve_choice(horizontal_direction, H_DIRECTIONS)
        v_direction = _resolve_choice(vertical_direction, V_DIRECTIONS)
        distance = _resolve_choice(distance, DISTANCES)
        joined = f"{h_direction} {v_direction} {distance}"
        if add_sks:
            return (f"{_SKS} {joined}",)
        return (joined,)
