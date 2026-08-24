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
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "添加到提示词（及每条组合）前面的文本",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "tooltip": "添加到提示词（及每条组合）后面的文本",
                }),
                "ordered": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "combinations 是否按固定顺序（h→v→d）输出；关闭时随机打乱",
                }),
            },
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("prompt", "combinations")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "相机角度组合：选择水平方向/垂直角度/景别（可选随机），可加前后缀，输出组合后的镜头提示词；combinations 输出全部 8x4x3=96 种组合的字符串列表（默认随机打乱，ordered 打开后按 h→v→d 固定顺序）"

    @classmethod
    def IS_CHANGED(cls, horizontal_direction, vertical_direction, distance, add_sks=True, prefix="", suffix="", ordered=False):
        if not ordered or _RANDOM in (horizontal_direction, vertical_direction, distance):
            return random.random()
        return (horizontal_direction, vertical_direction, distance, prefix, suffix)

    def execute(self, horizontal_direction, vertical_direction, distance, add_sks=True, prefix="", suffix="", ordered=False):
        h_direction = _resolve_choice(horizontal_direction, H_DIRECTIONS)
        v_direction = _resolve_choice(vertical_direction, V_DIRECTIONS)
        distance = _resolve_choice(distance, DISTANCES)
        trigger = f"{_SKS} " if add_sks else ""
        joined = f"{prefix}{trigger}{h_direction} {v_direction} {distance}{suffix}"
        combinations = [
            f"{prefix}{trigger}{h} {v} {d}{suffix}"
            for h in H_DIRECTIONS
            for v in V_DIRECTIONS
            for d in DISTANCES
        ]
        if not ordered:
            random.shuffle(combinations)
        return (joined, combinations)
