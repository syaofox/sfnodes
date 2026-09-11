"""SFConditioningConcat —— 多路 Conditioning 拼接。

等价于多个原生 Conditioning (Concat) 串联：原生 concat 把 from 首条
conditioning 沿 dim=1 拼到 to 的每一条之后（`torch.cat((t1, cond_from), 1)`，
to 各条的 dict 上下文逐条保留）。本节点把该语义推广到 N 路：slot1
（conditioning_1）为被拼接方逐条保留，conditioning_2..N 为拼接源，各取
首条沿 dim=1 依次拼接到每条 to 之后——与链式串联多个原生 Concat 等价
（dim1 拼接满足结合律，一次拼完还少了中间临时张量）。

动态槽位由前端 web/sf_conditioning_concat.js 管理（复用
sf_dynamic_slots.installDynamicSlots，初始 2、上限 20，同 combine）。
灵活 optional schema、槽名前缀、槽数上下限、槽序排序键全部复用
conditioning_combine 模块（语义同为 conditioning_N 动态多槽）。

与原生逐行对齐的取舍（经确认）：
- 某路 from 含多条时仅取首条并 logging.warning 点名该槽（原生同款警告）。
- conditioning_1 未连接时抛 ValueError 明示（原生此时是晦涩 TypeError）。
- embedding 维度不一致时由 torch.cat 原生抛错，不发明 padding。
- pooled_output 等 dict 内容沿用 to 方原样（原生不做加权混合）。
"""

import logging

import torch

from .conditioning_combine import (
    CONDITIONING_PREFIX,
    _ConditioningCombineInputs,
    conditioning_slot_key,
)

_CATEGORY = "sfnodes/model"


class SFConditioningConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _ConditioningCombineInputs(),
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "多路 Conditioning 拼接：等价于多个原生 Conditioning (Concat) 串联。"
        "conditioning_1 为被拼接方（逐条保留），conditioning_2 起为拼接源"
        "（各取首条沿 dim=1 依次拼接）。连上即用，输入槽会自动增删"
        "（初始 2，上限 20），未连接的拼接源自动跳过"
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # conditioning_N 槽由前端动态增删，超出静态 schema 时接管校验
        return True

    def execute(self, **kwargs):
        conditioning_to = kwargs.get("conditioning_1")
        if conditioning_to is None:
            raise ValueError(
                "[SFConditioningConcat] conditioning_1（被拼接方）未连接"
            )
        segments = []
        for key in sorted(kwargs, key=conditioning_slot_key):
            if key == "conditioning_1" or not key.startswith(CONDITIONING_PREFIX):
                continue
            value = kwargs[key]
            if value is None or len(value) == 0:
                continue
            if len(value) > 1:
                logging.warning(
                    "[SFConditioningConcat] %s 含 %d 条 conditioning，"
                    "仅首条参与拼接（与原生 ConditioningConcat 一致）",
                    key, len(value),
                )
            segments.append(value[0][0])
        out = [
            [torch.cat([cond] + segments, dim=1), ctx.copy()]
            for cond, ctx in conditioning_to
        ]
        return (out,)


__all__ = ["SFConditioningConcat"]
