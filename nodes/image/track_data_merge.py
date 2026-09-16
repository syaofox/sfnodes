"""SFTrackDataMerge：单节点完成 SAM3_TRACK_DATA 的逐路相减 + 并集叠加。

替代"`SFTrackDataSubtract` 串联 `SFTrackDataAdd`"的繁琐接线：一个基础
`track_data` + 若干动态槽 `track_1..20`，前端每槽可切换模式——
`-`（sub，逐对象相减、保留对象数）或 `+`（add，并集塌单身份）。

模式经 hidden STRING `SlotModes`（JSON，前端存 node.properties 并随工作流
保存、由 graphToPrompt 钩子注入）传入；后端按槽序收集后调纯逻辑
`sf_utils.track_data_ops.merge_track_data`（先减后加，等价串联，顺序无关）。
"""

import json

_CATEGORY = "sfnodes/image"

_MAX_SLOTS = 20


def _parse_modes(raw):
    """解析前端注入的槽模式 JSON 为 `{槽名: "add"|"sub"}`（容错，非法回退空）。"""
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): ("add" if v == "add" else "sub") for k, v in data.items()}


class SFTrackDataMerge:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"track_{i}": (
                "MASK,SAM3_TRACK_DATA",
                {"tooltip": f"第 {i} 路遮罩 / 追踪数据（未连接跳过）；模式见节点上的 -/+ 切换"},
            )
            for i in range(1, _MAX_SLOTS + 1)
        }
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA", {"tooltip": "基础追踪数据（各 -/+ 路的操作对象）"}),
            },
            "optional": optional,
            "hidden": {"SlotModes": ("STRING", {"default": "{}"})},
        }

    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    RETURN_NAMES = ("track_data",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "一个节点完成 SAM3_TRACK_DATA 的逐路相减与并集叠加（等价 SF Track Data "
        "Subtract 串联 SF Track Data Add，省去繁琐接线）。每路动态槽可切换模式："
        "- 逐对象排除该路并保留对象数，+ 把该路并集进来并合成单一身份。"
        "先应用全部 -（多路并集后逐对象相减），再有任一 + 时塌平为单身份"
        "（scores=[1.0]）；无 + 时保留对象数与 scores。输入槽随连接自动增删"
        "（初始 4，上限 20），模式随工作流保存"
    )

    def execute(self, track_data, SlotModes="{}", **kwargs):
        import torch
        from comfy.ldm.sam3.tracker import pack_masks, unpack_masks
        from ...sf_utils.track_data_ops import is_track_data, merge_track_data

        if not is_track_data(track_data):
            raise ValueError("SF Track Data Merge: 输入不是有效的 SAM3_TRACK_DATA")

        modes = _parse_modes(SlotModes)
        subtracts = []
        adds = []
        for i in range(1, _MAX_SLOTS + 1):
            value = kwargs.get(f"track_{i}")
            if value is None:
                continue
            if modes.get(f"track_{i}", "sub") == "add":
                adds.append(value)
            else:
                subtracts.append(value)

        return (merge_track_data(
            track_data,
            subtracts,
            adds,
            pack_masks=pack_masks,
            unpack_masks=unpack_masks,
            torch=torch,
            interpolate=torch.nn.functional.interpolate,
        ),)
