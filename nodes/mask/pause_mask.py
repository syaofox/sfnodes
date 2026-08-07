"""SFPauseMask - an inline MASK gate that pauses a workflow.

基于 SFPauseText / SFPauseImage 的闸门模式扩展：输入/输出是 MASK 张量
（[B, H, W]，与 ComfyUI 遮罩格式一致）。放在遮罩源（蒙版生成、分割、抠图）
与昂贵下游（局部重绘、精细遮罩后处理）之间。Pause 模式：工作流停在此节点并
显示遮罩；快照以灰度 PNG 存到 ComfyUI temp 目录。按 Continue 时前端把上游剪出
prompt，本节点读回快照——只有下游运行，精确用你预览过的那张遮罩。

与 SFPauseImage 的差异：快照为单通道灰度 PNG（L 模式，0-255 量化，对遮罩
可接受）；tensor 转换针对 [B,H,W] 无 C 通道的 MASK 帧。

决策在前端 JS（Pattern #9，双钩子同款）；本节点只对交给它的模式做出反应。
无 IS_CHANGED（同 pause_image.py 的理由）。
"""

import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

_CATEGORY = "sfnodes/mask"


# _json_safe 清洗 NaN/Inf 使 ui payload 保持合法 JSON（同 pause_image.py）
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
    return obj


def _mask_to_pil(frame):
    """HxW float [0,1] MASK 帧 -> PIL.Image（L 模式灰度）。

    0-255 量化：遮罩通常是二值/低精度，8bit 足够；与 ComfyUI 自身把遮罩存成
    灰度 PNG 的惯例一致。标准 MASK 帧为 [H,W]；防御非标准 [1,H,W]（部分
    节点输出带单例通道维），压平后 L 模式才接受 2D 数组。
    """
    arr = frame.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _pil_to_mask(pil):
    """PIL.Image -> 1xHxW float [0,1] MASK 张量。"""
    arr = np.array(pil.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _snapshot_path(node_id):
    """按节点 id 的确定性快照路径（ComfyUI temp 目录）。

    前缀 sf_pause_mask_（与 sf_pause_ 图片闸门隔离命名空间，语义清晰；
    节点 id 全局唯一，撞文件风险本就低，隔离是防御）。
    """
    safe = "".join(c for c in str(node_id) if c.isalnum() or c in "_-") or "node"
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"sf_pause_mask_{safe}.png")


class SFPauseMask:
    DESCRIPTION = (
        "SF Pause Mask - 内联遮罩闸门：在此节点处停下工作流，让你在运行昂贵的"
        "下一步（局部重绘、精细遮罩后处理）之前先看看遮罩。把任意 MASK 源接入"
        "输入，把下一节点接到输出。\n\n"
        "开关在 Pause 时，按 Run 停在这里并显示遮罩（灰度预览），工作流其余"
        "部分不运行。按 Continue 只有下游运行，喂给你刚看到的那张精确遮罩——"
        "上游（分割/抠图/蒙版生成）被跳过，所以很快。按 Regenerate 在此处掷"
        "一张新遮罩。开关拨到 Pass 则整条工作流一次跑完。\n\n"
        "快照以灰度 PNG 存在 ComfyUI 的 temp 目录（0-255 量化），重启 ComfyUI "
        "时被清空——重启后请先 Pause 一次再使用 Continue。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "要闸门的遮罩。接入你的遮罩源；闸门放行或继续时同一张遮罩原样从输出流出。",
                    },
                ),
            },
            "hidden": {
                # 前端 app.graphToPrompt hook 注入（Pattern #9）：
                # JSON 字符串 {"mode": "pause"|"continue"|"pass"}。
                "PauseState": ("STRING", {"default": ""}),
                "unique_id": "UNIQUE_ID",
                # Save 按钮把执行期工作流嵌入快照 PNG（与 pause_image 同款）。
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = (
        "继续往下游的遮罩：Pause/Pass 是实时输入，Continue 是重载的快照。",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    # 有意不设 IS_CHANGED——同 pause_image.py：NaN 会让节点缓存键折叠所有祖先，
    # 闸门下游每次 Run 全量重跑。模式在隐藏 PauseState 输入里、已属缓存键。

    def run(self, mask=None, PauseState="", unique_id=None,
            prompt=None, extra_pnginfo=None):
        try:
            state = json.loads(PauseState) if PauseState else {}
        except Exception:
            state = {}
        mode = state.get("mode", "pause")
        path = _snapshot_path(unique_id)

        frame = [{"filename": os.path.basename(path), "subfolder": "", "type": "temp"}]

        if mode == "continue":
            if not os.path.isfile(path):
                raise RuntimeError(
                    "SF Pause Mask: 快照已过期（ComfyUI 的 temp 目录被清空）。"
                    "请按 Run 重新 Pause，再按 Continue。"
                )
            try:
                with Image.open(path) as snap:
                    out = _pil_to_mask(snap)
            except Exception as e:
                raise RuntimeError(
                    "SF Pause Mask: 快照无法读取（可能不完整）。请按 Run 重新"
                    " Pause，再按 Continue。"
                ) from e
            return {"ui": {"sf_pause_mask_frame": frame}, "result": (out,)}

        # Pause 或 Pass：遮罩已接线。
        if mask is None:
            raise RuntimeError(
                "SF Pause Mask: 输入未连接遮罩。"
            )

        # 快照第一帧供 Continue 回放。保存失败不弄崩 run——遮罩照常透传。
        saved = False
        try:
            _mask_to_pil(mask[0]).save(path, "PNG")
            saved = True
        except OSError as e:
            print(f"[SF Pause Mask] snapshot save failed: {e}")
        if saved:
            workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
            frame[0]["_sf_pause_meta"] = _json_safe({"prompt": prompt, "workflow": workflow})
        ui = {"sf_pause_mask_frame": frame} if saved else {}
        return {"ui": ui, "result": (mask,)}
