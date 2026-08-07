"""SFPauseImage - an inline IMAGE gate that pauses a workflow.

复刻 Pixaroma Pause Image：放在图片源与昂贵下游工作（放大、二遍、重后期）之间。
Pause 模式：工作流停在此节点并显示图片；节点把快照存到 ComfyUI temp 目录。
按 Continue 时前端把上游从提交的 prompt 中剪掉，本节点读回快照——只有下游运行，
精确用你预览过的那张图，昂贵的上游完全跳过。

决策在前端 JS（Pattern #9，与 SFPauseText 同款双钩子）：app.graphToPrompt hook 注入
生效模式到隐藏 PauseState 并在提交时剪枝。本节点只对交给它的模式做出反应。

无 IS_CHANGED（与 pause_text.py 相同理由）：NaN 会让节点缓存键折叠所有祖先，
闸门下游每次 Run 全量重跑。
"""

import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

_CATEGORY = "sfnodes/image"


# _json_safe 清洗 NaN/Inf 使 ui payload 保持合法 JSON。Save 按钮把整个 prompt
# 嵌入快照 PNG，其中任何节点的 IS_CHANGED 返回 NaN 都会贡献 `is_changed: [NaN]`，
# 不是合法 JSON——前端 JSON.parse executed 消息会抛错并丢弃整个 payload。
# 本节点已无 IS_CHANGED，但同图的 PreviewImage / XY Plot 仍可能有，清洗保留。
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return str(obj)
    return obj


def _tensor_to_pil(frame):
    """HxWxC float [0,1] tensor frame -> PIL.Image (RGB)。"""
    arr = frame.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]  # 丢弃 alpha，pause/continue 往返保持 RGB
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(pil):
    """PIL.Image -> 1xHxWxC float [0,1] tensor。"""
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _snapshot_path(node_id):
    """按节点 id 的确定性快照路径（ComfyUI temp 目录）。

    pause run 与 continue run 拿到同一个节点 id（ComfyUI 的 UNIQUE_ID 对节点
    跨 run 稳定），所以 Pause 模式写入的文件正是 Continue 模式读回的文件。
    前缀用 sf_pause_（原版 pixaroma_pause_）——若同时安装 pixaroma 插件，
    同 node_id 会撞同名文件互相覆盖。
    """
    safe = "".join(c for c in str(node_id) if c.isalnum() or c in "_-") or "node"
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"sf_pause_{safe}.png")


class SFPauseImage:
    DESCRIPTION = (
        "SF Pause Image - 内联图片闸门：在此节点处停下工作流，让你在运行昂贵的"
        "下一步（放大、二遍、重后期）之前先看看图片。把任意 IMAGE 源接入输入，"
        "把下一节点接到输出。\n\n"
        "开关在 Pause 时，按 Run 停在这里并显示图片，工作流其余部分不运行。"
        "按 Continue 只有下游运行，喂给你刚看到的那张精确图片——模型、采样器与"
        "解码被跳过，所以很快。按 Regenerate 在此处掷一张新图（采样器种子在"
        "随机化时会得到不同图片）。开关拨到 Pass 则整条工作流一次跑完。\n\n"
        "快照存在 ComfyUI 的 temp 目录，重启 ComfyUI 时被清空——重启后请先"
        "Pause 一次再使用 Continue。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Optional 而非 required：Continue 模式前端剪掉此输入链接（上游被
                # 跳过），optional 输入让节点以 image=None 运行；Pause / Pass 模式
                # 图片在场。
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "要闸门的图片。接入你的图片源；闸门放行或继续时同一张图原样从输出流出。",
                    },
                ),
            },
            "hidden": {
                # 前端 app.graphToPrompt hook 注入（Pattern #9）：
                # JSON 字符串 {"mode": "pause"|"continue"|"pass"}。
                "PauseState": ("STRING", {"default": ""}),
                "unique_id": "UNIQUE_ID",
                # Save 按钮把执行期工作流嵌入快照 PNG，保存的图片可拖回 ComfyUI
                # 重建（拖动重建用）。
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = (
        "继续往下游的图片：Pause/Pass 是实时输入，Continue 是重载的快照。",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    # 有意不设 IS_CHANGED——见 pause_text.py 的同一注释（讨论 #76 为实测案例）。
    # 曾返回 float("nan")：NaN 永不等同于自己，节点缓存键折叠每个祖先的
    # IS_CHANGED（caching.py::get_node_signature）→ 闸门之后接的一切每次 Run
    # 失效。实测对照组：EmptyImage -> PreviewImage 完美缓存，中间插本节点后
    # 两者每次重跑。
    # 去掉安全：缓存命中仍会重发 ui payload（帧照样显示）；快照文件按节点稳定
    # 命名，磁盘副本仍是 Continue 要读的那张；模式在隐藏 PauseState 输入里、
    # 已属缓存键，切换模式或上游图变化仍会重跑本节点。

    def run(self, image=None, PauseState="", unique_id=None,
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
                    "SF Pause Image: 快照已过期（ComfyUI 的 temp 目录被清空）。"
                    "请按 Run 重新 Pause，再按 Continue。"
                )
            # 防御：快照可能损坏/截断（如 ComfyUI 保存中途被杀）。任何读失败都
            # 转成清晰消息，而不是用原始 PIL 回溯炸掉整个工作流。with 块同时
            # 释放文件句柄，下次 pause run 才能覆盖同一路径（Windows 文件锁）。
            try:
                with Image.open(path) as snap:
                    out = _pil_to_tensor(snap)
            except Exception as e:
                raise RuntimeError(
                    "SF Pause Image: 快照无法读取（可能不完整）。请按 Run 重新"
                    " Pause，再按 Continue。"
                ) from e
            return {"ui": {"sf_pause_frame": frame}, "result": (out,)}

        # Pause 或 Pass：图片已接线。
        if image is None:
            raise RuntimeError(
                "SF Pause Image: 输入未连接图片。"
            )

        # 快照第一帧供 Continue 回放。（v1 快照 frame 0；大于 1 的 batch 回放
        # 其第一帧。）保存失败（temp 只读、磁盘满）不能弄崩 run——图片照常透传；
        # 只是 Continue 拿不到新快照。
        saved = False
        try:
            _tensor_to_pil(image[0]).save(path, "PNG")
            saved = True
        except OSError as e:
            print(f"[SF Pause Image] snapshot save failed: {e}")
        if saved:
            # 把执行期工作流打到 frame 上，Save 按钮可嵌入正确的种子
            # （NaN 清洗见模块注释）。只在新鲜捕获（pause/pass）时——continue 的
            # prompt 是上游被剪的版本，嵌入它无法重建图片。
            workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
            frame[0]["_sf_pause_meta"] = _json_safe({"prompt": prompt, "workflow": workflow})
        ui = {"sf_pause_frame": frame} if saved else {}
        return {"ui": ui, "result": (image,)}
