"""SFPauseLatent - an inline LATENT gate that pauses a workflow.

放在分段采样（KSampler Advanced 两次采样）中间：把第一段采样器的 LATENT 输出
接入输入，把第二段采样器的 latent 输入接到输出。Pause 模式：工作流停在此节点
并显示预览图（把第一段采样的 VAEDecode 结果接入 image 预览输入）；节点把中间态
latent 整 batch 快照存到 ComfyUI temp 目录。按 Continue 时前端把第一段采样
（整条上游链）从提交的 prompt 中剪掉，本节点读回 latent 快照——只有下游运行，
第二段采样器从暂停时那份精确的中间态继续，第一段完全不重跑。

与 SFPauseImage 的差异：数据通道是 LATENT 而非 IMAGE，且快照是 latent 张量
（safetensors，全 batch 保存，含 noise_mask/batch_index 等张量键——继续采样
需要完整 batch 与重绘遮罩，不同于 image 闸门仅快照首帧）；预览走可选的 image
输入（VAEDecode 结果），Continue 时前端连同 latent 链接一起剪掉它，否则
VAEDecode 仍被消费会把第一段采样器拉活。

决策在前端 JS（Pattern #9，双钩子同款）；本节点只对交给它的模式做出反应。
无 IS_CHANGED（同 pause_image.py 的理由）。
"""

import json
import os

import folder_paths
import numpy as np
import safetensors.torch as sf
import torch

from .pause_image import _json_safe, _tensor_to_pil

_CATEGORY = "sfnodes/image"


def _latent_snapshot_path(node_id):
    """按节点 id 的确定性 latent 快照路径（ComfyUI temp 目录）。

    前缀 sf_pause_latent_（与 sf_pause_ 图片闸门 / sf_pause_mask_ 遮罩闸门
    隔离命名空间，语义清晰；节点 id 全局唯一，撞文件风险本就低，隔离是防御）。
    """
    safe = "".join(c for c in str(node_id) if c.isalnum() or c in "_-") or "node"
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"sf_pause_latent_{safe}.latent")


def _preview_snapshot_path(node_id):
    """预览快照路径（同 latent 快照的命名空间）。Continue 时回传给前端显示。"""
    safe = "".join(c for c in str(node_id) if c.isalnum() or c in "_-") or "node"
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"sf_pause_latent_{safe}.png")


def _save_latent(latent, path):
    """把 latent dict 存成 safetensors（与 ComfyUI SaveLatent 同格式）。

    保存 latent 中全部张量键（samples + noise_mask/batch_index 等）——继续
    采样需要完整 batch 与重绘遮罩。写入 latent_format_version_0 标记，官方
    LoadLatent 读回时 multiplier 为 1.0（旧格式无此键要除以 0.18215）。
    """
    out = {
        "latent_tensor": latent["samples"].contiguous(),
        "latent_format_version_0": torch.tensor([]),
    }
    for k, v in latent.items():
        if k != "samples" and isinstance(v, torch.Tensor):
            out[k] = v.contiguous()
    sf.save_file(out, path)


def _load_latent(path):
    """读回 _save_latent 写的快照，还原 latent dict（latent_tensor -> samples）。"""
    d = sf.load_file(path, device="cpu")
    out = {}
    for k, v in d.items():
        if k == "latent_format_version_0":
            continue
        out["samples" if k == "latent_tensor" else k] = v
    return out


class SFPauseLatent:
    DESCRIPTION = (
        "SF Pause Latent - 内联 latent 闸门：在分段采样（KSampler Advanced 两次"
        "采样）中间暂停。把第一段采样器的 LATENT 输出接入输入，把第二段采样器"
        "的 latent 输入接到输出，再把第一段采样的 VAEDecode 结果接入 image "
        "预览输入——Pause 时停在第一段采样结束处并显示预览图，Continue 时跳过"
        "第一段采样，从保存的中间态 latent 继续第二段，第一段完全不重跑。\n\n"
        "开关在 Pause 时，按 Run 停在这里并显示预览，工作流其余部分不运行。"
        "按 Continue 只有下游运行，喂给你暂停时那份精确的中间态 latent。按 "
        "Regenerate 重新采样第一段（采样器种子在随机化时会得到不同图片）。"
        "开关拨到 Pass 则整条工作流一次跑完。\n\n"
        "示例（8 步分两段，中间暂停检查）：\n"
        "  KSampler(A) [steps=8, start=0, end=4] -> 本节点 latent -> "
        "KSampler(B) [steps=8, start=4, end=8]\n"
        "  KSampler(A) -> VAEDecode -> 本节点 image（预览）\n\n"
        "快照存在 ComfyUI 的 temp 目录，重启 ComfyUI 时被清空——重启后请先"
        " Pause 一次再使用 Continue。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Optional 而非 required：Continue 模式前端剪掉此输入链接（第一段
                # 采样被跳过），optional 输入让节点以 latent=None 运行；Pause /
                # Pass 模式 latent 在场。
                "latent": (
                    "LATENT",
                    {
                        "tooltip": "要闸门的中间态 latent。接入第一段采样器的输出；闸门放行或继续时同一份 latent 原样从输出流出。",
                    },
                ),
                # 预览输入：Pause 时显示这里的图片（把第一段采样的 VAEDecode
                # 结果接进来）。Continue 时前端连同 latent 链接一起剪掉它。
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "预览输入（可选）：把第一段采样的 VAEDecode 结果接进来，暂停时在这里显示。",
                    },
                ),
            },
            "hidden": {
                # 前端 app.graphToPrompt hook 注入（Pattern #9）：
                # JSON 字符串 {"mode": "pause"|"continue"|"pass"}。
                "PauseState": ("STRING", {"default": ""}),
                "unique_id": "UNIQUE_ID",
                # Save 按钮把执行期工作流嵌入预览快照 PNG（与 pause_image 同款）。
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_TOOLTIPS = (
        "继续往下游的 latent：Pause/Pass 是实时输入，Continue 是重载的快照。",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = _CATEGORY

    # 有意不设 IS_CHANGED——同 pause_image.py：NaN 会让节点缓存键折叠所有祖先，
    # 闸门下游每次 Run 全量重跑。模式在隐藏 PauseState 输入里、已属缓存键。

    def run(self, latent=None, image=None, PauseState="", unique_id=None,
            prompt=None, extra_pnginfo=None):
        try:
            state = json.loads(PauseState) if PauseState else {}
        except Exception:
            state = {}
        mode = state.get("mode", "pause")
        lpath = _latent_snapshot_path(unique_id)
        ppath = _preview_snapshot_path(unique_id)

        if mode == "continue":
            if not os.path.isfile(lpath):
                raise RuntimeError(
                    "SF Pause Latent: 快照已过期（ComfyUI 的 temp 目录被清空）。"
                    "请按 Run 重新 Pause，再按 Continue。"
                )
            # 防御：快照可能损坏/截断（如 ComfyUI 保存中途被杀）。任何读失败都
            # 转成清晰消息，而不是用原始异常炸掉整个工作流。
            try:
                out = _load_latent(lpath)
            except Exception as e:
                raise RuntimeError(
                    "SF Pause Latent: 快照无法读取（可能不完整）。请按 Run 重新"
                    " Pause，再按 Continue。"
                ) from e
            # 预览 png 在 pause 时已存（如有 image 输入），continue 一并回传
            # 显示同一张图；被清空/从未存过则无 frame（预览早已看过，可接受）。
            if os.path.isfile(ppath):
                frame = [{"filename": os.path.basename(ppath), "subfolder": "", "type": "temp"}]
                return {"ui": {"sf_pause_latent_frame": frame}, "result": (out,)}
            return {"ui": {}, "result": (out,)}

        # Pause 或 Pass：latent 已接线。
        if latent is None:
            raise RuntimeError(
                "SF Pause Latent: 输入未连接 latent。"
            )

        # 快照中间态 latent（全 batch）供 Continue 回放。保存失败（temp 只读、
        # 磁盘满）不能弄崩 run——latent 照常透传；只是 Continue 拿不到新快照，
        # 且无预览（与 pause_image 的降级语义一致：不显示不可继续的预览）。
        ui = {}
        try:
            _save_latent(latent, lpath)
        except OSError as e:
            print(f"[SF Pause Latent] snapshot save failed: {e}")
        else:
            # 预览快照首帧：image 未接则无预览（ui 无 frame），但 latent 快照
            # 照存——暂停/继续采样不受影响。预览保存失败同样只是少一个 frame。
            if image is not None:
                frame = [{"filename": os.path.basename(ppath), "subfolder": "", "type": "temp"}]
                try:
                    _tensor_to_pil(image[0]).save(ppath, "PNG")
                except OSError as e:
                    print(f"[SF Pause Latent] preview save failed: {e}")
                else:
                    # 把执行期工作流打到 frame 上，Save 按钮可嵌入正确的种子
                    # （NaN 清洗见 pause_image 模块注释）。
                    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
                    frame[0]["_sf_pause_meta"] = _json_safe({"prompt": prompt, "workflow": workflow})
                    ui = {"sf_pause_latent_frame": frame}
        return {"ui": ui, "result": (latent,)}
