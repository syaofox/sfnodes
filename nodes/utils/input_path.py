"""SFInputPath：input 目录文件名 → 绝对路径（跨机器 / 启动目录无关）。

背景：VHS 的 Path 类加载节点（VHS_LoadVideoPath / VHS_LoadVideoFFmpegPath /
VHS_LoadAudio）用 `validate_path()`（`os.path.isfile`）校验路径——相对路径按
**进程 CWD** 解析。ComfyUI 从非根目录启动时 `input/xxx.mp4` 直接报
"video is not a valid path: input/xxx.mp4"（2026-10 远程实例实测：CWD=/root）。
本节点用 `folder_paths.get_annotated_filepath()` 把 input 相对名解析为绝对路径，
供这些 Path 类节点直连，使工作流与启动目录无关。

- 输入：input 目录下的文件名/子路径（`clip.mp4`、`sub/clip.mp4`），也接受
  `x [input]` 注解；已是存在的绝对路径时原样返回（normpath）。
- 输出：绝对路径 STRING；空输入输出空串（不报错，便于建图期空跑），
  解析失败（如目录穿越）抛带上下文的 ValueError。
"""

import os

import folder_paths

_CATEGORY = "sfnodes/utils"


class SFInputPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "",
                    "tooltip": "input 目录下的文件名/子路径（如 clip.mp4 / sub/clip.mp4），也接受 'x [input]' 注解；已是存在的绝对路径时原样返回",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "把 input 目录文件名解析为绝对路径（复用 folder_paths.get_annotated_filepath，"
        "含 [input] 注解与防目录穿越）；供 VHS Path 类加载节点直连，"
        "工作流不再依赖 ComfyUI 的启动目录"
    )

    def execute(self, name):
        name = (name or "").strip()
        if not name:
            return ("",)
        if os.path.isabs(name) and os.path.isfile(name):
            return (os.path.normpath(name),)
        try:
            resolved = folder_paths.get_annotated_filepath(name)
        except Exception as exc:
            raise ValueError(f"SF Input Path: 无法解析 {name!r}：{exc}") from exc
        return (resolved,)


NODE_CLASS_MAPPINGS = {"SFInputPath": SFInputPath}
NODE_DISPLAY_NAME_MAPPINGS = {"SFInputPath": "SF Input Path"}
