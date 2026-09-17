"""SFVideoConcat：把循环分段落盘的多段视频自动合并为单个文件。

分段跑长视频时 `VHS_VideoCombine` 每轮落盘一段，本节点在循环外用一次调用
完成合并（配合 `SF For Loop End` 输出的累加文件列表），避免把整段帧重新
装回内存：复用核心 Video API（`comfy_api.latest.InputImpl.VideoFromFile` /
`VideoFromList` + `Types.VideoContainer`/`VideoCodec`），各段与目标容器、
编码签名一致时纯 remux 不重编码；`audio` 输入（如 VHS_LoadVideo 的整段
音频）作为完整音轨注入（分段时各段通常不带音轨，避免音画错位）。

路径解析（兼容 VHS_FILENAMES 的 `(bool, [paths])`、SF Batch Anything 累加
后的 list/tuple 混合结构、跳过 PNG 与 -audio 中间文件）在纯逻辑
`sf_utils/video_concat.py::collect_segment_paths`（无依赖，可直测）。
"""

import os

_CATEGORY = "sfnodes/video"


class SFVideoConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("*", {"tooltip": "段视频来源：接 VHS Video Combine 的 Filenames 输出，或经 SF Batch Anything 在循环里累加后的列表（混合结构自动展平解析）"}),
                "filename_prefix": ("STRING", {"default": "Wan21_SCAIL2/final", "tooltip": "合并文件保存前缀（output 目录，自动编号不覆盖）"}),
                "format": (["mp4", "mkv", "webm"], {"default": "mp4", "tooltip": "输出容器；与段编码签名一致时纯 remux 不重编码，不一致才转码"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "可选完整音轨（如 VHS_LoadVideo 的音频输出）；分段时各段不带音轨，由这里统一注入，音轨长度过长时自动截到视频长度"}),
                "cleanup": ("BOOLEAN", {"default": False, "tooltip": "合并成功后删除各段文件（不可恢复，默认保留）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "把分段落盘的多段视频按顺序自动合并为一个文件（核心 Video API remux，不重编码），可选注入整段音轨并在合并后清理段文件"
    OUTPUT_NODE = True

    def execute(self, segments, filename_prefix="Wan21_SCAIL2/final", format="mp4", audio=None, cleanup=False):
        import folder_paths
        from comfy_api.latest import InputImpl, Types, io, ui

        from ...sf_utils.video_concat import collect_segment_paths

        paths = collect_segment_paths(segments)
        if not paths:
            raise ValueError(
                "SF Video Concat: 未从 segments 解析到任何视频文件路径（接 VHS Video Combine 的 Filenames 输出，或循环里经 SF Batch Anything 累加）"
            )
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise ValueError(f"SF Video Concat: 段文件不存在（可能已被清理或路径失效）：{missing[0]}")

        videos = [InputImpl.VideoFromFile(path) for path in paths]
        merged = InputImpl.VideoFromList(videos, complete_audio=audio, codec=Types.VideoCodec("auto"))
        width, height = merged.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        ext = Types.VideoContainer.get_extension(format)
        file = f"{filename}_{counter:05}_.{ext}"
        output_path = os.path.join(full_output_folder, file)
        merged.save_to(output_path, format=Types.VideoContainer(format), codec=Types.VideoCodec("auto"))

        if cleanup:
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        preview = ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)])
        return {"ui": preview, "result": (output_path,)}
